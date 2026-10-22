"""Reserve fresh source groups before selecting an improved scorer."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import polars as pl
import yaml
from fastembed import TextEmbedding

from arka.distillation import input_hash, record_text
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)

ROOT = Path(os.environ.get("ARKA_V2_EVAL_DIR", "/tmp/arka-165-v2"))
OLD = Path(os.environ.get("ARKA_BASE_EVAL_DIR", "/tmp/arka-165-real-eval"))
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/arka-165-embedding-cache"


def prepare():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Evaluation already prepared; use a fresh ARKA_V2_EVAL_DIR")
    old = OutputWriter().read_parquet(OLD / "unjudged.parquet")
    old_trees = {r.source.doc_id for r in old}
    old_texts = {input_hash(r) for r in old}
    messages = pl.read_parquet("/tmp/arka-165-oasst1-train.parquet").to_dicts()
    lookup = {m["message_id"]: m for m in messages}
    tokenizer = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", threads=4
    ).model.tokenizer
    tokenizer.no_truncation()
    eligible = []
    for m in messages:
        parent = lookup.get(m["parent_id"])
        if (
            m["message_tree_id"] in old_trees
            or m["role"] != "assistant"
            or m["lang"] != "en"
            or m["deleted"]
            or m["synthetic"]
            or not parent
            or parent["parent_id"] is not None
            or parent["lang"] != "en"
        ):
            continue
        labels = (
            dict(
                zip(
                    m["labels"]["name"],
                    zip(m["labels"]["value"], m["labels"]["count"], strict=True),
                    strict=True,
                )
            )
            if m["labels"]
            else {}
        )
        if (
            "quality" not in labels
            or labels["quality"][1] < 2
            or labels.get("pii", (0, 0))[0] > 0
        ):
            continue
        payload = ConversationPayload(
            instruction=parent["text"].strip(), response=m["text"].strip()
        )
        r = ConversationRecord(
            id=m["message_id"],
            content_hash=content_hash(payload),
            source=RecordSource(type="oasst1", doc_id=m["message_tree_id"]),
            lineage=RecordLineage(
                root_id=m["message_tree_id"], parent_ids=[m["parent_id"]]
            ),
            payload=payload,
            scores=RecordScores(),
            config_hash="arka-165-v2",
            created_at=m["created_date"],
        )
        if (
            input_hash(r) in old_texts
            or len(tokenizer.encode(record_text(r)).ids) > 240
        ):
            continue
        eligible.append((r, labels["quality"]))
    grouped = {}
    for r, h in eligible:
        grouped.setdefault(r.source.doc_id, []).append((r, h))
    ordered = sorted(
        grouped,
        key=lambda t: hashlib.sha256(("arka-165-v2-reserve:" + t).encode()).hexdigest(),
    )
    final = []
    final_trees = set()
    for t in ordered:
        final_trees.add(t)
        final.extend(sorted(grouped[t], key=lambda x: x[0].id))
        if len(final) >= 600:
            break
    final = final[:600]
    final_hashes = {input_hash(r) for r, h in final}
    development = sorted(
        [
            (r, h)
            for r, h in eligible
            if r.source.doc_id not in final_trees and input_hash(r) not in final_hashes
        ],
        key=lambda x: hashlib.sha256(
            ("arka-165-v2-development:" + x[0].id).encode()
        ).hexdigest(),
    )[:1800]
    assert len(final) == 600 and len(development) == 1800
    for name, selection in [("development", development), ("final", final)]:
        directory = ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((OLD / "config.yaml").read_text())
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        selected_records = [r for r, h in selection]
        human = {
            r.id: {"quality": h[0], "rater_count": h[1], "tree_id": r.source.doc_id}
            for r, h in selection
        }
        if name == "development":
            selected_records = old + selected_records
            human.update(json.loads((OLD / "human_labels.json").read_text()))
            (directory / "teacher.jsonl").write_text(
                (OLD / "teacher.jsonl").read_text()
            )
        OutputWriter().write_parquet(selected_records, directory / "unjudged.parquet")
        (directory / "human_labels.json").write_text(json.dumps(human, indent=2))
    protocol = {
        "old_600_role": "All original records, including the original holdout, are development data now",
        "development_count": 2400,
        "new_development_labels": 1800,
        "fresh_holdout_count": 600,
        "fresh_holdout_selected_before_model_choice": True,
        "fresh_holdout_labeling": "Only after selected feature format and alpha are written to locked_model_choice.json",
        "split_unit": "Conversation trees plus identical input hashes",
        "selection": "Tree SHA-256 arka-165-v2-reserve then message id, first 600 final records; development SHA-256 arka-165-v2-development first 1800 outside reserved trees",
        "holdout_ids": [r.id for r, h in final],
        "development_ids": [r.id for r in old] + [r.id for r, h in development],
        "final_trees": len({r.source.doc_id for r, h in final}),
        "development_trees": len(old_trees | {r.source.doc_id for r, h in development}),
        "eligibility": "Same English human first-turn, >=2 human raters, no positive PII annotation, <=240 encoder tokens as first evaluation",
        "candidate_selection": "Five group folds on development teacher labels only; maximize out-of-fold Spearman, break ties by lower complexity; no holdout tuning",
        "primary_metric": "Fresh holdout Spearman with LLM teacher; compare old representation retrained on same development records",
        "secondary_metrics": [
            "Human quality Spearman",
            "Top-half recall of human quality>=0.75",
            "RMSE and MAE",
            "Observed total judge requests, known cost, and elapsed time",
        ],
        "bootstrap": "1000 holdout tree resamples seed 1652; paired difference against retrained old representation",
        "pass_criterion": "Positive paired improvement in teacher rank agreement with 95% interval above zero; student human-good recall no more than 5 percentage points below direct teacher point estimate. Broader issue deployment quality remains separately judged.",
    }
    assert not set(protocol["holdout_ids"]) & set(protocol["development_ids"])
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(
        json.dumps(
            {k: v for k, v in protocol.items() if not k.endswith("ids")}, indent=2
        ),
        flush=True,
    )


def collect(directory):
    if directory == "final":
        assert (ROOT / "locked_model_choice.json").exists(), (
            "Model choice is not locked"
        )
    spec = importlib.util.spec_from_file_location(
        "original_evaluator", Path.cwd() / "docs/verification/issue-165/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT / directory
    module.llm_judge()
    p = ROOT / directory
    records = OutputWriter().read_parquet(p / "unjudged.parquet")
    labels = {r["id"]: r["label"] for r in module.read_lines("teacher.jsonl")}
    missing = [r.id for r in records if r.id not in labels]
    (p / "label_coverage.json").write_text(
        json.dumps(
            {
                "selected": len(records),
                "valid_labels": len(labels),
                "missing_ids": missing,
                "policy": "Only valid labels enter training or scored comparisons; failures remain explicitly recorded, with no substitute scores",
            },
            indent=2,
        )
    )
    judged = []
    for r in records:
        if r.id not in labels:
            continue
        label = labels[r.id]
        judged.append(
            r.model_copy(
                update={
                    "scores": RecordScores(
                        quality=label["overall"],
                        quality_per_dim=label["scores"],
                        rubric_hash=label["rubric_hash"],
                        rubric_version=label["rubric_version"],
                        judge_model=label["judge_model"],
                        judge_prompt_hash=label["judge_prompt_hash"],
                    )
                }
            )
        )
    OutputWriter().write_parquet(judged, p / "judged.parquet")
    print(
        f"Label coverage: {len(judged)}/{len(records)}; missing IDs remain in label_coverage.json",
        flush=True,
    )


def decision():
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from arka.decisions.client import DecisionClient, DecisionClientError
    from arka.decisions.models import DecisionConfig
    from arka.labeling.rubric import RubricLoader
    from arka.pipeline.decision_stages import record_state, rubric_questions

    assert (ROOT / "locked_model_choice.json").exists()
    p = ROOT / "final"
    records = OutputWriter().read_parquet(p / "unjudged.parquet")
    done = (
        {
            json.loads(line)["id"]
            for line in (p / "decision.jsonl").read_text().splitlines()
        }
        if (p / "decision.jsonl").exists()
        else set()
    )
    cfg = DecisionConfig(
        provider="openrouter",
        model="openai/gpt-6-luna-decisions",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/alpha",
        timeout_seconds=30,
        max_retries=1,
    )
    questions = rubric_questions(
        RubricLoader().load(Path.cwd() / "rubrics/sft_quality.yaml")
    )
    import threading

    lock = threading.Lock()

    def append(name, row):
        with lock, (p / name).open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    started = time.perf_counter()
    with DecisionClient(cfg) as client:

        def observed(response):
            response.read()
            try:
                raw = response.json()
            except ValueError:
                raw = {}
            append(
                "decision_http.jsonl",
                {
                    "status": response.status_code,
                    "usage": raw.get("usage", {}),
                    "error": raw.get("error"),
                    "model": raw.get("model"),
                },
            )

        client._http.event_hooks["response"] = [observed]

        def label(r):
            start = time.perf_counter()
            try:
                result = client.decide(record_state(r), questions)
                append(
                    "decision.jsonl",
                    {
                        "id": r.id,
                        "result": result.model_dump(mode="json"),
                        "elapsed_s": time.perf_counter() - start,
                    },
                )
            except DecisionClientError as exc:
                append("decision_errors.jsonl", {"id": r.id, "error": str(exc)})

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(label, r) for r in records if r.id not in done]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 100 == 0:
                    print(
                        f"Decision attempted {i}/{len(records) - len(done)}", flush=True
                    )
    with (ROOT / "timings.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "operation": "direct_decision_collection",
                    "seconds": time.perf_counter() - started,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "decision":
        decision()
    else:
        collect(sys.argv[1])
