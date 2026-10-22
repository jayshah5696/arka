"""Bounded real-data evaluation for Arka #165. Raw data stays outside the repo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import yaml
from fastembed import TextEmbedding
from openai import OpenAI

from arka.config.loader import ConfigLoader
from arka.decisions.client import DecisionClient, DecisionClientError
from arka.decisions.models import DecisionConfig
from arka.distillation import DistilledModel, _spearman, input_hash, record_text
from arka.labeling.judges import SingleLLMJudge
from arka.labeling.rubric import RubricLoader
from arka.llm.client import LLMClient
from arka.pipeline.decision_stages import record_state, rubric_questions
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)

ROOT = Path(os.environ.get("ARKA_EVAL_DIR", "/tmp/arka-165-real-eval"))
REPO = Path(os.environ.get("ARKA_EVAL_REPO", str(Path.cwd())))
MODEL = "google/gemini-3.1-flash-lite-preview"
DATA_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/arka-165-embedding-cache"
ROOT.mkdir(exist_ok=True)
LOCK = threading.Lock()
LOCAL = threading.local()


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def append(name, data):
    with LOCK, (ROOT / name).open("a") as handle:
        handle.write(json.dumps(data, allow_nan=False) + "\n")


def read_lines(name):
    path = ROOT / name
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def prepare():
    data_path = Path("/tmp/arka-165-oasst1-train.parquet")
    if not data_path.exists():
        response = httpx.get(DATA_URL, follow_redirects=True, timeout=90)
        response.raise_for_status()
        data_path.write_bytes(response.content)
    messages = pl.read_parquet(data_path).to_dicts()
    lookup = {m["message_id"]: m for m in messages}
    encoder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = encoder.model.tokenizer
    tokenizer.no_truncation()
    candidates = []
    for m in messages:
        parent = lookup.get(m["parent_id"])
        if (
            m["role"] != "assistant"
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
        record = ConversationRecord(
            id=m["message_id"],
            content_hash=content_hash(payload),
            source=RecordSource(type="oasst1", doc_id=m["message_tree_id"]),
            lineage=RecordLineage(
                root_id=m["message_tree_id"], parent_ids=[m["parent_id"]]
            ),
            payload=payload,
            scores=RecordScores(),
            config_hash="oasst1-real-eval-v1",
            created_at=m["created_date"],
        )
        length = len(tokenizer.encode(record_text(record)).ids)
        if length > 240:
            continue
        candidates.append((record, labels["quality"], length))
    candidates.sort(
        key=lambda item: hashlib.sha256(
            ("arka-165-v1:" + item[0].id).encode()
        ).hexdigest()
    )
    selected = candidates[:600]
    assert len(selected) == 600
    OutputWriter().write_parquet([r for r, _, _ in selected], ROOT / "unjudged.parquet")
    save(
        "human_labels.json",
        {
            r.id: {
                "quality": label[0],
                "rater_count": label[1],
                "tree_id": r.source.doc_id,
                "tokens": length,
            }
            for r, label, length in selected
        },
    )
    config = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": MODEL,
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 40,
            "max_retries": 2,
        },
        "executor": {"mode": "threadpool", "max_workers": 6},
        "pipeline": [],
        "embeddings": {
            "provider": "huggingface",
            "model": "all-MiniLM-L6-v2",
            "batch_size": 64,
        },
        "output": {"format": "jsonl", "path": "ranked.jsonl"},
    }
    (ROOT / "config.yaml").write_text(yaml.safe_dump(config))
    catalog = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()[
        "data"
    ]
    teacher = next(m for m in catalog if m["id"] == MODEL)
    save(
        "protocol.json",
        {
            "dataset": "OpenAssistant/oasst1",
            "dataset_url": DATA_URL,
            "download_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "eligible_records": len(candidates),
            "sample_count": 600,
            "selection": "SHA-256 order of arka-165-v1:<message_id>, first 600, no score stratification",
            "eligibility": "English human first-turn answers, >=2 quality raters, no annotated PII, <=240 encoder tokens",
            "teacher": MODEL,
            "teacher_pricing_usd_per_token": teacher["pricing"],
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "alpha": 1.0,
            "split": "Arka deterministic 80/20 connected source/lineage/text groups",
            "primary_metric": "Spearman agreement with held-out teacher rubric quality",
            "secondary_metrics": [
                "MAE vs teacher",
                "RMSE vs teacher",
                "Spearman vs existing human quality",
                "top-half human quality>=0.75 precision and recall",
                "teacher top-half overlap",
                "cost and elapsed time",
            ],
            "human_good_threshold": 0.75,
            "keep_fraction": 0.5,
            "uncertainty": "1000 bootstrap samples of holdout conversation trees, seed 165",
            "tuning": "No hyperparameter selection on holdout",
            "scope_limit": "Short English first-turn records from public data; not representative of all Arka runs",
        },
    )
    print(
        json.dumps(
            {
                "prepared": len(selected),
                "eligible": len(candidates),
                "trees": len({r.source.doc_id for r, _, _ in selected}),
                "human_mean": float(np.mean([q[0] for _, q, _ in selected])),
            }
        ),
        flush=True,
    )


def tracked_response(response):
    response.read()
    try:
        body = response.json()
    except ValueError:
        body = {}
    append(
        "http_usage.jsonl",
        {
            "record_id": getattr(LOCAL, "record_id", None),
            "status": response.status_code,
            "id": body.get("id"),
            "model": body.get("model"),
            "usage": body.get("usage", {}),
            "path": response.request.url.path,
        },
    )
    if getattr(LOCAL, "record_id", None):
        append("teacher_raw.jsonl", {"record_id": LOCAL.record_id, "response": body})


def llm_judge(limit=None):
    cfg = ConfigLoader().load(ROOT / "config.yaml")
    transport = httpx.Client(timeout=40, event_hooks={"response": [tracked_response]})
    client = LLMClient(
        cfg.llm,
        client_factory=lambda c: OpenAI(
            api_key=c.api_key.get_secret_value(),
            base_url=str(c.base_url),
            timeout=40,
            max_retries=0,
            http_client=transport,
        ),
    )
    judge = SingleLLMJudge(client)
    rubric = RubricLoader().load(REPO / "rubrics/sft_quality.yaml")
    records = OutputWriter().read_parquet(ROOT / "unjudged.parquet")
    completed = {r["id"] for r in read_lines("teacher.jsonl")}
    todo = [r for r in records if r.id not in completed]
    if limit:
        todo = todo[:limit]

    def label(r):
        LOCAL.record_id = r.id
        started = time.perf_counter()
        try:
            result = judge.label(r.payload.instruction, r.payload.response, rubric)
            assert all(1 <= v <= 5 for v in result.scores.values())
        except Exception as exc:
            append(
                "teacher_errors.jsonl",
                {"id": r.id, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        append(
            "teacher.jsonl",
            {
                "id": r.id,
                "label": result.model_dump(mode="json"),
                "elapsed_s": time.perf_counter() - started,
            },
        )
        return r.id

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(label, r) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    print(f"teacher failure: {type(exc).__name__}: {exc}", flush=True)
                if i % 25 == 0 or i == len(todo):
                    print(
                        f"teacher {len(completed) + i}/{len(records)}, batch elapsed {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        transport.close()
        append(
            "timings.jsonl",
            {
                "operation": "teacher",
                "requested": len(todo),
                "seconds": time.perf_counter() - started,
            },
        )
    labels = {r["id"]: r["label"] for r in read_lines("teacher.jsonl")}
    if len(labels) == len(records):
        judged = []
        for r in records:
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
        OutputWriter().write_parquet(judged, ROOT / "judged.parquet")


def decision_probe():
    record = OutputWriter().read_parquet(ROOT / "unjudged.parquet")[0]
    rubric = RubricLoader().load(REPO / "rubrics/sft_quality.yaml")
    for model in ["typesafe/jev-router", "openai/gpt-6-luna-decisions"]:
        cfg = DecisionConfig(
            provider="openrouter",
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/alpha",
            timeout_seconds=30,
            max_retries=1,
        )
        with DecisionClient(cfg) as client:
            try:
                started = time.perf_counter()
                result = client.decide(record_state(record), rubric_questions(rubric))
                save(
                    "decision_probe.json",
                    {
                        "status": "available",
                        "model": model,
                        "elapsed_s": time.perf_counter() - started,
                        "result": result.model_dump(mode="json"),
                    },
                )
                print("Decision baseline available: " + model, flush=True)
                return
            except DecisionClientError as exc:
                append("decision_errors.jsonl", {"model": model, "error": str(exc)})
                print("Decision probe: " + str(exc), flush=True)
    save(
        "decision_probe.json",
        {"status": "unavailable", "errors": read_lines("decision_errors.jsonl")},
    )


def decision_judge():
    probe = json.loads((ROOT / "decision_probe.json").read_text())
    if probe["status"] != "available":
        return
    records = OutputWriter().read_parquet(ROOT / "unjudged.parquet")
    if not (ROOT / "decision.jsonl").exists():
        append(
            "decision.jsonl",
            {
                "id": records[0].id,
                "result": probe["result"],
                "elapsed_s": probe["elapsed_s"],
                "pilot": True,
            },
        )
    completed = {r["id"] for r in read_lines("decision.jsonl")}
    cfg = DecisionConfig(
        provider="openrouter",
        model=probe["model"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/alpha",
        timeout_seconds=30,
        max_retries=2,
    )
    rubric = RubricLoader().load(REPO / "rubrics/sft_quality.yaml")
    questions = rubric_questions(rubric)
    started = time.perf_counter()
    with DecisionClient(cfg) as client:

        def tracked(response):
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            append(
                "decision_http_usage.jsonl",
                {
                    "status": response.status_code,
                    "usage": payload.get("usage", {}),
                    "model": payload.get("model"),
                    "error": payload.get("error"),
                },
            )

        client._http.event_hooks["response"] = [tracked]

        def label(r):
            start = time.perf_counter()
            result = client.decide(record_state(r), questions)
            append(
                "decision.jsonl",
                {
                    "id": r.id,
                    "result": result.model_dump(mode="json"),
                    "elapsed_s": time.perf_counter() - start,
                },
            )

        todo = [r for r in records if r.id not in completed]
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(label, r) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 25 == 0 or i == len(todo):
                    print(
                        f"decision {len(completed) + i}/600, batch elapsed {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    append(
        "timings.jsonl",
        {
            "operation": "decision",
            "requested": len(todo),
            "seconds": time.perf_counter() - started,
        },
    )


def command(name, args):
    started = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", *args],
        cwd=REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    (ROOT / f"{name}.log").write_text(result.stdout + result.stderr)
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stdout + result.stderr
    append("timings.jsonl", {"operation": name, "seconds": elapsed})
    print(f"{name}: {elapsed:.3f}s", flush=True)


def fit_and_infer():
    model_path = ROOT / "model.json"
    if not model_path.exists():
        command(
            "training",
            [
                "arka-train-scorer",
                "--config",
                str(ROOT / "config.yaml"),
                "--records",
                str(ROOT / "judged.parquet"),
                "--output",
                str(model_path),
                "--features",
                "instruction_response_json_v1",
            ],
        )
    model = DistilledModel.model_validate_json(model_path.read_text())
    validation = set(model.validation_ids)
    records = OutputWriter().read_parquet(ROOT / "judged.parquet")
    holdout = [r for r in records if r.id in validation]
    (ROOT / "holdout.jsonl").write_text(
        "".join(json.dumps(r.export_payload()) + "\n" for r in holdout)
    )
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["pipeline"] = [
        {"type": "seed_source", "path": "holdout.jsonl"},
        {"type": "normalize_conversation"},
        {
            "type": "distilled_ridge",
            "model_path": "model.json",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        },
        {
            "type": "select",
            "target_count": len(holdout) // 2,
            "weights": {"distilled_ridge": 1},
        },
    ]
    (ROOT / "inference.yaml").write_text(yaml.safe_dump(cfg))
    command(
        "inference",
        ["arka", "--config", str(ROOT / "inference.yaml"), "--run-id", "real-holdout"],
    )
    command(
        "resume",
        [
            "arka",
            "--config",
            str(ROOT / "inference.yaml"),
            "--run-id",
            "real-holdout",
            "--resume",
        ],
    )


def metrics(teacher, student, human):
    keep = max(1, len(teacher) // 2)
    selected = np.argsort(-student, kind="stable")[:keep]
    teacher_selected = np.argsort(-teacher, kind="stable")[:keep]
    good = human >= 0.75
    return {
        "spearman_teacher_student": _spearman(teacher, student),
        "mae_teacher_student": float(np.mean(np.abs(teacher - student))),
        "rmse_teacher_student": float(np.sqrt(np.mean((teacher - student) ** 2))),
        "spearman_human_student": _spearman(human, student),
        "spearman_human_teacher": _spearman(human, teacher),
        "student_top_half_human_good_precision": float(np.mean(good[selected])),
        "student_top_half_human_good_recall": float(
            np.sum(good[selected]) / np.sum(good)
        )
        if np.sum(good)
        else None,
        "teacher_top_half_human_good_precision": float(np.mean(good[teacher_selected])),
        "teacher_top_half_human_good_recall": float(
            np.sum(good[teacher_selected]) / np.sum(good)
        )
        if np.sum(good)
        else None,
        "top_half_overlap": len(set(selected) & set(teacher_selected)) / keep,
        "all_human_good_fraction": float(np.mean(good)),
    }


def analyze():
    model = DistilledModel.model_validate_json((ROOT / "model.json").read_text())
    labels = json.loads((ROOT / "human_labels.json").read_text())
    judged = OutputWriter().read_parquet(ROOT / "judged.parquet")
    validation = set(model.validation_ids)
    holdout = [r for r in judged if r.id in validation]
    training = [r for r in judged if r.id not in validation]
    assert not {r.source.doc_id for r in training} & {r.source.doc_id for r in holdout}
    assert not {input_hash(r) for r in training} & {input_hash(r) for r in holdout}
    scored = OutputWriter().read_parquet(
        ROOT / "runs/real-holdout/stages/02s_distilled_ridge/data.parquet"
    )
    predictions = {input_hash(r): r.scores.distilled_ridge for r in scored}
    teacher = np.array([r.scores.quality for r in holdout])
    student = np.array([predictions[input_hash(r)] for r in holdout])
    human = np.array([labels[r.id]["quality"] for r in holdout])
    result = metrics(teacher, student, human)
    mean = np.mean([r.scores.quality for r in training])
    result["training_mean_baseline_mae"] = float(np.mean(np.abs(teacher - mean)))
    result["training_mean_baseline_rmse"] = float(
        np.sqrt(np.mean((teacher - mean) ** 2))
    )
    groups = {}
    for i, r in enumerate(holdout):
        groups.setdefault(r.source.doc_id, []).append(i)
    grouped = list(groups.values())
    rng = np.random.default_rng(165)
    samples = []
    for _ in range(1000):
        ids = [
            i for g in rng.integers(0, len(grouped), len(grouped)) for i in grouped[g]
        ]
        samples.append(metrics(teacher[ids], student[ids], human[ids]))
    intervals = {
        k: [
            float(v)
            for v in np.percentile(
                [m[k] for m in samples if m[k] is not None], [2.5, 97.5]
            )
        ]
        for k in result
        if k in samples[0] and any(m[k] is not None for m in samples)
    }
    usage = read_lines("http_usage.jsonl")
    successful = [r for r in usage if r["status"] == 200]
    costs = [r["usage"].get("cost", r["usage"].get("total_cost")) for r in successful]
    cost_total = sum(c for c in costs if c is not None)
    tokens = {
        k: sum(r["usage"].get(k, 0) for r in successful)
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    upstream_costs = [
        r["usage"].get("cost_details", {}).get("upstream_inference_cost")
        for r in successful
    ]
    decision_rows = {r["id"]: r for r in read_lines("decision.jsonl")}
    decision_report = {"status": "unavailable"}
    if decision_rows:
        covered = np.array([r.id in decision_rows for r in holdout])
        decisions = np.array(
            [
                sum(
                    (
                        1
                        + decision_rows[r.id]["result"]["answers"][
                            f"labeling_engine.{name}"
                        ]["score"]
                    )
                    * weight
                    for name, weight in [
                        ("instruction_clarity", 0.4),
                        ("response_quality", 0.6),
                    ]
                )
                for r in holdout
                if r.id in decision_rows
            ]
        )
        result["spearman_human_decision"] = _spearman(human[covered], decisions)
        result["spearman_teacher_decision"] = _spearman(teacher[covered], decisions)
        result["spearman_student_decision"] = _spearman(student[covered], decisions)
        k = len(decisions) // 2
        good = human[covered] >= 0.75
        decision_selected = np.argsort(-decisions, kind="stable")[:k]
        result["decision_top_half_human_good_precision"] = float(
            np.mean(good[decision_selected])
        )
        result["decision_top_half_human_good_recall"] = float(
            np.sum(good[decision_selected]) / np.sum(good)
        )
        decision_http = read_lines("decision_http_usage.jsonl")
        decision_report = {
            "status": "measured_with_failures",
            "records": len(decision_rows),
            "holdout_records_scored": int(covered.sum()),
            "missing_ids": [r.id for r in judged if r.id not in decision_rows],
            "models": sorted({r["result"]["model"] for r in decision_rows.values()}),
            "reported_cost_usd": sum(
                r["result"]["usage"]["cost_usd"] or 0 for r in decision_rows.values()
            ),
            "upstream_cost_usd": None,
            "prompt_tokens": sum(
                r["result"]["usage"]["prompt_tokens"] for r in decision_rows.values()
            ),
            "successful_requests": sum(
                r["result"]["requests_sent"] for r in decision_rows.values()
            ),
            "http_status_counts_after_pilot": {
                str(status): sum(r["status"] == status for r in decision_http)
                for status in {r["status"] for r in decision_http}
            },
            "per_record_median_seconds": float(
                np.median([r["elapsed_s"] for r in decision_rows.values()])
            ),
            "probe": json.loads((ROOT / "decision_probe.json").read_text()),
        }
    detailed = []
    selected_ids = set(np.argsort(-student, kind="stable")[: len(holdout) // 2])
    expected_payloads = {input_hash(holdout[i]) for i in selected_ids}
    exported = [
        json.loads(line) for line in (ROOT / "ranked.jsonl").read_text().splitlines()
    ]
    # JSONL also contains null optional fields. Compare only the declared feature pair.
    actual_payloads = {
        hashlib.sha256(
            json.dumps(
                {k: p[k] for k in ["instruction", "response"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for p in exported
    }
    assert actual_payloads == expected_payloads
    for i, (r, a, b, c) in enumerate(
        zip(holdout, teacher, student, human, strict=True)
    ):
        decision = None
        if r.id in decision_rows:
            decision = sum(
                (
                    1
                    + decision_rows[r.id]["result"]["answers"][
                        f"labeling_engine.{name}"
                    ]["score"]
                )
                * weight
                for name, weight in [
                    ("instruction_clarity", 0.4),
                    ("response_quality", 0.6),
                ]
            )
        detailed.append(
            {
                "id": r.id,
                "tree_id": r.source.doc_id,
                "teacher": float(a),
                "student": float(b),
                "decision": decision,
                "human_quality": float(c),
                "absolute_error": float(abs(a - b)),
                "human_raters": labels[r.id]["rater_count"],
                "student_selected": i in selected_ids,
            }
        )
    with (ROOT / "holdout_scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    report = {
        "protocol": json.loads((ROOT / "protocol.json").read_text()),
        "model_sha256": hashlib.sha256((ROOT / "model.json").read_bytes()).hexdigest(),
        "split": model.evaluation.model_dump(mode="json"),
        "teacher_summary": {
            "mean": float(teacher.mean()),
            "std": float(teacher.std()),
            "min": float(teacher.min()),
            "max": float(teacher.max()),
        },
        "metrics": result,
        "bootstrap_95_percent_intervals": intervals,
        "teacher_usage": {
            "http_attempts": len(usage),
            "successful_requests": len(successful),
            "requests_with_known_cost": sum(c is not None for c in costs),
            "reported_cost_usd": cost_total
            if all(c is not None for c in costs)
            else None,
            "upstream_inference_cost_usd": sum(upstream_costs)
            if all(c is not None for c in upstream_costs)
            else None,
            **tokens,
        },
        "timings": read_lines("timings.jsonl"),
        "decision_baseline": decision_report,
        "largest_teacher_student_errors": sorted(
            detailed, key=lambda r: -r["absolute_error"]
        )[:10],
    }
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "fastembed",
                "onnxruntime",
                "openai",
                "polars",
                "pydantic",
            ]
        },
        "encoder_files_sha256": {
            str(p.relative_to("/tmp/arka-165-embedding-cache")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in Path("/tmp/arka-165-embedding-cache").rglob("*")
            if p.name
            in ["model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json"]
        },
    }
    report["teacher_usage"]["per_record_median_seconds"] = float(
        np.median([r["elapsed_s"] for r in read_lines("teacher.jsonl")])
    )
    report["teacher_usage"]["models"] = sorted(
        {r["label"]["judge_model"] for r in read_lines("teacher.jsonl")}
    )
    report["collection_failures"] = (
        json.loads((ROOT / "collection_failures.json").read_text())
        if (ROOT / "collection_failures.json").exists()
        else None
    )
    report["evidence_directory"] = str(ROOT)
    report["cli_selected_payloads_verified"] = len(actual_payloads)
    save("results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=[
            "prepare",
            "pilot",
            "judge",
            "decision",
            "decision-judge",
            "fit",
            "analyze",
        ],
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "pilot": lambda: llm_judge(2),
        "judge": llm_judge,
        "decision": decision_probe,
        "decision-judge": decision_judge,
        "fit": fit_and_infer,
        "analyze": analyze,
    }[args.operation]()
