"""Develop on 2400 records, lock two artifacts, then evaluate a fresh 600."""

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from arka.config.loader import ConfigLoader
from arka.distillation import (
    DistilledModel,
    _spearman,
    features_for_records,
    input_hash,
)
from arka.pipeline.output import OutputWriter

ROOT = Path(os.environ.get("ARKA_V2_EVAL_DIR", "/tmp/arka-165-v2"))
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/arka-165-embedding-cache"


def load_json(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(name, args):
    started = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", *args], capture_output=True, text=True, env=os.environ.copy()
    )
    (ROOT / f"{name}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    elapsed = time.perf_counter() - started
    with (ROOT / "timings.jsonl").open("a") as h:
        h.write(json.dumps({"operation": name, "seconds": elapsed}) + "\n")
    print(f"{name}: {elapsed:.2f}s", flush=True)


def ridge_cv(x, y, folds):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    predictions = {alpha: np.zeros(len(y)) for alpha in alphas}
    for fold in range(5):
        train, validation = folds != fold, folds == fold
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1
        xt, xv = (x[train] - mean) / scale, (x[validation] - mean) / scale
        ym = y[train].mean()
        u, singular, vt = np.linalg.svd(xt, full_matrices=False)
        transformed = xv @ vt.T
        projected = u.T @ (y[train] - ym)
        for alpha in alphas:
            predictions[alpha][validation] = (
                transformed @ ((singular / (singular**2 + alpha)) * projected) + ym
            )
    return [
        {
            "alpha": a,
            "spearman": _spearman(y, p),
            "mae": float(np.mean(abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        }
        for a, p in predictions.items()
    ]


def develop_and_lock():
    assert not (ROOT / "locked_model_choice.json").exists(), (
        "Model choice is already locked"
    )
    directory = ROOT / "development"
    recs = OutputWriter().read_parquet(directory / "judged.parquet")
    cfg = ConfigLoader().load(directory / "config.yaml")
    vectors = features_for_records(recs, cfg, feature_format="pair_quality_v2")
    np.save(ROOT / "development_features.npy", vectors)
    target = np.array([r.scores.quality for r in recs])
    folds = np.array(
        [
            int(hashlib.sha256(r.source.doc_id.encode()).hexdigest()[:8], 16) % 5
            for r in recs
        ]
    )
    scores = ridge_cv(vectors, target, folds)
    save(ROOT / "development_cross_validation.json", scores)
    chosen = max(scores, key=lambda r: r["spearman"])
    print("Chosen from development only: " + json.dumps(chosen), flush=True)
    for name, features, alpha in [
        ("improved", "pair_quality_v2", chosen["alpha"]),
        ("baseline", "instruction_response_json_v1", 1.0),
    ]:
        command(
            f"train_{name}",
            [
                "arka-train-scorer",
                "--config",
                str(directory / "config.yaml"),
                "--records",
                str(directory / "judged.parquet"),
                "--output",
                str(ROOT / f"{name}.json"),
                "--features",
                features,
                "--alpha",
                str(alpha),
            ],
        )
    improved = DistilledModel.model_validate_json((ROOT / "improved.json").read_text())
    baseline = DistilledModel.model_validate_json((ROOT / "baseline.json").read_text())
    assert improved.training_ids == baseline.training_ids
    assert improved.validation_ids == baseline.validation_ids
    protocol = load_json(ROOT / "protocol.json")
    final = set(protocol["holdout_ids"])
    assert not final & set(improved.training_ids + improved.validation_ids)
    save(
        ROOT / "locked_model_choice.json",
        {
            "feature_format": "pair_quality_v2",
            "alpha": chosen["alpha"],
            "development_oof": chosen,
            "development_records": len(recs),
            "actual_fit_records": len(improved.training_ids),
            "training_ids_sha256": hashlib.sha256(
                json.dumps(improved.training_ids).encode()
            ).hexdigest(),
            "holdout_ids_sha256": hashlib.sha256(
                json.dumps(protocol["holdout_ids"]).encode()
            ).hexdigest(),
            "improved_sha256": hashlib.sha256(
                (ROOT / "improved.json").read_bytes()
            ).hexdigest(),
            "baseline_sha256": hashlib.sha256(
                (ROOT / "baseline.json").read_bytes()
            ).hexdigest(),
            "relevance_model": "Omitted; development benefit was negligible",
            "feature_count": 12,
        },
    )
    print("Locked. Fresh holdout labels may now be collected.", flush=True)


def infer():
    choice = load_json(ROOT / "locked_model_choice.json")
    directory = ROOT / "final"
    records = OutputWriter().read_parquet(directory / "judged.parquet")
    assert len(records) > 0
    (ROOT / "holdout.jsonl").write_text(
        "".join(json.dumps(r.export_payload()) + "\n" for r in records)
    )
    for name in ["improved", "baseline"]:
        assert (
            hashlib.sha256((ROOT / f"{name}.json").read_bytes()).hexdigest()
            == choice[f"{name}_sha256"]
        )
        cfg = yaml.safe_load((directory / "config.yaml").read_text())
        cfg["pipeline"] = [
            {"type": "seed_source", "path": "holdout.jsonl"},
            {"type": "normalize_conversation"},
            {
                "type": "distilled_ridge",
                "model_path": f"{name}.json",
                "model_sha256": choice[f"{name}_sha256"],
            },
            {
                "type": "select",
                "target_count": len(records) // 2,
                "weights": {"distilled_ridge": 1},
            },
        ]
        cfg["output"]["path"] = f"{name}-ranked.jsonl"
        path = ROOT / f"{name}-inference.yaml"
        path.write_text(yaml.safe_dump(cfg))
        command(f"infer_{name}", ["arka", "--config", str(path), "--run-id", name])
        command(
            f"resume_{name}",
            ["arka", "--config", str(path), "--run-id", name, "--resume"],
        )


def metrics(teacher, prediction, human):
    selected = np.argsort(-prediction, kind="stable")[: len(teacher) // 2]
    good = human >= 0.75
    return {
        "teacher_spearman": _spearman(teacher, prediction),
        "human_spearman": _spearman(human, prediction),
        "mae": float(np.mean(abs(teacher - prediction))),
        "rmse": float(np.sqrt(np.mean((teacher - prediction) ** 2))),
        "human_good_recall": float(np.sum(good[selected]) / sum(good)),
        "human_good_precision": float(np.mean(good[selected])),
    }


def analyze():
    records = OutputWriter().read_parquet(ROOT / "final/judged.parquet")
    development = OutputWriter().read_parquet(ROOT / "development/judged.parquet")
    assert not {r.source.doc_id for r in records} & {
        r.source.doc_id for r in development
    }
    assert not {input_hash(r) for r in records} & {input_hash(r) for r in development}
    labels = load_json(ROOT / "final/human_labels.json")
    teacher = np.array([r.scores.quality for r in records])
    human = np.array([labels[r.id]["quality"] for r in records])
    predictions = {}
    for name in ["improved", "baseline"]:
        scored = OutputWriter().read_parquet(
            ROOT / f"runs/{name}/stages/02s_distilled_ridge/data.parquet"
        )
        mapping = {input_hash(r): r.scores.distilled_ridge for r in scored}
        predictions[name] = np.array([mapping[input_hash(r)] for r in records])
        selected = {
            input_hash(records[i])
            for i in np.argsort(-predictions[name], kind="stable")[: len(records) // 2]
        }
        exported = [
            json.loads(line)
            for line in (ROOT / f"{name}-ranked.jsonl").read_text().splitlines()
        ]
        hashes = {
            hashlib.sha256(
                json.dumps(
                    {k: r[k] for k in ["instruction", "response"]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for r in exported
        }
        assert hashes == selected
    measured = {name: metrics(teacher, p, human) for name, p in predictions.items()}
    measured["direct_teacher"] = metrics(teacher, teacher, human)
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r.source.doc_id, []).append(i)
    blocks = list(groups.values())
    rng = np.random.default_rng(1652)
    draws = {name: [] for name in measured}
    differences = []
    recall_differences = []
    for _ in range(1000):
        idx = [i for b in rng.integers(0, len(blocks), len(blocks)) for i in blocks[b]]
        m = {
            name: metrics(teacher[idx], p[idx], human[idx])
            for name, p in {**predictions, "direct_teacher": teacher}.items()
        }
        for name, value in m.items():
            draws[name].append(value)
        differences.append(
            m["improved"]["teacher_spearman"] - m["baseline"]["teacher_spearman"]
        )
        recall_differences.append(
            m["improved"]["human_good_recall"]
            - m["direct_teacher"]["human_good_recall"]
        )
    intervals = {
        name: {
            k: np.percentile(
                [r[k] for r in rows if r[k] is not None], [2.5, 97.5]
            ).tolist()
            for k in rows[0]
        }
        for name, rows in draws.items()
    }
    difference_interval = np.percentile(differences, [2.5, 97.5]).tolist()
    recall_difference = (
        measured["improved"]["human_good_recall"]
        - measured["direct_teacher"]["human_good_recall"]
    )
    collection = {}
    for name in ["development", "final"]:
        p = ROOT / name
        usage = [
            json.loads(line)
            for line in (p / "http_usage.jsonl").read_text().splitlines()
        ]
        known = [
            (r.get("usage") or {})
            .get("cost_details", {})
            .get("upstream_inference_cost")
            for r in usage
        ]
        collection[name] = {
            "http_requests": len(usage),
            "upstream_cost_usd": sum(known)
            if all(c is not None for c in known)
            else None,
            "known_upstream_cost_subtotal_usd": sum(c for c in known if c is not None),
            "requests_with_known_upstream_cost": sum(c is not None for c in known),
            "prompt_tokens": sum(
                (r.get("usage") or {}).get("prompt_tokens", 0) for r in usage
            ),
            "completion_tokens": sum(
                (r.get("usage") or {}).get("completion_tokens", 0) for r in usage
            ),
            "prior_labels_reused": 600 if name == "development" else 0,
            "collection_timings": [
                json.loads(line)
                for line in (p / "timings.jsonl").read_text().splitlines()
            ],
        }
    rows = []
    for i, r in enumerate(records):
        rows.append(
            {
                "id": r.id,
                "tree_id": r.source.doc_id,
                "teacher": float(teacher[i]),
                "improved": float(predictions["improved"][i]),
                "baseline": float(predictions["baseline"][i]),
                "human": float(human[i]),
            }
        )
    with (ROOT / "holdout_scores.csv").open("w") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision_rows = (
        {
            r["id"]: r
            for r in [
                json.loads(line)
                for line in (ROOT / "final/decision.jsonl").read_text().splitlines()
            ]
        }
        if (ROOT / "final/decision.jsonl").exists()
        else {}
    )
    decision_report = {
        "scored_records": len(decision_rows),
        "selected_records": 600,
        "upstream_cost_usd": None,
    }
    if decision_rows:
        mask = np.array([r.id in decision_rows for r in records])
        direct = np.array(
            [
                sum(
                    (
                        1
                        + decision_rows[r.id]["result"]["answers"][
                            f"labeling_engine.{n}"
                        ]["score"]
                    )
                    * w
                    for n, w in [
                        ("instruction_clarity", 0.4),
                        ("response_quality", 0.6),
                    ]
                )
                for r in records
                if r.id in decision_rows
            ]
        )
        decision_report["matched_holdout_count"] = int(mask.sum())
        decision_report["metrics"] = metrics(teacher[mask], direct, human[mask])
        decision_report["teacher_human_spearman_on_same_subset"] = _spearman(
            teacher[mask], human[mask]
        )
        decision_report["improved_human_spearman_on_same_subset"] = _spearman(
            predictions["improved"][mask], human[mask]
        )
        decision_report["requests"] = len(
            (ROOT / "final/decision_http.jsonl").read_text().splitlines()
        )
        decision_report["prompt_tokens"] = sum(
            r["result"]["usage"]["prompt_tokens"] for r in decision_rows.values()
        )
        decision_report["reported_cost_usd"] = sum(
            r["result"]["usage"].get("cost_usd") or 0 for r in decision_rows.values()
        )
    report = {
        "protocol": load_json(ROOT / "protocol.json"),
        "locked_model_choice": load_json(ROOT / "locked_model_choice.json"),
        "metrics": measured,
        "bootstrap_95_percent_intervals": intervals,
        "paired_teacher_spearman_improvement_95_interval": difference_interval,
        "student_minus_teacher_good_recall": recall_difference,
        "student_minus_teacher_good_recall_95_interval": np.percentile(
            recall_differences, [2.5, 97.5]
        ).tolist(),
        "predeclared_quality_check_passed": bool(
            difference_interval[0] > 0 and recall_difference >= -0.05
        ),
        "collection": collection,
        "label_coverage": {
            name: load_json(ROOT / name / "label_coverage.json")
            for name in ["development", "final"]
        },
        "decision_baseline": decision_report,
        "timings": [
            json.loads(line)
            for line in (ROOT / "timings.jsonl").read_text().splitlines()
        ],
        "source_overlap": 0,
        "feature_text_overlap": 0,
        "exported_selection_verified": len(records) // 2,
        "evidence_directory": str(ROOT),
    }
    save(ROOT / "results.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["protocol"]}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    import sys

    {"develop": develop_and_lock, "infer": infer, "analyze": analyze}[sys.argv[1]]()
