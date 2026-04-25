"""Compute comparison metrics on a run output.

Usage:
    uv run python scratch/simula-eval/metrics.py <run_dir> <output_jsonl>

Writes <run_dir>/../metrics.json next to the slice's dataset.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _extract_text(rec: dict[str, Any]) -> str:
    """Best-effort extraction of human-readable text from a chatml or raw record."""
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        return "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in msgs if isinstance(m, dict)
        )
    instr = rec.get("instruction", "")
    resp = rec.get("response", "")
    if instr or resp:
        return f"{instr}\n{resp}"
    return json.dumps(rec, sort_keys=True)


def _avg_pairwise_cosine(texts: list[str], k_max: int = 200) -> float | None:
    """Average pairwise cosine distance via fastembed (matches arka's runner). None if deps missing."""
    if len(texts) < 2:
        return 0.0
    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError:
        return None

    sample = texts[:k_max]
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vecs = np.array(list(model.embed(sample)))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    sims = vecs @ vecs.T
    n = len(sample)
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sims[iu]
    return float(np.mean(distances))


def _stage_stats(run_dir: Path) -> list[dict[str, Any]]:
    """Read stats.json from each stage directory (under run_dir/stages/<name>/)."""
    out: list[dict[str, Any]] = []
    stages_root = run_dir / "stages"
    if not stages_root.exists():
        return out
    for stage_dir in sorted(stages_root.iterdir()):
        if not stage_dir.is_dir():
            continue
        stats_path = stage_dir / "stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text())
                stats["__stage"] = stage_dir.name
                out.append(stats)
            except json.JSONDecodeError:
                pass
    return out


def compute(run_dir: Path, output_jsonl: Path, slice_name: str) -> dict[str, Any]:
    records = _load_jsonl(output_jsonl) if output_jsonl.exists() else []
    texts = [_extract_text(r) for r in records]

    stage_stats = _stage_stats(run_dir)
    by_stage = {s["__stage"]: s for s in stage_stats}

    def _count(stage: str) -> int | None:
        s = by_stage.get(stage)
        if not s:
            return None
        return s.get("output_records") or s.get("kept") or s.get("count")

    metrics = {
        "slice": slice_name,
        "run_dir": str(run_dir),
        "output_jsonl": str(output_jsonl),
        "final_count": len(records),
        "stage_counts": {s["__stage"]: s for s in stage_stats},
        "avg_pairwise_cosine_distance": _avg_pairwise_cosine(texts),
    }

    if records:
        char_lens = [len(t) for t in texts]
        metrics["text_chars_min"] = min(char_lens)
        metrics["text_chars_max"] = max(char_lens)
        metrics["text_chars_mean"] = statistics.mean(char_lens)
        metrics["text_chars_median"] = statistics.median(char_lens)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("output_jsonl", type=Path)
    ap.add_argument("--name", required=True, help="Slice name, e.g. '00-baseline'")
    ap.add_argument("--out", type=Path, default=None, help="Where to write metrics.json")
    args = ap.parse_args()

    metrics = compute(args.run_dir, args.output_jsonl, args.name)
    out = args.out or (args.output_jsonl.parent / "metrics.json")
    out.write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
