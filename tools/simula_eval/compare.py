"""Render a side-by-side comparison markdown table from N metrics.json files.

Usage:
    uv run python tools/simula_eval/compare.py \
        scratch/simula-eval/00-baseline/metrics.json \
        scratch/simula-eval/01-double-critic/metrics.json \
        --out scratch/simula-eval/comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _stage_yields(m: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    rows: list[tuple[str, int, int, int]] = []
    for stage_name, stats in sorted(m.get("stage_counts", {}).items()):
        rows.append(
            (
                stage_name,
                int(stats.get("count_in", 0)),
                int(stats.get("count_out", 0)),
                int(stats.get("dropped_count", 0)),
            )
        )
    return rows


def render(metrics: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Simula slice comparison\n")

    # Headline metrics row
    lines.append("## Headline metrics\n")
    headers = ["metric"] + [m["slice"] for m in metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    def _row(name: str, getter):
        cells = [name] + [str(getter(m)) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")

    _row("final_count", lambda m: m.get("final_count"))
    _row(
        "avg_pairwise_cosine_distance",
        lambda m: f"{m.get('avg_pairwise_cosine_distance', 0):.4f}"
        if m.get("avg_pairwise_cosine_distance") is not None
        else "n/a",
    )
    _row("text_chars_median", lambda m: m.get("text_chars_median"))
    _row("text_chars_min", lambda m: m.get("text_chars_min"))
    _row("text_chars_max", lambda m: m.get("text_chars_max"))

    # Per-stage yields
    lines.append("\n## Stage yields (count_in -> count_out [dropped])\n")
    all_stages = sorted(
        {s for m in metrics for s in (m.get("stage_counts") or {})}
    )
    headers = ["stage"] + [m["slice"] for m in metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for stage in all_stages:
        cells = [stage]
        for m in metrics:
            sc = (m.get("stage_counts") or {}).get(stage)
            if sc is None:
                cells.append("—")
            else:
                cells.append(
                    f"{sc.get('count_in', 0)} → {sc.get('count_out', 0)} "
                    f"[{sc.get('dropped_count', 0)}]"
                )
        lines.append("| " + " | ".join(cells) + " |")

    # Drop reasons
    lines.append("\n## Drop reasons by slice\n")
    for m in metrics:
        lines.append(f"### {m['slice']}")
        any_reasons = False
        for stage_name, stats in sorted((m.get("stage_counts") or {}).items()):
            reasons = stats.get("drop_reasons") or {}
            if not reasons:
                continue
            any_reasons = True
            for reason, count in reasons.items():
                lines.append(f"- `{stage_name}` / `{reason}`: {count}")
        if not any_reasons:
            lines.append("- (no drops)")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_files", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    metrics = [_load(p) for p in args.metrics_files]
    text = render(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
