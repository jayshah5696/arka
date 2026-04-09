from __future__ import annotations

import argparse
import json
import textwrap
import uuid
from collections.abc import Sequence
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.models import RunResult
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arka")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolve_run_id(cli_run_id: str | None, config_run_id: str | None) -> str:
    """Return the run_id from CLI, config, or auto-generate a UUID4."""
    if cli_run_id is not None:
        return cli_run_id
    if config_run_id is not None:
        return config_run_id
    return str(uuid.uuid4())


def _print_run_summary(result: RunResult) -> None:
    # DX: Print human-readable summary at end of run so users immediately see results and costs
    try:
        report = json.loads(result.run_report_path.read_text())
    except Exception:
        return

    summary = []
    summary.append("")
    summary.append("=" * 65)
    summary.append(f" Pipeline Run Completed: {result.run_id}")
    summary.append("=" * 65)

    stage_yields = report.get("stage_yields", [])
    if stage_yields:
        summary.append(
            f" {'Stage':<25} {'In':>8} -> {'Out':>8} {'Dropped':>8} {'Cost ($)':>10}"
        )
        summary.append("-" * 65)
        for stat in stage_yields:
            stage = stat.get("stage", "unknown")
            count_in = stat.get("count_in", 0)
            count_out = stat.get("count_out", 0)
            dropped = stat.get("dropped_count", 0)
            cost = stat.get("cost_usd")
            cost_str = f"{cost:.4f}" if cost is not None else "0.0000"
            summary.append(
                f" {stage:<25} {count_in:>8} -> {count_out:>8} {dropped:>8} {cost_str:>10}"
            )

    summary.append("-" * 65)
    summary.append(f" Final Records: {report.get('final_count', 0)}")

    cost_usd = report.get("cost_usd")
    if cost_usd is not None:
        summary.append(f" Total Cost:    ${cost_usd:.4f}")

    summary.append(f" Dataset Path:  {report.get('dataset_path') or 'None'}")
    summary.append(f" Report Path:   {result.run_report_path}")
    summary.append("=" * 65)
    summary.append("")

    print(textwrap.indent("\n".join(summary), ""))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).expanduser().resolve()
    project_root = config_path.parent
    config = ConfigLoader().load(config_path)
    run_id = _resolve_run_id(args.run_id, config.run_id)
    stages = StageBuilder(config=config, project_root=project_root).build()

    result = PipelineRunner(project_root=project_root).run(
        config=config,
        stages=stages,
        run_id=run_id,
        resume=args.resume,
    )
    _print_run_summary(result)
