from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Sequence
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder


# DX: [Print run summary] [Developers had to manually open run_report.json to see stage yields, drops, and costs. A terminal summary removes this friction.]
def _print_summary(run_id: str, project_root: Path) -> None:
    report_path = project_root / "runs" / run_id / "report" / "run_report.json"
    if not report_path.exists():
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return

    print(f"\n--- Pipeline Summary ({report.get('status', 'unknown')}) ---")
    print(f"Run ID: {report.get('run_id', run_id)}")
    print(f"Final Count: {report.get('final_count', 0)} records")

    cost = report.get("cost_usd")
    if cost is not None:
        print(f"Total Cost: ${cost:.6f}")

    print("\nStage Yields:")
    for stage in report.get("stage_yields", []):
        name = stage.get("stage", "unknown")
        count_in = stage.get("count_in", 0)
        count_out = stage.get("count_out", 0)
        dropped = stage.get("dropped_count", 0)
        status = stage.get("status", "unknown")
        print(
            f"  {name}: {count_in} in -> {count_out} out (dropped {dropped}) [{status}]"
        )

        drop_reasons = stage.get("drop_reasons", {})
        if drop_reasons:
            for reason, count in drop_reasons.items():
                print(f"    - {reason}: {count}")

    print(f"\nFull report written to: {report_path}")


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).expanduser().resolve()
    project_root = config_path.parent
    config = ConfigLoader().load(config_path)
    run_id = _resolve_run_id(args.run_id, config.run_id)
    stages = StageBuilder(config=config, project_root=project_root).build()

    try:
        PipelineRunner(project_root=project_root).run(
            config=config,
            stages=stages,
            run_id=run_id,
            resume=args.resume,
        )
    finally:
        _print_summary(run_id, project_root)
