from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arka",
        description="Arka: A config-driven synthetic data generation framework.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional unique identifier for the run. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run from the last completed stage",
    )
    # DX: added --dry-run to allow users to validate config and preview stages without running the pipeline
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and list pipeline stages without executing them",
    )
    return parser


def _resolve_run_id(cli_run_id: str | None, config_run_id: str | None) -> str:
    """Return the run_id from CLI, config, or auto-generate a UUID4."""
    if cli_run_id is not None:
        return cli_run_id
    if config_run_id is not None:
        return config_run_id
    return str(uuid.uuid4())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).expanduser().resolve()
    project_root = config_path.parent

    try:
        config = ConfigLoader().load(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    run_id = _resolve_run_id(args.run_id, config.run_id)
    stages = StageBuilder(config=config, project_root=project_root).build()

    if args.dry_run:
        print(f"Configuration is valid. Run ID: {run_id}")
        print("Pipeline stages to be executed:")
        for i, stage in enumerate(stages, 1):
            print(f"  {i}. {stage.name}")
        return 0

    result = PipelineRunner(project_root=project_root).run(
        config=config,
        stages=stages,
        run_id=run_id,
        resume=args.resume,
    )

    if result is None:
        return 1

    # DX: print a clear summary at completion so users don't have to hunt for the report JSON
    print("\nPipeline completed successfully!")
    print(f"  Run ID: {result.run_id}")
    print(f"  Final records: {result.final_count}")
    print(f"  Dataset saved to: {result.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
