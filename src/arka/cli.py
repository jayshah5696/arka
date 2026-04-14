from __future__ import annotations

import argparse
import uuid
from collections.abc import Sequence
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder


def build_parser() -> argparse.ArgumentParser:
    # DX: added clear, descriptive help text for CLI arguments to improve discoverability
    parser = argparse.ArgumentParser(prog="arka", description="Arka synthetic data generation pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the pipeline configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Custom identifier for this run. Auto-generated if not provided."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the pipeline from the latest successful stage checkpoint."
    )
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

    PipelineRunner(project_root=project_root).run(
        config=config,
        stages=stages,
        run_id=run_id,
        resume=args.resume,
    )
