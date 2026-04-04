from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.runner import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arka")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-id", default="manual-run")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).expanduser().resolve()
    project_root = config_path.parent
    config = ConfigLoader().load(config_path)
    PipelineRunner(project_root=project_root).run(
        config=config,
        stages=[],
        run_id=args.run_id,
        resume=args.resume,
    )
