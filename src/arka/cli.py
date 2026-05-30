from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import click

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder


def _print_summary(
    run_id: str, project_root: Path, duration_secs: float | None = None
) -> None:
    """Print a human-readable pipeline run summary to stdout."""
    report_path = project_root / "runs" / run_id / "report" / "run_report.json"
    if not report_path.exists():
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return

    print(f"\n--- Pipeline Summary ({report.get('status', 'unknown')}) ---")
    print(f"Run ID: {report.get('run_id', run_id)}")
    if duration_secs is not None:
        if duration_secs < 60:
            print(f"Duration: {duration_secs:.1f}s")
        else:
            mins = int(duration_secs // 60)
            secs = duration_secs % 60
            print(f"Duration: {mins}m {secs:.1f}s")
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
        # DX: If a stage failed, print the error type and message for better visibility
        if status == "failed" and "error" in stage:
            err = stage["error"]
            print(f"    - Failed: {err.get('type')}: {err.get('message')}")

        drop_reasons = stage.get("drop_reasons", {})
        if drop_reasons:
            for reason, count in drop_reasons.items():
                print(f"    - {reason}: {count}")

    print(f"\nFull report written to: {report_path}")


@click.command(name="arka")
@click.option(
    "--config",
    default="config.yaml",
    help="Path to the YAML configuration file (default: config.yaml)",
)
@click.option(
    "--run-id",
    default=None,
    help="Optional unique identifier for the run (overrides config run_id)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume a previously interrupted run using checkpoints",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Load config and preview stages without executing the pipeline",
)
@click.option(
    "--validate-config",
    is_flag=True,
    help="Check if the configuration is valid and exit without running",
)
@click.option(
    "--list-stages",
    is_flag=True,
    help="Alias for --dry-run: Load config and preview stages without executing the pipeline",
)
def cli(
    config: str,
    run_id: str | None,
    resume: bool,
    dry_run: bool,
    validate_config: bool,
    list_stages: bool,
) -> None:
    """Arka: A config-driven synthetic data generation framework."""
    start_time = time.time()
    config_path = Path(config).expanduser().resolve()
    project_root = config_path.parent

    # DX: Print clean error messages for config issues instead of Python tracebacks.
    try:
        loaded_config = ConfigLoader().load(config_path)
    except FileNotFoundError:
        click.echo(f"Error: Configuration file not found at {config_path}", err=True)
        sys.exit(1)
    except ConfigValidationError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    # DX: Add --validate-config flag allows checking YAML syntax and schema without side effects or running the pipeline
    if validate_config:
        click.echo(f"Configuration is valid: {config_path}")
        sys.exit(0)

    resolved_run_id = _resolve_run_id(run_id, loaded_config.run_id)
    stages = StageBuilder(config=loaded_config, project_root=project_root).build()

    if dry_run or list_stages:
        click.echo(f"Dry run enabled. Loaded config: {config_path}")
        click.echo(f"Resolved run ID: {resolved_run_id}")
        click.echo("Stages to execute:")
        for i, stage in enumerate(stages, 1):
            click.echo(f"  {i}. {stage.name}")
        return

    try:
        PipelineRunner(project_root=project_root).run(
            config=loaded_config,
            stages=stages,
            run_id=resolved_run_id,
            resume=resume,
        )
    except Exception as exc:
        # DX: Catch pipeline execution errors to prevent raw Python tracebacks.
        # This provides a clean, human-readable error message to the user.
        click.echo(f"Error: Pipeline execution failed - {exc}", err=True)
        sys.exit(1)
    finally:
        duration_secs = time.time() - start_time
        _print_summary(resolved_run_id, project_root, duration_secs)


def _resolve_run_id(cli_run_id: str | None, config_run_id: str | None) -> str:
    """Return the run_id from CLI, config, or auto-generate a UUID4."""
    if cli_run_id is not None:
        return cli_run_id
    if config_run_id is not None:
        return config_run_id
    return str(uuid.uuid4())


def main(argv: Sequence[str] | None = None) -> None:
    try:
        cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.exceptions.Exit as e:
        if e.exit_code != 0:
            sys.exit(e.exit_code)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
