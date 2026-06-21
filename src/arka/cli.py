from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder

if TYPE_CHECKING:
    from arka.config.models import ResolvedConfig
    from arka.pipeline.stages import Stage


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

    dataset_path = report.get("dataset_path")
    if dataset_path:
        print(f"\nDataset Output: {dataset_path}")

    error_info = report.get("error")
    if error_info:
        # DX: Explicitly highlight the failed stage and error in the summary report
        print(
            f"Fatal Error: Stage '{error_info.get('stage', 'unknown')}' failed - {error_info.get('message', 'Unknown error')}"
        )

    print("\nStage Yields:")
    for stage in report.get("stage_yields", []):
        name = stage.get("stage", "unknown")
        count_in = stage.get("count_in", 0)
        count_out = stage.get("count_out", 0)
        dropped = stage.get("dropped_count", 0)
        status = stage.get("status", "unknown")
        cost_usd = stage.get("cost_usd")
        cost_str = f" (${cost_usd:.6f})" if cost_usd is not None else ""

        # DX: Explicitly show lost records in the CLI summary when a stage fails
        if status == "failed":
            lost = count_in - count_out
            print(
                f"  {name}: {count_in} in -> {count_out} out (lost {lost} records) [{status}]{cost_str}"
            )
            # DX: print error details on the stage that failed
            error = stage.get("error")
            if error:
                print(f"    - Failed: {error.get('type')}: {error.get('message')}")
        else:
            print(
                f"  {name}: {count_in} in -> {count_out} out (dropped {dropped}) [{status}]{cost_str}"
            )

        drop_reasons = stage.get("drop_reasons", {})
        if drop_reasons:
            for reason, count in drop_reasons.items():
                print(f"    - {reason}: {count}")

    print(f"\nFull report written to: {report_path}")


def _load_config(config_path: Path) -> ResolvedConfig:
    """Load config and print clean error messages for config issues instead of Python tracebacks."""
    try:
        return ConfigLoader().load(config_path)
    except FileNotFoundError:
        click.echo(f"Error: Configuration file not found at {config_path}", err=True)
        sys.exit(1)
    except ConfigValidationError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


def _validate_config(
    loaded_config: ResolvedConfig, config_path: Path, project_root: Path
) -> None:
    """Validate config including stage building without side effects or running the pipeline."""
    try:
        # Build stages to ensure they are valid and can be constructed
        StageBuilder(config=loaded_config, project_root=project_root).build()
    except Exception as exc:
        click.echo(f"Configuration is invalid: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Configuration is valid: {config_path}")
    sys.exit(0)


def _dry_run_or_list_stages(
    *,
    dry_run: bool,
    list_stages: bool,
    config_path: Path,
    resolved_run_id: str,
    stages: list[Stage],
    loaded_config: ResolvedConfig,
    project_root: Path,
) -> None:
    """Preview stages without executing the pipeline."""
    if dry_run:
        click.echo(f"Dry run enabled. Loaded config: {config_path}")
        click.echo(f"Resolved run ID: {resolved_run_id}")
    else:
        click.echo(f"Loaded config: {config_path}")
    click.echo("Stages to execute:")
    for i, stage in enumerate(stages, 1):
        click.echo(f"  {i}. {stage.name}")

    # DX: Print the expected dataset output path during dry-runs so the user knows where the file will end up
    output_path = project_root / loaded_config.output.path
    click.echo(f"\nExpected Dataset Output: {output_path}")


def _run_pipeline(
    *,
    project_root: Path,
    loaded_config: ResolvedConfig,
    stages: list[Stage],
    resolved_run_id: str,
    resume: bool,
    start_time: float,
) -> None:
    """Execute the pipeline stages and display execution summary."""
    error_to_report = None
    try:
        PipelineRunner(project_root=project_root).run(
            config=loaded_config,
            stages=stages,
            run_id=resolved_run_id,
            resume=resume,
        )
    except Exception as exc:
        error_to_report = exc
    finally:
        duration_secs = time.time() - start_time
        try:
            _print_summary(resolved_run_id, project_root, duration_secs)
        except Exception:
            pass  # Prevent summary errors from swallowing pipeline errors

    if error_to_report is not None:
        # DX: Print the fatal error message after the summary so it is the last thing
        # the user sees, preventing them from having to scroll up to find the failure cause.
        click.echo(f"\nError: Pipeline execution failed - {error_to_report}", err=True)
        sys.exit(1)


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

    loaded_config = _load_config(config_path)

    if validate_config:
        _validate_config(loaded_config, config_path, project_root)

    resolved_run_id = _resolve_run_id(run_id, loaded_config.run_id)
    stages = StageBuilder(config=loaded_config, project_root=project_root).build()

    if dry_run or list_stages:
        _dry_run_or_list_stages(
            dry_run=dry_run,
            list_stages=list_stages,
            config_path=config_path,
            resolved_run_id=resolved_run_id,
            stages=stages,
            loaded_config=loaded_config,
            project_root=project_root,
        )
        return

    _run_pipeline(
        project_root=project_root,
        loaded_config=loaded_config,
        stages=stages,
        resolved_run_id=resolved_run_id,
        resume=resume,
        start_time=start_time,
    )


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
