from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import click

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.config.plan import ExecutionPlan
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

    click.secho(
        f"\n--- Pipeline Summary ({report.get('status', 'unknown')}) ---",
        bold=True,
        fg="cyan",
    )
    click.secho(f"Run ID: {report.get('run_id', run_id)}")
    if duration_secs is not None:
        if duration_secs < 60:
            click.secho(f"Duration: {duration_secs:.1f}s")
        else:
            mins = int(duration_secs // 60)
            secs = duration_secs % 60
            click.secho(f"Duration: {mins}m {secs:.1f}s")
    click.secho(f"Final Count: {report.get('final_count', 0)} records", bold=True)

    cost = report.get("cost_usd")
    if cost is not None:
        click.secho(f"Total Cost: ${cost:.6f}")

    decisions = report.get("decision_usage")
    if decisions is not None:
        click.secho(
            f"Decision requests: {decisions['requests']}; input tokens: {decisions['input_tokens']}"
        )
        if decisions.get("cost_usd") is not None:
            click.secho(f"Decision cost: ${decisions['cost_usd']:.6f}")
        escalation_counts = report.get("escalation_counts", {})
        click.secho(f"Unresolved escalations: {escalation_counts.get('unresolved', 0)}")

    dataset_path = report.get("dataset_path")
    if dataset_path:
        click.secho(f"\nDataset Output: {dataset_path}", fg="green")

    error_info = report.get("error")
    if error_info:
        # DX: Explicitly highlight the failed stage and error in the summary report
        click.secho(
            f"Fatal Error: Stage '{error_info.get('stage', 'unknown')}' failed - {error_info.get('message', 'Unknown error')}",
            fg="red",
            bold=True,
        )

    click.secho("\nStage Yields:", bold=True)
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
            click.secho(
                f"  {name}: {count_in} in -> {count_out} out (lost {lost} records) [{status}]{cost_str}",
                fg="red",
            )
            # DX: print error details on the stage that failed
            error = stage.get("error")
            if error:
                click.secho(
                    f"    - Failed: {error.get('type')}: {error.get('message')}",
                    fg="red",
                )
        else:
            click.secho(
                f"  {name}: {count_in} in -> {count_out} out (dropped {dropped}) [{status}]{cost_str}"
            )

        drop_reasons = stage.get("drop_reasons", {})
        if drop_reasons:
            for reason, count in drop_reasons.items():
                click.secho(f"    - {reason}: {count}", fg="yellow")

    click.secho(f"\nFull report written to: {report_path}")


def _load_config(config_path: Path) -> ResolvedConfig:
    """Load config and print clean error messages for config issues instead of Python tracebacks."""
    try:
        return ConfigLoader().load(config_path)
    except FileNotFoundError:
        click.secho(
            f"Error: Configuration file not found at {config_path}", fg="red", err=True
        )
        sys.exit(1)
    except IsADirectoryError:
        click.secho(
            f"Error: Expected a configuration file, but found a directory at {config_path}",
            fg="red",
            err=True,
        )
        sys.exit(1)
    except PermissionError:
        click.secho(
            f"Error: Permission denied when reading configuration file at {config_path}",
            fg="red",
            err=True,
        )
        sys.exit(1)
    except ConfigValidationError as exc:
        click.secho(f"Error: {str(exc)}", fg="red", err=True)
        sys.exit(1)


def _validate_config(
    loaded_config: ResolvedConfig, config_path: Path, project_root: Path
) -> None:
    """Validate config including stage building without side effects or running the pipeline."""
    try:
        # Build stages to ensure they are valid and can be constructed
        stages = StageBuilder(config=loaded_config, project_root=project_root).build()
        if loaded_config._experiment is not None:
            ExecutionPlan.build(
                loaded_config, project_root, [stage.name for stage in stages]
            )
    except Exception as exc:
        click.secho(f"Error: Configuration is invalid: {exc}", fg="red", err=True)
        sys.exit(1)

    click.secho(f"Configuration is valid: {config_path}", fg="green")
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
        click.secho(f"Dry run enabled. Loaded config: {config_path}", fg="cyan")
        click.secho(f"Resolved run ID: {resolved_run_id}", fg="cyan")
    else:
        click.secho(f"Loaded config: {config_path}", fg="cyan")
    click.secho("Stages to execute:", bold=True)
    for i, stage in enumerate(stages, 1):
        click.secho(f"  {i}. {stage.name}")

    # DX: Print the expected dataset output path during dry-runs so the user knows where the file will end up
    output_path = project_root / loaded_config.output.path
    click.secho(f"\nExpected Dataset Output: {output_path}", fg="green")


def _run_pipeline(
    *,
    project_root: Path,
    loaded_config: ResolvedConfig,
    stages: list[Stage],
    resolved_run_id: str,
    resume: bool,
    start_time: float,
    plan: ExecutionPlan | None = None,
) -> None:
    """Execute the pipeline stages and display execution summary."""
    # DX: Print the resolved Run ID explicitly at the start so users can track the run directory immediately
    click.secho(
        f"Starting pipeline execution with Run ID: {resolved_run_id}",
        fg="cyan",
        bold=True,
    )
    error_to_report = None
    try:
        PipelineRunner(project_root=project_root).run(
            config=loaded_config,
            stages=stages,
            run_id=resolved_run_id,
            resume=resume,
            **({"plan": plan} if plan is not None else {}),
        )
    except KeyboardInterrupt:
        # DX: Gracefully handle Ctrl+C to prevent raw tracebacks and show an actionable error message
        error_to_report = "Pipeline execution interrupted by user"
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
        click.secho(
            f"\nError: Pipeline execution failed - {error_to_report}",
            fg="red",
            err=True,
        )
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
# DX: Add --version flag [what] so users can easily check the installed framework version without looking at package metadata [why]
@click.version_option(package_name="arka")
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
    try:
        stages = StageBuilder(config=loaded_config, project_root=project_root).build()
        plan = (
            ExecutionPlan.build(
                loaded_config, project_root, [stage.name for stage in stages]
            )
            if loaded_config._experiment is not None
            else None
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    if dry_run or list_stages:
        if plan is not None:
            click.echo(f"Experiment fingerprint: {plan.fingerprint}")
            for path, fingerprint in plan.dependencies.items():
                click.echo(f"Input: {path} ({fingerprint})")
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
        plan=plan,
    )


def _resolve_run_id(cli_run_id: str | None, config_run_id: str | None) -> str:
    """Return the run_id from CLI, config, or auto-generate a UUID4."""
    if cli_run_id is not None:
        return cli_run_id
    if config_run_id is not None:
        return config_run_id
    return str(uuid.uuid4())


@click.command(name="arka-train-scorer")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--records",
    "records_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--target", default="quality", show_default=True)
@click.option("--alpha", default=1.0, type=float, show_default=True)
@click.option(
    "--features",
    "feature_format",
    default="pair_quality_v2",
    show_default=True,
    type=click.Choice(["pair_quality_v2", "instruction_response_json_v1"]),
)
def train_scorer(
    config_path: Path,
    records_path: Path,
    output: Path,
    target: str,
    alpha: float,
    feature_format: str,
) -> None:
    """Train a reusable score regressor from judged Arka data.parquet records."""
    from arka.distillation import embedding_identity, features_for_records, train_model
    from arka.pipeline.output import OutputWriter
    from arka.records.models import ConversationRecord

    if output.exists():
        raise click.ClickException(f"Model output already exists: {output}")
    try:
        config = ConfigLoader().load(config_path)
        records = OutputWriter().read_parquet(records_path)
        if not records or any(not isinstance(r, ConversationRecord) for r in records):
            raise ValueError("Training requires judged conversation records")
        from arka.privacy.records import protect_record
        from arka.privacy.runtime import build_policy

        privacy = build_policy(config.pii, config_path.resolve().parent)
        if privacy is not None:
            for record in records:
                protected, summary = protect_record(record, privacy)
                if summary.drop_reason or protected.payload != record.payload:
                    raise ValueError(
                        "Training text needs privacy protection; protect and judge it before training"
                    )
        vectors = features_for_records(records, config, feature_format=feature_format)
        model = train_model(
            records,
            vectors,
            embedding_identity(config),
            target=target,
            alpha=alpha,
            feature_format=feature_format,
            privacy_fingerprint=privacy.fingerprint if privacy else None,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        data = model.model_dump_json(indent=2).encode()
        with output.open("xb") as handle:
            handle.write(data)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved model: {output}")
    click.echo(f"model_sha256: {hashlib.sha256(data).hexdigest()}")
    click.echo(model.evaluation.model_dump_json(indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    try:
        cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.exceptions.Exit as e:
        if e.exit_code != 0:
            sys.exit(e.exit_code)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
