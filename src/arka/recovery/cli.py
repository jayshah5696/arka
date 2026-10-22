from __future__ import annotations

import uuid
from pathlib import Path

import click

from arka.recovery.models import RecoveryConfig, RecoveryReport
from arka.recovery.service import RecoveryService


def _service(config_path: Path, project_root: Path | None) -> RecoveryService:
    config_path = config_path.resolve()
    return RecoveryService(
        RecoveryConfig.load(config_path),
        project_root=project_root or config_path.parent,
        config_root=config_path.parent,
    )


def _summary(report: RecoveryReport) -> None:
    click.echo(f"Run ID: {report.run_id}")
    for source in report.sources:
        click.echo(
            f"{source.key}: {source.failed_count}/{source.sampled_count} sampled failures ({source.failure_rate:.1%}); {source.total_count} records"
        )
    if report.operation == "rebuild":
        click.echo(
            f"Accepted replacements: {report.accepted_replacements}; exhausted: {report.exhausted_count}; unresolved audit failures: {report.unresolved_count}"
        )
        click.echo(f"Dataset: {report.dataset_path} ({report.final_count} records)")
    click.echo(
        f"Completed client calls: {report.requests}; tokens: {report.usage.total_tokens}"
    )
    if report.usage.cost_usd is not None:
        qualifier = "total" if report.cost_complete else "known portion"
        click.echo(f"Cost ({qualifier}): ${report.usage.cost_usd:.6f}")


@click.command(name="arka-audit")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--project-root", type=click.Path(file_okay=False, path_type=Path))
@click.option("--parent-run", required=True)
@click.option(
    "--stage", default=None, help="Saved stage to audit; defaults to the final stage."
)
@click.option("--run-id", default=None)
@click.option("--resume", is_flag=True)
def audit(
    config_path: Path,
    project_root: Path | None,
    parent_run: str,
    stage: str | None,
    run_id: str | None,
    resume: bool,
) -> None:
    """Sample records by source from an existing Arka run."""
    if resume and not run_id:
        raise click.ClickException("--resume requires --run-id")
    resolved_id = run_id or str(uuid.uuid4())
    click.echo(f"Audit run: {resolved_id}")
    try:
        report = _service(config_path, project_root).audit(
            parent_run, resolved_id, stage=stage, resume=resume
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    _summary(report)


@click.command(name="arka-rebuild")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--project-root", type=click.Path(file_okay=False, path_type=Path))
@click.option("--audit-run", required=True)
@click.option(
    "--source",
    "sources",
    multiple=True,
    required=True,
    help="Explicit source key from the audit report. Repeat to select several sources.",
)
@click.option("--run-id", default=None)
@click.option("--resume", is_flag=True)
def rebuild(
    config_path: Path,
    project_root: Path | None,
    audit_run: str,
    sources: tuple[str, ...],
    run_id: str | None,
    resume: bool,
) -> None:
    """Judge selected sources and replace failed answers with bounded attempts."""
    if resume and not run_id:
        raise click.ClickException("--resume requires --run-id")
    resolved_id = run_id or str(uuid.uuid4())
    click.echo(f"Rebuild run: {resolved_id}")
    try:
        report = _service(config_path, project_root).rebuild(
            audit_run, resolved_id, sources=list(sources), resume=resume
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    _summary(report)
