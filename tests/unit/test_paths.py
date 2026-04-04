from __future__ import annotations

from pathlib import Path

from arka.core.paths import RunPaths


def test_bootstrap_creates_expected_run_layout(tmp_path: Path) -> None:
    run_paths = RunPaths.bootstrap(root_dir=tmp_path, run_id="run-123")

    assert run_paths.run_dir == tmp_path / "runs" / "run-123"
    assert run_paths.stages_dir.exists()
    assert run_paths.report_dir.exists()
    assert run_paths.manifest_path.parent.exists()
    assert run_paths.run_report_path.parent.exists()
    assert run_paths.sqlite_path == tmp_path / "state.db"
