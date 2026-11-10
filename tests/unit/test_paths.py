from __future__ import annotations

from pathlib import Path

import pytest

from arka.core.paths import RunPaths


def test_bootstrap_creates_expected_run_layout(tmp_path: Path) -> None:
    run_paths = RunPaths.bootstrap(root_dir=tmp_path, run_id="run-123")

    assert run_paths.run_dir == tmp_path / "runs" / "run-123"
    assert run_paths.stages_dir.exists()
    assert run_paths.report_dir.exists()
    assert run_paths.manifest_path.parent.exists()
    assert run_paths.run_report_path.parent.exists()
    assert run_paths.sqlite_path == tmp_path / "state.db"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "/tmp/outside"])
def test_run_paths_reject_invalid_identifiers(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        RunPaths.bootstrap(root_dir=tmp_path, run_id=run_id)
    assert not (tmp_path / "runs").exists()


def test_run_paths_reject_symlink_outside_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runs / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        RunPaths.bootstrap(root_dir=tmp_path, run_id="linked")
    assert list(outside.iterdir()) == []
