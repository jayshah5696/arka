"""Failure evidence must survive a failed verification command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.verify_local import run_command


def test_failed_command_preserves_full_output(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"
    result = run_command(
        [sys.executable, "-c", "import sys; print('failure detail'); sys.exit(7)"],
        log,
    )
    assert result["exit_code"] == 7
    assert "failure detail" in log.read_text()


def test_missing_executable_is_reported_without_losing_evidence(tmp_path: Path) -> None:
    log = tmp_path / "missing.log"
    result = run_command([str(tmp_path / "missing-executable")], log)
    assert result["exit_code"] != 0
    assert "missing-executable" in log.read_text()
