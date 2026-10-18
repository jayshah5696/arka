"""Run Arka's deterministic CLI checks and retain evidence without provider keys."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], log: Path) -> dict:
    """Keep complete child output on disk, including command launch failures."""
    started = time.monotonic()
    with log.open("w") as stream:
        try:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
            exit_code = result.returncode
        except OSError as error:
            stream.write(f"{error}\n")
            exit_code = 1
    return {
        "command": command,
        "exit_code": exit_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New evidence directory")
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="arka-verify-"))
    print(f"Evidence: {output}", flush=True)
    summary = {
        "status": "running",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "evidence": str(output),
        "checks": [],
    }
    cli = str(Path(sys.executable).with_name("arka"))
    checks = [
        ("version", [cli, "--version"]),
        (
            "validate-examples",
            [sys.executable, str(ROOT / "scripts/validate_examples.py")],
        ),
        (
            "dry-run-examples",
            [sys.executable, str(ROOT / "scripts/dry_run_examples.py")],
        ),
        (
            "cleanup",
            [
                sys.executable,
                str(ROOT / "scripts/verify_cleanup_cli.py"),
                "--output",
                str(output / "cleanup"),
            ],
        ),
        (
            "decisions",
            [
                sys.executable,
                str(ROOT / "scripts/verify_decisions_cli.py"),
                "--output",
                str(output / "decisions"),
            ],
        ),
    ]
    result_path = output / "result.json"
    result_path.write_text(json.dumps(summary, indent=2))
    for name, command in checks:
        result = run_command(command, output / f"{name}.log")
        summary["checks"].append({"name": name, **result})
        failed = result["exit_code"] != 0
        summary["status"] = "failed" if failed else "running"
        result_path.write_text(json.dumps(summary, indent=2))
        print(f"{name}: {'failed' if failed else 'passed'}", flush=True)
        if failed:
            print(f"Read the full log: {result['log']}", file=sys.stderr)
            return 1
    summary["status"] = "passed"
    result_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
