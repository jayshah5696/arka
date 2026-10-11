"""Check each open stack PR at its own head without changing the user's checkout."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
repository = Path.cwd()
prs = json.loads(
    subprocess.check_output(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,headRefName,baseRefName",
        ],
        text=True,
    )
)
results = []
for pr in sorted(
    (pr for pr in prs if 184 <= pr["number"] <= 191), key=lambda pr: pr["number"]
):
    number = pr["number"]
    head = subprocess.check_output(
        ["git", "rev-parse", pr["headRefName"]], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/" + pr["headRefName"]], text=True
    ).strip()
    if head != remote:
        raise RuntimeError(f"PR {number}: local and fetched remote heads differ")
    output = root / f"pr-{number}"
    output.mkdir()
    checkout = root / f"checkout-{number}"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout), head],
        check=True,
        capture_output=True,
    )
    checks = []
    try:
        commands = [["uv", "sync", "--frozen"]]
        if number == 191:
            commands[0].extend(["--extra", "language"])
        commands.extend(
            [["just", "check"], ["uv", "build", "--out-dir", str(output / "dist")]]
        )
        for index, command in enumerate(commands):
            run = subprocess.run(
                command,
                cwd=checkout,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=180,
            )
            log = run.stdout + run.stderr
            (output / f"{index}.log").write_text(log)
            checks.append({"command": command, "exit_code": run.returncode})
            if run.returncode:
                print(log, flush=True)
                raise RuntimeError(f"PR {number}: {command} failed")
        check_log = (output / "1.log").read_text()
        passed = re.search(r"(\d+) passed", check_log)
        skipped = re.search(r"(\d+) skipped", check_log)
        result = {
            **pr,
            "head_sha": head,
            "passed": int(passed[1]) if passed else None,
            "skipped": int(skipped[1]) if skipped else 0,
            "checks": checks,
        }
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            f"PR #{number}: {result['passed']} passed, {result['skipped']} skipped; build passed ({head[:7]})",
            flush=True,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", str(checkout)],
            cwd=repository,
            check=True,
            capture_output=True,
        )
if len(results) != 8:
    raise RuntimeError(f"Expected all eight PRs; checked {len(results)}")
