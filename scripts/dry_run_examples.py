import os
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    examples_dir = project_root / "examples"

    # Mock keys if not present so dry-run environment check passes
    env = os.environ.copy()
    if "OPENROUTER_API_KEY" not in env:
        env["OPENROUTER_API_KEY"] = "mock_openrouter_key"
    if "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "mock_openai_key"

    yaml_files = sorted(examples_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} examples to dry-run...")

    failed = []
    for f in yaml_files:
        print(f"\nDry-running {f.name}...")
        cmd = ["uv", "run", "arka", "--config", str(f), "--dry-run"]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ Failed: {f.name}")
            print(res.stderr or res.stdout)
            failed.append(f.name)
        else:
            print("✅ Success")
            print(res.stdout.strip())

    if failed:
        print(f"\n❌ The following examples failed to dry-run: {failed}")
        sys.exit(1)
    else:
        print("\n🎉 All core examples dry-ran successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
