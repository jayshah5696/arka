import os
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    examples_dir = project_root / "examples"

    env = os.environ.copy()
    if "OPENROUTER_API_KEY" not in env:
        print("❌ Error: OPENROUTER_API_KEY must be set to run execution verification.")
        sys.exit(1)

    yaml_files = sorted(examples_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} examples to execute...")

    failed = []
    for f in yaml_files:
        run_id = f"verify-{f.stem}"
        print("\n==================================================")
        print(f"Executing {f.name} (run_id: {run_id})...")
        print("==================================================")
        cmd = ["uv", "run", "arka", "--config", str(f), "--run-id", run_id]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ Execution failed for {f.name}:")
            print(res.stderr or res.stdout)
            failed.append(f.name)
        else:
            print(f"✅ Execution succeeded for {f.name}!")
            print(res.stdout.strip())

            # Verify output dataset exists and is non-empty
            dataset_name = f"{f.stem}-dataset.jsonl"

            out_file = examples_dir / "output" / dataset_name
            if out_file.exists():
                lines = out_file.read_text().strip().split("\n")
                lines = [line for line in lines if line]
                print(f"Generated dataset at: {out_file.relative_to(project_root)}")
                print(f"Total records in dataset: {len(lines)}")
                if not lines:
                    print("❌ Error: Generated dataset is empty!")
                    failed.append(f.name)
            else:
                print(f"❌ Error: Output dataset not found at {out_file}")
                failed.append(f.name)

    if failed:
        print(f"\n❌ The following examples failed execution: {failed}")
        sys.exit(1)
    else:
        print("\n🎉 All core examples executed successfully and generated datasets!")
        sys.exit(0)


if __name__ == "__main__":
    main()
