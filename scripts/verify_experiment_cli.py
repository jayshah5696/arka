"""Verify generated experiment configs, named model dispatch, and interrupted resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    work = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="arka-experiment-"))
    )
    if args.output:
        work.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {work}", flush=True)
    cli = str(Path(sys.executable).with_name("arka"))
    requests: list[dict] = []
    fail = threading.Event()
    fail.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            index = len(requests)
            if fail.is_set() and index >= 2:
                body = json.dumps(
                    {
                        "error": {
                            "message": "fixture interruption",
                            "type": "authentication_error",
                        }
                    }
                ).encode()
                status = 401
            else:
                body = json.dumps(
                    {
                        "id": f"fixture-{index}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "instruction": f"Explain example {index}",
                                            "response": f"Example {index} has enough distinct detail to pass the length filter.",
                                        }
                                    ),
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    }
                ).encode()
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {**os.environ, "ARKA_API_KEY": "experiment-fixture-secret"}

    def execute(name: str, command: list[str], success: bool = True) -> str:
        result = subprocess.run(
            command, cwd=work, env=env, capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        (work / f"{name}.log").write_text(output)
        assert (result.returncode == 0) == success, output
        return output

    try:
        path = work / "experiment.yaml"
        execute(
            "init",
            [
                str(Path(sys.executable).with_name("arka-init")),
                "--preset",
                "sft",
                "--output",
                str(path),
                "--source",
                "seeds.jsonl",
                "--model",
                "fixture-writer",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--target-count",
                "3",
            ],
        )
        data = yaml.safe_load(path.read_text())
        data["models"]["writer"]["supports_json_schema"] = True
        data["models"]["unused"] = {
            **data["models"]["writer"],
            "model": "must-not-call",
            "base_url": "http://127.0.0.1:1/v1",
        }
        data["default_model"] = "unused"
        data["execution"] = {"max_workers": 1, "timeout_seconds": 5, "max_retries": 1}
        data["pipeline"][2]["config"]["generation_multiplier"] = 1
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        seed = work / "seeds.jsonl"
        seed.write_text(
            '{"instruction":"Explain water","response":"Water is a liquid."}\n'
        )
        command = [cli, "--config", str(path), "--run-id", "experiment"]
        execute("validate", [*command, "--validate-config"])
        preview = execute("plan", [*command, "--dry-run"])
        assert "Experiment fingerprint:" in preview
        assert not (work / "runs").exists() and not requests
        execute("interrupted", command, success=False)
        run = work / "runs/experiment"
        raw_path = run / "stages/generate/raw_responses.jsonl"
        first_row = raw_path.read_text().splitlines()[0]
        assert json.loads(first_row)["model"] == "fixture-writer"
        initial_config = (run / "config.resolved.yaml").read_bytes()
        initial_plan = (run / "plan.json").read_bytes()
        fail.clear()
        # Keep one worker so the request delta also proves completed slot reuse.
        data["execution"].update(timeout_seconds=15, max_retries=2)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        count_before_resume = len(requests)
        execute("resume-interrupted", [*command, "--resume"])
        assert len(requests) - count_before_resume == 2
        assert raw_path.read_text().splitlines()[0] == first_row
        dataset = work / "output/dataset.jsonl"
        original_dataset = dataset.read_bytes()
        assert len(original_dataset.splitlines()) == 3
        data["execution"]["max_workers"] = 3
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        calls = len(requests)
        execute("resume-completed", [*command, "--resume"])
        assert len(requests) == calls
        assert dataset.read_bytes() == original_dataset
        assert (run / "config.resolved.yaml").read_bytes() == initial_config
        assert (run / "plan.json").read_bytes() == initial_plan
        assert len((run / "execution.history.jsonl").read_text().splitlines()) == 3
        snapshot = {
            p.relative_to(work): p.read_bytes()
            for p in [*run.rglob("*"), work / "state.db"]
            if p.is_file()
        }
        original_seed = seed.read_bytes()
        seed.write_text(
            '{"instruction":"Different source","response":"Different answer"}\n'
        )
        rejected = execute("reject-source", [*command, "--resume"], success=False)
        assert "new run" in rejected.lower()
        seed.write_bytes(original_seed)
        data["models"]["writer"]["model"] = "changed-writer"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        rejected = execute("reject-model", [*command, "--resume"], success=False)
        assert "new run" in rejected.lower()
        assert len(requests) == calls
        assert all(
            (work / name).read_bytes() == content for name, content in snapshot.items()
        )
        assert (
            "experiment-fixture-secret" not in (initial_config + initial_plan).decode()
        )
        assert all(request["model"] == "fixture-writer" for request in requests)
        (work / "requests.json").write_text(json.dumps(requests, indent=2))
        (work / "result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "export_count": 3,
                    "provider_calls": len(requests),
                    "interrupted_resume_calls": 2,
                    "completed_resume_calls": 0,
                    "incompatible_resume_calls": 0,
                    "original_config_preserved": True,
                    "rejected_run_state_preserved": True,
                },
                indent=2,
            )
        )
        print(f"Verified generated experiments and resume: {work}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
