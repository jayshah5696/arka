"""Exercise the production decision transport and CLI without paid API calls.

Run with `uv run python scripts/verify_decisions_cli.py`. The temporary run
contains resolved config, stage artifacts, the dataset, and request payloads.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="arka-decisions-"))
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/decisions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(payload)
            answers = []
            for question in payload["questions"]:
                name, kind = question["name"], question["type"]
                answer = {"name": name, "type": kind}
                if kind == "predicate":
                    answer["probability"] = (
                        0.05 if name.endswith("is_incorrect") else 0.95
                    )
                else:
                    options = (
                        list(range(len(question["levels"])))
                        if kind == "score"
                        else [c["value"] for c in question["choices"]]
                    )
                    value = options[-1] if kind == "score" else options[0]
                    answer.update(
                        {
                            kind: value,
                            "confidence": 1,
                            "probabilities": [
                                {"value": option, "probability": float(option == value)}
                                for option in options
                            ],
                        }
                    )
                answers.append(answer)
            body = json.dumps(
                {"model": "fixture", "answers": answers, "usage": {"input_tokens": 10}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (work / "seeds.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "instruction": f"Explain topic {i}",
                    "response": f"Answer for topic {i}",
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    config = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "unused",
            "api_key": "fixture-secret",
            "base_url": "http://127.0.0.1:1/v1",
        },
        "decisions": {
            "provider": "openai",
            "model": "fixture",
            "api_key": "fixture-secret",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "input_cost_per_million": 1,
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "pipeline": [
            {"type": "seed_source", "path": str(work / "seeds.jsonl")},
            {"type": "normalize_conversation"},
            {"type": "seed_gate", "on_uncertain": "drop"},
            {"type": "decision_panel"},
            {
                "type": "labeling_engine",
                "backend": "decision",
                "rubric_path": str(root / "rubrics/sft_quality.yaml"),
                "min_overall_score": 3.5,
            },
            {"type": "complexity_elo", "backend": "decision"},
            {"type": "reward_model", "backend": "decision", "min_score": 6},
            {
                "type": "double_critic",
                "backend": "decision",
                "on_uncertain": "escalate",
            },
            {
                "type": "select",
                "target_count": 2,
                "weights": {"decisions.reward_model.helpfulness": 1},
                "confidence_weighting": True,
            },
        ],
        "output": {"format": "chatml", "path": "./dataset.jsonl"},
    }
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    command = [
        "uv",
        "run",
        "--project",
        str(root),
        "arka",
        "--config",
        str(config_path),
        "--run-id",
        "decision-cli",
    ]
    try:
        completed = subprocess.run(
            command, cwd=work, capture_output=True, text=True, timeout=60
        )
        (work / "cli.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        manifest = json.loads((work / "runs/decision-cli/manifest.json").read_text())
        assert manifest["final_count"] == 2
        assert manifest["decision_usage"]["requests"] == 9
        assert manifest["decision_usage"]["input_tokens"] == 90
        assert "Decision requests: 9" in completed.stdout
        assert len(requests) == 9
        inverse = [
            p
            for p in requests
            if any(q["name"] == "double_critic.is_incorrect" for q in p["questions"])
        ]
        assert len(inverse) == 3 and all(len(p["questions"]) == 1 for p in inverse)
        assert (
            "fixture-secret"
            not in (work / "runs/decision-cli/config.resolved.yaml").read_text()
        )
        scored = pl.read_parquet(
            work / "runs/decision-cli/stages/03_double_critic/data.parquet"
        )
        assert scored.height == 3
        assert len((work / "dataset.jsonl").read_text().splitlines()) == 2
        resumed = subprocess.run(
            [*command, "--resume"], cwd=work, capture_output=True, text=True, timeout=60
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert len(requests) == 9
        (work / "requests.json").write_text(json.dumps(requests, indent=2))
        print(f"Verified CLI decisions and resume. Evidence: {work}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
