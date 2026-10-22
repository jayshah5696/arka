"""Exercise installed commands and both provider protocols on loopback HTTP.

Fail before implementation if packaging, source lookup, candidate persistence,
backend score scales, dataset assembly, or CLI resume is broken.
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


@pytest.mark.parametrize("backend", ["llm", "decision"])
def test_installed_audit_rebuild_and_resume(tmp_path, backend):
    receipts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            receipts.append(body)
            if "questions" in body:
                state = json.loads(body["input"])
                level = 2 if state["response"].startswith("good") else 0
                result = {
                    "model": "fixture",
                    "answers": [
                        {
                            "name": q["name"],
                            "type": "score",
                            "score": level,
                            "probabilities": [
                                {"value": i, "probability": float(i == level)}
                                for i in range(3)
                            ],
                            "confidence": 1,
                        }
                        for q in body["questions"]
                    ],
                    "usage": {"input_tokens": 10},
                }
            else:
                state = json.loads(body["messages"][-1]["content"])
                schema = body["response_format"]["json_schema"]["schema"]
                if "scores" in schema["properties"]:
                    result_data = {
                        "scores": {
                            "correct": 3 if state["response"].startswith("good") else 1
                        },
                        "reasoning": "Fixture assessment",
                    }
                else:
                    result_data = {"response": "good replacement", "reasoning": None}
                result = {
                    "id": "fixture",
                    "object": "chat.completion",
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(result_data),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        llm = {
            "provider": "openai",
            "model": "fixture",
            "api_key": "fixture-key",
            "base_url": base_url,
            "supports_json_schema": True,
            "max_retries": 1,
        }
        judge = {
            "backend": backend,
            "rubric_path": "rubric.yaml",
            "min_overall_score": 3,
        }
        judge["llm" if backend == "llm" else "decision"] = (
            llm
            if backend == "llm"
            else {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture-key",
                "base_url": base_url,
                "max_retries": 1,
            }
        )
        cfg = {"judge": judge, "replacement": {"llm": llm, "max_attempts": 2}}
        (tmp_path / "recovery.yaml").write_text(yaml.safe_dump(cfg))
        (tmp_path / "rubric.yaml").write_text(
            yaml.safe_dump(
                {
                    "version": "1",
                    "description": "quality",
                    "dimensions": [
                        {
                            "name": "correct",
                            "description": "correctness",
                            "scale_min": 1,
                            "scale_max": 3,
                            "criteria": {1: "wrong", 2: "partial", 3: "correct"},
                        }
                    ],
                    "overall_weights": {"correct": 1},
                }
            )
        )
        records = []
        for name, answer, source in [
            ("fix", "bad", "a"),
            ("keep", "good original", "a"),
            ("outside", "bad", "b"),
        ]:
            payload = ConversationPayload(instruction=name, response=answer)
            records.append(
                ConversationRecord(
                    id=name,
                    content_hash=content_hash(payload),
                    payload=payload,
                    source=RecordSource(type="seed", seed_file_hash=source),
                    lineage=RecordLineage(root_id=name, parent_ids=[]),
                    scores=RecordScores(),
                    config_hash="parent",
                    created_at="2026-10-21T00:00:00Z",
                )
            )
        parent_dir = tmp_path / "runs/parent"
        OutputWriter().write_parquet(records, parent_dir / "stages/final/data.parquet")
        (parent_dir / "manifest.json").write_text(
            json.dumps({"run_id": "parent", "stage_names": ["final"]})
        )

        def run(command, *args):
            result = subprocess.run(
                [
                    command,
                    "--config",
                    str(tmp_path / "recovery.yaml"),
                    "--project-root",
                    str(tmp_path),
                    *args,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            return result

        run("arka-audit", "--parent-run", "parent", "--run-id", "audit")
        args = ("--audit-run", "audit", "--run-id", "rebuilt", "--source", "seed:a")
        run("arka-rebuild", *args)
        count = len(receipts)
        run("arka-rebuild", *args, "--resume")
        assert len(receipts) == count == 5
        output = [
            json.loads(line)
            for line in (tmp_path / "runs/rebuilt/dataset.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [x["response"] for x in output] == [
            "good replacement",
            "good original",
            "bad",
        ]
        report = json.loads(
            (tmp_path / "runs/rebuilt/report/recovery_report.json").read_text()
        )
        assert report["accepted_replacements"] == 1 and report["unresolved_count"] == 1
        assert report["final_count"] == 3
        assert all(
            "fixture-key" not in p.read_text()
            for p in (tmp_path / "runs").glob("*/operation.json")
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
