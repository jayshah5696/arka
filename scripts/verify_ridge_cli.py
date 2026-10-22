"""Verify separate ridge training, scoring, privacy rejection and resume via CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash
from arka.records.models import ConversationRecord


class Embeddings(BaseHTTPRequestHandler):
    requests = 0

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return
        type(self).requests += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vectors = []
        for text in body["input"]:
            instruction = (
                json.loads(text)["instruction"] if text.startswith("{") else text
            )
            vectors.append([float(instruction.split()[-1]), 1.0])
        payload = json.dumps(
            {
                "object": "list",
                "model": "fixture-embedding-v1",
                "data": [
                    {"object": "embedding", "index": i, "embedding": v}
                    for i, v in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="arka-ridge-proof-"))
    )
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Embeddings)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    command_index = 0

    def run(name, *args, expected=0):
        nonlocal command_index
        command_index += 1
        executable = str(Path(sys.executable).with_name(name))
        result = subprocess.run([executable, *args], capture_output=True, text=True)
        (root / f"command-{command_index}.log").write_text(
            result.stdout + result.stderr
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout + result.stderr

    try:
        records = [
            ConversationRecord.model_validate(
                {
                    "id": str(i),
                    "content_hash": "pending",
                    "source": {"type": "fixture", "doc_id": f"doc-{i // 2}"},
                    "lineage": {"root_id": f"root-{i // 2}", "parent_ids": []},
                    "payload": {
                        "instruction": f"Question {i}",
                        "response": f"Answer {i}",
                    },
                    "scores": {
                        "quality": float(i),
                        "rubric_hash": "fixture-rubric",
                        "judge_model": "fixture-judge",
                    },
                    "config_hash": "fixture",
                    "created_at": "2026-10-21T00:00:00Z",
                }
            )
            for i in range(30)
        ]
        for record in records:
            record.content_hash = content_hash(record.payload)
        OutputWriter().write_parquet(records, root / "judged.parquet")
        (root / "seeds.jsonl").write_text(
            "".join(json.dumps(r.export_payload()) + "\n" for r in records)
        )
        config = {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "unused",
                "api_key": "fixture",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "embeddings": {"provider": "openai", "model": "fixture-embedding-v1"},
            "pipeline": [],
            "output": {"format": "jsonl", "path": "ranked.jsonl"},
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        for version, features in [
            (1, "instruction_response_json_v1"),
            (2, "pair_quality_v2"),
        ]:
            model_path = root / f"model-{version}.json"
            run(
                "arka-train-scorer",
                "--config",
                str(path),
                "--records",
                str(root / "judged.parquet"),
                "--output",
                str(model_path),
                "--features",
                features,
            )
            model = json.loads(model_path.read_text())
            assert model["version"] == version
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            config["pipeline"] = [
                {"type": "seed_source", "path": "seeds.jsonl"},
                {"type": "normalize_conversation"},
                {
                    "type": "distilled_ridge",
                    "model_path": model_path.name,
                    "model_sha256": digest,
                },
                {
                    "type": "select",
                    "target_count": 3,
                    "weights": {"distilled_ridge": 1},
                },
            ]
            path.write_text(yaml.safe_dump(config))
            run("arka", "--config", str(path), "--run-id", f"proof-{version}")
            scored = OutputWriter().read_parquet(
                root / f"runs/proof-{version}/stages/02s_distilled_ridge/data.parquet"
            )
            assert len(scored) == 30
            assert all(
                r.scores.distilled_ridge_prediction.model_sha256 == digest
                for r in scored
            )
            output = [
                json.loads(line)
                for line in (root / "ranked.jsonl").read_text().splitlines()
            ]
            assert len(output) == 3
            if version == 1:
                assert [r["instruction"] for r in output] == [
                    "Question 29",
                    "Question 28",
                    "Question 27",
                ]
            before = Embeddings.requests
            run(
                "arka",
                "--config",
                str(path),
                "--run-id",
                f"proof-{version}",
                "--resume",
            )
            assert Embeddings.requests == before
            model_path.write_text(model_path.read_text() + " ")
            run(
                "arka",
                "--config",
                str(path),
                "--run-id",
                f"proof-{version}",
                "--resume",
                expected=1,
            )
            assert Embeddings.requests == before
        config["pipeline"] = []
        config["pii"] = {"backend": "regex", "languages": ["en"], "entities": ["email"]}
        path.write_text(yaml.safe_dump(config))
        records[0].payload.response = "Contact person@example.com"
        records[0].content_hash = content_hash(records[0].payload)
        OutputWriter().write_parquet(records, root / "private.parquet")
        before = Embeddings.requests
        failure = run(
            "arka-train-scorer",
            "--config",
            str(path),
            "--records",
            str(root / "private.parquet"),
            "--output",
            str(root / "private-model.json"),
            expected=1,
        )
        assert "privacy" in failure.lower()
        assert Embeddings.requests == before
        assert not (root / "private-model.json").exists()
        result = {
            "status": "passed",
            "records": 30,
            "versions": [1, 2],
            "selected_per_version": 3,
            "resume_extra_requests": 0,
            "changed_model_rejected_before_requests": True,
            "private_training_rejected_before_requests": True,
            "embedding_requests": Embeddings.requests,
            "quality_claim": "Deterministic transport and behavior proof only",
        }
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"evidence": str(root), **result}, indent=2))
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
