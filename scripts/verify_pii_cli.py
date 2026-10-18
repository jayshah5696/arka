"""Verify PII boundaries through the CLI, SDK and a loopback provider.

Supply --model-path to include real, offline GLiNER2 inference. Model files
must be provisioned separately. This helper makes no paid provider calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from arka.pipeline.output import OutputWriter
from arka.privacy.config import PIIConfig
from arka.privacy.runtime import build_policy


def write_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cli = str(Path(sys.executable).with_name("arka"))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": payload})
            if self.path == "/v1/decisions":
                result = {
                    "model": "fixture",
                    "usage": {"input_tokens": 10},
                    "answers": [
                        {"name": q["name"], "type": "predicate", "probability": 0.01}
                        for q in payload["questions"]
                    ],
                }
            else:
                assert self.path == "/v1/chat/completions"
                schema = payload["response_format"]["json_schema"]["name"]
                if payload["model"] == "malformed":
                    content = '{"instruction":"broken\\u0040example.com"'
                elif schema == "EvolvedInstruction":
                    content = json.dumps(
                        {
                            "instruction": "Explain gravity to John Smith at evolved@example.com with detailed formulas and clear examples."
                        }
                    )
                elif schema == "EvolvedResponse":
                    content = json.dumps(
                        {
                            "response": "Contact John Smith at generated@example.com for the formulas."
                        }
                    )
                elif schema == "TransformResponse":
                    content = json.dumps(
                        {"text": "Contact generated@example.com for the answer."}
                    )
                else:
                    content = json.dumps(
                        {
                            "instruction": "Contact John Smith at generated@example.com. He lives at 123 Main Street, London.",
                            "response": "The requested explanation is ready.",
                            "reasoning": "Ask reason@example.com first.",
                        }
                    )
                result = {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    evidence = []

    def run_case(
        name: str,
        backend: str,
        *,
        pdf=False,
        evol=False,
        malformed=False,
        variant="prompt",
    ):
        request_start = len(requests)
        directory = root / name
        directory.mkdir()
        source_text = "Contact John Smith at source@example.com. He lives at 123 Main Street, London."
        seed_text = "Explain gravity" if evol else source_text
        (directory / "seeds.jsonl").write_text(
            json.dumps(
                {"instruction": seed_text, "response": "Gravity attracts masses."}
            )
            + "\n"
        )
        if pdf:
            write_pdf(directory / "source.pdf", source_text)
        pii = {"backend": backend, "entities": ["email"], "languages": ["en"]}
        if backend == "gliner2":
            pii.update(
                entities=["person", "email", "address"],
                model_path=str(args.model_path.resolve()),
            )
        pipeline = [
            {
                "type": "pdf_source",
                "path": "source.pdf",
                "decision_filter": {"enabled": True},
            }
            if pdf
            else {"type": "seed_source", "path": "seeds.jsonl"},
            {
                "type": "evol_instruct_generator",
                "rounds": 1,
                "branching_factor": 1,
                "operators": ["deepen"],
            }
            if evol
            else {
                "type": "prompt_based_generator",
                "target_count": 1,
                "generation_multiplier": 1,
            },
        ]
        if variant == "transform":
            pipeline[-1] = {
                "type": "transform_generator",
                "input_field": "payload.instruction",
                "output_field": "payload.response",
                "prompt_template": "Rewrite {input_text}",
            }
        elif variant == "taxonomy":
            (directory / "taxonomy.json").write_text(
                json.dumps(
                    {
                        "factors": [
                            {
                                "factor": "topic",
                                "root": {
                                    "name": "topic",
                                    "children": [{"name": "science"}],
                                },
                            }
                        ]
                    }
                )
            )
            pipeline = [
                {
                    "type": "taxonomy_generator",
                    "taxonomy_path": "taxonomy.json",
                    "target_count": 1,
                    "generation_multiplier": 1,
                }
            ]
        config = {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "malformed" if malformed else name,
                "api_key": "fixture",
                "base_url": url,
                "supports_json_schema": True,
                "max_retries": 1,
            },
            "decisions": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture",
                "base_url": url,
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pii": pii,
            "pipeline": pipeline,
            "output": {"format": "jsonl", "path": "dataset.jsonl"},
        }
        path = directory / "config.json"
        path.write_text(json.dumps(config, indent=2))
        command = [cli, "--config", str(path), "--run-id", name]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180, env=env
        )
        (directory / "cli.log").write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        original_export = (directory / "dataset.jsonl").read_bytes()
        calls = len(requests)
        resumed = subprocess.run(
            [*command, "--resume"], capture_output=True, text=True, timeout=180, env=env
        )
        (directory / "resume.log").write_text(resumed.stdout + resumed.stderr)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert len(requests) == calls
        assert (directory / "dataset.jsonl").read_bytes() == original_export
        forbidden = [
            "source@example.com",
            "generated@example.com",
            "reason@example.com",
            "evolved@example.com",
            "broken@example.com",
            "broken\\u0040example.com",
        ]
        if backend == "gliner2":
            forbidden += ["John Smith", "123 Main Street"]
        # Source input files are intentionally outside the protection contract.
        for artifact in directory.rglob("*"):
            if not artifact.is_file() or artifact.name in {"source.pdf", "seeds.jsonl"}:
                continue
            if artifact.suffix == ".parquet":
                content = json.dumps(pl.read_parquet(artifact).to_dicts())
            elif artifact.name == "state.db":
                with sqlite3.connect(artifact) as db:
                    content = "\n".join(db.iterdump())
            elif artifact.suffix in {".json", ".jsonl", ".log", ".yaml"}:
                content = artifact.read_text()
            else:
                continue
            assert not any(value in content for value in forbidden), artifact
        rows = [json.loads(line) for line in original_export.splitlines()]
        if pdf:
            source = OutputWriter().read_parquet(
                directory / f"runs/{name}/stages/01_source/data.parquet"
            )
            assert source[0].source.char_start == 0
            assert source[0].source.char_end == len(source_text)
        if malformed:
            assert not rows
            raw_path = next(directory.glob(f"runs/{name}/stages/*/raw_responses.jsonl"))
            raw = json.loads(raw_path.read_text())
            assert raw["usage"]["total_tokens"] == 30
            assert raw["privacy"]["drop_reason"] == "pii_unparseable_response"
        else:
            assert rows and "[email]" in original_export.decode()
        # Inspect requests at the remote boundary as well as stored artifacts.
        assert not any(
            value in json.dumps(requests[request_start:]) for value in forbidden
        )
        evidence.append(
            {
                "case": name,
                "backend": backend,
                "status": "passed",
                "export_count": len(rows),
                "resume_provider_calls": 0,
            }
        )
        print(f"Passed {name}", flush=True)

    try:
        run_case("regex-pdf", "regex", pdf=True)
        run_case("malformed", "regex", malformed=True)
        run_case("regex-evol", "regex", evol=True)
        run_case("regex-transform", "regex", variant="transform")
        run_case("regex-taxonomy", "regex", variant="taxonomy")
        if args.model_path:
            run_case("gliner-pdf", "gliner2", pdf=True)
            run_case("gliner-evol", "gliner2", evol=True)
            policy = build_policy(
                PIIConfig(
                    backend="gliner2",
                    languages=["en"],
                    entities=["person", "email"],
                    model_path=str(args.model_path.resolve()),
                ),
                root,
            )
            original = (
                "Contact John Smith at first@example.com. "
                + "A general sentence about gravity. " * 130
                + "Contact John Smith at last@example.com."
            )
            protected = policy.protect(original)
            assert (
                "John Smith" not in protected.value
                and "@example.com" not in protected.value
            )
            assert protected.value.count("[email]") == 2
            evidence.append(
                {
                    "case": "gliner-long-repeated-spans",
                    "status": "passed",
                    "input_characters": len(original),
                    "categories": protected.categories,
                }
            )
        (root / "requests.json").write_text(json.dumps(requests, indent=2))
        (root / "result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "cases": evidence,
                    "provider_calls": len(requests),
                    "gliner_model_verified": bool(args.model_path),
                },
                indent=2,
            )
        )
        print(f"PII verification passed. Evidence: {root}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
