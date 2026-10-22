"""Prove source audit, PDF rebuilding, and interrupted resume through installed CLIs.

All provider calls use a loopback fixture. Preserve configs, requests, reports,
Parquet artifacts, and command logs in a fresh --output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from arka.pipeline.output import OutputWriter
from arka.records.identity import file_hash
from arka.recovery.journal import write_json


def make_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
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
    stream.set_data(
        b"BT /F1 12 Tf 20 200 Td (The reference answer is forty-two.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def verify(root: Path, backend: str) -> dict:
    root.mkdir()
    receipts = []
    state = {"source": "pdf", "generated": 0, "interrupted": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            receipt = {"path": self.path, "body": body}
            receipts.append(receipt)
            decision = "questions" in body
            properties = (
                {}
                if decision
                else body["response_format"]["json_schema"]["schema"]["properties"]
            )
            judging = decision or "scores" in properties
            if decision:
                record = json.loads(body["input"])
            elif "instruction" not in properties:
                record = json.loads(body["messages"][-1]["content"])
            else:
                record = None
            if (
                judging
                and record["response"] == "good repaired"
                and not state["interrupted"]
            ):
                state["interrupted"] = True
                receipt["status"] = 503
                self.reply(
                    503,
                    {
                        "error": {
                            "message": "Fixture interruption",
                            "type": "server_error",
                        }
                    },
                )
                return
            receipt["status"] = 200
            if decision:
                level = 2 if record["response"].startswith("good") else 0
                result = {
                    "model": "local-fixture",
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
                if "instruction" in properties:
                    result_data = {
                        "instruction": f"Explain the {state['source']} fact.",
                        "response": "bad original",
                        "reasoning": None,
                    }
                elif judging:
                    result_data = {
                        "scores": {
                            "correct": 3 if record["response"].startswith("good") else 1
                        },
                        "reasoning": "Local assessment",
                    }
                else:
                    state["generated"] += 1
                    result_data = {
                        "response": "bad retry"
                        if state["generated"] == 1
                        else "good repaired",
                        "reasoning": None,
                    }
                result = {
                    "id": f"fixture-{len(receipts)}",
                    "object": "chat.completion",
                    "model": "local-fixture",
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
            self.reply(200, result)

        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm = {
            "provider": "openai",
            "model": "local-fixture",
            "api_key": "local-fixture",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "max_retries": 1,
            "supports_json_schema": True,
        }

        def run(name, command, *args, success=True):
            result = subprocess.run(
                [str(Path(sys.executable).with_name(command)), *args],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (root / f"{name}.log").write_text(result.stdout + result.stderr)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result

        pdf = root / "source.pdf"
        make_pdf(pdf)
        seed = root / "seed.jsonl"
        seed.write_text(
            json.dumps(
                {"instruction": "Explain the seed fact.", "response": "Seed input"}
            )
            + "\n"
        )
        artifacts = OutputWriter()
        originals = []
        source_records = []
        for source, stage in [
            ("pdf", {"type": "pdf_source", "path": str(pdf)}),
            ("seed", {"type": "seed_source", "path": str(seed)}),
        ]:
            state["source"] = source
            config = {
                "version": "1",
                "llm": llm,
                "executor": {"mode": "threadpool", "max_workers": 1},
                "pipeline": [
                    stage,
                    {
                        "type": "prompt_based_generator",
                        "target_count": 1,
                        "generation_multiplier": 1,
                    },
                ],
                "output": {"format": "jsonl", "path": f"{source}-dataset.jsonl"},
            }
            config_path = root / f"{source}.json"
            write_json(config_path, config)
            run(
                source,
                "arka",
                "--config",
                str(config_path),
                "--run-id",
                f"{source}-parent",
            )
            directory = root / "runs" / f"{source}-parent"
            manifest = json.loads((directory / "manifest.json").read_text())
            source_records.extend(
                artifacts.read_parquet(
                    directory / "stages" / manifest["stage_names"][0] / "data.parquet"
                )
            )
            originals.extend(
                artifacts.read_parquet(
                    directory / "stages" / manifest["stage_names"][-1] / "data.parquet"
                )
            )
        # Assemble one fixture population from the two real pipeline runs so
        # source selection can prove that an unselected source stays unchanged.
        parent = root / "runs/parent"
        artifacts.write_parquet(source_records, parent / "stages/source/data.parquet")
        artifacts.write_parquet(originals, parent / "stages/final/data.parquet")
        write_json(
            parent / "manifest.json",
            {"run_id": "parent", "stage_names": ["source", "final"]},
        )
        protected_parent = root / "runs/protected-parent"
        artifacts.write_parquet(
            source_records, protected_parent / "stages/source/data.parquet"
        )
        artifacts.write_parquet(
            originals, protected_parent / "stages/final/data.parquet"
        )
        write_json(
            protected_parent / "manifest.json",
            {"run_id": "protected-parent", "stage_names": ["source", "final"]},
        )
        write_json(
            protected_parent / "config.resolved.yaml",
            {"pii": {"backend": "regex", "languages": ["en"], "entities": ["email"]}},
        )
        rubric = {
            "version": "1",
            "description": "Answer correctness",
            "dimensions": [
                {
                    "name": "correct",
                    "description": "Does the answer match the source evidence?",
                    "scale_min": 1,
                    "scale_max": 3,
                    "criteria": {1: "Wrong", 2: "Partial", 3: "Correct"},
                }
            ],
            "overall_weights": {"correct": 1},
        }
        write_json(root / "rubric.json", rubric)
        judge = {
            "backend": backend,
            "rubric_path": "rubric.json",
            "min_overall_score": 3,
        }
        judge["llm" if backend == "llm" else "decision"] = (
            llm
            if backend == "llm"
            else {
                key: value
                for key, value in llm.items()
                if key != "supports_json_schema"
            }
        )
        config_path = root / "recovery.json"
        write_json(
            config_path,
            {"judge": judge, "replacement": {"llm": llm, "max_attempts": 2}},
        )
        before_protected = len(receipts)
        protected = run(
            "protected-parent",
            "arka-audit",
            "--config",
            str(config_path),
            "--parent-run",
            "protected-parent",
            "--run-id",
            "protected-audit",
            success=False,
        )
        assert "PII" in protected.stderr and len(receipts) == before_protected
        run(
            "audit",
            "arka-audit",
            "--config",
            str(config_path),
            "--parent-run",
            "parent",
            "--run-id",
            "audit",
        )
        source_key = f"pdf:{file_hash(pdf)}"
        args = [
            "--config",
            str(config_path),
            "--audit-run",
            "audit",
            "--run-id",
            "rebuilt",
            "--source",
            source_key,
        ]
        run("interrupted", "arka-rebuild", *args, success=False)
        assert state["generated"] == 2
        run("resumed", "arka-rebuild", *args, "--resume")
        assert state["generated"] == 2, "Resume generated a saved replacement again"
        count = len(receipts)
        dataset_path = root / "runs/rebuilt/dataset.jsonl"
        before = dataset_path.read_bytes()
        run("completed-resume", "arka-rebuild", *args, "--resume")
        assert len(receipts) == count and dataset_path.read_bytes() == before
        directory = root / "runs/rebuilt"
        records = artifacts.read_parquet(directory / "stages/rebuild/data.parquet")
        assert records[0].payload.instruction == originals[0].payload.instruction
        assert records[0].payload.response == "good repaired"
        assert records[1] == originals[1]
        assert records[0].lineage.parent_ids == [originals[0].id]
        attempts = artifacts.read_parquet(directory / "stages/rebuild/attempts.parquet")
        assert len(attempts) == 2
        assert [r.stage_events[-1].reason_code for r in attempts] == [
            "quality_failed",
            "accepted",
        ]
        report = json.loads((directory / "report/recovery_report.json").read_text())
        assert report["attempt_count"] == 2 and report["accepted_replacements"] == 1
        assert report["unresolved_count"] == 1 and report["final_count"] == 2
        expected_evidence = next(
            r.payload.text for r in source_records if r.source.type == "pdf_chunk"
        )
        evidence_requests = []
        for receipt in receipts:
            body = receipt["body"]
            state_json = body.get("input")
            if state_json is None and "messages" in body:
                state_json = body["messages"][-1]["content"]
            try:
                record = json.loads(state_json)
            except (ValueError, TypeError):
                continue
            if record.get("instruction") == originals[0].payload.instruction:
                assert record["source_evidence"] == expected_evidence
                evidence_requests.append(receipt)
        assert len(evidence_requests) == 6
        # A rebuilt run retains enough ancestry to become another audit input.
        run(
            "reaudit",
            "arka-audit",
            "--config",
            str(config_path),
            "--parent-run",
            "rebuilt",
            "--run-id",
            "reaudit",
        )
        write_json(root / "requests.json", receipts)
        return {
            "backend": backend,
            "status": "passed",
            "requests_before_reaudit": count,
            "generation_attempts": state["generated"],
            "exact_evidence_requests": len(evidence_requests),
            "final_count": report["final_count"],
            "unresolved_count": report["unresolved_count"],
            "report": str(directory / "report/recovery_report.json"),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = [verify(root / backend, backend) for backend in ["llm", "decision"]]
    write_json(root / "result.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
