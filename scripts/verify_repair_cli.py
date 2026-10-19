"""Verify bounded repair through the installed CLI and local HTTP providers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from arka.pipeline.output import OutputWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    work = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="arka-repair-"))
    )
    if args.output:
        work.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {work}", flush=True)
    requests = []
    interrupted = False
    cli = str(Path(sys.executable).with_name("arka"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal interrupted
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "payload": payload})
            if self.path == "/v1/decisions":
                state = payload["input"]
                score = 1 if '"response": "BAD"' in state else 4
                answers = []
                for question in payload["questions"]:
                    value = score if question["name"].endswith("correctness") else 4
                    answers.append(
                        {
                            "name": question["name"],
                            "type": "score",
                            "score": value,
                            "confidence": 1,
                            "probabilities": [
                                {"value": i, "probability": float(i == value)}
                                for i in range(5)
                            ],
                        }
                    )
                self.send_json(
                    {
                        "model": "fixture",
                        "answers": answers,
                        "usage": {"input_tokens": 10},
                    }
                )
                return
            assert self.path == "/v1/chat/completions", self.path
            name = payload["response_format"]["json_schema"]["name"]
            user = payload["messages"][-1]["content"]
            if name == "GeneratedConversation":
                result = {"instruction": "Explain grounded answer", "response": "BAD"}
            elif name == "RepairResponse":
                result = {
                    "response": "Still BAD" if "Exhaust" in user else "Fixed answer"
                }
                if "Privacy" in user:
                    result = {"response": "Contact alice@example.com"}
            else:
                assert name == "JudgeResponse", name
                is_bad = (
                    "Response: <text>BAD" in user or "Response: <text>Still BAD" in user
                )
                if "Interrupt" in user and not is_bad and not interrupted:
                    interrupted = True
                    self.send_json(
                        {
                            "error": {
                                "message": "fixture interruption",
                                "type": "server_error",
                            }
                        },
                        status=503,
                    )
                    return
                result = {
                    "scores": {
                        "correctness": 1 if is_bad else 4,
                        "clarity": 1 if "Unclear" in user else 4,
                    },
                    "reasoning": "Correct the failed criterion",
                }
            self.send_json(
                {
                    "id": f"fixture-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(result),
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "total_cost": 0.01,
                    },
                }
            )

        def send_json(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    rubric = {
        "version": "1",
        "description": "Answer quality",
        "dimensions": [
            {
                "name": name,
                "description": name,
                "scale_min": 0,
                "scale_max": 4,
                "criteria": {0: "Wrong", 4: "Good"},
            }
            for name in ("correctness", "clarity")
        ],
        "overall_weights": {"correctness": 0.5, "clarity": 0.5},
    }
    summary = {"status": "running", "scenarios": [], "evidence": str(work)}
    (work / "result.json").write_text(json.dumps(summary, indent=2))

    def run(directory, config, name, *, resume=False, expect_success=True):
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        command = [cli, "--config", str(path), "--run-id", "repair"] + (
            ["--resume"] if resume else []
        )
        completed = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, timeout=90
        )
        (directory / f"{name}.log").write_text(completed.stdout + completed.stderr)
        if (completed.returncode == 0) != expect_success:
            raise RuntimeError(completed.stdout + completed.stderr)
        return completed

    def scenario(
        name,
        seeds,
        *,
        max_repairs=3,
        decision=False,
        pdf=False,
        missing_evidence=False,
        pii=False,
    ):
        directory = work / name
        directory.mkdir()
        (directory / "rubric.yaml").write_text(yaml.safe_dump(rubric))
        (directory / "seeds.jsonl").write_text(
            "".join(
                json.dumps({"instruction": i, "response": r}) + "\n" for i, r in seeds
            )
        )
        source = [
            {"type": "seed_source", "path": "./seeds.jsonl"},
            {"type": "normalize_conversation"},
        ]
        if pdf:
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            stream = DecodedStreamObject()
            stream.set_data(
                b"BT /F1 12 Tf 20 100 Td (The permitted answer is forty two.) Tj ET"
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/Font"),
                                    NameObject("/Subtype"): NameObject("/Type1"),
                                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                                }
                            )
                        }
                    )
                }
            )
            writer.write(directory / "source.pdf")
            source = [
                {"type": "pdf_source", "path": "./source.pdf"},
                {
                    "type": "prompt_based_generator",
                    "target_count": 1,
                    "generation_multiplier": 1,
                },
            ]
        repair = {
            "type": "quality_repair",
            "rubric_path": "./rubric.yaml",
            "min_scores": {"correctness": 3, "clarity": 3},
            "repair_dimensions": ["correctness"],
            "max_repairs": max_repairs,
        }
        if missing_evidence:
            repair["evidence_stage"] = "missing"
        config = {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture-secret",
                "base_url": base_url,
                "max_retries": 1,
                "timeout_seconds": 10,
                "supports_json_schema": True,
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pipeline": [*source, repair, {"type": "exact"}],
            "output": {"format": "chatml", "path": "./dataset.jsonl"},
        }
        if decision:
            config["decisions"] = {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture-secret",
                "base_url": base_url,
                "max_retries": 1,
                "input_cost_per_million": 1,
            }
            repair["backend"] = "decision"
            # The panel must not prefetch evidence-dependent repair judgments.
            config["pipeline"].insert(len(source), {"type": "decision_panel"})
        if pii:
            config["pii"] = {
                "backend": "regex",
                "languages": ["en"],
                "entities": ["email"],
                "blocked_entities": ["email"],
            }
        return directory, config

    def artifacts(directory):
        stage = directory / "runs/repair/stages/03_quality_repair"
        return (
            stage,
            json.loads((stage / "stats.json").read_text()),
            OutputWriter().read_parquet(stage / "data.parquet"),
        )

    def resume_proof(directory, config):
        before = len(requests)
        dataset = (directory / "dataset.jsonl").read_bytes()
        run(directory, config, "resume", resume=True)
        assert len(requests) == before
        assert (directory / "dataset.jsonl").read_bytes() == dataset
        assert (
            "fixture-secret"
            not in (directory / "runs/repair/config.resolved.yaml").read_text()
        )

    try:
        directory, config = scenario(
            "bounded",
            [
                ("Explain shared", "Fixed answer"),
                ("Explain shared", "BAD"),
                ("Rescue", "BAD"),
                ("Exhaust", "BAD"),
                ("Unclear", "BAD"),
                ("Limit", "BAD"),
            ],
        )
        start = len(requests)
        run(directory, config, "cli")
        stage, stats, records = artifacts(directory)
        assert (
            stats["initially_passing"] == 1
            and stats["attempted_repairs"] == 3
            and stats["accepted_repairs"] == 2
        )
        assert stats["drop_reasons"] == {
            "repair_exhausted": 1,
            "repair_ineligible_dimensions": 1,
            "repair_limit": 1,
        }
        assert len(requests) - start == 12 and stats["llm_usage"]["requests"] == 12
        assert (
            len(records) == 3
            and len((directory / "dataset.jsonl").read_text().splitlines()) == 2
        )
        assert (
            len(
                OutputWriter().read_parquet(
                    directory / "runs/repair/stages/02c_exact_dedup/dropped.parquet"
                )
            )
            == 1
        )
        assert (
            records[1].lineage.operator == "quality_repair"
            and records[1].payload.instruction == "Explain shared"
        )
        source_parquet = directory / "runs/repair/stages/01_source/data.parquet"
        source_bytes = source_parquet.read_bytes()
        before = len(requests)
        missing_resume = run(
            directory, config, "missing-resume", resume=False, expect_success=False
        )
        assert (
            "Quality repair artifacts exist; resume or use a new run ID"
            in missing_resume.stderr
            and len(requests) == before
            and source_parquet.read_bytes() == source_bytes
        )
        resume_proof(directory, config)
        rubric_path = directory / "rubric.yaml"
        original_rubric = rubric_path.read_text()
        rubric_path.write_text(
            original_rubric.replace("Answer quality", "Changed quality")
        )
        changed = run(
            directory, config, "changed-rubric", resume=True, expect_success=False
        )
        assert "new run ID" in changed.stderr
        rubric_path.write_text(original_rubric)
        input_path = directory / "runs/repair/stages/02_normalize/data.parquet"
        original_inputs = input_path.read_bytes()
        inputs = OutputWriter().read_parquet(input_path)
        inputs[0] = inputs[0].model_copy(
            update={
                "payload": inputs[0].payload.model_copy(
                    update={"response": "Changed input"}
                )
            }
        )
        OutputWriter().write_parquet(inputs, input_path)
        before = len(requests)
        changed = run(
            directory, config, "changed-input", resume=True, expect_success=False
        )
        assert "new run ID" in changed.stderr and len(requests) == before
        input_path.write_bytes(original_inputs)
        summary["scenarios"].append(
            {"name": "bounded", "requests": 12, "final_count": 2}
        )

        directory, config = scenario("baseline", [("Rescue", "BAD")], max_repairs=0)
        run(directory, config, "cli")
        assert artifacts(directory)[1]["drop_reasons"] == {"repair_limit": 1}
        assert not (directory / "dataset.jsonl").read_bytes()
        resume_proof(directory, config)
        summary["scenarios"].append({"name": "baseline", "final_count": 0})

        for name, options in [
            ("pdf", {"pdf": True}),
            ("missing-evidence", {"pdf": True, "missing_evidence": True}),
            ("decision", {"decision": True}),
            ("privacy", {"pii": True}),
        ]:
            instruction = "Privacy" if name == "privacy" else "Rescue"
            directory, config = scenario(name, [(instruction, "BAD")], **options)
            start = len(requests)
            run(directory, config, "cli")
            stage, stats, records = artifacts(directory)
            if name in {"pdf", "decision"}:
                assert len(records) == 1 and stats["accepted_repairs"] == 1
            elif name == "missing-evidence":
                assert (
                    stats["drop_reasons"] == {"repair_missing_evidence": 1}
                    and len(requests) - start == 1
                )
            else:
                assert not records and len(requests) - start == 2
                assert any(
                    reason.startswith("pii_") for reason in stats["drop_reasons"]
                )
                assert stats["llm_usage"]["requests"] == 2
                for path in stage.rglob("*.json"):
                    assert "alice@example.com" not in path.read_text()
            if name == "pdf":
                evidence = "The permitted answer is forty two."
                assert all(
                    evidence in json.dumps(r["payload"]) for r in requests[start:]
                )
            if name == "decision":
                assert len(requests) - start == 3
                assert stats["decision_usage"]["requests"] == 2
                assert stats["llm_usage"]["requests"] == 1
                panel = json.loads(
                    (
                        directory / "runs/repair/stages/02p_decision_panel/stats.json"
                    ).read_text()
                )
                assert panel["decision_usage"]["requests"] == 0
            resume_proof(directory, config)
            summary["scenarios"].append(
                {
                    "name": name,
                    "final_count": len(records),
                    "requests": len(requests) - start,
                }
            )

        directory, config = scenario("interrupted", [("Interrupt", "BAD")], pii=True)
        start = len(requests)
        run(directory, config, "interrupted", expect_success=False)
        generation_calls = sum(
            r["payload"].get("response_format", {}).get("json_schema", {}).get("name")
            == "RepairResponse"
            for r in requests[start:]
        )
        assert generation_calls == 1
        saved = json.loads(
            next(
                (directory / "runs/repair/stages/03_quality_repair").glob(
                    "attempt-*.json"
                )
            ).read_text()
        )
        assert saved["candidate"] is not None and saved["final"] is None
        before = len(requests)
        run(directory, config, "resume-after-interruption", resume=True)
        assert len(requests) == before + 1
        assert artifacts(directory)[1]["accepted_repairs"] == 1
        resume_proof(directory, config)
        summary["scenarios"].append(
            {"name": "interrupted", "generation_calls": 1, "resume_requests": 1}
        )
        summary["status"] = "passed"
        summary["requests"] = len(requests)
        summary["request_paths"] = dict(Counter(r["path"] for r in requests))
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (work / "requests.json").write_text(json.dumps(requests, indent=2))
        (work / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
