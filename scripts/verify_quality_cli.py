"""Prove distilled scoring and source audits through loopback HTTP and the CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
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
        else Path(tempfile.mkdtemp(prefix="arka-quality-"))
    )
    if args.output:
        work.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {work}", flush=True)
    requests = []
    interrupt = False
    interrupted = False
    interrupt_target = ["Task interrupt-1"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            nonlocal interrupted
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "payload": payload})
            if self.path == "/v1/embeddings":
                # Deliberately separable fixture vectors test mechanics, not model quality.
                self.send_json(
                    {
                        "object": "list",
                        "model": "fixture-embedding",
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [1.0, 0.0]
                                if '"GOOD"' in text
                                else [0.0, 1.0],
                            }
                            for i, text in enumerate(payload["input"])
                        ],
                        "usage": {"prompt_tokens": 10, "total_tokens": 10},
                    }
                )
                return
            if self.path == "/v1/decisions":
                state = payload["input"]
                score = 4 if '"response": "GOOD"' in state else 0
                uncertain = "Uncertain" in state
                self.send_json(
                    {
                        "model": "fixture",
                        "answers": [
                            {
                                "name": q["name"],
                                "type": "score",
                                "score": score,
                                "confidence": 0.1 if uncertain else 1,
                                "probabilities": [
                                    {"value": i, "probability": float(i == score)}
                                    for i in range(5)
                                ],
                            }
                            for q in payload["questions"]
                        ],
                        "usage": {"input_tokens": 10},
                    }
                )
                return
            assert self.path == "/v1/chat/completions", self.path
            user = payload["messages"][-1]["content"]
            if interrupt and not interrupted and interrupt_target[0] in user:
                interrupted = True
                self.send_json(
                    {
                        "error": {
                            "message": "fixture interruption",
                            "type": "server_error",
                        }
                    },
                    503,
                )
                return
            value = {
                "scores": {"correctness": 4 if "Response: <text>GOOD" in user else 0},
                "reasoning": "alice@example.com"
                if "Private judgment" in user
                else "Fixture judgment",
            }
            if (
                payload["response_format"]["json_schema"]["name"]
                == "GeneratedConversation"
            ):
                value = {"instruction": "Grounded task", "response": "BAD"}
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
                                "content": json.dumps(value),
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    cli = str(Path(sys.executable).with_name("arka"))
    rubric = {
        "version": "1",
        "description": "Quality",
        "dimensions": [
            {
                "name": "correctness",
                "description": "Correct answer",
                "scale_min": 0,
                "scale_max": 4,
                "criteria": {0: "Bad", 4: "Good"},
            }
        ],
        "overall_weights": {"correctness": 1},
    }
    (work / "rubric.yaml").write_text(yaml.safe_dump(rubric))
    summary = {
        "status": "running",
        "evidence": str(work),
        "scenarios": [],
        "commands": [],
    }

    def config(pipeline):
        return {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture-secret",
                "base_url": base_url,
                "max_retries": 1,
                "timeout_seconds": 5,
                "supports_json_schema": True,
            },
            "decisions": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture-secret",
                "base_url": base_url,
                "max_retries": 1,
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "embeddings": {
                "provider": "openai",
                "model": "fixture-embedding",
                "base_url": base_url,
                "max_retries": 0,
            },
            "pipeline": pipeline,
            "output": {"format": "jsonl", "path": "dataset.jsonl"},
        }

    def run(name, cfg, *, resume=False, success=True, log="cli"):
        directory = work / name
        directory.mkdir(exist_ok=True)
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        command = [cli, "--config", str(path), "--run-id", "quality"] + (
            ["--resume"] if resume else []
        )
        start = time.monotonic()
        completed = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, timeout=60
        )
        log_path = directory / f"{log}.log"
        log_path.write_text(completed.stdout + completed.stderr)
        summary["commands"].append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "log": str(log_path),
            }
        )
        if (completed.returncode == 0) != success:
            raise RuntimeError(f"Read {log_path}\n{completed.stdout}{completed.stderr}")
        return directory

    def seeds(name, count=4, special=None):
        directory = work / name
        directory.mkdir(exist_ok=True)
        path = directory / "seeds.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "instruction": special or f"Task {name}-{i}",
                        "response": "GOOD" if i % 2 else "BAD",
                    }
                )
                + "\n"
                for i in range(count)
            )
        )
        return {"type": "seed_source", "path": str(path)}

    def artifacts(directory, stage):
        path = directory / f"runs/quality/stages/{stage}"
        return (
            path,
            json.loads((path / "stats.json").read_text()),
            OutputWriter().read_parquet(path / "data.parquet"),
        )

    def resume_proof(name, cfg):
        before = len(requests)
        dataset = (work / name / "dataset.jsonl").read_bytes()
        run(name, cfg, resume=True, log="resume")
        assert len(requests) == before
        assert (work / name / "dataset.jsonl").read_bytes() == dataset
        assert (
            "fixture-secret"
            not in (work / name / "runs/quality/config.resolved.yaml").read_text()
        )

    try:
        training_paths = []
        for i in range(4):
            name = f"labels-{i}"
            labeling = {
                "type": "labeling_score",
                "rubric_path": str(work / "rubric.yaml"),
                "backend": "decision" if i % 2 else "llm",
            }
            cfg = config([seeds(name), labeling])
            directory = run(name, cfg)
            path, stats, records = artifacts(directory, "02s_label_score")
            assert len(records) == 4
            training_paths.append(str(path / "data.parquet"))
            resume_proof(name, cfg)
        summary["scenarios"].append(
            {"name": "teacher-labels", "records": 16, "backends": ["llm", "decision"]}
        )
        contract = {
            "rubric_path": str(work / "rubric.yaml"),
            "min_scores": {"correctness": 3},
        }
        train_cfg = config(
            [
                {
                    "type": "distilled_quality_train",
                    "training_paths": training_paths,
                    **contract,
                }
            ]
        )
        before = len(requests)
        directory = run("train", train_cfg)
        stage, stats, _ = artifacts(directory, "03_distilled_quality_train")
        model_path = stage / "model.json"
        model = json.loads(model_path.read_text())
        assert set(model["train_groups"]).isdisjoint(model["holdout_groups"])
        assert stats["evaluation"]["useful_record_recall"] == 1
        assert all(r["path"] == "/v1/embeddings" for r in requests[before:])
        resume_proof("train", train_cfg)
        summary["scenarios"].append(
            {
                "name": "train",
                "training": stats["training_count"],
                "holdout": stats["holdout_count"],
            }
        )
        scoring = {
            "type": "distilled_quality_score",
            "model_path": str(model_path),
            **contract,
        }
        score_cfg = config([seeds("score"), scoring])
        before = len(requests)
        directory = run("score", score_cfg)
        _, stats, records = artifacts(directory, "03_distilled_quality_score")
        assert len(records) == 4 and stats["scored_count"] == 4
        assert all(r.scores.quality is None for r in records)
        assert all(
            (r.scores.distilled_quality >= 0.5) == (r.payload.response == "GOOD")
            for r in records
        )
        assert all(r["path"] == "/v1/embeddings" for r in requests[before:])
        resume_proof("score", score_cfg)
        # Filtering occurs before direct judging of survivors, on a new source.
        filter_cfg = config(
            [
                seeds("prefilter"),
                {**scoring, "min_probability": 0.5},
                {"type": "labeling_score", "rubric_path": contract["rubric_path"]},
            ]
        )
        before = len(requests)
        directory = run("prefilter", filter_cfg)
        _, stats, records = artifacts(directory, "03_distilled_quality_score")
        assert len(records) == stats["dropped_count"] == 2
        assert sum(r["path"] == "/v1/chat/completions" for r in requests[before:]) == 2
        resume_proof("prefilter", filter_cfg)
        summary["scenarios"].append(
            {"name": "prefilter", "retained": 2, "later_judge_calls": 2}
        )
        # Independent offline evaluation reuses complete teacher labels.
        eval_cfg = config(
            [
                seeds("evaluate"),
                {"type": "labeling_score", "rubric_path": contract["rubric_path"]},
                {**scoring, "evaluate": True},
            ]
        )
        directory = run("evaluate", eval_cfg)
        _, stats, records = artifacts(directory, "03_distilled_quality_score")
        assert (
            stats["evaluation_independent"]
            and stats["evaluation"]["false_rejections"] == 0
        )
        assert all(r.scores.quality in (0, 4) for r in records)
        summary["scenarios"].append({"name": "evaluate", "independent": True})
        # Changed external dependencies must fail before any HTTP request.
        original_model = model_path.read_bytes()
        model["version"] = 99
        model_path.write_text(json.dumps(model))
        before = len(requests)
        run("score", score_cfg, resume=True, success=False, log="changed-model")
        assert len(requests) == before
        model_path.write_bytes(original_model)
        source_path = Path(training_paths[0])
        original_source = source_path.read_bytes()
        OutputWriter().write_parquet(
            OutputWriter().read_parquet(source_path)[:-1], source_path
        )
        run("train", train_cfg, resume=True, success=False, log="changed-labels")
        assert len(requests) == before
        source_path.write_bytes(original_source)
        # Audit all records in this small source without changing the output.
        audit = {"type": "source_audit", **contract, "sample_size": 10}
        audit_cfg = config([seeds("audit"), audit])
        directory = run("audit", audit_cfg)
        stage, stats, records = artifacts(directory, "03_source_audit")
        originals = OutputWriter().read_parquet(
            directory / "runs/quality/stages/01_source/data.parquet"
        )
        assert [r.payload for r in records] == [r.payload for r in originals]
        assert all(r.scores.quality is None for r in records)
        group = next(iter(stats["by_source"].values()))
        assert group["census"] and group["failed"] == group["passed"] == 2
        resume_proof("audit", audit_cfg)
        saved_rubric = (work / "rubric.yaml").read_bytes()
        (work / "rubric.yaml").write_text(
            yaml.safe_dump({**rubric, "description": "Changed"})
        )
        before = len(requests)
        run("audit", audit_cfg, resume=True, success=False, log="changed-rubric")
        assert len(requests) == before
        (work / "rubric.yaml").write_bytes(saved_rubric)
        summary["scenarios"].append({"name": "audit", "sampled": 4, "failed": 2})
        for name, evidence_stage in [
            ("grounded-audit", "01_source"),
            ("missing-evidence-audit", "missing"),
        ]:
            directory = work / name
            directory.mkdir()
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            stream = DecodedStreamObject()
            stream.set_data(
                b"BT /F1 12 Tf 20 100 Td (Permitted reference evidence.) Tj ET"
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
            cfg = config(
                [
                    {"type": "pdf_source", "path": str(directory / "source.pdf")},
                    {
                        "type": "prompt_based_generator",
                        "target_count": 1,
                        "generation_multiplier": 1,
                    },
                    {**audit, "evidence_stage": evidence_stage},
                ]
            )
            before = len(requests)
            run(name, cfg)
            _, stats, records = artifacts(directory, "03_source_audit")
            group = next(iter(stats["by_source"].values()))
            assert len(records) == 1
            if evidence_stage == "missing":
                assert group["missing_evidence"] == 1 and group["judged"] == 0
            else:
                assert group["judged"] == group["failed"] == 1
                judge_calls = [
                    r
                    for r in requests[before:]
                    if r["payload"]
                    .get("response_format", {})
                    .get("json_schema", {})
                    .get("name")
                    == "JudgeResponse"
                ]
                assert len(
                    judge_calls
                ) == 1 and "Permitted reference evidence." in json.dumps(judge_calls[0])
            resume_proof(name, cfg)
            summary["scenarios"].append({"name": name, "judged": group["judged"]})
        # Force the second deterministic sample to fail once; persisted work survives.
        interrupt_cfg = config([seeds("interrupt"), {**audit, "sample_size": 4}])
        # Sampling order is deterministic; target the second selected instruction.
        from arka.pipeline.quality_artifacts import digest
        from arka.records.identity import file_hash, record_id
        from arka.records.models import ConversationPayload

        seed_rows = [
            json.loads(line)
            for line in (work / "interrupt/seeds.jsonl").read_text().splitlines()
        ]
        source_group = "seeds:" + file_hash(work / "interrupt/seeds.jsonl")
        ordered = sorted(
            seed_rows,
            key=lambda row: digest(
                [0, source_group, record_id(ConversationPayload(**row))]
            ),
        )
        failed_instruction = ordered[1]["instruction"]
        interrupt = True
        # Use an explicit fixture control instead of changing application behavior.
        interrupt_target[0] = failed_instruction
        directory = run("interrupt", interrupt_cfg, success=False, log="interrupted")
        state = json.loads(
            (directory / "runs/quality/stages/03_source_audit/audit.json").read_text()
        )
        assert len(state["entries"]) == 1
        before = len(requests)
        run("interrupt", interrupt_cfg, resume=True, log="resume-interrupted")
        assert len(requests) == before + 3
        resume_proof("interrupt", interrupt_cfg)
        summary["scenarios"].append({"name": "interrupted-audit", "resume_requests": 3})
        for name, special, options in [
            (
                "private-audit",
                "Private judgment",
                {
                    "pii": {
                        "backend": "regex",
                        "languages": ["en"],
                        "entities": ["email"],
                        "blocked_entities": ["email"],
                    }
                },
            ),
            ("uncertain-audit", "Uncertain", {}),
        ]:
            cfg = config(
                [
                    seeds(name, count=1, special=special),
                    {
                        **audit,
                        "backend": "decision"
                        if name.startswith("uncertain")
                        else "llm",
                    },
                ]
            )
            cfg.update(options)
            directory = run(name, cfg)
            stage, stats, records = artifacts(directory, "03_source_audit")
            group = next(iter(stats["by_source"].values()))
            assert (
                len(records) == 1
                and group["failure_rate"] is None
                and group["incomplete"]
            )
            assert (
                group["privacy_rejected" if name.startswith("private") else "uncertain"]
                == 1
            )
            assert all(
                "alice@example.com" not in p.read_text() for p in stage.glob("*.json")
            )
            resume_proof(name, cfg)
            summary["scenarios"].append({"name": name, "failure_rate": None})
        summary["status"] = "passed"
        summary["requests"] = len(requests)
        summary["request_paths"] = dict(Counter(r["path"] for r in requests))
        summary["quality_claim"] = (
            "Fixture behavior only; representative quality and total cost are unverified"
        )
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
