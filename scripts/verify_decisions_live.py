"""Run a small paid experiment against OpenRouter's Decisions API.

Requires OPENROUTER_API_KEY. Use --output with a new directory. The transport
sends real HTTP and saves synthetic request/response bodies without headers.
Failures to detect: malformed answers, hidden LLM calls, excess requests,
failed stages, missing scores, cache misses, and resume making new requests.
This is a functional experiment, not a calibration benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from arka.config.loader import ConfigLoader
from arka.decisions.client import DecisionClient
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder

SAMPLES = [
    ("What is 2 + 2?", "4", True),
    ("What is 2 + 2?", "5", False),
    ("What is the capital of France?", "Paris", True),
    ("What is the capital of France?", "London", False),
    ("What is 7 multiplied by 8?", "56", True),
    ("What is 7 multiplied by 8?", "63", False),
]


class LiveTransport(httpx.BaseTransport):
    """Count real requests and enforce a hard budget across all clients."""

    def __init__(self, output: Path):
        self.output = output
        self.http = httpx.HTTPTransport()
        self.lock = threading.Lock()
        self.sent = 0
        self.receipts = []

    def handle_request(self, request):
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        with self.lock:
            if self.sent >= 48:
                raise RuntimeError("Live request budget exhausted")
            self.sent += 1
            index = self.sent
        started = time.perf_counter()
        response = self.http.handle_request(request)
        response.read()
        receipt = {
            "index": index,
            "request": json.loads(request.content),
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "response": response.json(),
        }
        with self.lock:
            self.receipts.append(receipt)
            (self.output / "receipts.json").write_text(
                json.dumps(sorted(self.receipts, key=lambda r: r["index"]), indent=2)
            )
        return response

    def close(self):
        # Stage clients share this transport. Its owner closes the HTTP pool.
        pass


def config(root: Path, repo: Path, panel: bool, filtering: bool) -> dict:
    judges = [
        {
            "type": "labeling_engine",
            "backend": "decision",
            "rubric_path": str(repo / "rubrics/sft_quality.yaml"),
            "min_overall_score": 3.5 if filtering else None,
            "min_confidence": 0,
        },
        {"type": "complexity_elo", "backend": "decision", "min_confidence": 0},
        {
            "type": "reward_model",
            "backend": "decision",
            "min_score": 6 if filtering else None,
            "min_confidence": 0,
        },
        {
            "type": "double_critic",
            "backend": "decision",
            "accept_at": 0.75 if filtering else 0.99,
            "reject_at": 0.4 if filtering else 0,
            "on_uncertain": "escalate" if filtering else "keep",
        },
    ]
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "unused",
            "api_key": "unused",
            "base_url": "http://127.0.0.1:1/v1",
        },
        "decisions": {
            "provider": "openrouter",
            "model": "openai/gpt-6-luna-decisions",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/alpha",
            "timeout_seconds": 45,
            "max_retries": 1,
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "pipeline": [
            {"type": "seed_source", "path": str(root / "seeds.jsonl")},
            {"type": "normalize_conversation"},
            *([{"type": "decision_panel"}] if panel else []),
            *judges,
        ],
        "output": {
            "format": "chatml",
            "path": "filtered.jsonl" if filtering else "dataset.jsonl",
            "on_unresolved_escalation": "drop" if filtering else "keep",
        },
    }


def run(root, repo, transport, *, panel, filtering=False, resume=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "seeds.jsonl").write_text(
        "".join(
            json.dumps({"instruction": instruction, "response": response}) + "\n"
            for instruction, response, _ in SAMPLES
        )
    )
    run_id = "filtered" if filtering else "scored"
    config_path = root / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config(root, repo, panel, filtering)))
    resolved = ConfigLoader().load(config_path)
    checkpoint = CheckpointManager(root / "state.db")

    def factory(cfg):
        return DecisionClient(cfg, checkpoint=checkpoint, transport=transport)

    sent_before = transport.sent
    started = time.perf_counter()
    try:
        # A live Decisions experiment must never instantiate an LLM client.
        with patch(
            "arka.pipeline.models.StageContext.llm_client",
            side_effect=AssertionError("Unexpected LLM call"),
        ) as llm:
            result = PipelineRunner(root, decision_client_factory=factory).run(
                resolved,
                StageBuilder(resolved, root).build(),
                run_id,
                resume=resume,
            )
            assert llm.call_count == 0
    finally:
        checkpoint.close()
    elapsed = round(time.perf_counter() - started, 3)
    manifest = json.loads(result.manifest_path.read_text())
    run_dir = result.manifest_path.parent
    assert manifest["status"] == "completed"
    for path in (run_dir / "stages").glob("*/stats.json"):
        stats = json.loads(path.read_text())
        assert stats.get("decision_failures", 0) == 0, path
    assert (
        os.environ["OPENROUTER_API_KEY"]
        not in (run_dir / "config.resolved.yaml").read_text()
    )
    records = OutputWriter().read_parquet(
        run_dir / "stages/03_double_critic/data.parquet"
    )
    samples = []
    for record in records:
        assert len(record.scores.decisions) == 6
        samples.append(
            {
                "instruction": record.payload.instruction,
                "response": record.payload.response,
                "expected_correct": next(
                    correct
                    for instruction, response, correct in SAMPLES
                    if instruction == record.payload.instruction
                    and response == record.payload.response
                ),
                "quality": record.scores.quality,
                "reward": record.scores.reward_model,
                "p_correct": record.scores.decisions[
                    "double_critic.is_correct"
                ].probability,
                "p_incorrect": record.scores.decisions[
                    "double_critic.is_incorrect"
                ].probability,
            }
        )
    requests = transport.sent - sent_before
    assert requests == (0 if resume or filtering else 12 if panel else 30)
    assert result.final_count == len(records)
    if not filtering:
        assert len(records) == 6
    else:
        assert records and all(sample["expected_correct"] for sample in samples)
    return {
        "requests": requests,
        "elapsed_seconds": elapsed,
        "final_count": result.final_count,
        "usage": manifest["decision_usage"],
        "samples": samples,
        "llm_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    transport = LiveTransport(output)
    try:
        results = {}
        for mode in ("panel", "direct"):
            results[mode] = run(output / mode, repo, transport, panel=mode == "panel")
            (output / "result.json").write_text(json.dumps(results, indent=2))
        results["filtered"] = run(
            output / "panel", repo, transport, panel=True, filtering=True
        )
        results["resumed"] = run(
            output / "panel", repo, transport, panel=True, resume=True
        )
        results["http_requests"] = transport.sent
        results["models"] = sorted(
            {receipt["response"]["model"] for receipt in transport.receipts}
        )
        results["reported_cost_usd"] = sum(
            receipt["response"]["usage"]["cost"] for receipt in transport.receipts
        )
        latencies = sorted(receipt["latency_ms"] for receipt in transport.receipts)
        results["p50_latency_ms"] = statistics.median(latencies)
        results["p95_latency_ms"] = latencies[int(len(latencies) * 0.95)]
        (output / "result.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print(f"Evidence: {output}")
    finally:
        transport.http.close()


if __name__ == "__main__":
    main()
