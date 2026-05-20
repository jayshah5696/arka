from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import BaseModel

from arka.cli import main
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.output import OutputWriter


def _install_fake_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation_payloads: list[dict[str, str]],
    judge_payloads: list[dict[str, Any]],
) -> None:
    generation_iter = iter(generation_payloads)
    judge_iter = iter(judge_payloads)

    class FakeLLMClient:
        def __init__(self, config) -> None:
            self.config = config

        def complete_structured(
            self,
            messages,
            schema: type[BaseModel],
            **_: Any,
        ) -> LLMOutput:
            if schema.__name__ == "GeneratedConversation":
                payload = next(generation_iter)
            elif schema.__name__ == "JudgeResponse":
                payload = next(judge_iter)
            else:
                raise AssertionError(f"Unexpected schema: {schema.__name__}")

            parsed = schema.model_validate(payload)
            return LLMOutput(
                text=parsed.model_dump_json(),
                parsed=parsed,
                usage=TokenUsage(total_tokens=10),
                finish_reason="stop",
                model=self.config.model,
                provider=self.config.provider,
                request_id="req-matrix",
                latency_ms=1,
                error=None,
            )

    monkeypatch.setattr("arka.llm.factory.LLMClient", FakeLLMClient)


def _write_seed_file(tmp_path: Path, seed_format: str) -> Path:
    if seed_format == "jsonl":
        path = tmp_path / "seeds.jsonl"
        path.write_text('{"instruction":"Seed hello","response":"Hello"}\n')
        return path

    path = tmp_path / "seeds.csv"
    path.write_text("instruction,response\nSeed hello,Hello\n")
    return path


def _write_rubric(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "description": "SFT quality rubric",
                "dimensions": [
                    {
                        "name": "instruction_clarity",
                        "description": "Clear instruction",
                        "scale_min": 1,
                        "scale_max": 5,
                        "criteria": {1: "bad", 5: "good"},
                    },
                    {
                        "name": "response_quality",
                        "description": "Good response",
                        "scale_min": 1,
                        "scale_max": 5,
                        "criteria": {1: "bad", 5: "good"},
                    },
                ],
                "overall_weights": {
                    "instruction_clarity": 0.4,
                    "response_quality": 0.6,
                },
                "few_shot": [
                    {
                        "instruction": "What is 2+2?",
                        "response": "4",
                        "scores": {
                            "instruction_clarity": 5,
                            "response_quality": 5,
                        },
                        "reasoning": "Clear and correct.",
                        "expected_verdict": "pass",
                    },
                    {
                        "instruction": "Tell me stuff",
                        "response": "Stuff.",
                        "scores": {
                            "instruction_clarity": 1,
                            "response_quality": 1,
                        },
                        "reasoning": "Vague and weak.",
                        "expected_verdict": "fail",
                    },
                ],
            },
            sort_keys=False,
        )
    )


def _write_config(
    path: Path,
    *,
    data_source_path: str,
    output_format: str,
    executor_mode: str,
    target_count: int,
    generation_multiplier: int = 1,
    exact_dedup: bool = False,
    near_dedup: bool = False,
    quality_filter: bool = False,
) -> None:
    config: dict[str, Any] = {
        "version": "1",
        "run_id": None,
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.openai.com/v1",
            "timeout_seconds": 30,
            "max_retries": 3,
        },
        "executor": {
            "mode": executor_mode,
            "max_workers": 1,
        },
        "data_source": {
            "type": "seeds",
            "path": data_source_path,
        },
        "generator": {
            "type": "prompt_based",
            "target_count": target_count,
            "generation_multiplier": generation_multiplier,
        },
        "dedup": [
            *([{"type": "exact"}] if exact_dedup else []),
            *([{"type": "near"}] if near_dedup else []),
        ],
        "filters": {
            "target_count": target_count,
        },
        "embeddings": {
            "provider": "huggingface",
            "model": "all-MiniLM-L6-v2",
        },
        "output": {
            "format": output_format,
            "path": "./output/dataset.jsonl",
        },
    }

    if quality_filter:
        config["filters"].setdefault("stages", []).append(
            {
                "type": "labeling_engine",
                "rubric_path": "./rubrics/sft_quality.yaml",
                "min_overall_score": 3.5,
            }
        )
        config["labeling_engine"] = {
            "rubric_path": "./rubrics/sft_quality.yaml",
            "mode": "single",
        }

    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _assert_formatted_payload(
    payload: dict[str, Any],
    *,
    output_format: str,
    instruction: str,
    response: str,
) -> None:
    if output_format == "jsonl":
        assert payload == {
            "instruction": instruction,
            "response": response,
            "system": None,
            "turns": None,
        }
        return
    if output_format == "chatml":
        assert payload == {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        }
        return
    if output_format == "alpaca":
        assert payload == {
            "instruction": instruction,
            "input": "",
            "output": response,
        }
        return
    raise AssertionError(f"Unexpected output format: {output_format}")


@pytest.mark.parametrize("seed_format", ["jsonl", "csv"])
@pytest.mark.parametrize("output_format", ["jsonl", "chatml", "alpaca"])
@pytest.mark.parametrize("executor_mode", ["threadpool", "realtime", "provider_batch"])
def test_supported_source_output_and_executor_matrix_runs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_format: str,
    output_format: str,
    executor_mode: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "arka.embeddings.embedder.Embedder._embed_huggingface",
        lambda self, texts: None,
    )
    seed_path = _write_seed_file(tmp_path, seed_format)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        data_source_path=f"./{seed_path.name}",
        output_format=output_format,
        executor_mode=executor_mode,
        target_count=1,
    )
    _install_fake_llm(
        monkeypatch,
        generation_payloads=[
            {
                "instruction": "Generated hello",
                "response": "Generated hello response",
            }
        ],
        judge_payloads=[],
    )

    main(["--config", str(config_path), "--run-id", "matrix-run"])

    dataset_path = tmp_path / "output" / "dataset.jsonl"
    resolved_config_path = tmp_path / "runs" / "matrix-run" / "config.resolved.yaml"
    report_path = tmp_path / "runs" / "matrix-run" / "report" / "run_report.json"

    assert dataset_path.exists()
    assert resolved_config_path.exists()
    assert report_path.exists()

    payload = json.loads(dataset_path.read_text().strip())
    _assert_formatted_payload(
        payload,
        output_format=output_format,
        instruction="Generated hello",
        response="Generated hello response",
    )

    resolved = yaml.safe_load(resolved_config_path.read_text())
    assert resolved["executor"]["mode"] == executor_mode
    assert resolved["data_source"]["path"] == f"./{seed_path.name}"

    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["final_count"] == 1


@pytest.mark.parametrize(
    ("exact_dedup", "near_dedup", "generation_payloads", "expected_final_count"),
    [
        (
            False,
            False,
            [
                {"instruction": "Unique one", "response": "Response one"},
                {"instruction": "Unique two", "response": "Response two"},
            ],
            2,
        ),
        (
            True,
            False,
            [
                {"instruction": "Exact duplicate", "response": "Same response"},
                {"instruction": "Exact duplicate", "response": "Same response"},
            ],
            1,
        ),
        (
            False,
            True,
            [
                {
                    "instruction": "Explain machine learning in simple terms for beginners with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
                    "response": "Machine learning lets computers learn from data.",
                },
                {
                    "instruction": "Explain machine learning in simple terms for beginner with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
                    "response": "ML means computers learn from examples.",
                },
            ],
            1,
        ),
        (
            True,
            True,
            [
                {
                    "instruction": "Explain machine learning in simple terms for beginners with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
                    "response": "Machine learning lets computers learn from data.",
                },
                {
                    "instruction": "Explain machine learning in simple terms for beginners with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
                    "response": "Machine learning lets computers learn from data.",
                },
                {
                    "instruction": "Explain machine learning in simple terms for beginner with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
                    "response": "ML means computers learn from examples.",
                },
            ],
            1,
        ),
    ],
    ids=["no-dedup", "exact-only", "near-only", "exact-and-near"],
)
def test_supported_dedup_matrix_runs_and_records_expected_drop_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_dedup: bool,
    near_dedup: bool,
    generation_payloads: list[dict[str, str]],
    expected_final_count: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "arka.embeddings.embedder.Embedder._embed_huggingface",
        lambda self, texts: None,
    )
    seed_path = _write_seed_file(tmp_path, "jsonl")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        data_source_path=f"./{seed_path.name}",
        output_format="jsonl",
        executor_mode="threadpool",
        target_count=len(generation_payloads),
        exact_dedup=exact_dedup,
        near_dedup=near_dedup,
    )
    _install_fake_llm(
        monkeypatch,
        generation_payloads=generation_payloads,
        judge_payloads=[],
    )

    main(["--config", str(config_path), "--run-id", "dedup-run"])

    report_path = tmp_path / "runs" / "dedup-run" / "report" / "run_report.json"
    report = json.loads(report_path.read_text())
    assert report["final_count"] == expected_final_count
    assert report["status"] == "completed"

    exact_stats_path = (
        tmp_path / "runs" / "dedup-run" / "stages" / "02c_exact_dedup" / "stats.json"
    )
    near_stats_path = (
        tmp_path / "runs" / "dedup-run" / "stages" / "02d_near_dedup" / "stats.json"
    )

    if exact_dedup:
        assert exact_stats_path.exists()
        exact_stats = json.loads(exact_stats_path.read_text())
        expected_exact_drops = (
            1 if generation_payloads[0] == generation_payloads[1] else 0
        )
        assert exact_stats["dropped_count"] == expected_exact_drops
    else:
        assert not exact_stats_path.exists()

    if near_dedup:
        assert near_stats_path.exists()
        near_stats = json.loads(near_stats_path.read_text())
        expected_near_drops = (
            1
            if expected_final_count == 1 and not (exact_dedup and not near_dedup)
            else 0
        )
        if exact_dedup and near_dedup:
            expected_near_drops = 1
        assert near_stats["dropped_count"] == expected_near_drops
    else:
        assert not near_stats_path.exists()


@pytest.mark.parametrize("output_format", ["jsonl", "chatml", "alpaca"])
def test_generation_quality_gate_run_writes_expected_report_and_filtered_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "arka.embeddings.embedder.Embedder._embed_huggingface",
        lambda self, texts: np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    seed_path = _write_seed_file(tmp_path, "jsonl")
    _write_rubric(tmp_path / "rubrics" / "sft_quality.yaml")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        data_source_path=f"./{seed_path.name}",
        output_format=output_format,
        executor_mode="threadpool",
        target_count=3,
        quality_filter=True,
    )
    _install_fake_llm(
        monkeypatch,
        generation_payloads=[
            {
                "instruction": "Strong instruction one",
                "response": "Strong response one",
            },
            {
                "instruction": "Strong instruction two",
                "response": "Strong response two",
            },
            {
                "instruction": "Weak instruction",
                "response": "Weak response",
            },
        ],
        judge_payloads=[
            {
                "scores": {
                    "instruction_clarity": 5,
                    "response_quality": 5,
                },
                "reasoning": "excellent",
            },
            {
                "scores": {
                    "instruction_clarity": 4,
                    "response_quality": 4,
                },
                "reasoning": "good",
            },
            {
                "scores": {
                    "instruction_clarity": 1,
                    "response_quality": 1,
                },
                "reasoning": "poor",
            },
            {
                "scores": {
                    "instruction_clarity": 5,
                    "response_quality": 5,
                },
                "reasoning": "good canary",
            },
            {
                "scores": {
                    "instruction_clarity": 1,
                    "response_quality": 1,
                },
                "reasoning": "bad canary",
            },
        ],
    )

    main(["--config", str(config_path), "--run-id", "quality-run"])

    dataset_path = tmp_path / "output" / "dataset.jsonl"
    report_path = tmp_path / "runs" / "quality-run" / "report" / "run_report.json"
    quality_stage_path = (
        tmp_path
        / "runs"
        / "quality-run"
        / "stages"
        / "03_label_quality"
        / "data.parquet"
    )

    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["final_count"] == 2
    assert report["drop_reasons"] == {"low_quality_score": 1}
    assert report["quality_distribution"] == {
        "mean": 3.3333,
        "std": 1.6997,
        "min": 1.0,
        "max": 5.0,
    }
    assert report["canaries"]["status"] == "pass"
    assert isinstance(report["diversity_score"], float)
    assert Path(report["samples_path"]).exists()
    assert Path(report["canaries_path"]).exists()

    kept_records = OutputWriter().read_parquet(quality_stage_path)
    assert [record.scores.quality for record in kept_records] == [5.0, 4.0]

    output_lines = [
        json.loads(line)
        for line in dataset_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(output_lines) == 2
    _assert_formatted_payload(
        output_lines[0],
        output_format=output_format,
        instruction="Strong instruction one",
        response="Strong response one",
    )
    _assert_formatted_payload(
        output_lines[1],
        output_format=output_format,
        instruction="Strong instruction two",
        response="Strong response two",
    )
