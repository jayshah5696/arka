from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from arka.cli import main
from arka.llm.models import LLMOutput, TokenUsage


class FakeGeneratorLLMClient:
    def __init__(self, config) -> None:
        self.config = config

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        parsed = schema.model_validate(
            {
                "instruction": "Generated hello",
                "response": "Generated hello response",
            }
        )
        return LLMOutput(
            text=parsed.model_dump_json(),
            parsed=parsed,
            usage=TokenUsage(total_tokens=10),
            finish_reason="stop",
            model=self.config.model,
            provider=self.config.provider,
            request_id="req-smoke",
            latency_ms=1,
            error=None,
        )


def test_smoke_pipeline_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "arka.llm.factory.LLMClient", FakeGeneratorLLMClient
    )

    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"  Say hello  ","response":"  Hello there  "}\n'
    )
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "smoke.yaml"
    (tmp_path / "smoke.yaml").write_text(fixture_path.read_text())

    main(["--config", "smoke.yaml", "--run-id", "smoke-run"])

    dataset_path = tmp_path / "output" / "smoke-dataset.jsonl"
    report_path = tmp_path / "runs" / "smoke-run" / "report" / "run_report.json"
    stage_path = (
        tmp_path / "runs" / "smoke-run" / "stages" / "02_generate" / "data.parquet"
    )

    assert dataset_path.exists()
    assert report_path.exists()
    assert stage_path.exists()
    assert (
        dataset_path.read_text().strip()
        == '{"messages":[{"role":"user","content":"Generated hello"},{"role":"assistant","content":"Generated hello response"}]}'
    )
