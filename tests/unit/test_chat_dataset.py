import json

import pytest

from arka.config.loader import ConfigLoader
from arka.pipeline.output import OutputWriter
from arka.pipeline.source_stages import SeedSourceStage
from arka.records.models import Record


def test_text_chat_roundtrip(tmp_path):
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", "Be concise."),
            ("user", "Name a prime"),
            ("assistant", "Seven"),
            ("user", "  Another?"),
            ("assistant", "  eleven\n"),
        ]
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"messages": messages}))
    reader = SeedSourceStage(tmp_path)
    records = reader._read_jsonl(path, "config")
    for output_format in ["jsonl", "chatml"]:
        OutputWriter().write_jsonl(records, path, output_format=output_format)
        records = reader._read_jsonl(path, "config")
    assert json.loads(path.read_text()) == {"messages": messages}


def test_invalid_export_preserves_prior_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("prior dataset")
    with pytest.raises(ValueError, match="chatml"):
        OutputWriter().write_jsonl(
            [Record.model_construct(payload={})], path, output_format="chatml"
        )
    assert path.read_text() == "prior dataset"


@pytest.mark.parametrize("version", ["1", "2"])
def test_local_config_needs_no_model(version):
    stage = {"type": "seed_source", "path": "seeds.jsonl"}
    cfg = ConfigLoader().load_dict(
        {
            "version": version,
            "pipeline": [
                stage if version == "1" else {"id": "source", "config": stage}
            ],
            "output": {"format": "chatml", "path": "dataset.jsonl"},
        }
    )
    assert cfg.llm is None
    assert cfg.executor.max_workers == 4


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "missing answer"}],
        [
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "content": "ok"},
        ],
        [
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "assistant", "content": "ok"},
        ],
    ],
)
def test_invalid_chat_reports_row_number(tmp_path, messages):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": messages}))
    with pytest.raises(ValueError, match=r"bad\.jsonl:1:"):
        SeedSourceStage(tmp_path)._read_jsonl(path, "config")


def test_legacy_duplicate_filter_rejected_before_resume(tmp_path):
    from click.testing import CliRunner

    from arka.cli import cli

    path = tmp_path / "config.yaml"
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "pipeline": [
                    {"type": "length"},
                    {"type": "length", "min_response_chars": 20},
                ],
                "output": {"format": "chatml", "path": "data.jsonl"},
            }
        )
    )
    result = CliRunner().invoke(
        cli, ["--config", str(path), "--run-id", "collision", "--resume"]
    )
    assert result.exit_code != 0
    assert "unique stage IDs" in result.output
    assert not (tmp_path / "runs").exists()


def test_generation_requires_model():
    with pytest.raises(
        ValueError, match="llm config is required for prompt_based_generator"
    ):
        ConfigLoader().load_dict(
            {
                "version": "1",
                "pipeline": [
                    {
                        "type": "prompt_based_generator",
                        "generation_multiplier": 1,
                        "target_count": 1,
                    }
                ],
                "output": {"format": "chatml", "path": "data.jsonl"},
            }
        )


def test_answer_only_export_keeps_literal_answer_text(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        json.dumps(
            {"instruction": "Q", "response": "literal <think>", "reasoning": "trace"}
        )
    )
    records = SeedSourceStage(tmp_path)._read_jsonl(path, "config")
    OutputWriter().write_jsonl(
        records, path, output_format="chatml", include_reasoning=False
    )
    assert json.loads(path.read_text())["messages"][-1]["content"] == "literal <think>"
    assert records[0].payload.reasoning == "trace"


def test_incompatible_export_reports_failed_run(tmp_path):
    from click.testing import CliRunner

    from arka.cli import cli

    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "version": "1",
                "pipeline": [{"type": "seed_source", "path": "seed.jsonl"}],
                "output": {"format": "alpaca", "path": "data.jsonl"},
            }
        )
    )
    (tmp_path / "seed.jsonl").write_text(
        json.dumps(
            {
                "system": "Use metric units",
                "instruction": "Distance?",
                "response": "One km",
            }
        )
    )
    (tmp_path / "data.jsonl").write_text("existing")
    result = CliRunner().invoke(
        cli, ["--config", str(config), "--run-id", "incompatible"]
    )
    assert result.exit_code != 0
    assert (tmp_path / "data.jsonl").read_text() == "existing"
    manifest = json.loads((tmp_path / "runs/incompatible/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "alpaca" in manifest["error"]["message"]
