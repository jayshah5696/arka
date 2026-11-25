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


@pytest.mark.parametrize("option", ["--validate-config", "--dry-run", "--resume"])
def test_legacy_duplicate_filter_rejected_before_execution(tmp_path, option):
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
        cli, ["--config", str(path), "--run-id", "collision", option]
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


def test_export_permissions_follow_normal_creation_and_preserve_existing_mode(tmp_path):
    from stat import S_IMODE

    reference = tmp_path / "reference"
    reference.touch(mode=0o666)
    target = tmp_path / "dataset.jsonl"
    writer = OutputWriter()
    writer.write_jsonl([], target)
    assert S_IMODE(target.stat().st_mode) == S_IMODE(reference.stat().st_mode)
    target.chmod(0o640)
    writer.write_jsonl([], target)
    assert S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.parametrize(
    "row",
    [
        {
            "messages": [
                {"role": "system", "content": "x" * 16385},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "<text>"},
                {"role": "assistant", "content": "earlier"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        },
        {
            "system": "context",
            "instruction": "Q",
            "response": "A",
            "reasoning": "x" * 16385,
        },
    ],
)
def test_chat_seed_reports_out_of_bounds_content(tmp_path, row):
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=r"seed.jsonl:1:.*(16384|delimiter)"):
        SeedSourceStage(tmp_path)._read_jsonl(seed, "config")
