"""Slice 3 \u2014 TaxonomyGeneratorStage.

For each (strategy, target_count // num_strategies) combination, sample one
node-set, build a meta-prompt, generate one ConversationRecord. The lineage
must carry the sampled nodes for downstream coverage analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _bundle_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tax.yaml"
    p.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain_root
      children:
        - name: email
        - name: blog
  - name: tone
    root:
      name: tone_root
      children:
        - name: casual
        - name: professional
strategies:
  - name: default
    include_factors: [domain, tone]
""".strip()
    )
    return p


def _config(tmp_path: Path, *, taxonomy_path: Path, target_count: int = 8) -> Any:
    from arka.config.loader import ConfigLoader

    return ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "taxonomy_prompt",
                "target_count": target_count,
                "generation_multiplier": 1,
                "taxonomy_path": str(taxonomy_path),
            },
            "filters": {"target_count": target_count, "stages": []},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )


def _ctx(config, tmp_path: Path):
    from arka.pipeline.models import StageContext

    return StageContext(
        run_id="run-1",
        stage_name="02_generate",
        work_dir=tmp_path / "stages" / "02_generate",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


class _FakeLLM:
    """Returns a deterministic conversation for each call."""

    def __init__(self) -> None:
        self.calls = 0
        self.received_messages: list[Any] = []

    def complete_structured(self, messages, schema, **kwargs):
        from arka.llm.models import LLMOutput, TokenUsage

        self.calls += 1
        self.received_messages.append(list(messages))
        instr = f"call-{self.calls}-instruction"
        resp = f"call-{self.calls}-response"
        text = json.dumps({"instruction": instr, "response": resp})
        parsed = schema(instruction=instr, response=resp)
        return LLMOutput(
            text=text,
            parsed=parsed,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req-{self.calls}",
            latency_ms=5,
            error=None,
        )


def test_taxonomy_generator_config_validates(tmp_path: Path) -> None:
    """The generator config must accept type='taxonomy_prompt' + taxonomy_path."""
    config = _config(tmp_path, taxonomy_path=_bundle_yaml(tmp_path))
    assert config.generator.type == "taxonomy_prompt"
    assert config.generator.taxonomy_path is not None


def test_taxonomy_generator_rejects_missing_taxonomy_path() -> None:
    """type='taxonomy_prompt' without taxonomy_path is a config error."""
    from arka.config.loader import ConfigLoader, ConfigValidationError

    with pytest.raises((ConfigValidationError, ValueError)):
        ConfigLoader().load_dict(
            {
                "version": "1",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "k",
                    "base_url": "https://api.openai.com/v1",
                },
                "executor": {"mode": "threadpool", "max_workers": 1},
                "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
                "generator": {
                    "type": "taxonomy_prompt",
                    "target_count": 4,
                    "generation_multiplier": 1,
                    # MISSING taxonomy_path
                },
                "filters": {"target_count": 4, "stages": []},
                "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
            }
        )


def test_taxonomy_generator_produces_target_count_records(tmp_path: Path) -> None:
    from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage

    config = _config(tmp_path, taxonomy_path=_bundle_yaml(tmp_path), target_count=4)
    fake = _FakeLLM()
    stage = TaxonomyGeneratorStage(llm_client=fake)
    ctx = _ctx(config, tmp_path)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    out = stage.run([], ctx)

    assert len(out) == 4
    assert all(isinstance(r, ConversationRecord) for r in out)
    assert fake.calls == 4


def test_taxonomy_generator_meta_prompt_contains_sampled_nodes(tmp_path: Path) -> None:
    """The constructed meta-prompt must mention each sampled node value so the
    LLM is steered by the taxonomy choices, not by the seeds."""
    from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage

    config = _config(tmp_path, taxonomy_path=_bundle_yaml(tmp_path), target_count=2)
    fake = _FakeLLM()
    stage = TaxonomyGeneratorStage(llm_client=fake)
    ctx = _ctx(config, tmp_path)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    stage.run([], ctx)

    assert len(fake.received_messages) == 2
    for messages in fake.received_messages:
        prompt = " ".join(m["content"] for m in messages).lower()
        # Every meta-prompt should reference one domain leaf and one tone leaf
        assert any(d in prompt for d in ("email", "blog")), (
            f"meta-prompt missing a domain leaf: {prompt[:200]}"
        )
        assert any(t in prompt for t in ("casual", "professional")), (
            f"meta-prompt missing a tone leaf: {prompt[:200]}"
        )


def test_taxonomy_generator_records_carry_lineage_with_sampled_nodes(
    tmp_path: Path,
) -> None:
    """Each generated record's lineage must carry which taxonomy nodes were
    sampled. This is what slice 4 (level-ratio coverage metric) will read.
    """
    from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage

    config = _config(tmp_path, taxonomy_path=_bundle_yaml(tmp_path), target_count=3)
    fake = _FakeLLM()
    stage = TaxonomyGeneratorStage(llm_client=fake)
    ctx = _ctx(config, tmp_path)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    out = stage.run([], ctx)
    for record in out:
        # We use scores.quality_per_dim['taxonomy_nodes'] for the audit trail \u2014
        # mirrors the double_critic audit pattern. Slice 4 reads from there.
        sampled = record.scores.quality_per_dim.get("taxonomy_nodes")
        assert sampled is not None
        # one node sampled per included factor (domain + tone)
        assert "domain" in sampled and "tone" in sampled


def test_taxonomy_generator_passthroughs_existing_records(tmp_path: Path) -> None:
    """Existing records (seeds) flow through; the generator only adds records on top.

    Mirrors the slice-0 baseline contract.

    """
    from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage

    config = _config(tmp_path, taxonomy_path=_bundle_yaml(tmp_path), target_count=2)
    fake = _FakeLLM()
    stage = TaxonomyGeneratorStage(llm_client=fake)
    ctx = _ctx(config, tmp_path)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    seed = ConversationRecord(
        id="s1",
        content_hash="h",
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id="r-s1", parent_ids=[]),
        payload=ConversationPayload(instruction="x", response="y"),
        scores=RecordScores(),
        config_hash="c",
        created_at="2026-01-01T00:00:00Z",
    )

    out = stage.run([seed], ctx)
    assert len(out) == 1 + 2  # seed + 2 generated
    assert out[0].id == "s1"
