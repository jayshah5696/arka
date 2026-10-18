from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from arka.privacy.config import PIIConfig
from arka.privacy.runtime import build_policy


@pytest.mark.parametrize(
    "changes",
    [
        {"entities": []},
        {"languages": []},
        {"languages": ["zh"]},
        {"entities": ["invented"]},
        {"blocked_entities": ["secret"]},
        {"max_redacted_ratio": 1.1},
        {"placeholder_template": "{unknown}"},
    ],
)
def test_invalid_coverage_or_policy_fails_configuration(changes: dict) -> None:
    with pytest.raises(ValidationError):
        PIIConfig.model_validate(
            {
                "backend": "gliner2",
                "languages": ["en"],
                "entities": ["person"],
                "model_path": "model",
                **changes,
            }
        )


def test_missing_model_fails_before_import_or_source_read(tmp_path: Path) -> None:
    config = PIIConfig(
        backend="gliner2", languages=["en"], entities=["person"], model_path="missing"
    )
    with pytest.raises(ValueError, match="local.*model"):
        build_policy(config, tmp_path)


def test_gliner_uses_local_checkpoint_long_api_and_global_offsets(
    tmp_path: Path, monkeypatch
) -> None:
    for name in [
        "config.json",
        "encoder_config/config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        path = tmp_path / "model" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    calls = []

    class Model:
        def eval(self):
            return self

        def extract_entities_long(self, text, entities, **kwargs):
            calls.append((text, entities, kwargs))
            start = text.rindex("Alice")
            return {
                "entities": {
                    "person": [
                        {
                            "text": "Alice",
                            "start": start,
                            "end": start + 5,
                            "confidence": 0.9,
                        }
                    ]
                }
            }

    def load(path, **kwargs):
        assert Path(path) == tmp_path / "model"
        assert kwargs["local_files_only"] is True
        return Model()

    monkeypatch.setitem(
        sys.modules,
        "gliner2",
        SimpleNamespace(GLiNER2=SimpleNamespace(from_pretrained=load)),
    )
    config = PIIConfig(
        backend="gliner2",
        languages=["en", "fr"],
        entities=["person"],
        model_path="model",
    )
    policy = build_policy(config, tmp_path)
    text = "Alice " + "word " * 1000 + "Alice"
    result = policy.protect(text)

    assert result.value == text[:-5] + "[person]"
    assert calls[0][1] == ["person"]
    assert calls[0][2]["include_spans"] is True
    assert calls[0][2]["chunk_overlap"] > 0
    first_identity = policy.fingerprint
    (tmp_path / "model/model.safetensors").write_text("changed")
    assert build_policy(config, tmp_path).fingerprint != first_identity


def test_regex_policy_has_stable_effective_identity(tmp_path: Path) -> None:
    config = PIIConfig(backend="regex", languages=["en"], entities=["email"])
    first = build_policy(config, tmp_path)
    assert first.fingerprint == build_policy(config, tmp_path).fingerprint
    changed = config.model_copy(update={"placeholder_template": "<{entity}>"})
    assert first.fingerprint != build_policy(changed, tmp_path).fingerprint


@pytest.mark.parametrize(
    "changes",
    [
        {"languages": ["e1"]},
        {"placeholder_template": "{entity}"},
        {"placeholder_template": "[{entity}]\n"},
    ],
)
def test_regex_rejects_invalid_language_codes_or_unsafe_placeholders(changes):
    with pytest.raises(ValidationError):
        PIIConfig(
            backend="regex",
            languages=changes.get("languages", ["en"]),
            entities=["email"],
            placeholder_template=changes.get("placeholder_template", "[{entity}]"),
        )


def test_gliner_exception_logging_does_not_disclose_input(caplog):
    import logging

    from arka.privacy.policy import PIIPolicy, PrivacyError
    from arka.privacy.runtime import GLiNERDetector

    class Model:
        def extract_entities_long(self, text, *args, **kwargs):
            try:
                raise ValueError(text)
            except ValueError:
                logging.getLogger("gliner2.inference.runtime").exception(
                    "failed %s", text
                )
                raise

    detector = GLiNERDetector(
        Model(),
        PIIConfig(
            backend="gliner2",
            languages=["en"],
            entities=["person"],
            model_path="unused",
        ),
    )
    with pytest.raises(PrivacyError):
        PIIPolicy(detector).protect("private@example.com")
    assert "private@example.com" not in caplog.text
    assert "GLiNER2 extraction failed" in caplog.text
