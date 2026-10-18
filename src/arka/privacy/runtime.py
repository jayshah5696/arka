from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

from arka.privacy.config import PIIConfig
from arka.privacy.detectors import RegexDetector
from arka.privacy.policy import PIIPolicy, PrivacyError, Span

MODEL_FILES = (
    "config.json",
    "encoder_config/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


class _SafeExtractionLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # GLiNER2 logs exception messages before re-raising extraction errors.
        # Those messages can contain input text. Keep a payload-free diagnostic.
        record.msg = "GLiNER2 extraction failed while PII protection was enabled"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


class GLiNERDetector:
    """Serialize local model inference and request offsets over the full input."""

    def __init__(self, model, config: PIIConfig) -> None:
        self.model = model
        self.config = config
        self._lock = Lock()
        logger = logging.getLogger("gliner2.inference.runtime")
        if not any(isinstance(item, _SafeExtractionLog) for item in logger.filters):
            logger.addFilter(_SafeExtractionLog())

    def detect(self, text: str) -> list[Span]:
        with self._lock:
            result = self.model.extract_entities_long(
                text,
                sorted(set(self.config.entities)),
                threshold=self.config.threshold,
                include_spans=True,
                include_confidence=True,
                chunk_size=256,
                chunk_overlap=64,
                batch_size=1,
            )
        entities = result["entities"]
        if not isinstance(entities, dict) or set(entities) - set(self.config.entities):
            raise PrivacyError("Invalid GLiNER2 entity result")
        spans = []
        for entity, values in entities.items():
            for value in values:
                start, end = value["start"], value["end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or text[start:end] != value["text"]
                ):
                    raise PrivacyError("Invalid GLiNER2 offsets")
                spans.append(Span(start, end, entity))
        return spans


def build_policy(config: PIIConfig | None, project_root: Path) -> PIIPolicy | None:
    """Preflight before source reads. Local model files must already exist."""
    if config is None:
        return None
    identity = {"version": 1, "config": config.model_dump(mode="json")}
    if config.backend == "regex":
        detector = RegexDetector(set(config.entities))
    else:
        model_path = (project_root / config.model_path).resolve()
        if any(not (model_path / name).is_file() for name in MODEL_FILES):
            raise ValueError(
                "PII requires a complete local GLiNER2 model; provision model_path before running"
            )
        digest = hashlib.sha256()
        for name in MODEL_FILES:
            digest.update(name.encode())
            with (model_path / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        identity["model_sha256"] = digest.hexdigest()
        try:
            from gliner2 import GLiNER2
        except ImportError:
            raise ValueError(
                "GLiNER2 PII requires the optional extra: uv sync --extra pii"
            ) from None
        try:
            model = GLiNER2.from_pretrained(
                str(model_path),
                local_files_only=True,
                map_location="cpu",
                attention_backend="standard",
            ).eval()
        except Exception:
            raise ValueError(
                "Cannot load local GLiNER2 PII model; verify the checkpoint and optional extra"
            ) from None
        detector = GLiNERDetector(model, config)
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    policy = PIIPolicy(
        detector,
        placeholder_template=config.placeholder_template,
        max_redacted_ratio=config.max_redacted_ratio,
        blocked_entities=set(config.blocked_entities),
    )
    policy.fingerprint = fingerprint
    return policy
