"""Resolve experiment identity and external dependencies before opening run state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from arka.config.models import PDFSourceConfig, ResolvedConfig, SeedSourceConfig


def experiment_settings(config: ResolvedConfig) -> dict[str, Any]:
    """Only typed connection locations contain transport settings or secrets."""
    if config._experiment is None:
        return BaseModel.model_dump(config, mode="json")
    data = config._experiment.model_dump(mode="json")
    data.pop("execution")
    for connection in [
        *data["models"].values(),
        data.get("decisions"),
        data.get("embeddings"),
    ]:
        if connection:
            connection.pop("timeout_seconds", None)
            connection.pop("max_retries", None)
    return data


def execution_settings(config: ResolvedConfig) -> dict[str, Any]:
    data = {"executor": config.executor.model_dump(mode="json")}
    if config._experiment is not None:
        data["transport"] = config._experiment.execution.model_dump(
            exclude={"max_workers"}
        )
    return data


def dependency_hash(path: Path) -> str:
    """Hash files in bounded chunks, or every regular file in a local model directory."""
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for item in paths:
        if not item.is_file():
            continue
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def dependencies(config: ResolvedConfig, root: Path) -> dict[str, str]:
    """Inventory input paths from the typed configuration, excluding output paths."""
    found: dict[str, str] = {}

    def visit(value: Any, location: str = "") -> None:
        if isinstance(value, BaseModel):
            for key in type(value).model_fields:
                item = getattr(value, key)
                where = f"{location}.{key}" if location else key
                is_input = key in {
                    "rubric_path",
                    "taxonomy_path",
                    "model_path",
                    "lid_model_path",
                    "training_paths",
                    "eval_paths",
                } or (
                    key == "path"
                    and isinstance(value, SeedSourceConfig | PDFSourceConfig)
                )
                if is_input and isinstance(item, (str, list)):
                    for name in item if isinstance(item, list) else [item]:
                        if not isinstance(name, str):
                            continue
                        path = (root / name).expanduser().resolve()
                        if not path.exists():
                            raise ValueError(
                                f"Input dependency {where}: {path} does not exist; restore it or use a new run"
                            )
                        found[str(path)] = dependency_hash(path)
                else:
                    visit(item, where)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    # Traverse typed input settings, never arbitrary user dictionaries or export paths.
    for index, stage in enumerate(config.pipeline):
        visit(stage, f"pipeline[{index}]")
    visit(config.labeling_engine, "labeling_engine")
    if config.pii:
        visit(config.pii, "pii")
    return found


@dataclass(frozen=True)
class ExecutionPlan:
    settings: dict[str, Any]
    dependencies: dict[str, str]
    stages: tuple[str, ...]
    fingerprint: str

    @classmethod
    def build(
        cls, config: ResolvedConfig, root: Path, names: list[str]
    ) -> ExecutionPlan:
        if len(names) != len(set(names)):
            raise ValueError("Stage names collide; use unique stage IDs")
        settings = experiment_settings(config)
        inputs = dependencies(config, root)
        identity = {"settings": settings, "dependencies": inputs, "stages": names}
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return cls(settings, inputs, tuple(names), fingerprint)

    def dump(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "dependencies": self.dependencies,
            "stages": self.stages,
            "fingerprint": self.fingerprint,
        }

    def validate_existing(self, run_dir: Path, *, resume: bool) -> None:
        path = run_dir / "plan.json"
        if not path.exists():
            if (run_dir / "config.resolved.yaml").exists():
                raise ValueError(
                    "Existing run has no dependency plan; use a new run ID"
                )
            if resume:
                raise ValueError("No saved experiment to resume; start a new run")
            return
        if not resume:
            raise ValueError("Run already exists; use --resume or a new run ID")
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != self.fingerprint:
            changed = [
                key
                for key in ("settings", "dependencies", "stages")
                if previous[key] != json.loads(json.dumps(self.dump()[key]))
            ]
            raise ValueError(
                f"Experiment changed ({', '.join(changed)}); use a new run ID"
            )
