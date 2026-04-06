from __future__ import annotations

import os
import re
from pathlib import Path

from arka.config.loader import ConfigLoader

HEADER_FIELDS = [
    "WHAT",
    "WHEN",
    "REQUIRES",
    "RUN",
    "ARTIFACTS",
    "TEACHES",
    "COST",
]
ALLOWED_COSTS = {"free", "low", "medium", "high"}
_OPENROUTER_PATTERN = re.compile(r"openrouter\.ai", re.IGNORECASE)
_HEADER_PATTERN = re.compile(r"^#\s*([A-Z]+):\s*(.*)$")
_TODO_PATTERN = re.compile(
    r"^#\s*TODO:\s*.*(slice|milestone)", re.IGNORECASE | re.MULTILINE
)


def example_yaml_paths(project_root: Path) -> list[Path]:
    examples_dir = project_root / "examples"
    return sorted(
        path
        for path in examples_dir.rglob("*.yaml")
        if path.is_file() and "runs" not in path.parts
    )


def load_example_config(path: Path):
    os.environ.setdefault("OPENROUTER_API_KEY", "example-openrouter-key")
    os.environ.setdefault("OPENAI_API_KEY", "example-openai-key")
    return ConfigLoader().load(path)


def top_comment_block(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    block: list[str] = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            block.append(line)
            continue
        break
    return block


def header_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in top_comment_block(path):
        match = _HEADER_PATTERN.match(line)
        if match is None:
            continue
        values[match.group(1)] = match.group(2).strip()
    return values


def validate_example_file(path: Path) -> list[str]:
    errors: list[str] = []
    raw_text = path.read_text(encoding="utf-8")
    headers = header_values(path)

    missing_headers = [field for field in HEADER_FIELDS if field not in headers]
    if missing_headers:
        errors.append(f"{path}: missing header fields: {', '.join(missing_headers)}")

    cost = headers.get("COST", "")
    if cost and cost not in ALLOWED_COSTS:
        errors.append(f"{path}: COST must be one of {sorted(ALLOWED_COSTS)}")

    if _OPENROUTER_PATTERN.search(raw_text):
        if "${OPENROUTER_API_KEY}" not in raw_text:
            errors.append(
                f"{path}: OpenRouter config must reference OPENROUTER_API_KEY"
            )
        if "${OPENAI_API_KEY}" in raw_text:
            errors.append(
                f"{path}: OpenRouter config must not reference OPENAI_API_KEY"
            )

    if path.parent.name == "future" and _TODO_PATTERN.search(raw_text) is None:
        errors.append(
            f"{path}: future example must include a TODO comment with target slice or milestone"
        )

    resolved = load_example_config(path)
    if not resolved.output.path.startswith("./"):
        errors.append(f"{path}: output.path must start with './'")

    seed_path = resolved.data_source.path
    if seed_path is not None and "/seeds/" in seed_path:
        candidate = (path.parent / seed_path).resolve()
        if not candidate.exists():
            errors.append(f"{path}: referenced seed file does not exist: {seed_path}")

    return errors


def validate_examples(project_root: Path) -> list[str]:
    errors: list[str] = []
    for path in example_yaml_paths(project_root):
        errors.extend(validate_example_file(path))
    return errors
