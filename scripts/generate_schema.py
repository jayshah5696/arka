from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from arka.config.experiment import ExperimentConfig
from arka.config.models import ResolvedConfig
from arka.recovery.models import RecoveryConfig


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    schema = TypeAdapter(ResolvedConfig | ExperimentConfig).json_schema()

    schema_path = project_root / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Generated schema saved to {schema_path}")
    recovery_path = project_root / "recovery.schema.json"
    recovery_path.write_text(
        json.dumps(RecoveryConfig.model_json_schema(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated recovery schema saved to {recovery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
