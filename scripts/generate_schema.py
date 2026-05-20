from __future__ import annotations

import json
from pathlib import Path

from arka.config.models import ResolvedConfig


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    schema = ResolvedConfig.model_json_schema()

    schema_path = project_root / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Generated schema saved to {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
