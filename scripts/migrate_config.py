#!/usr/bin/env python3
"""Offline migration script to convert legacy Arka configs to modern unified pipeline configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

from arka.config.models import ResolvedConfig


def migrate_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        content = input_path.read_text()
        data = yaml.safe_load(content) or {}
    except Exception as e:
        print(f"Error parsing input YAML: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform migration
    try:
        migrated_data = ResolvedConfig.migrate_old_config(data)
    except Exception as e:
        print(f"Error migrating config: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        output_path.write_text(yaml.safe_dump(migrated_data, sort_keys=False))
        print(f"Successfully migrated {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error writing output YAML: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a legacy Arka configuration file to the modern pipeline format."
    )
    parser.add_argument("input", type=str, help="Path to the legacy YAML config file")
    parser.add_argument("output", type=str, help="Path to save the migrated YAML config file")
    args = parser.parse_args()

    migrate_file(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
