from __future__ import annotations

from pathlib import Path

from arka.examples_validation import validate_examples


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    errors = validate_examples(project_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"\n{len(errors)} validation error(s) found.")
        return 1
    print("All example validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
