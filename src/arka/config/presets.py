"""Generate ordinary, editable experiment files. Presets are never loaded at run time."""

from pathlib import Path

import click
import yaml

from arka.config.experiment import ExperimentConfig


@click.command(name="arka-init")
@click.option("--preset", type=click.Choice(["sft", "pdf-sft"]), required=True)
@click.option(
    "--output", type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option(
    "--source", required=True, help="Input path, relative to the generated config."
)
@click.option("--model", required=True, help="Provider model ID.")
@click.option("--base-url", required=True, help="OpenAI-compatible API endpoint.")
@click.option(
    "--target-count", type=click.IntRange(min=1), default=100, show_default=True
)
def init_config(
    preset: str, output: Path, source: str, model: str, base_url: str, target_count: int
) -> None:
    """Create a complete experiment config without reading credentials or calling providers."""
    pipeline = [
        {
            "id": "source",
            "config": {
                "type": "pdf_source" if preset == "pdf-sft" else "seed_source",
                "path": source,
            },
        }
    ]
    if preset == "sft":
        pipeline.append(
            {"id": "normalize", "config": {"type": "normalize_conversation"}}
        )
    pipeline.extend(
        [
            {
                "id": "generate",
                "model": "writer",
                "config": {
                    "type": "prompt_based_generator",
                    "target_count": target_count,
                    "generation_multiplier": 2,
                },
            },
            {"id": "deduplicate", "config": {"type": "exact"}},
            {"id": "length", "config": {"type": "length", "min_response_chars": 20}},
        ]
    )
    data = {
        "version": "2",
        "models": {
            "writer": {
                "provider": "openai",
                "model": model,
                "base_url": base_url,
                "api_key": "${ARKA_API_KEY}",
            }
        },
        "default_model": "writer",
        "execution": {"max_workers": 4, "timeout_seconds": 30, "max_retries": 3},
        "pipeline": pipeline,
        "output": {"format": "jsonl", "path": "output/dataset.jsonl"},
    }
    try:
        ExperimentConfig.model_validate(data)
        with output.open("x") as stream:
            stream.write(
                "# Set ARKA_API_KEY before running. All relative paths use this file's directory.\n"
            )
            stream.write(
                "# This is a starting pipeline. Evaluate quality before using its dataset.\n"
            )
            stream.write(yaml.safe_dump(data, sort_keys=False))
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"Created experiment config: {output}")
