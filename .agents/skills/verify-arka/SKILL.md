---
name: verify-arka
description: Verify Arka's CLI cleanup stack, artifacts, resume behavior, and reasoning exports with local fixtures.
---

# Verify Arka

Use `principle-prove-it-works` for evidence and `show-me` for a short behavior diagram in each PR. Preserve user edits. This workflow changes no product code and makes no paid provider calls.

## Launch

Run from the repository root. Install with `uv sync --frozen --extra language`. Run `uv run arka --help` to confirm the installed CLI is ready. There is no long-running app process. Each helper starts its own loopback HTTP fixture on an OS-selected port and shuts it down on exit. Choose a fresh absolute output directory for each invocation.

The reasoning checks need a local official `lid.176.ftz` model. Set `ARKA_TEST_LID_MODEL` to its absolute path. Obtain it separately from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz and retain its CC-BY-SA 3.0 attribution. The helpers and pipeline do not download this model.

## Doctor

Run `git status --short`, `uv --version`, `just --list`, `uv run arka --version`, and `uv run arka --help`. Confirm the chosen evidence directory does not exist. For native checks, run `uv run --extra language python -c 'import os, pathlib, fasttext; p = pathlib.Path(os.environ["ARKA_TEST_LID_MODEL"]); assert p.is_file(); fasttext.load_model(str(p)); print(p)'`.

The fixtures supply a dummy key and their own loopback URL. No user token is required. They use isolated config directories, SQLite state, and run IDs. Nothing binds a fixed port. Do not infer native verification from a skipped test. The PR-head helper requires `gh` read access, fetched Git refs, and local branches matching their fetched remote heads.

## Drive

1. Run `ARKA_TEST_LID_MODEL="$ARKA_TEST_LID_MODEL" just check` and `uv build`.
2. Run `uv run python scripts/verify_cleanup_cli.py --output /tmp/arka-cleanup-proof`. This drives installed CLI subprocesses and actual SDK HTTP requests for PDF generation, cleanup, taxonomy selection, and resume.
3. Run `uv run --extra language python scripts/verify_stack_cli.py --output /tmp/arka-stack-proof`. This drives PDF boilerplate removal, generated reasoning, language checks, benchmark filtering, deduplication, and resume.
4. Run `uv run --extra language --with transformers --with datasets --with jinja2 python scripts/verify_hf_exports.py --output /tmp/arka-stack-proof`. This loads all three exports and checks Qwen3 tokenization. The tokenizer may download from Hugging Face on its first run. These packages stay outside project dependencies.
5. For this stack's eight branch checks, run `git fetch origin`, then `uv run python scripts/verify_pr_heads.py --output /tmp/arka-pr-head-proof`. It runs frozen dependency sync, `just check`, and `uv build` at each head. It prints failures and preserves logs before removing only the checkouts it created.

Use a new path if a prior run exists. Do not reuse cached run output as fresh proof.

## Evidence

Inspect `result.json`, exported JSONL, `cli.log`, `resume.log`, and stage `stats.json` and Parquet files. The cleanup helper also saves PDF and taxonomy configs. The stack helper saves benchmark input, all export formats, and request receipts. The HF helper saves `hf-result.json` and `hf-rendered.txt`. The branch helper saves head SHA, base branch, commands, exit codes, counts, build files, and full logs.

For each PR, put a small Mermaid input-to-output diagram next to an actual fixture example. Cite the exact checked head, command results, and committed evidence links. Identify whole-stack checks separately from tests at that PR's head. Keep the 500-record repetition audit and 1,000-record identity audit pending until representative generated data is available. Fixture performance is not a production false-positive estimate. Token preservation is not proof of training quality.

## Cleanup

Helpers stop only their own fixture server. The PR-head helper removes only its own detached checkouts, without force. On failure, retain the evidence directory and inspect the full log. If checkout cleanup fails, inspect `git worktree list` and the reported path before acting. No helper deletes evidence or model files. Confirm `result.json` and exports remain after the process exits.

## Helpers

- `scripts/verify_cleanup_cli.py --output PATH`: PDF and taxonomy generation through CLI/SDK/HTTP, filters, lineage, artifacts, resume.
- `scripts/verify_stack_cli.py --output PATH`: complete cleanup stack with a supplied native language model.
- `scripts/verify_hf_exports.py --output PATH`: existing stack exports through HF datasets and Qwen3 tokenizer.
- `scripts/verify_pr_heads.py --output PATH`: PRs #184 through #191 at their own heads. This is deliberately stack-specific; update the explicit PR range for a later stack.

See [features/README.md](features/README.md). Maintain this map with `maintain-verification-skill` when behavior or commands change.
