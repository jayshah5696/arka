---
name: verify-arka
description: Verify Arka CLI configuration, generation, cleanup, decision scoring, exports, and resume with local HTTP fixtures. Use after relevant behavior changes or for an Arka verification request.
---

# Verify Arka

Verify observable CLI behavior and retain fresh evidence. This skill covers the recipes in [features/README.md](features/README.md). It does not prove every pipeline option or dataset quality. Keep product fixes separate from verification changes and report product failures without hiding them.

## Launch

Run from the repository root:

```sh
uv sync --frozen
uv run arka --version
uv run arka --help
just verify
```

The command prints a fresh absolute `/tmp/arka-verify-*` evidence directory. To choose a location, use `just verify --output /tmp/arka-local-proof`; that directory must not exist. Arka is a batch CLI. Successful version/help commands establish readiness. Each fixture binds loopback on an OS-selected port and stops when its helper exits. There is no persistent app process.

## Doctor

Run `git status --short`, `git rev-parse HEAD`, `uv --version`, `just --list`, `uv run arka --version`, and `uv run arka --help`. Confirm that an explicit evidence path does not exist. Default checks require no user credentials, model download, or fixed port. They create synthetic input, isolated configs, run directories, and SQLite state. Separate runs can execute concurrently with separate evidence paths.

The first dependency sync needs access to the configured package registry. The default helpers only call local HTTP fixtures; example validation and stage previews do not execute providers. Do not print environment tokens.

Optional native checks require `uv sync --frozen --extra language` and an existing official `lid.176.ftz` model. Set `ARKA_TEST_LID_MODEL` to its absolute path. Check it with:

```sh
uv run --extra language python -c 'import os, pathlib, fasttext; p = pathlib.Path(os.environ["ARKA_TEST_LID_MODEL"]); assert p.is_file(); fasttext.load_model(str(p)); print(p)'
```

Obtain the model separately from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz and retain its CC-BY-SA 3.0 attribution. Missing native prerequisites mean that recipe is unverified.

## Drive

Run `just check` and `uv build` for code changes. Then run `just verify`. It checks CLI readiness, validates all current example configs, previews their stages, and runs two installed-CLI workflows:

- PDF generation, text cleanup, taxonomy coverage selection, and resume.
- Decision seed gating, shared panel scoring, critics, selection, usage reporting, and resume.

Both use actual SDK HTTP requests to deterministic loopback fixtures. No internal setter substitutes for the CLI path.

Run optional checks only when relevant:

```sh
uv run --extra language python scripts/verify_stack_cli.py --output /tmp/arka-native-proof
uv run --extra language --with transformers --with datasets --with jinja2 python scripts/verify_hf_exports.py --output /tmp/arka-native-proof
```

The second command consumes the first command's existing exports. It may download the Qwen3 tokenizer from Hugging Face. These checks are outside the default dependency set.

Live provider verification requires task authorization for paid calls and an existing `OPENROUTER_API_KEY`. Its recipe is `uv run python scripts/verify_decisions_live.py --output /tmp/arka-live-proof`. It limits execution to 48 HTTP requests. This helper drives the production pipeline with an instrumented transport, rather than the installed CLI. Do not describe it as CLI proof or infer a spending cap from its request limit.

## Evidence

Inspect the root `result.json`. It records the Git revision, working tree status, commands, exit codes, elapsed times, and full log paths. A failure stops subsequent checks and preserves the summary and logs. A terminated run can retain status `running`; that is incomplete proof.

Under `cleanup/`, inspect `result.json`, `requests.json`, source PDF/taxonomy, and each scenario's config, `cli.log`, `resume.log`, dataset, manifest, stats, and Parquet artifacts. Expect 14 initial requests, two exports per scenario, and zero additional requests on resume.

Under `decisions/`, inspect the corresponding files. Three seeds produce two exports through nine requests. Resume must preserve dataset bytes and make zero new requests. The resolved config must redact the dummy key. Fixture cost fields use synthetic rates and are not actual charges.

Report the exact command, revision, action, resulting state, and artifact paths. Separate local fixture proof, optional native/HF proof, live provider proof, and representative dataset audits. Do not claim skipped checks passed.

## Cleanup

Normal helper exit shuts down only its own loopback server. No helper deletes evidence. After the command returns, confirm both child `result.json` files, datasets, and `resume.log` files remain. If interrupted, inspect this run's process tree before stopping only its processes. Preserve partial logs and do not reuse the directory for fresh proof. Do not use broad process kills or remove model files.

## Helpers

- `just verify [--output PATH]` runs `uv run python scripts/verify_local.py` with the same optional argument.
- `uv run python scripts/verify_cleanup_cli.py --output PATH` runs PDF cleanup and taxonomy selection.
- `uv run python scripts/verify_decisions_cli.py [--output PATH]` runs decision scoring and resume.
- `uv run --extra language python scripts/verify_stack_cli.py --output PATH` runs the native cleanup/reasoning recipe.
- `uv run --extra language --with transformers --with datasets --with jinja2 python scripts/verify_hf_exports.py --output PATH` checks existing stack exports.
- `uv run python scripts/verify_decisions_live.py --output PATH` runs the separately authorized live provider recipe.

`scripts/verify_pr_heads.py` is a historical PR #184 through #191 audit. It is not part of current default verification. Use `maintain-verification-skill` for a broader feature-map drift audit.
