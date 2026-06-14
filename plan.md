1. **Understand the Goal**: Identify and implement ONE small Developer Experience (DX) improvement that makes `arka` easier to use, debug, or understand. Must be under 50 lines, no new dependencies, follow existing style. Add `# DX: [what] [why]` comment.
2. **Review Checklist & Findings**:
    - `cli.py` has a clean `_print_summary` block, handles `ConfigValidationError` cleanly.
    - Missing config field errors name the field.
    - Environment variable missing handles via `ConfigValidationError` cleanly.
    - File not found in `01_source` gives the full path.
    - Invalid YAML keys point out the exact invalid key.
    - Missing YAML field points out the field.
    - There is a `--dry-run` / `--list-stages` to preview stages.
    - Progress is logged per-stage (`print(f"Running stage {i}/{len(stages)}: {stage.name} ({count_in} records in)...")`).
    - Run summary handles dropped records.
    - *What's missing / could be better?*
        - What happens when a fatal error like `FileNotFoundError` for the main config file occurs?
        - `ConfigValidationError` prints `ConfigValidationError(self._format_validation_error(exc, data)) from exc`. When printed using `str(exc)`, it prints cleanly without tracebacks.
        - Looking closely at `src/arka/pipeline/runner.py`:
          ```python
          except Exception as exc:
                failed_stage_name = stage.name
                failed_error = StageErrorInfo(
                    type=exc.__class__.__name__,
                    message=str(exc),
                )
                ...
                raise RuntimeError(f"Stage '{stage.name}' failed - {exc}") from exc
          ```
        - In `src/arka/cli.py`:
          ```python
          except Exception as exc:
              # DX: Catch pipeline execution errors to prevent raw Python tracebacks.
              # This provides a clean, human-readable error message to the user.
              click.echo(f"Error: Pipeline execution failed - {exc}", err=True)
              sys.exit(1)
          finally:
              duration_secs = time.time() - start_time
              _print_summary(resolved_run_id, project_root, duration_secs)
          ```
        - If `_load_config(config_path)` raises `FileNotFoundError`, it prints a clean error. What about missing `project_root / resolved_config.output.path`? It will create directories, so that's fine.
        - *Wait! Look at the LLM client.* What if an API failure occurs after retries? It raises `LLMClientError("retryable_api_error", str(exc))`. This bubbles up and is caught by `PipelineRunner`, which raises `RuntimeError(f"Stage '{stage.name}' failed - {exc}")`. And `cli.py` prints it cleanly.
        - *Wait! What about the `--help` text in `src/arka/cli.py`?*
        - Look at the "Config discoverability" section in the checklist: "Are all config fields documented with inline comments in config.example.yaml?". The file `config.example.yaml` doesn't exist! There are files in `examples/`. Memory says: "Example configuration files for different workflows are stored in the examples/ directory rather than as a single catch-all config.example.yaml file in the root."
        - Wait, is there a way to generate a default config or get a sample one easily?
        - Look at the checklist again:
            - "run_report.json has rich data but no human-readable summary printed at the end" -> wait, I saw `_print_summary` prints this. Wait, does it print `dropped` count overall? It prints per stage.
            - "LLM API failures after retries - does it tell the user which stage failed and how many records were lost?" -> Currently it raises `RuntimeError(f"Stage '{stage.name}' failed - {exc}") from exc` and exits. The final summary says `0 out` but wait, if it fails, it drops *all* remaining records for that stage, and maybe we want to explicitly state how many were lost due to failure? Actually `runner.py` says:
              ```python
                    stage_stats.append(
                        self._build_stage_stat(
                            stage_name=stage.name,
                            count_in=count_in,
                            count_out=len(records), # Wait, len(records) here is from before the stage!
                            ...
              ```
              Wait, look at `runner.py` `except Exception as exc:` block:
              ```python
                    stage_stats.append(
                        self._build_stage_stat(
                            stage_name=stage.name,
                            count_in=count_in,
                            count_out=len(records), # <--- BUG! records is the input list, so it records count_out = count_in even though the stage failed!
                            status="failed",
                            resumed=False,
                            stats_path=run_paths.stage_stats_path(stage.name),
                            error=failed_error,
                        )
                    )
              ```
              If a stage fails, it produced 0 valid output records, but `count_out=len(records)` writes the *input* count! The memory says: "Pipeline stage results must explicitly differentiate between 'dropped' records (filtered intentionally) and 'lost' records (due to a failed stage). On stage failure, `count_out` must be recorded as 0 in checkpoints and stats to accurately reflect that the stage produced zero valid records."
              I should fix this!
3. **Plan**:
   - Update `src/arka/pipeline/runner.py` in the `except Exception as exc:` block for stage execution.
   - Change `count_out=len(records)` to `count_out=0` in `checkpoint_manager.save_stage` and `self._build_stage_stat` when a stage fails.
   - Add `# DX: On stage failure, record count_out as 0 so the run report accurately reflects that 0 valid records were produced, rather than duplicating the input count.`
4. **Pre-commit checks**: Include testing and verifying `just check`.
