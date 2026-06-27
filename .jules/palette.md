## 2024-06-21 - Added explicit dataset output path to --dry-run / --list-stages
* **What**: Updated `_dry_run_or_list_stages` in `src/arka/cli.py` to print the expected dataset output file path when running a dry run or listing stages.
* **Why**: To remove developer friction by explicitly showing where the dataset will end up before the run even starts, improving transparency and understanding of the pipeline.
* **Before**: Output stopped after listing the stages to execute.
* **After**: Output explicitly prints the expected dataset output path at the end.

## 2024-11-20 - Palette: Error visibility formatting in CLI config errors and pipeline stage failures
* **What**: Appended `Error: ` prefix to `ConfigValidationError` raised in CLI to improve configuration error visibility. Appended the grammatically correct number of lost records (e.g., `.{count_in} record was lost.` or `.{count_in} records were lost.`) to the RuntimeError in the pipeline runner stage error handling block.
* **Why**: To quickly distinguish configuration loading issues from other runtime failures by explicitly tagging them, and to provide developers with clear, grammatically correct visibility on data loss/blast radius if a stage crashes.
* **Before**: Output error was `Configuration is invalid: ...`, and RuntimeError was `Stage 'generate' failed - API Error`.
* **After**: Output error is `Error: Configuration is invalid: ...`, and RuntimeError is `Stage 'generate' failed - API Error. 1 record was lost.` or `Stage 'generate' failed - API Error. 150 records were lost.`


## 2024-12-05 - Palette: Format PyYAML syntax errors for cleaner CLI output
* **What**: Updated the `ConfigLoader` in `src/arka/config/loader.py` to extract `line`, `column`, and file path from `yaml.YAMLError`'s `problem_mark` when parsing YAML configurations. Wrapped it in a `ConfigValidationError` to prevent raw stack traces from reaching the user.
* **Why**: To remove developer friction when debugging configuration files. Previously, if a user made a YAML syntax error (like an indentation issue or an invalid mapping), the CLI would output a generic string representation of the PyYAML error without context or crash with a stack trace. This provides actionable location hints directly in the CLI.
* **Before**: Error output was a raw Python stack trace or a generic message with `<unicode string>` as the file name.
* **After**: Error output explicitly states: `Error: YAML syntax error in /path/to/file.yaml at line X, column Y: expected <block end>, but found '<block mapping start>'`
