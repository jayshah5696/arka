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

## 2024-06-24 - Palette: Raise explicit FileNotFoundError for missing PDF source files
* **What**: Changed the `ValueError` raised in `PDFSourceStage` when a source file is missing to a `FileNotFoundError`, matching the behavior of `SeedSourceStage`, and updated the error message to include the full evaluated path. Added a unit test to verify this behavior.
* **Why**: To remove developer friction by providing a standard, clear, and actionable error (matching other source stages) when a required input file cannot be found. This consistency reduces cognitive load during debugging.
* **Before**: The CLI threw a generic `ValueError` without an explicit `FileNotFoundError` context when a PDF file was missing.
* **After**: The CLI catches and formats the `FileNotFoundError` consistently, printing "Error: Pipeline execution failed - Stage '01_source' failed - PDF source file not found at expected path: [full_path]. [count] records were lost."

## 2024-11-20 - Palette: Improved YAML syntax error formatting
* **What**: Updated `ConfigLoader.load` in `src/arka/config/loader.py` to catch `yaml.YAMLError` and extract `problem_mark.name`, `line`, and `column` to provide a clear, developer-friendly error message. Also updated `test_yaml_syntax_error_includes_filename` in `tests/unit/test_config_loader.py` to assert the new message format.
* **Why**: To remove developer friction when configuring pipelines. Previously, raw YAML syntax errors from PyYAML were opaque and difficult to debug, often just returning the raw exception string without clear line/column pointers if the user didn't know what to look for.
* **Before**: `Configuration is invalid: while parsing a flow node expected the node content, but found '-' in "/app/bad_yaml.yaml", line 12, column 3` (or similar raw traceback/string representations).
* **After**: `Error: YAML syntax error in /app/bad_yaml.yaml at line 12, column 3: expected the node content, but found '-'`

## $(date +%Y-%m-%d) - Palette: Colorized output for CLI successes and failures
* **What**: Replaced `click.echo` with `click.secho` in `src/arka/cli.py` to add color coding to error messages (`fg="red"`) and success messages (`fg="green"`).
* **Why**: To remove developer friction by making critical output (successes and errors) visually distinct and easier to parse in the terminal.
* **Before**: Output was uniform plain text, making it harder to quickly spot success or failure states.
* **After**: Error messages are highlighted in red and success messages in green, providing immediate visual feedback.
