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


## 2024-06-26 - Palette: Better YAML syntax error formatting
* **What**: Updated `_format_yaml_error` in `src/arka/config/loader.py` to intercept PyYAML syntax errors and extract specific line, column, and file names from the exception's `problem_mark`.
* **Why**: To remove developer friction by providing clean, actionable hints directly pointing developers to syntax errors in their configuration files, without exposing confusing `while scanning...` raw stack traces or default `<unicode string>` filenames.
* **Before**: Output error was `Configuration is invalid: while scanning a simple key... in "<unicode string>", line 19, column 1...`.
* **After**: Output error is `Error: Invalid YAML syntax in 'config.yaml' at line 19, column 1: could not find expected ':' (context: while scanning a simple key)`.
