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


## 2024-08-16 - Palette: Enhance YAML syntax error formatting
* **What**: Updated `_format_validation_error` exception block in `src/arka/config/loader.py` to extract `line`, `column`, and `name` from `yaml.YAMLError` using its `problem_mark` attribute.
* **Why**: To remove developer friction when writing YAML configurations by providing actionable location hints for syntax errors rather than raw stack traces with unreadable line markers.
* **Before**: Output error omitted accurate line details when the stream was explicitly named, or just printed Python stack traces or opaque `<unicode string>` identifiers.
* **After**: Output explicitly prints the exact file name, line, and column of the syntax error (e.g. `YAML syntax error in /path/to/config.yaml at line 4, column 1:`).
