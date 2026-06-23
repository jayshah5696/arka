## 2024-11-20 - Palette: [DX improvement] Include prefix Error to config validation error and number of lost records in stage failures
* **What**: Appended `Error: ` prefix to `ConfigValidationError` raised in CLI to avoid just a raw exception dump, and added `.{count_in} records were lost.` to the RuntimeError in the pipeline runner stage error handling to expose count of lost records clearly to the developer.
* **Why**: To quickly figure out that there was a configuration error when reading the CLI output by seeing an explicit `Error: ` tag. Added the lost records count so that debugging a stage failure allows a developer to directly understand the impact on their dataset before inspecting the run report json.
* **Before**: Output error of `Configuration is invalid: ...`, and `RuntimeError: Stage 'stage' failed - ...`
* **After**: Output error of `Error: Configuration is invalid: ...`, and `RuntimeError: Stage 'stage' failed - .... 10 records were lost.`
