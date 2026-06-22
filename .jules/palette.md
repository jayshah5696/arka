## 2023-10-24 - Include lost records count in stage failure CLI error
* **What**: Updated the `RuntimeError` wrapper in `PipelineRunner` to include the number of records lost (`count_in`) when a stage fails.
* **Why**: When a stage crashes (e.g. LLM API failures after retries), the CLI error tells the user what failed but left ambiguity on the blast radius. Knowing how many records were lost helps the developer decide whether to debug the stage or ignore it.
* **Before**: `Error: Pipeline execution failed - Stage 'generate' failed - API Error`
* **After**: `Error: Pipeline execution failed - Stage 'generate' failed - API Error. 150 records were lost.`
