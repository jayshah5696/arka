1. **Hoist config_hash computation out of loops in generator stages.**
   - In `TransformGeneratorStage.run` (src/arka/pipeline/generator_stages.py), the `config_hash=self._config_hash(ctx)` is called inside the `for record in transformable_records:` loop. Since `ctx.config` is invariant per stage execution, computing its hash via JSON serialization on every single record adds massive redundant overhead. I will compute `config_hash = self._config_hash(ctx)` before the loop and pass it in.

2. **Run tests.**
   - `uv run pytest tests/` to ensure everything is green.

3. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
   - Run the checks to make sure the code is compliant.

4. **Submit PR.**
   - With PR title "Bolt: [optimization summary]".
