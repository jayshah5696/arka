## 2025-02-23 - Prevent Prompt Injection in RewardModelFilterStage
* **Vulnerability:** Unsanitized record instructions and responses were directly passed into LLM prompt messages in `RewardModelFilterStage`.
* **Learning:** This existed because the reward model stage was likely built assuming inputs were already clean or were only evaluated structurally, overlooking that malicious content could override the judge's scoring behavior.
* **Prevention:** Always enforce the use of `sanitize_for_prompt` and explicit `<text>` tags wrapping for all untrusted data passed to LLMs, as explicitly stated in the newly added memory rule.
