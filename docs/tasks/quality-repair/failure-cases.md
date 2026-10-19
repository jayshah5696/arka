# Quality repair failure cases

The first version judges single-turn answers, repairs explicitly eligible failures once, and judges the replacement with the same rubric and evidence. It retains passing originals and sends replacements through the configured downstream validators and deduplication.

Write focused tests before product code for these failures:

- A passing original is rewritten, or its input order changes.
- Repair changes the instruction or system message, retains stale reasoning or response scores, or loses instruction taxonomy metadata.
- A replacement fails its second judgment but is accepted, or an exhausted record receives another attempt.
- The repair limit is exceeded, an excluded rubric failure is repaired, or a multi-turn example is partially rewritten.
- A grounded record is repaired or judged without its exact permitted source chunk. A mismatched root or chunk must not supply evidence.
- A provider fails after generation and resume repeats the persisted generation or completed initial judgment.
- Changed inputs, rubric contents, effective settings or evidence reuse an incompatible attempt, including a completed-stage CLI resume.
- An invalid judge score, incomplete score response or malformed generation enters the dataset.
- A low-confidence decision judgment is treated as a definite quality failure and sent for repair.
- PII-rejected generation is judged again, or unprotected text enters attempt artifacts. Usage from rejected output must remain available.
- Unknown rubric dimensions and out-of-scale thresholds start provider requests.
- A repair duplicates a passing original and survives configured downstream deduplication.

CLI proof will also inspect request counts, per-attempt artifacts, reports, final exports and resume with no extra requests. Use deterministic local HTTP fixtures. Representative quality gains and paid-provider behavior remain separate evaluations.
