# gemm_ar_bf16: evaluation results

One row per run of the case, appended by `evaluate.py` (step 7 of the
[README](README.md)); the full record is the run's `evaluation.json`. Compare a change to
the harness against these rows: same case, same target, fewer rounds, GPU hours or owner
prompts, or a pass where there was a fail.

- **Result:** `pass`, `fail`, `incomplete` (no verdict), `unaudited`, `pending-review`
  (audit items nobody has cleared) or `void` (the audit found a leak).
- **Candidate, Existing, Gap:** medians at the target bucket (M=16384) from the verifier,
  in the same launch; the target is a gap of at most +5%.
- **GPU h:** time in run.py launches times ranks; reserved node time is in the record
  when the owner gave it.
- **Owner prompts:** the owner's messages after the kickoff.

No run yet.
