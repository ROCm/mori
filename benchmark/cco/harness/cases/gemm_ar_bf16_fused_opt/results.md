# gemm_ar_bf16_fused_opt: evaluation results

One row per run of the case, appended by `../evaluate.py`; the full record is the run's
`evaluation.json`, which also holds the start candidate's time and the improvement on it.
The target is a gap of at most 0% to the existing op at M=16384, in the same launch. The
columns are those of [gemm_ar_bf16's results](../gemm_ar_bf16/results.md).
