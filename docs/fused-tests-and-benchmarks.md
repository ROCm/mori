# Running the tests and benchmarks

Every operator has a numerics test of its kernels and a multi-rank test of the
public operator (2 ranks for `gemm_ar`, 8 for the others). CI runs both in the CCO job. They need the environment
described in [What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_gemm_ar_op.py
pytest tests/python/cco/test_gemm_a2a.py tests/python/cco/test_gemm_a2a_op.py
pytest tests/python/cco/test_gemm_ag.py tests/python/cco/test_gemm_ag_op.py
pytest tests/python/cco/test_ag_gemm.py tests/python/cco/test_ag_gemm_op.py
pytest tests/python/cco/test_a2a_gemm.py tests/python/cco/test_a2a_gemm_op.py
```

Each benchmark under `benchmark/cco/flydsl/<op>/` checks its modes against a
reference and prints one `RESULT_JSON` line per configuration. All of them have
`gemm-only`, `split-rccl`, `split-sdma` and `fused-sdma`; `gemm_ar` adds
`split-lsa` and `fused-lsa`, `gemm_a2a` adds `gemm-staging` and `gemm-to-window`,
and `gemm_ag` adds `gemm-to-window`. Only `bench_a2a_gemm.py` runs all its modes
by default (`--modes`). The others run one `--mode` per invocation, by default
`fused-sdma` (`gemm_ag`: `split-sdma`). For example:

```bash
torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048
```

Each operator's README has the exact commands behind its tables. Before
trusting a number of your own:

- Keep the node otherwise idle.
- Interleave the configurations you compare across rounds.
- Time CUDA-graph replays of many calls, not one.
- Pin `--quant`.
- Allow about 10 s between back-to-back 8-rank runs, so the SDMA queues are
  released.

The `gemm_ar` README's
[Measurement traps](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md#measurement-traps)
lists the ones that have produced wrong numbers here.
