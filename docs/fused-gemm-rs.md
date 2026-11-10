# GEMM + reduce-scatter

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`GemmReduceScatterOp` is the [GEMM + all-reduce](fused-gemm-ar.md) pipeline
stopped before the gather, for sequence parallelism. The GEMM's epilogue pushes
each destination's row band as it is produced, and the reduce sums this rank's
band. Rank `r` keeps rows `r*M/P` onward of the sum: the dim-0 split
`reduce_scatter_tensor` makes, and what a sequence-parallel layer keeps after a
RowParallelLinear. It lives in the same module, `mori.ops.gemm_ar`.

```python
from mori.ops.gemm_ar import GemmReduceScatterOp, preshuffle_a_scale, preshuffle_b

with GemmReduceScatterOp(comm, n=5120, k=2048, m_max=16384, quant="mxfp8") as op:
    b_shuffled = preshuffle_b(b_fp8)                       # once per weight
    m_pad = op.padded_m(x.shape[0])
    a_fp8, a_exps = quantize_mxfp8(op.pad_rows(x, m_pad))  # your quantiser
    band = op(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale)   # [m_pad / P, N] bf16
```

- **Quantisation**: as for GEMM + all-reduce: `quant="blockscale"` (the
  default), `"mxfp8"`, `"mxfp8_raw"` or `"bf16"`.
- **No wire options**: the only leg is the scatter, which is bf16, so the
  constructor takes none of the all-reduce's `gather_dtype`,
  `gather_transport`, `scatter_dtype` or `schedule`.
- **Shapes**: K a multiple of 128 and at least 256; N a multiple of 256; 2 to
  8 ranks.
- **Window**: the all-reduce's. It holds a whole `[m_max, N]` output region, of
  which this operator writes `1/P`; size it with
  `GemmReduceScatterOp.window_bytes_for`.

DeepSeek-V4.1-Flash `wo_b` (N=5120, K=2048, mxfp8) on 4x MI355X. Graph replay,
maximum over ranks, median of 3 interleaved rounds. µs:

| M | GEMM only | GEMM + `reduce_scatter_tensor` | split SDMA | fused | vs RCCL |
|---:|---:|---:|---:|---:|---:|
| 2048 | 39.6 | 161.7 | 152.8 | 156.6 | -3.2% |
| 4096 | 66.4 | 295.4 | 269.9 | **262.9** | -11.0% |
| 8192 | 104.4 | 540.5 | 484.2 | **460.2** | -14.9% |
| 16384 | 174.9 | 1065.3 | 917.3 | **834.5** | **-21.7%** |

The [`gemm_ar` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md), next to the code, has the full
design, every measurement table and the negative results.
