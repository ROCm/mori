# GEMM + all-reduce

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`GemmAllReduceOp` writes the GEMM's C straight into the window. The copy
engine pushes each destination's slice as soon as its last tile is written,
so the reduce-scatter half of the all-reduce overlaps the rest of the GEMM.
The reduce and all-gather run after it. The same pipeline stopped before the
gather is [GEMM + reduce-scatter](fused-gemm-rs.md).

```python
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

with GemmAllReduceOp(comm, n=7168, k=2048, m_max=16384, quant="blockscale") as op:
    b_shuffled = preshuffle_b(b_fp8)                       # once per weight
    m_pad = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))   # your quantiser
    out = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]]   # [M, N] bf16
```

- **Quantisation**: `quant="blockscale"` (the default), `"mxfp8"`,
  `"mxfp8_raw"` (row-major uint8 scales staged through LDS), or `"bf16"` (bf16
  A and B with no scales, for layers the checkpoint keeps in bf16; the scatter
  wire stays bf16). There is no `"ptpc"`.
- **fp8 on the wire**: `gather_dtype="fp8"` sends the all-gather leg as e4m3
  with one fp32 scale per row, halving its bytes. `gather_transport="lsa"` lets
  the CUs pull and widen in one step. This is lossy: relL2 2.5e-2 against
  2.4e-3 with a bf16 wire.
- **Shapes**: K a multiple of 128 and at least 256; N a multiple of 256 (and
  of 1024 with an fp8 gather); 2 to 8 ranks.
- **`schedule="wo_b"`** selects the measured padding and chunk profile for the
  two `wo_b` targets below. Pass `logical_m=` to the call when using it.
- **Standalone GEMM**: `Mxfp8GemmOp` and `Mxfp8GemvOp` (decode, M ≤ 32) are
  the same mxfp8 GEMM with nothing fused, for a ColumnParallelLinear that has
  no collective to fuse with.

`wo_b` at M=16384, maximum over ranks. µs. V4-Pro is on 8x MI355X, median of
11. V4.1-Flash is on 4x MI355X, from the README's fp8-wire comparison; its
fused bf16-wire cell comes from a different run and is left out here:

| | DeepSeek-V4-Pro (TP8, blockscale) | DeepSeek-V4.1-Flash (TP4, mxfp8) |
|---|---:|---:|
| split SDMA, bf16 wire | 1472.9 | 1592.5 |
| fused, bf16 wire | 1148.8 | — |
| fused, fp8 wire | **979.3** | **1192.1** (over LSA) |
| GEMM + NCCL, as the model ships | 1419.5 | — |

The two models do not reach the same conclusion. On V4-Pro the overlap
dominates. On V4.1-Flash the GEMM is 178 µs against 1414 µs of communication,
so most of the gain comes from the fp8 wire (-15.6% on its own) and the mxfp8
GEMM, not from fusing. In an SGLang server, V4.1-Flash prefill throughput at
batch 4-16 rises **+5.6% to +6.0%** with the fused `wo_b` on the fp8 wire and
**+7.4% to +7.7%** with the standalone mxfp8 GEMM added. V4-Pro's GPU-busy time
over a profiled prefill falls **5.0%**.

The [`gemm_ar` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md), next to the code, has the full
design, every measurement table and the negative results.
