# All-gather + GEMM

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`AgGemmOp`: sequence parallelism leaves every rank with its own tokens
`A_r [M, K]`, and a ColumnParallelLinear needs every rank's tokens times this
rank's weight shard: `C_r = concat_s(A_s) @ B_r.T`, `[P*M, N]`. The copy
engines push this rank's rows to every peer in row bands. The GEMM starts on
its own rows, and each remote tile starts as soon as the band it needs has
landed.

```python
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

N, K, M_MAX = 6288, 7168, 4096
vmm = 2 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
        op.self_test()
        b_shuffled = preshuffle_b(b_fp8)                   # this rank's [N, K] shard
        m = op.padded_m(x.shape[0])
        rows, scales = op.local_rows(m)                    # quantise straight into the window
        quantize_per_token_into(op.pad_rows(x, m), rows, scales)  # your quantiser
        c = op(rows, b_shuffled, scales, b_scale)          # [P*m, N] bf16
```

- **Quantisation**: fp8 ptpc. The per-token A scale travels with its rows.
- **Shapes**: N any multiple of 16. The last column tile is masked, so a
  shard like 6288 goes in without padding the weight. M a multiple of 256.
- **Zero copy**: `local_rows(m)` is this rank's slot in the window. Quantising
  into it skips the copy into the window, and `out=` skips the copy out.
- **Bands**: by default the most bands per peer, up to 4, that still hold 20
  output tiles each. `chunks=` pins the count.

Kimi-K3 column-parallel projections at TP8 (K=7168), fp8 in, bf16 out, 8x
MI355X. Median of 2 interleaved rounds, maximum over ranks. µs:

| Projection | global M | split RCCL | split SDMA | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|
| KDA, N=6400 | 4096 | 257.5 | 258.7 | **237.9** | -8.0% | -7.6% |
| | 32768 | 1751.5 | 1693.1 | **1322.3** | -21.9% | -24.5% |
| MLA, N=3840 | 4096 | 188.2 | 178.8 | **176.8** | -1.1% | -6.1% |
| | 32768 | 1316.6 | 1212.5 | **896.9** | -26.0% | **-31.9%** |

The [`ag_gemm` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/ag_gemm/README.md), next to the code, has the full
design, every measurement table and the negative results.
