# GEMM + all-gather

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`GemmAgOp`: every rank computes `C = A @ B.T` on its own rows and every rank
receives `[P*M, N]`, source rank `r` at rows `[r*M, (r+1)*M)`. It was built
for DeepSeek-V4's compressor `wkv_gate`, a projection on a context-parallel
token shard whose output every rank consumes. The GEMM writes C into this
rank's receive slot, and its epilogue pushes each finished chunk of rows to
every peer.

```python
from mori.ops.gemm_ag import GemmAgOp

N, K, M_MAX = 2048, 7168, 16384
vmm = 2 * GemmAgOp.window_bytes_for(world, n=N, m_max=M_MAX, out_dtype="fp32") + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, in_dtype="bf16", out_dtype="fp32") as op:
        op.self_test()
        m = op.padded_m(x.shape[0])
        c = op(op.pad_rows(x, m), w)                       # [P*m, N] fp32
```

- **Precisions**: `in_dtype="bf16"` takes bf16 `a`/`b` (plain row-major
  weight), no scales, and returns bf16 or fp32. `in_dtype="fp8"` takes
  `quant` = `ptpc` / `blockscale` / `mxfp8`, with the scale layouts under
  "fp8 operands" in [What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common),
  and returns bf16.
- **`mode="auto"`** (the default) fuses an M once `ceil(M/256) * ceil(N/256)`
  is at least the device's CU count, and splits below that, per M on one
  instance. `"fused"` and `"split"` pin a path.
- **Shapes**: N a multiple of 256. K a multiple of 128 (fp8, at least 256) or
  of 64 (bf16, at least 128). M a multiple of 128 (256 under mxfp8).

Through the operator, Pro `wkv_gate` (bf16 in, fp32 out). Graph-timed, maximum
over ranks. µs:

| P | M | `auto` | fused | split |
|---:|---:|---:|---:|---:|
| 8 | 2048 | **351.3** (split) | 362.8 | 351.3 |
| 8 | 4096 | **666.2** (split) | 685.5 | 666.1 |
| 8 | 8192 | **1239.2** (fused) | 1240.0 | 1288.7 |
| 4 | 8192 | **1231.3** (fused) | 1231.3 | 1282.2 |

Against Torch + RCCL on the same shapes (8x MI355X, three interleaved rounds),
the best MORI path is 10-16% faster on V4-Pro and 6-11% faster on V4.1-Flash.
For V4-Pro: M=2048 373.4 against 415.1 µs; M=16384 2527.7 against 3008.2 µs.
The steady part is the transport: SDMA's all-gather alone is about 13% faster
than RCCL's.

The [`gemm_ag` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ag/README.md), next to the code, has the full
design, every measurement table and the negative results.
