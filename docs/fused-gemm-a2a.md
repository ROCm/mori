# GEMM + all-to-all

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`GemmA2AOp`: every rank computes the whole `C = A @ B.T` and sends column
block `j` to rank `j`. Rank `d` gets `[P*M, N/P]`, source rank `r` at rows
`[r*M, (r+1)*M)`. This is a QKV projection under Ulysses sequence
parallelism: N is the QKV width, `N/P` one rank's head shard, and `M = S/P`.
The epilogue stages C per destination, and the copy engine pushes each chunk
as soon as its tiles are done.

```python
from mori.ops.gemm_a2a import GemmA2AOp
from mori.ops.gemm_ar import preshuffle_b

with GemmA2AOp(comm, n=10240, k=8192, m_max=16384, quant="ptpc") as op:
    op.self_test()
    b_shuffled = preshuffle_b(w_qkv)                       # [N, K], same on every rank
    m = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m)) # your quantiser
    qkv = op(a_fp8, b_shuffled, a_scale, b_scale)          # [P*m, N/P] bf16
```

- **Quantisation**: `ptpc` or `mxfp8`.
- **Shapes**: N a multiple of `P * 256`, so each destination's columns are
  whole GEMM tiles. At P=8 that is N a multiple of 2048, which Llama-3.1-8B /
  70B / 405B, Qwen2.5-72B and Qwen3-32B satisfy. A model with fewer KV heads
  than ranks (Qwen3-235B at P=8) does not fit.
- `chunks=` pins the number of pushes per destination; the tile and the chunk
  count are otherwise picked from M.

8x MI355X, fp8 ptpc in, bf16 out. Every case ran in one process, with three
interleaved rounds and a median launch gap, maximum over ranks. µs:

| Shape | M (S) | GEMM only | split SDMA | split RCCL | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-70B, N=10240 K=8192 | 2048 (16k) | 151 | 244 | 344 | **203** | -16.6% | -40.9% |
| | 4096 (32k) | 257 | 441 | 513 | **328** | -25.5% | -36.0% |
| | 8192 (64k) | 451 | 813 | 930 | **578** | -28.8% | -37.8% |
| | 16384 (128k) | 918 | 1619 | 1824 | **1138** | **-29.7%** | -37.6% |
| Llama-3.1-405B, N=18432 K=16384 | 2048 (16k) | 502 | 681 | 738 | **568** | -16.6% | -23.1% |
| | 16384 (128k) | 3383 | 4694 | 4998 | **3678** | **-21.6%** | -26.4% |

The mxfp8 path gains 11-24% over split SDMA at the same shapes. On MiniMax-M3's
QKV at P=4, the operator's defaults are 1.6-10.4% ahead of aiter's fused
`sp_qkv` kernels.

The [`gemm_a2a` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_a2a/README.md), next to the code, has the full
design, every measurement table and the negative results.
