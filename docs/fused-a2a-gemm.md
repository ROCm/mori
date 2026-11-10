# All-to-all + GEMM

The example below assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

`A2aGemmOp`: after Ulysses attention every rank holds every token for its own
head shard, `[P*M, K/P]`. The o-projection needs this rank's M tokens over all
of K. Block 0 of the GEMM posts every band to every peer. The GEMM starts on
this rank's own K segment, and each remote segment is read as soon as its band
of rows has landed. This is the reverse of `GemmA2AOp`.

```python
from mori.ops.a2a_gemm import A2aGemmOp
from mori.ops.gemm_ar import preshuffle_b

with A2aGemmOp(comm, n=N, k=K, m_max=max_tokens_per_rank) as op:
    op.self_test()
    b = preshuffle_b(w_o)                                  # [N, K], same on every rank
    m = op.padded_m(tokens_per_rank)
    a = op.send_buffer(m)                                  # [P*m, K/P] fp8, in the window
    quantize_attention_output_into(a)                      # your quantiser
    c = op(a, b, a_scale, b_scale)                         # [m, N] bf16
```

- **Quantisation**: fp8 ptpc. The per-token scale is taken over all heads
  before quantisation, from the row maxima gathered across ranks, so it is
  local and never travels.
- **Shapes**: N a multiple of 256; K a multiple of `P * 128`; M a multiple of
  256.
- **Bands and waits** are picked per M from the measurements below (`chunks=`
  and `wait=` override them). `send_buffer(m)` lets attention write its output
  straight into the window.

8x MI355X (4 for M3), fp8 ptpc in, bf16 out. Three interleaved rounds, median
launch gap, maximum over ranks. µs:

| Shape | M per rank | GEMM only | split RCCL | split SDMA | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M3 o-proj, P=4, N=K=6144 | 2048 | 63.0 | 144.0 | 132.9 | **121.7** | -8.5% | -15.5% |
| | 4096 | 125.1 | 264.9 | 244.0 | **184.1** | -24.5% | -30.5% |
| | 8192 | 208.6 | 479.0 | 438.7 | **300.2** | -31.6% | **-37.3%** |
| Llama-3.1-70B o-proj, P=8, N=K=8192 | 2048 | 94.3 | 156.2 | 145.8 | **134.4** | -7.8% | -14.0% |
| | 4096 | 182.6 | 294.6 | 269.6 | **258.6** | -4.1% | -12.2% |
| | 8192 | 356.4 | 552.5 | 506.4 | **435.2** | -14.0% | -21.2% |

Through the operator with its defaults, the 70B shape lands within 1.3% of the
best fused column (136.6 / 262.0 / 430.9 µs). At P=8 the fused path still
trails the GEMM alone by 40-80 µs. The first wave of tiles needs half of the
transferred bytes, and they arrive at the link's rate. Closing that gap needs
the bands posted before the GEMM starts, from attention's epilogue.

The [`a2a_gemm` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/a2a_gemm/README.md), next to the code, has the full
design, every measurement table and the negative results.
