# cco-LSA intranode MoE dispatch / combine (ops v2)

Intranode MoE dispatch/combine built on MORI CCO LSA. The public entry point is
`EpDispatchCombineOp`, with two interchangeable backends:

- **flydsl** (default): gather/scatter combine, bf16/f32/fp8/fp4, quantization,
  StdMoE conversion, scale forwarding, replay, fused BF16-to-FP8 dispatch.
- **hip**: JIT v2 C++/HIP kernels. Gather combine only; dispatch transports
  bf16/f32/fp8/fp4 and supports asymmetric dispatch/combine dtypes.

Select a backend with `cfg.kernel_backend` or `MORI_V2_KERNEL_BACKEND`. Backend
modules are imported lazily, so the HIP path does not require FlyDSL.

## Layout

| file | role |
|---|---|
| `dispatch_combine_op.py` | backend-neutral config, selector, routing handle, shared dispatch/combine/reset/lifecycle |
| `flydsl_backend.py` | CCO-LSA arena layout and FlyDSL kernel adapters |
| `hip_backend.py` | JIT v2 HIP plan binding and capability validation |
| `intranode_kernels.py` | FlyDSL dispatch/combine and StdMoE kernel factories |
| `ep_plans.py` | EP-specific JIT plan wrappers |
| `symm_arena.py` | one symmetric CCO window divided into named regions |
| `tuning_configs.py` | FlyDSL geometry and size-gated cache policy |
| `hip_tuning_configs.py` | independent HIP geometry table |

## Supported behavior

- Dispatch deduplicates top-k routes targeting the same PE, allocates one remote
  receive slot, publishes reverse routing/weights/indices/scales, and copies the
  token through the flat LSA VA.
- Normal dispatch derives `total_recv` directly from `tok_off`; replay retains
  the per-source count handshake because it reuses an existing slot map.
- Gather combine reads expert output remotely and accumulates in FP32. Scatter
  combine writes results back to source ranks.
- `dispatch_fp8_blockwise=True` fuses BF16 quantization and FP8 transport inside
  the token-centric kernel, forwarding one FP32 scale per 128 elements.
- `expert_output_buffer()` lets expert GEMM write directly into the symmetric
  combine input, avoiding a staging copy.
- Geometry is always hybrid: use measured table entries when available and the
  analytical fallback for an unmeasured FlyDSL shape. HIP uses its own table.

## Shipped performance policy

The production policy is intentionally small:

1. `tok_off` is the normal-dispatch receive count path.
2. On FlyDSL versions that honor buffer cache modifiers, token stores bypass
   cache through 512 input tokens and route metadata through 256. Larger
   messages stay cached to preserve write combining.
3. Fused BF16-to-FP8 dispatch uses the token-centric implementation with cyclic
   peer phasing. Generic BF16 dispatch remains work-centric.
4. Expert output should be written through `expert_output_buffer()` when the
   caller controls the GEMM destination.

Measured on EP8 gfx950, hidden=7168, top-k=8:

- `tok_off` plus the size-gated cache policy improves dispatch by roughly
  9-12% at 128 tokens and 4-5% at 512 without a large-message regression.
- fused FP8 blockwise dispatch improves 35.7% at 512 and 44.4% at 2048 versus
  BF16 transport while eliminating a separate quantization kernel.
- direct expert output improves the full dispatch/expert/combine path by
  6.1-6.7%.
- analytical fallback stayed within about 1.3% of measured tables and improved
  an untuned hidden=4096/top-k=8/400-token combine by 36.1%.

FlyDSL 0.3 currently ignores the measured cache hints; its tuning table sets
both thresholds to zero while retaining the same correctness path.

## Removed experiments

The following compile-time experiments were removed after correctness and
interleaved performance testing:

- route payload prefetch and deferred destination-counter atomics: neutral or
  slower, including a large-shape regression;
- replay fast decode: about 2% only at the smallest replay case and neutral at
  larger sizes;
- generic BF16 token-centric auto-selection: small CUDA-graph wins but eager
  regressions and substantial kernel/config complexity; token-centric remains
  only for fused FP8;
- dispatch/combine peer rotations: neutral for random routing and useful only
  for specially synchronized hotspot patterns;
- token-granular CCO SDMA payload copy: about 14x slower than LSA at 512 BF16
  tokens. A future SDMA design should batch many peer tokens into contiguous
  transfers rather than issue one copy per token.

## Running tests and benchmarks

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v
torchrun --standalone --nproc_per_node=8 test_op.py
BACKENDS=flydsl,hip torchrun --standalone --nproc_per_node=8 bench_ep.py
```

`test_op.py` environment controls include `HIDDEN`, `TOPK`, `EPR`, `SWEEP`,
`DTYPE`, `COMBINE`, `QUANT`, `STDMOE`, `SCALE_DIM`, `TUNED`, `REPLAY`,
`RESET`, `MAXRECV`, `DISPATCH_FP8_BLOCKWISE`, `ZERO_COPY_EXPERT_OUTPUT`, and
`MORI_V2_KERNEL_BACKEND`.

`bench_ep.py` accepts `BACKENDS`, `MODES`, `SWEEP`, `ITERS`, `WARMUP`, `DISP`,
`COMBINE_IN`, `CHECK`, `HIDDEN`, `TOPK`, `EPR`, and geometry pins
`DBN`/`DWPB`/`CBN`/`CWPB`.

## Reference performance

EP8 gfx950, hidden=7168, top-k=8, BF16, eager op API, inplace combine,
30 warmups and 300 iterations:

| tokens/rank | dispatch | combine | pair |
|---:|---:|---:|---:|
| 128 | 73.2 us | 87.3 us | 160.5 us |
| 512 | 113.8 us | 144.0 us | 257.7 us |
| 2048 | 410.5 us | 521.4 us | 931.9 us |
| 8192 | 1605.0 us | 1560.7 us | 3165.7 us |

Every benchmark point is correctness-gated with an identity expert before
latency or bandwidth is reported.
