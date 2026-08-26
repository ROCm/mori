# cco-LSA intranode MoE dispatch / combine (ops v2)

Intranode (single-node, EP8) MoE dispatch + combine built on **mori-cco LSA**
(intra-node P2P over the flat symmetric VA). One op class, `EpDispatchCombineOp`,
behind two interchangeable kernel backends:

- **flydsl** (default): FlyDSL device kernels, the full feature set (gather +
  scatter combine, fp8/fp4, quant, StdMoE, per-token scales, routing replay).
- **hip**: C++/HIP kernels JIT-compiled by the v2 JIT framework. Gather combine
  only, in bf16/fp32; the dispatch leg also carries fp8 and fp4 (transport only —
  it moves an already-quantized payload). A dedicated gfx125x TDM body is selected
  by arch. Works on a machine with no FlyDSL. See `docs/MORI_JIT_V2_DESIGN.md`.

Select with `cfg.kernel_backend` or `MORI_V2_KERNEL_BACKEND`. Peer addresses are
computed in-kernel over the flat LSA VA, no host P2P tables. Reference =
ROCm/FlyDSL PR #522 (`dispatch_combine_intranode_{kernel,op}.py`).

Supported token dtypes: **bf16**, **f32**, **fp8** (gather-only; OCP e4m3 on
gfx950, e4m3**fnuz** max 240 on gfx942) and **fp4** (e2m1, gather-only,
**gfx950-only** — the `cvt_scalef32_*_fp4` intrinsics don't exist on gfx942).
Combine: gather (UseP2PRead) **and** scatter (`_nop2p`); weighted combine
(`out_weights`); StdMoE (ConvertDispatchOutput / ConvertCombineInput, standalone
+ wired into the op); fp8 combine-wire **quant** (`fp8_direct_cast` **and**
`fp8_blockwise`, scatter-only — distinct from the plain fp8 token dtype, which
keeps a bf16 external payload); per-token scales forwarding;
`max_total_recv_tokens` cap; mori-parity host op-layer + per-device,
dtype-aware tuning table. Not done: `skip_stage1` (FlyDSL-only).

This is a real package: `from mori.ops.dispatch_combine_v2 import
EpDispatchCombineConfig, EpDispatchCombineOp`. Importing it pulls in **no** kernel
backend — the base and config live in `dispatch_combine_op.py`; each backend is imported
lazily, only when selected, so the package imports without FlyDSL installed.

## Layout

| file | role |
|---|---|
| `dispatch_combine_op.py` | backend-agnostic base + entry: `EpDispatchCombineConfig` (+`.tuned()`), `EpDispatchCombineOp` (backend selector + shared dispatch/combine/reset/lifecycle), `EpDispatchRoutingHandle`, `KernelSet` |
| `flydsl_backend.py` | **flydsl** backend subclass (`EpDispatchCombineOpFlyDSL`): arena layout + FlyDSL kernel binding for the full feature set |
| `hip_backend.py` | **hip** backend subclass (`EpDispatchCombineOpHip`): arena layout + C++/JIT plan binding, gather only; rejects unsupported configs at construction |
| `ep_plans.py` | EP-specific shim: loads `libmori_ops_v2.so` and exposes `EpDispatchPlan`/`EpCombinePlan`. The generic ctypes binding it calls lives in `mori.jit.v2.plan_api` (the plan_api C ABI), not here |
| `symm_arena.py` | `SymmArena`: one cco-LSA window carved into named regions |
| `flydsl_prims.py` | FlyDSL device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | FlyDSL kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `tuning_configs.py` | **flydsl** kernel geometry: per-(world,hidden,topk) block/warp lookup |
| `hip_tuning_configs.py` | **hip** kernel geometry, separate table (never borrows flydsl's); same `lookup` contract. Independent dispatch/combine tables, keyed by device, shape, topk and (dispatch only) dtype; an unswept shape gets a single-shot default |

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay). `MORI_V2_KERNEL_BACKEND=hip` runs it against the HIP kernels; FlyDSL experiment toggles cover SDMA, cache policy, token-centric dispatch, and peer-order variants |
| `test_ep_backend_parity.py` | runs both backends in one process on the same input and compares element for element |
| `test_jit_binding.py` | JIT plan binding: schemas, request/args round-trip, cache behaviour. No GPU peers needed |
| `test_graph_capture.py` | captures dispatch → identity expert → combine as one HIP graph and replays it |
| `test_asym_dtype.py` | asymmetric dtype legs (fp8/fp4 dispatch + bf16 combine) |
| `bench_ep.py` | the perf bench, for every backend. Alternating dispatch/combine pairs, eager + CUDA graph, each point gated on an identity-expert check and non-zero exit on failure. Envs: `BACKENDS=flydsl,hip`, `MODES=eager,graph`, `SWEEP`, `ITERS`, `DISP=bf16\|fp8\|fp4`, `COMBINE_IN=inplace\|staged`, `CHECK=0`, `SDMA_TOKEN_COPY`/`SDMA_QUEUES` (FlyDSL only), `DBN`/`DWPB`/`CBN`/`CWPB` to pin geometry, `HIDDEN`/`TOPK`/`EPR` |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. Intranode only (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
BACKENDS=flydsl,hip torchrun --standalone --nproc_per_node=8 bench_ep.py   # perf, both backends
```

Config via env (`test_op.py` / pytest): `HIDDEN`, `TOPK`, `EPR`, `SWEEP`,
`DTYPE`, `COMBINE`, `QUANT`, `STDMOE`, `SCALE_DIM`, `TUNED`, `REPLAY`,
`RESET`, `MAXRECV`, and `MORI_V2_KERNEL_BACKEND`.

Config via env (`bench_ep.py`): `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DISP`,
`COMBINE_IN`, `BACKENDS`, `MODES`, `ITERS`, `WARMUP`, `CHECK`, and
`SDMA_TOKEN_COPY`/`SDMA_QUEUES`, `DBN`/`DWPB`/`CBN`/`CWPB`.

CCO's `cco_lsa_ptr` restores the global address-space provenance erased by the
scalar DSL ABI, so `{winBase, stride4G}` metadata loads use LLVM `addrspace(1)`.
Dispatch unconditionally reuses one peer base per work item. A 2026-08-10
gfx950 rerun retained this path after measuring 1.8%-6.1% lower dispatch latency
across BF16/FP8/FP4, top-k 4/8/9, gather/scatter, and scale forwarding at
128-2048 tokens; 8-token medians showed no repeatable regression. The
dispatch-stall experiments below are selected with `PREFETCH_ROUTE_PAYLOAD=1` and
`DEFER_DEST_CTR_ATOMIC=1`. The 2026-08-11 atomic/cache experiments use
`USE_TOK_OFF_TOTAL_RECV`, `UNCACHED_TOKEN_STORE`,
`UNCACHED_METADATA_STORE`, `REPLAY_FAST_PATH`, and `REPLAY_BENCH`. Token-centric
experiments use `TOKEN_CENTRIC_DISPATCH` and `TOKEN_CENTRIC_ROTATE_PEERS`.
Peer-order experiments use
`ROTATE_DISPATCH_SLOT_ORDER=1`, `ROTATE_COMBINE_PEER_ORDER=1`, and
`ROUTING_PATTERN=random|aligned|round_robin|hotspot`.

## Design notes

- **dispatch**: per (token, k) dedup same-dest-PE via ballot; lane0 remote
  `atomic_add` allocates a recv slot; publish origin id + idx/wts + 16B dual-issue
  token copy to the peer; grid barrier; per-peer count signal; collect `total_recv`.
- **combine** (gather, = mori `UseP2PRead`): cross-device entry barrier, then each
  local token gathers its k expert outputs **remotely** from `peer.out_tok[dest_tok_id]`
  and reduces in f32. Register-light i32 reads (2 bf16 / `v2f32` accumulate) + 2-way
  unroll keep VGPRs low so 16 warps/block run at high occupancy to hide xGMI read
  latency; remote reads are latency-bound so combine needs ~128 blocks, while
  dispatch's posted writes saturate at ~64 blocks (half the CUs).
- Self-written volatile/atomic spin-waits (`flydsl_prims.spin_until_*`) — mori-shmem's
  `wait_until_*` assert on a cco-only stack. Counters self-reset in-kernel → CUDAGraph-safe.

## Dispatch atomic/cache A/B (MI355X / gfx950, 2026-08-11)

Six independent ideas were tested against the same BF16/top-k=8 baseline. The
fast gate used pinned geometry (128: 64x8, 512: 96x8, 2048: 160x8), random
routing, 20 warmups, 300 graph iterations, and three forward/reverse-interleaved
process runs. Rejected paths were removed after recording the result.

### Independent results

Latency is dispatch µs; percentages are relative to each row's baseline.

| experiment | 128 tokens | 512 tokens | 2048 tokens | decision |
|---|---:|---:|---:|---|
| baseline | 45.01 | 122.32 | 417.93 | control |
| `tok_off` as `total_recv` | 43.28 (-3.85%) | 120.88 (-1.18%) | 415.10 (-0.68%) | keep |
| reuse one top-k index load | 45.21 (-0.20%) | 122.10 (-0.07%) | 417.70 (+0.09%) | remove |
| uncached token stores | 43.07 (-4.98%) | 117.90 (-4.05%) | 428.81 (+2.67%) | token-gated |
| uncached token + metadata stores | 41.67 (-8.08%) | 118.24 (-3.77%) | 427.62 (+2.39%) | token-gated |
| replay fast decode | 43.69 (-2.01%) | 117.61 (+0.02%) | 408.81 (+0.35%) | replay-only |
| LDS token staging | 44.57 (-1.29%) | 120.78 (-1.31%) | 413.41 (-1.28%) | remove (<2% gate) |

The two-pass source-count/prefix allocator was also correct (including top-k=6
and capped receive buffers), but its extra route pass and cross-rank handshake
cost 148.91 vs 122.46 us at 512 tokens (+21.6%) and 528.55 vs 416.91 us at
2048 (+26.8%). It was removed.

`tok_off` already counts all destination slot allocations. After every source's
ready flag is observed, normal dispatch can set
`total_recv=min(tok_off,max_recv)` and remove both the per-publish
`dest_pe_ctr` atomic and the Phase-3 count reduction. Replay still uses its
cached layout. The replay fast path decodes PE/slot/sentinel directly from
`tok_map`, avoiding redundant route/dedup loads.

For LSA peer stores, FlyDSL cache modifier 3 lowers to `sc0 nt` on gfx950.
Bypassing cache helps latency at small/mid payloads but loses large-message
write combining. Boundary measurements with `tok_off` enabled were:

| tokens | cached | uncached token | uncached token+metadata |
|---:|---:|---:|---:|
| 256 | 69.80 | 66.69 (-4.46%) | 66.17 (-5.20%) |
| 1024 | 218.16 | 221.59 (+1.57%) | 220.82 (+1.22%) |

The tuned policy therefore uses uncached token stores through 512 tokens,
uncached metadata stores through 256, and cached stores above 512. These
variants compile lazily and are selected from the actual runtime token count.
FlyDSL >=0.3 currently ignores this cache hint, so it keeps the pre-cache-policy
geometry and disables the cache thresholds while retaining the `tok_off` fix.

### Winner interaction and re-tuning

| tokens | old baseline | selected combination | change |
|---:|---:|---:|---:|
| 128 | 45.35 | 39.86 (`tok_off` + all uncached) | -12.11% |
| 512 | 122.59 | 116.85 (`tok_off` + token uncached) | -4.69% |
| 2048 | 417.11 | 417.24 (`tok_off`, cached) | +0.03% |

After the dependency chain changed, the 109-128 BF16/top-k=8 bucket was
re-tuned. `96x4` averaged 39.56 us, versus 39.79 for `80x4`, 40.22 for the old
`64x8`, and 40.25 for `64x4`; the schedule now uses `96x4`. Top-k=6 and
untuned shapes keep their old schedule/options.

Correctness passed BF16 top-k=6/8, FP8, FP4, gather/scatter, hotspot routing,
scale forwarding, replay, local expert count, and capped receive tests.

### Final ATT

Source-mapped traces use the same 96x4 geometry:

- baseline: `profiles/ep-v2-att-final-baseline`
- optimized: `profiles/ep-v2-att-final-optimized`

The optimized ISA removes the non-returning `dest_pe_ctr`
`global_atomic_add`, and peer payload stores change from plain
`buffer_store_dwordx4` to `buffer_store_dwordx4 ... sc0 nt`. Normalized
weight/index publish wait fell from 2127 to 539 cycles/hit (-74.7%), while the
Phase-2 barrier fell from 3164 to 2531 (-20.0%). Token-copy and slot-allocation
waits moved upward, confirming that the gain comes from removing the second
atomic/queue dependency and changing store policy rather than making the
remaining remote fetch-add faster. Both kernels remain at or below `v17` and
emit no scratch loads/stores.

## Token-centric dispatch A/B (MI355X / gfx950, 2026-08-12)

The work-centric kernel assigns one wave to each `(token, top-k slot)`, so the
same local token row is loaded once per unique destination PE. The token-centric
variant assigns one workgroup to a token:

1. wave 0 loads the full top-k row once, deduplicates by PE, and allocates one
   destination slot per unique PE;
2. a small PE-indexed LDS table publishes `(valid, recv_slot)` to the workgroup;
3. every thread loads one 16-byte token chunk once and stores that value to all
   valid peers; waves phase-shift their peer order to avoid synchronized incast;
4. the existing grid/rank completion and `tok_off` total-count path is reused.

The kernel keeps normal/replay, sentinel/overflow, TIS, idx/wts, scales,
scatter, and StdMoE semantics. `TOKEN_CENTRIC_DISPATCH=1` forces it for A/B;
add `TOKEN_CENTRIC_ROTATE_PEERS=1` to reproduce the measured phase-shifted
variant. The tuned policy only enables both for measured BF16/top-k=8 buckets.

### Correctness

Correctness passed random/aligned/round-robin/hotspot routing at 8/128/512
tokens, BF16 top-k=6/8, FP8, FP4, gather/scatter, scales, replay, StdMoE,
local-expert-count, and capped receives. Capped runs intentionally print the
full-count LEC mismatch while their dedicated `OP-CAP` checks pass.

### Geometry and latency

Baseline and token-centric measurements use identical `tok_off` and cache
policies. Values are means of three interleaved 20-warmup/300-iteration graph
runs, with each kernel using its best measured geometry.

| routing / tokens | work-centric | token-centric | change | token geometry |
|---|---:|---:|---:|---:|
| random / 128 | 40.42 us | 42.85 us | +6.00% | 128x8 |
| random / 512 | 116.59 us | 113.58 us | -2.58% | 64x8 |
| random / 1024 | 218.18 us | 215.42 us | -1.26% | 96x8 |
| random / 2048 | 415.31 us | 409.59 us | -1.38% | 160x8 |
| hotspot / 512 | 148.15 us | 145.81 us | -1.58% | 64x8 |
| hotspot / 2048 | 558.97 us | 559.27 us | +0.05% | 160x8 |

Single expansion runs also showed aligned/512 -3.15%, BF16 top-k=6/512
-3.28%, and FP8/512 -0.99%. At 256 tokens the gain was only 0.81%; at 128
tokens insufficient active token workgroups make the new route barriers and
larger kernel more expensive than the saved loads. Phase-shifting peer order
across workgroup waves improved random/512 by another 0.59%.

The tuned MI355X BF16/top-k=8 policy therefore uses:

| token range | kernel | dispatch geometry |
|---:|---|---:|
| <=256 | work-centric | existing schedule |
| 257-511 | work-centric | existing schedule |
| 512 | token-centric | 64x8 |
| 513-1024 | token-centric | 96x8 |
| 1025-4096 | token-centric | 160x8 |
| >4096 | work-centric | existing schedule (unmeasured) |

FlyDSL >=0.3 currently ignores the cache hints used by this measured policy, so
both the cache thresholds and automatic token-centric schedule remain disabled
there; the explicit A/B flag remains available.

### ATT

Source-mapped 512-token traces use the same 64x8 geometry:

- baseline: `profiles/ep-v2-att-token-centric-baseline`
- variant: `profiles/ep-v2-att-token-centric-variant`

The sampled payload `buffer_load_dwordx4` hit count fell from 2051 to 784
(-61.8%). Baseline token-copy wait was about 2340 cycles/hit; the token-centric
store-many sequence moved corresponding waits to roughly 1200-1700 cycles/hit.
The work-centric final Phase-2 barrier was about 45118 cycles/hit; the
token-centric final barrier fell to 368, but it adds per-token route and LDS
reuse barriers at about 3106 and 8816 cycles/hit. Its remote slot atomic wait
rose from about 3517 to 4721 cycles/hit, and register use rose from `v17` to
`v55`, but neither kernel emitted scratch traffic. The wall-clock win therefore
comes from fewer local payload loads and better workgroup balance, partially
offset by route barriers, serial store-many waits, and higher VGPR pressure.

### Follow-up optimization A/B

Five follow-up directions were implemented as isolated compile-time variants,
tested, and then removed because none produced a stable wall-clock win. All
512-token values below use BF16/top-k=8, random routing, 64x8, the same
`tok_off`/cache policy, and interleaved graph runs.

| experiment | baseline | variant | result |
|---|---:|---:|---:|
| raw addrspace(1) global payload store | 114.89 us | 115.49 us | +0.52% |
| fixed peer order | 115.21 us | 115.39 us | +0.16% |
| even/odd forward-reverse peer order | 115.21 us | 115.67 us | +0.40% |
| fixed order + peer-base hoist | 115.21 us | 115.38 us | +0.15% |
| explicit wave-uniform peer scalarization | 115.99 us | 115.60 us | -0.33% |
| weight load after slot atomic | 115.81 us | 115.23 us | -0.50% |
| skip final LDS-reuse barrier | 115.81 us | 115.62 us | -0.16% |
| weight reorder + barrier skip | 115.81 us | 115.66 us | -0.13% |
| two-chunk software pipeline | 115.45 us | 115.44 us | -0.01% |
| four-chunk software pipeline | 115.45 us | 115.54 us | +0.08% |
| ping-pong route double buffer | 115.45 us | 115.48 us | +0.03% |

The raw-store path did generate `global_store_dwordx4` and removed the buffer
descriptor form, but LLVM did not preserve the requested gfx950 `sc0 nt` policy;
the lost cache policy offset the removed descriptor waterfall. It was neutral at
2048 (-0.14%) and regressed 1024 by 0.26%.

Moving weight behind the atomic showed only 0.31% at 1024 and was neutral at
2048. Chunk streams 1/2/4 were indistinguishable, including combinations with
the raw-store path. The route ping-pong implementation correctly overlapped
wave-0 routing of token N+1 with data-wave copy of token N and passed
replay/scales, but losing wave 0 from the copy team and its larger kernel
cancelled the removed barrier.

Conclusion: retain cyclic peer rotation and the original one-chunk,
buffer-store token-centric implementation. The remaining ATT stalls are not
independently removable at useful wall-clock scale on this workload.

## DeepEP-local optimization A/B (MI355X / gfx950, 2026-08-13)

Four DeepEP-inspired ideas that can be evaluated inside EPv2 were tested:
fused dispatch quantization, formal zero-copy expert output, CCO-SDMA payload
offload, and an analytical geometry fallback.

### Fused BF16 to FP8 blockwise dispatch

`DISPATCH_FP8_BLOCKWISE=1` takes BF16 input, computes one FP32 scale per 128
elements in the token-centric kernel, sends E4M3 payload plus scales, and leaves
combine/expert output in BF16. The scale convention matches EPv2 combine
blockwise quantization; dispatch wire bytes at hidden=7168 fall from 14336 to
7392 bytes/token.

Three-run graph means with tuned geometry:

| tokens | BF16 dispatch | fused FP8 blockwise | change | prequantized FP8 lower bound |
|---:|---:|---:|---:|---:|
| 512 | 113.67 us | 73.11 us | -35.7% | 71.75 us |
| 2048 | 409.62 us | 227.87 us | -44.4% | 217.60 us |

The fused path uses token-centric `128x16`, uncached token stores through 4096,
and passes normal/replay plus scale-active (`INSCALE=500`) correctness. The
prequantized arm excludes external quantization cost, so it is a lower bound;
fused dispatch is within about 2-5% while removing the separate quant kernel.

### Zero-copy expert output to combine

`expert_output_buffer()` formalizes the existing `out_tok` arena view. Expert
GEMM/synthetic expert writes directly into it and passes the same view to
`combine()`, avoiding the extra external-output to `out_tok` D2D copy.

| tokens | external output + staging copy | direct expert output | change |
|---:|---:|---:|---:|
| 512 | 274.62 us | 257.77 us | -6.1% |
| 2048 | 1092.98 us | 1019.95 us | -6.7% |

These are full graph `dispatch -> identity expert -> combine` times. The API is
gather-only; scatter retains its dedicated staging layout.

### Analytical geometry fallback

`GEOMETRY=analytical` scales MI355X block/warp bucket thresholds by expected
unique-peer bytes (`world_size`, top-k, hidden and dtype). `hybrid`—now the
default—uses measured table entries when present and analytical sizing only for
missing gfx950 shapes; other architectures retain their old fallback.

On the tuned hidden=7168 matrix analytical sizing stayed within about 1.3% of
the table. On the previously untuned hidden=4096, top-k=8, 400-token point it
selected combine `64x8` instead of `32x16`: combine latency fell from 97.18 to
62.06 us (-36.1%), while dispatch improved about 0.8%.

### CCO-SDMA payload experiment

The original EP8 failure was not a device-queue deadlock: `ccoDevCommCreate`
successfully opened every peer signal IPC mapping, but
`hipIpcOpenMemHandle(..., hipIpcMemLazyEnablePeerAccess)` left the benign
`hipErrorPeerAccessAlreadyEnabled` in HIP's sticky last-error slot. The next
Torch allocation reported that stale setup error. `ccoDevCommCreate` now
consumes the sticky error immediately after each IPC open; an EP8 regression
test creates an SDMA DevComm and then performs a Torch allocation.

The restored opt-in `SDMA_TOKEN_COPY=1` path also needs two data-plane rules:

- every lane system-releases its staged token before lane 0 rings the SDMA
  doorbell (`waitcnt` alone only retires VMEM and allowed the engine to copy
  stale zeros);
- self-routes use the normal CU copy because anvil queues represent GPU pairs;
  remote blocks are striped across `SDMA_QUEUES` by block id and drain only
  their `(peer, queue)` pairs.

EP8 correctness passes at 8, 128, 512, and 2048 tokens/rank, including
`MORI_DISABLE_P2P=1`; the 2048-token stress run completes without a hang.
The path remains off by default because this token-granular shape is a poor
SDMA workload: at 512 BF16 tokens it measured 1725.20 us versus 123.29 us for
the matched LSA token-centric path. Each roughly 14 KiB token/peer transfer pays
the copy-engine packet cost, plus a local staging copy and system fence. A useful
SDMA design would first pack many tokens per peer into large contiguous batches.

## Dispatch stall A/B (MI355X / gfx950, 2026-08-07)

This experiment tested two low-risk scheduling changes suggested by source-mapped
ATT. All variants used the retained peer-base reuse and CCO's addrspace(1) LSA
accessor, EP8, hidden=7168, 256 experts, dispatch=64 blocks × 16 warps, and graph replay.
The switches remain **off by default**.

| variant | `PREFETCH_ROUTE_PAYLOAD` | `DEFER_DEST_CTR_ATOMIC` | change |
|---|---:|---:|---|
| A | 0 | 0 | current order |
| B1 | 1 | 0 | prefetch non-duplicate weight/index before the slot-allocation atomic |
| B2 | 0 | 1 | move `dest_ctr` atomic after route metadata/scales, before token copy |
| B3 | 1 | 1 | B1 + B2 |

Correctness: A/B1/B2/B3 all passed BF16 gather at 8/128/512 tokens. B3 also
passed dispatch replay and `SCALE_DIM=32` forwarding.

### Latency

Graph dispatch latency in µs, shown as `mean [min,max] (change vs A)`. BF16
top-k=8 used three interleaved repetitions; top-k=4/9 used two. FP8 used three
(a third run was added after an outlier). Each run used 20 warmups + 300 timed
iterations. Negative change is an improvement.

| dtype / top-k | tokens | A | B1 prefetch | B2 defer | B3 combined |
|---|---:|---:|---:|---:|---:|
| BF16 / 8 | 128 | 47.40 [47.05,47.70] | 47.98 [47.69,48.40] (+1.22%) | 49.31 [48.79,49.79] (+4.04%) | 49.03 [48.59,49.33] (+3.44%) |
| BF16 / 8 | 512 | 124.53 [123.80,125.13] | 123.96 [123.43,124.76] (-0.45%) | 126.33 [125.54,127.48] (+1.45%) | 126.42 [126.16,126.81] (+1.52%) |
| BF16 / 8 | 2048 | 419.04 [418.29,419.52] | 418.91 [418.63,419.13] (-0.03%) | 420.89 [419.85,421.74] (+0.44%) | 420.22 [419.79,420.47] (+0.28%) |
| BF16 / 4 | 128 | 35.52 [35.38,35.65] | 35.71 [35.65,35.77] (+0.55%) | 36.23 [36.12,36.35] (+2.03%) | 36.23 [36.13,36.34] (+2.03%) |
| BF16 / 4 | 512 | 91.94 [91.74,92.13] | 92.28 [92.23,92.33] (+0.38%) | 94.65 [94.39,94.91] (+2.95%) | 94.41 [93.94,94.88] (+2.69%) |
| BF16 / 9 | 128 | 52.70 [52.28,53.12] | 52.00 [51.91,52.10] (-1.32%) | 53.80 [53.22,54.39] (+2.10%) | 53.58 [53.57,53.59] (+1.67%) |
| BF16 / 9 | 512 | 134.05 [133.67,134.43] | 134.03 [133.72,134.34] (-0.01%) | 135.05 [134.84,135.25] (+0.74%) | 136.36 [135.67,137.05] (+1.72%) |
| FP8 / 8 | 128 | 33.99 [33.39,35.17] | 34.31 [33.47,34.78] (+0.93%) | 38.06 [34.90,43.42] (+11.96%) | 36.35 [35.86,36.60] (+6.94%) |
| FP8 / 8 | 512 | 83.83 [83.70,83.95] | 83.98 [83.70,84.42] (+0.18%) | 85.24 [85.09,85.50] (+1.68%) | 85.19 [84.93,85.34] (+1.62%) |

### ATT stall movement

BF16/top-k=8/128-token ATT was captured once per variant with device DWARF.
Values below are accumulated stall cycles divided by that instruction's
hitcount; traces covered different wave counts, so only normalized trends are
used. `0` means the compiler emitted no distinct wait at that location.

| normalized stall / hit | A | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| slot-atomic result wait | 2961 | 2650 | 2853 | 2714 |
| route weight+index store waits | 6349 | 7097 | 2465 | 1813 |
| extra wait before token-loop entry | 0 | 4011 | 0 | 2264 |
| first vec4 token-store wait | 1369 | 1046 | 2252 | 1864 |
| Phase-2-entry `s_barrier` | 7741 | 9141 | 8198 | 9756 |

ATT output:

- A: `profiles/ep-v2-att-stall-a`
- B1: `profiles/ep-v2-att-stall-b1-prefetch`
- B2: `profiles/ep-v2-att-stall-b2-defer`
- B3: `profiles/ep-v2-att-stall-b3-combined`

All four kernels stayed at or below `v17` in the decoded ISA and emitted no
scratch loads/stores, so the regressions are not explained by a VGPR spill or an
obvious occupancy-class drop. ATT did not hit the block-0 cross-block/rank spin
path, so it does not measure that synchronization component.

Result: B1 reduced the token-copy wait but introduced a large wait before the
token loop and increased barrier imbalance; its small wins at BF16 top-k=8/512
(-0.45%) and top-k=9/128 (-1.32%) were not general. B2 successfully moved stall
out of route stores, but the pending atomic then delayed the token-copy VMEM
queue, producing consistent latency regressions. B3 inherited the same transfer
and a larger barrier stall. Keep both switches disabled; the next useful
direction is reducing/aggregating atomics rather than moving them within the same
VMEM dependency chain.

## Peer-order A/B (MI355X / gfx950, 2026-08-07)

OPUS fused-A2A experiments showed that phase-shifting each communication WG's
peer sequence avoids synchronized incast (5.6% at its default shape and 18.8%
at a larger shape), while source-rank rotation without striping was at most 1%.
EP dispatch has no explicit peer loop, so this experiment tests the cheaper
approximation before adding a compact per-peer queue:

- A: original dispatch and gather order.
- D: dispatch maps logical slot `j` to
  `(j + rank + src_token) % topk`; the mapping is a per-token bijection and
  `tok_map` remains indexed by the original slot.
- C: each gather warp rotates its K decoded peers by
  `(rank + global_warp_id) % topk`.
- DC: D + C.

All variants used the retained peer-base reuse and CCO's addrspace(1) LSA
accessor, disabled the prefetch/defer experiments above, and used EP8,
hidden=7168, 64 blocks × 16
warps, graph replay, 20 warmups, and 300 timed iterations. The route generators
are:

- `random`: independent random expert IDs (`seed=1234+rank`, the historical bench).
- `aligned`: every rank uses the same random route matrix.
- `round_robin`: `peer=(token+slot)%8`, balanced but slot-synchronous.
- `hotspot`: the first half of slots target peer 0; the rest rotate over peers 1-7.

Correctness passed a 15-case matrix covering A/D/C/DC, top-k 4/8/9,
8/128/512 tokens, all four route patterns, and dispatch rotation with scatter.
A separate combined smoke case also passed. Gather accumulation order changes,
so checks use the existing numeric tolerance rather than requiring bitwise
identity.

### Top-k=8 latency

Three forward/reverse-interleaved process runs per cell. Values are
`mean [min,max]` µs; percentages are relative to A. D columns report dispatch,
C columns report combine, and DC reports both latency changes.

| routing | tokens | A dispatch | D dispatch | A combine | C combine | DC change (disp/comb) |
|---|---:|---:|---:|---:|---:|---:|
| random | 128 | 47.56 [47.32,47.87] | 47.51 [46.96,47.85] (-0.11%) | 41.36 [41.17,41.49] | 41.37 [41.33,41.42] (+0.02%) | -0.34% / +0.00% |
| random | 512 | 124.52 [123.85,124.94] | 125.24 [125.00,125.67] (+0.58%) | 120.24 [119.93,120.62] | 119.86 [119.52,120.04] (-0.31%) | +0.88% / -0.29% |
| random | 2048 | 418.77 [418.47,418.99] | 421.06 [420.67,421.38] (+0.55%) | 415.77 [415.68,415.87] | 415.51 [415.25,415.78] (-0.06%) | +0.57% / -0.13% |
| aligned | 128 | 48.33 [47.59,48.88] | 46.89 [46.73,47.16] (-2.99%) | 41.62 [41.53,41.74] | 41.63 [41.48,41.91] (+0.03%) | -1.97% / -0.28% |
| aligned | 512 | 124.18 [123.84,124.46] | 124.66 [124.44,124.97] (+0.39%) | 120.02 [119.88,120.23] | 120.10 [119.76,120.49] (+0.07%) | +0.74% / +0.48% |
| aligned | 2048 | 423.08 [422.39,423.74] | 425.34 [424.48,426.69] (+0.53%) | 415.84 [415.30,416.21] | 415.56 [415.24,415.73] (-0.07%) | +0.70% / -0.05% |
| round-robin | 128 | 63.42 [62.47,64.76] | 62.31 [62.03,62.71] (-1.76%) | 47.50 [47.18,48.04] | 47.46 [47.20,47.86] (-0.08%) | -1.18% / -0.34% |
| round-robin | 512 | 170.48 [169.36,171.65] | 169.94 [169.70,170.21] (-0.32%) | 155.45 [155.39,155.57] | 155.64 [155.18,156.19] (+0.12%) | -0.26% / +0.33% |
| round-robin | 2048 | 597.99 [597.25,598.75] | 597.66 [597.12,598.00] (-0.05%) | 558.84 [558.68,559.05] | 559.62 [558.83,560.84] (+0.14%) | -0.05% / +0.17% |
| hotspot | 128 | 55.00 [54.47,55.92] | 53.75 [53.67,53.90] (-2.28%) | 48.86 [48.47,49.29] | 46.54 [46.50,46.57] (-4.75%) | -3.15% / -4.70% |
| hotspot | 512 | 159.55 [156.71,163.65] | 156.65 [156.27,157.40] (-1.82%) | 148.85 [148.61,149.04] | 146.71 [146.35,146.95] (-1.44%) | -1.71% / -1.31% |
| hotspot | 2048 | 553.39 [552.48,555.09] | 552.97 [552.73,553.16] (-0.08%) | 538.64 [537.96,539.54] | 530.82 [530.62,530.96] (-1.45%) | +0.16% / -1.36% |

### Top-k sensitivity

Two interleaved runs per cell. Values after A are latency changes versus A;
the isolated D dispatch and C combine columns are the relevant comparisons.

| top-k / routing | tokens | A disp/comb (µs) | D dispatch | C combine | DC disp/comb |
|---|---:|---:|---:|---:|---:|
| 4 / random | 128 | 35.53 / 30.55 | -1.65% | +0.88% | -1.51% / +1.15% |
| 4 / random | 512 | 91.76 / 83.50 | -0.19% | -0.80% | -0.60% / +0.52% |
| 4 / aligned | 128 | 36.01 / 31.58 | -3.35% | -1.96% | -3.17% / +0.08% |
| 4 / aligned | 512 | 89.98 / 79.95 | +0.17% | +0.13% | +0.37% / +0.48% |
| 9 / random | 128 | 52.59 / 42.81 | -1.31% | +0.85% | -0.52% / +0.98% |
| 9 / random | 512 | 134.15 / 121.44 | -1.10% | +1.05% | -1.15% / +1.14% |
| 9 / aligned | 128 | 50.24 / 42.53 | +0.89% | +0.25% | +0.74% / +0.24% |
| 9 / aligned | 512 | 134.19 / 123.12 | -0.15% | +0.57% | -0.01% / +0.94% |

### Source-mapped ATT

ATT values are stall cycles divided by instruction hitcount; traces covered
different target-CU wave counts, so they explain movement rather than wall-clock
speedup.

| pattern / normalized stall | A | rotated | change |
|---|---:|---:|---:|
| aligned dispatch slot-atomic wait | 2629 | 2577 | -2.0% |
| aligned dispatch route-store wait | 9029 | 7599 | -15.8% |
| aligned dispatch token-copy wait | 1413 | 1393 | -1.4% |
| aligned dispatch Phase-2 barrier | 10131 | 8327 | -17.8% |
| hotspot dispatch slot-atomic wait | 3190 | 2663 | -16.5% |
| hotspot dispatch route-store wait | 6483 | 7050 | +8.7% |
| hotspot dispatch token-copy wait | 1712 | 1328 | -22.4% |
| hotspot dispatch Phase-2 barrier | 10610 | 8428 | -20.6% |
| aligned combine dominant gather wait | 1306 | 1247 | -4.6% |
| hotspot combine dominant gather wait | 1531 | 1287 | -15.9% |

ATT projects are under `profiles/ep-v2-att-peer-{aligned,hotspot}-{a,dispatch-rot,combine-rot}`.
Dispatch stayed at or below `v17`; combine's highest observed VGPR changed from
`v78` to `v79`. No variant emitted scratch loads/stores, so there was no spill
or apparent occupancy-class regression.

Result: cheap rotation is distribution-sensitive, exactly as expected from the
OPUS record. Independent-random top-k=8 changed by at most 0.58% in the isolated
D/C paths. Dispatch rotation helps synchronized 128-token traffic
(aligned -2.99%, round-robin -1.76%, hotspot -2.28%) but fades or reverses at
larger aligned shapes. Combine rotation is useful specifically under persistent
hotspot traffic (-4.75%/-1.44%/-1.45% at 128/512/2048). Keep both switches
off by default and opt in only with matching router distributions. The hotspot
result passes the gate for a future per-peer compact queue + Latin-square WG
phase experiment; simple slot rotation alone does not reproduce OPUS's large
striped-A2A gains.

## Perf (EP8, hidden=7168, top-k=8, 256 experts; dispatch 64blk / combine 128blk × 16warp, CUDA-graph, bf16)

Per-rank bandwidth = `recv_tok * per_token_bytes / time` (the bench sizes the
payload per dtype, `hidden*2` for bf16). Indicative bf16 numbers on **MI308X
(gfx942)** xGMI:

| tok/rank | dispatch | combine |
|---:|---:|---:|
| 512  | 268 GB/s | 213 GB/s |
| 2048 | 306 GB/s | 294 GB/s |
| 8192 | 314 GB/s | 323 GB/s |

Cross-impl (v2 vs mori v1) latency tables for fp8/fp4 are in PR ROCm/mori#448.
