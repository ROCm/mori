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

The flydsl backend's dispatch leg has two payload transports, chosen with
`cfg.dispatch_transport`:

- **vector** (default): per-lane vec4 copies. Every dtype and feature above.
- **tdm**: the gfx1250 tensor data mover, a FlyDSL port of the C++
  `intranode_1250x.hpp` path. bf16/f32 payload only — no scales, no fp4, no
  replay — and it produces the same state as vector, so combine and the routing
  handle do not care which ran. It costs ~12us more fixed overhead and moves
  roughly 3x the bandwidth once the payload dominates: at hidden 7168 EP4,
  32768 tokens, 540 GB/s -> 1690 (3.1x); on perfectly balanced routing, 630 ->
  1640 (2.6x). Below about a thousand tokens both sit on the same cross-rank
  rendezvous floor; with the TDM-specific launch schedule (128x8 through 2048
  tokens, then 64x16/128x16) the remaining gap to HIP is ~10% there, so `auto`
  resolves to vector; `MORI_EP_DISPATCH_TDM=1` flips that default for A/B runs.

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
| `tdm_prims.py` | gfx1250 TDM descriptor construction: the GROUP0/GROUP1 bit layouts and the `tdm_load`/`tdm_store`/`tdm_wait` wrappers |
| `intranode_kernels_tdm.py` | `make_dispatch_tdm`: the TDM dispatch transport, plus the host-side sizing helpers the op layer needs (`tdm_stage_capacity`, `tdm_max_warps`, `tdm_lds_bytes`) |
| `tuning_configs.py` | **flydsl** kernel geometry: per-(world,hidden,topk) block/warp lookup |
| `hip_tuning_configs.py` | **hip** kernel geometry, separate table (never borrows flydsl's); same `lookup` contract. Independent dispatch/combine tables, keyed by device, shape, topk and (dispatch only) dtype; an unswept shape gets a single-shot default |

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay). `MORI_V2_KERNEL_BACKEND=hip` runs it against the HIP kernels |
| `test_ep_backend_parity.py` | runs both backends in one process on the same input and compares element for element |
| `test_jit_binding.py` | JIT plan binding: schemas, request/args round-trip, cache behaviour. No GPU peers needed |
| `test_graph_capture.py` | captures dispatch → identity expert → combine as one HIP graph and replays it |
| `test_asym_dtype.py` | asymmetric dtype legs (fp8/fp4 dispatch + bf16 combine) |
| `test_dispatch_tdm.py` | runs one routing through both dispatch transports, checks each against a CPU model and prints an A/B latency + bandwidth table. Envs: `HIDDEN`/`TOPK`/`EPR`/`SWEEP`, `BENCH=1`, `ITERS`/`WARMUP`, `BALANCED=1` (perfectly balanced routing), `DISPATCH_BW="1,8"` to pin the grid |
| `compile_dispatch_tdm.py` | compile-only check of `make_dispatch_tdm` across five geometries. No GPU, seconds not minutes — the first thing to run after touching the kernel |
| `bench_ep.py` | the perf bench, for every backend. Alternating dispatch/combine pairs, eager + CUDA graph, each point gated on an identity-expert check and non-zero exit on failure. Envs: `BACKENDS=flydsl,hip`, `MODES=eager,graph`, `SWEEP`, `ITERS`, `DISP=bf16\|fp8\|fp4`, `COMBINE_IN=inplace\|staged`, `CHECK=0`, `DBN`/`DWPB`/`CBN`/`CWPB` to pin geometry, `HIDDEN`/`TOPK`/`EPR` |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. Intranode only (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
BACKENDS=flydsl,hip torchrun --standalone --nproc_per_node=8 bench_ep.py   # perf, both backends

python compile_dispatch_tdm.py                                        # TDM kernel, compile only
HIDDEN=7168 EPR=32 SWEEP=2048,8192,32768 BENCH=1 \
  torchrun --standalone --nproc_per_node=4 test_dispatch_tdm.py       # TDM vs vector, correctness + A/B
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DISP`, `COMBINE`, `QUANT`,
`BACKENDS`, `MODES`, `ITERS`, `DBN`/`DWPB`/`CBN`/`CWPB`.

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
