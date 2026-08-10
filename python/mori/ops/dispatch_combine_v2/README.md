# cco-LSA intranode MoE dispatch / combine (ops v2, FlyDSL)

Intranode (single-node, EP8) MoE dispatch + combine built on **mori-cco LSA**
(intra-node P2P over the flat symmetric VA) and **FlyDSL** device kernels. A
mori-parity reimplementation that swaps the mori-shmem provider for cco-LSA:
peer addresses are computed in-kernel via `cco.Window(arena).lsa_ptr(pe, off)`,
no host P2P tables. Reference = ROCm/FlyDSL PR #522
(`dispatch_combine_intranode_{kernel,op}.py`).

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

> **Test-only, not a mori API (yet).** These modules import each other by
> top-level name (`from intranode_kernels import ...`, `import flydsl_prims as P`)
> and have no `__init__.py`; they only resolve after the tests do
> `sys.path.insert(0, <this dir>)`. There is no `mori.ops.dispatch_combine_v2`
> package export, so `import mori.ops.dispatch_combine_v2.dispatch_combine_op`
> will fail. Use it via the test/bench harnesses below. Wiring it in as a real
> package (relative imports + `__init__.py` + `ops/__init__.py` export) is a
> follow-up.

## Layout

| file | role |
|---|---|
| `flydsl_prims.py` | device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | all FlyDSL intranode kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `dispatch_combine_op.py` | `SymmArena` + `EpDispatchCombineOp` / `EpDispatchCombineConfig` (+`.tuned()`) / `EpDispatchRoutingHandle` — mori-parity host op-layer (scales, scatter/quant combine, StdMoE, recv cap, LEC, reset, replay) |
| `tuning_configs.py` | per-(world,hidden,topk) block/warp lookup |

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay) |
| `bench_dispatch_combine.py` | eager + CUDA-graph perf bench + e2e correctness. Envs: `DTYPE=bf16\|f32\|fp8\|fp4`, `COMBINE=gather\|scatter`, `QUANT=none\|fp8_direct_cast\|fp8_blockwise`, `STDMOE=1`, `SCALE_DIM`, `SWEEP`, `DISP_BLOCK`/`COMB_BLOCK`, `WARP_NUM`/`COMB_WARP`, `MODE`, `TUNED`, `PREFETCH_ROUTE_PAYLOAD`, `DEFER_DEST_CTR_ATOMIC`, `ROTATE_DISPATCH_SLOT_ORDER`, `ROTATE_COMBINE_PEER_ORDER`, `ROUTING_PATTERN` |
| `run_bench.sh` | bench launcher (runs `bench_dispatch_combine.py` in the container) |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. Intranode only (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
torchrun --standalone --nproc_per_node=8 bench_dispatch_combine.py    # perf + e2e correctness
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DTYPE`, `COMBINE`, `QUANT`,
`DISP_BLOCK`/`COMB_BLOCK`, `WARP_NUM`/`COMB_WARP`, `MODE=eager|graph|both`, `TUNED`.
CCO's `cco_lsa_ptr` restores the global address-space provenance erased by the
scalar DSL ABI, so `{winBase, stride4G}` metadata loads use LLVM `addrspace(1)`.
Dispatch unconditionally reuses one peer base per work item. A 2026-08-10
gfx950 rerun retained this path after measuring 1.8%-6.1% lower dispatch latency
across BF16/FP8/FP4, top-k 4/8/9, gather/scatter, and scale forwarding at
128-2048 tokens; 8-token medians showed no repeatable regression. The
dispatch-stall experiments below are selected with `PREFETCH_ROUTE_PAYLOAD=1` and
`DEFER_DEST_CTR_ATOMIC=1`. Peer-order experiments use
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
