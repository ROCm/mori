# cco MoE dispatch / combine (ops v2)

MoE dispatch + combine built on **mori-cco**: intranode (single-node, EP8) over
the flat symmetric LSA VA, and internode (multi-node, e.g. 2×8) over CCO/GDA
RDMA. One op class, `EpDispatchCombineOp`, behind two interchangeable kernel
backends:

- **flydsl** (default): FlyDSL device kernels, the full feature set (gather +
  scatter combine, fp8/fp4, quant, StdMoE, per-token scales, routing replay).
  Intranode only.
- **hip**: C++/HIP kernels JIT-compiled by the v2 JIT framework. Gather combine
  only, in bf16/fp32; the dispatch leg also carries fp8 and fp4 (transport only —
  it moves an already-quantized payload). A dedicated gfx125x TDM body is selected
  by arch. **The only backend with an internode path** (bf16/f32/fp8, no fp4, no
  quant/StdMoE/scatter). Works on a machine with no FlyDSL. See
  `docs/MORI_JIT_V2_DESIGN.md`.

Select with `cfg.kernel_backend` or `MORI_V2_KERNEL_BACKEND`. Same-node peer
addresses are computed in-kernel over the flat LSA VA, no host P2P tables; a peer
on another node is reached with GDA RDMA. Reference =
ROCm/FlyDSL PR #522 (`dispatch_combine_intranode_{kernel,op}.py`).

Supported token dtypes: **bf16**, **f32**, **fp8** (gather-only; OCP e4m3 on
gfx950, e4m3**fnuz** max 240 on gfx942) and **fp4** (e2m1, gather-only,
**gfx950 and gfx1250** — the conversion intrinsics don't exist on gfx942. The
two differ: gfx950 uses the pair-at-a-time `cvt_scalef32_pk_*_fp4` family,
gfx1250 a single pack-of-8 per i32, so `_accum_funcs` branches on `WAVE`).
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
| `hip_backend.py` | **hip** backend subclass (`EpDispatchCombineOpHip`): arena layout + C++/JIT plan binding, gather only; rejects unsupported configs at construction. Holds the internode path too — device communicator, per-geometry plan sets for the `copystaging`/`dispatch{,_ll}` and `combinesync`/`combinesyncbarrier`/`combine{,_ll}`/`combineall` pass sequences, and its own unsupported-config gate |
| `ep_plans.py` | EP-specific shim: loads `libmori_ops_v2.so` and exposes `EpDispatchPlan`/`EpCombinePlan`. The generic ctypes binding it calls lives in `mori.jit.v2.plan_api` (the plan_api C ABI), not here. A gfx125x `EpDispatchPlan` whose caller binds no `staging_base` gets one at its first launch, shared by the unbound plans of its layout, with a `FutureWarning`; instead, allocate `info["stagingBytes"]` bytes once, keep them alive, and `bind(staging_base=buf)` on every dispatch plan (a tensor is checked against that size and held; an int address is taken on trust). Sizes come from the plans: `EpDispatchPlan.describe(**request)` / `EpCombinePlan.describe(...)` return the info a plan would report without compiling it, and `arena_regions(*infos, region_names=...)` turns their `region.<stem>` keys into a `SymmArena` region list; binding checks an arena that has `size()` |
| `symm_arena.py` | `SymmArena`: one cco-LSA window carved into named regions |
| `internode_regions.py` | `internode_regions(cfg)`: the internode op's `SymmArena` region list — 17 named `(name, nbytes)` regions (16 plus `out_scales` when scales are on), in v1's allocation order and transcribed from what `EpDispatchCombineHandle` allocated |
| `flydsl_prims.py` | FlyDSL device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | FlyDSL kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `tuning_configs.py` | **flydsl** kernel geometry: per-(world,hidden,topk) block/warp lookup |
| `hip_tuning_configs.py` | **hip** kernel geometry, separate table (never borrows flydsl's); same `lookup` contract. Independent dispatch/combine tables, keyed by device, shape and topk, plus the dispatch dtype (`dispatch_dtype_str`: bf16/fp8/fp4) for dispatch and `quant_type` for combine; an unswept shape gets a single-shot default |
| `internode_tuning_configs.py` | **hip** internode kernel geometry, a third table: token-count buckets keyed by device, shape, topk and dispatch dtype, carrying `(block_num, rdma_block_num, warp_num)` **per phase** — dispatch and combine are tuned to different values over one shared arena. The internode plans are compiled per geometry, so the backend walks the whole table at build time and `lookup` only picks a prebuilt bucket; `block_num` is clamped to the CU count |

## gfx125x dispatch: slot reservation

The slot allocator word lives in the cco window, and the blocks of a rank
reserve their slots together: each block adds its token count for
every destination, this rank included, into the rank's total for it, and the last
block to arrive reserves the total with one RMW on that destination's word and
publishes the first slot to the others. In the window the RMWs on one word
serialize, so a rank makes one per destination per call instead of one per block.
The blocks meet in a few hundred bytes of a `__device__` array in the dispatch
kernel's own module, so neither the arena nor the caller allocates anything for it.

At 8 ranks or fewer there is no RMW. `tok_off` also holds, from byte 256, a count
table of a word per source and destination: the last block of a rank stores its
total for p into p's word of its own row in the table of every higher-numbered
rank, and each block takes as its first slot on p the sum of the totals the
lower-numbered ranks stored into its own table. p's rows then run in source rank
order, the same on every call, and no block waits on a fabric round trip. A rank
clears its table at the end of the call; a lower rank stores the next call's row
only after the combine between the two calls.

A call small enough to give every block at most 8 warp work units (512 tokens on
the default 64 × 16 geometry at top-k 6) is spread over all the blocks instead of
filling the low half: block b takes units [8b, 8b + 8) on its first 8 warps. Where
the payload pass walks the token map, the warps left without a token then send all
of the metadata, so the token warps go straight to the payload. Geometries with at
most 8 warps per block, such as EP4's tuned 64 × 8, keep the plain map.

## Internode config

There is no kernel-type enum: the internode path is selected by
`gpu_per_node < world_size` (`cfg.is_internode`, i.e. `cfg.nodes > 1`), and only
the **hip** backend implements it.

| field | default | role |
|---|---|---|
| `gpu_per_node` | `None` → `world_size`, i.e. one node | GPUs per physical node; `world_size` must be a positive multiple of it. Setting it smaller is what selects the internode path. It is EP's own idea of a node and is checked against the communicator's LSA team (`lsa_size`, `lsa_rank`) at construction |
| `internode_kernel` | `"auto"` | Which internode kernel family runs. `"v2"` = the general path (chunked, deduplicating, sized for wide tokens); `"v2_ll"` = low latency (no dedup across expert slots, one entry per node per token). They are separate JIT modules, so naming one compiles only that one and it cannot fall back; `"auto"` compiles both and chooses per launch |
| `internode_auto_ll_max_tokens` | `512` | The `"auto"` crossover, compared against **this call's** token count (`input.shape[0]` for dispatch, `routing.cur_rank_num_token` for combine): `<=` runs `v2_ll`, `>` runs `v2`. One op therefore alternates as the batch changes. Unrelated to `max_num_inp_token_per_rank`, which is the capacity. Read only when `internode_kernel == "auto"`; must be >= 0 |
| `num_qp_per_pe` | `2` | QPs per peer on the RDMA leg. Only the internode path reads it, and 1 starves it (~1.5x), so the default is what that path wants; the intranode path ignores it. Must be >= 1 |

Two further internode-only rules `__post_init__` applies: `quant_type` must be
`"none"` (the internode combine's fp8 staging path is incomplete and returns
wrong tokens), and `max_num_inp_token_per_rank` is rounded up to a multiple of
the wavefront width, because send slots are handed out in whole wavefronts.

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_dispatch_combine_v2_internode.py` | the internode entry: a torchrun script (not a pytest wrapper) for correctness, bench and tuning over CCO/GDA. `--cmd test\|bench\|tuning\|stress`, `--max-tokens`, `--hidden-dim`, `--topk`, `--dtype`/`--combine-dtype`, `--num-qp` (default 1), `--kernel-type auto\|v2\|v2_ll`, `--auto-ll-max-tokens`, `--rounds`, `--spawn`. **Needs two nodes**: the op refuses a config whose node grouping disagrees with the communicator's LSA team, so one host cannot emulate it |
| `test_internode_regions.py` | pure-Python invariants of `internode_regions()`: the name contract with the backend and the capacity bounds the kernel's indexing implies. No GPU, no process group |
| `test_op_lifecycle.py` | arena-leak regression: rebuilding the op on one long-lived `Communicator`; `close()` must free and untrack the window. `torchrun --standalone --nproc_per_node=2` |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay). `MORI_V2_KERNEL_BACKEND=hip` runs it against the HIP kernels. `RAGGED=1` gives the ranks different token counts, two of them none; `NOSELF=1` routes no token to its own rank |
| `test_ep_backend_parity.py` | runs both backends in one process on the same input and compares element for element |
| `test_routing_snapshot.py` | `dispatch(..., snapshot=True)` handles outlive later dispatches: combine (twice) and replay with an older handle, both backends. `torchrun --standalone --nproc_per_node=2` |
| `test_ll128_split.py` | an op past the LL128 table's `max_tok` runs small calls on its LL128 plans and the rest on the schedule's: one call sequence crossing `max_tok` both ways, synced (delivery, reverse map, bf16 sum), back to back eager and graph-replayed, and as snapshot handles, each combine equal to the synced one bit for bit. hip, gfx1250. `torchrun --standalone --nproc_per_node=4` |
| `test_jit_binding.py` | JIT plan binding: schemas, request/args round-trip, cache behaviour. No GPU peers needed |
| `test_graph_capture.py` | captures dispatch → identity expert → combine as one HIP graph and replays it |
| `test_asym_dtype.py` | asymmetric dtype legs (fp8/fp4 dispatch + bf16 combine) |
| `bench_ep.py` | the perf bench, for every backend. Alternating dispatch/combine pairs, eager + CUDA graph, each point gated on an identity-expert check and non-zero exit on failure. Envs: `BACKENDS=flydsl,hip`, `MODES=eager,graph`, `SWEEP`, `ITERS`, `DISP=bf16\|fp8\|fp4`, `COMB_MODE=push\|pull` (default `push`, the fp4 combine: hip backend on gfx125x only), `COMBINE_IN=inplace\|staged`, `CHECK=0`, `DBN`/`DWPB`/`CBN`/`CWPB` to pin geometry, `HIDDEN`/`TOPK`/`EPR`, `ROUTE=rand\|ring\|noself`, `LATE_US` (rank 0 reaches every timed dispatch that many µs after the others) |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. These are the intranode paths (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
pytest test_internode_regions.py -v                                   # arena layout, no GPU
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
BACKENDS=flydsl,hip COMB_MODE=pull torchrun --standalone --nproc_per_node=8 bench_ep.py   # perf, both backends
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DISP`, `COMBINE`, `QUANT`,
`BACKENDS`, `MODES`, `ITERS`, `DBN`/`DWPB`/`CBN`/`CWPB`.

## Run internode (two nodes, 8 GPUs each)

The internode entry is a plain torchrun script, one launch per node; `--cmd`
selects correctness, bench or the geometry sweep.

```bash
# rank 0 of 2, from the repo root. --spawn defaults to 8, so torchrun runs ONE
# process per node and the script builds the 8 ranks; pass --spawn 0 to use
# --nproc_per_node=8 instead (the two topologies together are refused).
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=<ip> --master_port=<port> \
    tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
    --cmd test --max-tokens 128 --num-qp 2

# or drive one rank through the shared runner (--entry defaults to the v1 harness)
tools/run_internode_test.sh --rank 0 --master-addr <ip> --ifname <nic> \
    --cmd test --max-tokens 128 \
    --entry tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py
```

Set `MORI_RDMA_TC` / `MORI_RDMA_SL` to match the fabric's RoCE class; unset, every
transfer goes out on SL 1 / DSCP 0. `tools/env_setup.sh` derives both from
`ROCE_DSCP`/`ROCE_PRIO`.

## Design notes (intranode kernels)

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

## Perf (AsyncLL)

AsyncLL (`async_ll=True`) splits each phase into a send half and a recv half, so
a caller can put compute between them. The nine kernels and the transports are
described in `src/ops/dispatch_combine_v2/ep_asyncll_kernel.hpp`.

**Read the numbers below with one caveat.** The benchmark calls `send` and then
`recv` with nothing in between, which is the one thing AsyncLL exists to fill.
It therefore measures the cost of the protocol -- a payload staged three times
(input -> staging, staging -> peer, peer -> output), because that is the shape an
RDMA leg needs -- and none of the benefit. These tables say the port is correct
and what it costs; whether AsyncLL pays for itself needs an end-to-end MoE run
with real expert compute in the seam, which is not measured here.

### Cross-node, EP16 (2 x MI355X gfx950, 8 GPUs each)

hidden 7168, top-k 8, bf16, `--cmd bench`. Totals in us (dispatch + combine);
each token count's three arms were run interleaved in one window, so a node
stolen mid-batch biases all three equally.

| tok/rank | v2 asyncll | v2_ll | v1 async_ll | vs v1 |
|---:|---:|---:|---:|---:|
| 64   |  320.1 |  252.9 |  561.7 | -43.0% |
| 128  |  419.9 |  212.6 |  430.6 |  -2.4% |
| 256  |  684.0 |  322.5 |  694.0 |  -1.4% |
| 512  | 1243.9 |  521.0 | 1232.1 |  +0.9% |
| 1024 | 2618.8 | 1585.9 | 2441.2 |  +7.3% |
| 2048 | 4559.6 | 1755.7 | 4545.2 |  +0.3% |

**Against v1: parity.** Within ±7% from 128 tokens up, +0.3% at 2048. The two are
the same protocol over different transports, so tracking this closely is the
evidence that the cco port preserves it. RDMA bandwidth agrees digit for digit
at 2048 -- 25.2/26.2 GB/s for v2 against 25.7/25.8 for v1.

The -43% at 64 tokens is **not** a win. v1 launches with `block=auto warp=auto
rdma=auto` and its autotune picks badly in that band: v1 is *slower* at 64
tokens (561.7 us) than at 128 (430.6 us). Our geometry is fixed, so we do not
share the dip.

**Against v2_ll: 1.3x to 2.6x slower, and that is topological, not a tuning
gap.** v2_ll's RDMA bandwidth at 2048 is 64/71 GB/s against our 25/26. Its proxy
model has each rank talk only to its own-numbered rank on the far node, so the
traffic rides one fat same-rail link; AsyncLL addresses peers individually
(`SendBufSlotOffset` is per-PE), which spreads the same bytes over eight thinner
cross-rail QPs that never fill. That is the cost of the per-PE addressing, not
something a knob reaches. See `_make_dev_comm` in `hip_backend.py` for why
AsyncLL has to ask for `GDA_CONNECTION_CROSSNODE` where the others take RAIL.

### Single-node, EP4 (gfx1250, wave32)

hidden 7168, top-k 8, bf16, CUDA graph, 50 iters. `pair_us` = dispatch +
combine, median of three interleaved rounds, no excluded samples.

| tok/rank | v2 native | asyncll sdma | asyncll lsa (fused) | asyncll lsa (unfused) |
|---:|---:|---:|---:|---:|
| 64   |  65.7 |  90.9 |  676.4 | 1071.3 |
| 128  |  70.3 | 100.6 |  646.2 | 1053.8 |
| 256  |  74.3 | 122.0 |  677.7 | 1108.8 |
| 512  |  95.2 | 155.8 |  717.6 | 1207.2 |
| 1024 | 149.1 | 222.1 |  789.2 | 1329.7 |
| 2048 | 246.1 | 355.4 |  972.9 | 1706.9 |

The fused LSA column was re-measured after the `Unroll=4` change to the
peer-facing copies (see the LSA section of `ep_asyncll_kernel.hpp`), three
interleaved rounds against an unmodified tree on the same node and harness. The
unmodified arm reproduced the column this replaces to within ~1% at five of the
six points -- 676.4/655.5/717.2/821.3/1357.2 against 676.5/659.3/714.5/826.5/1370.2
-- so the two are comparable; 1024 is the exception (991.1 against 1130.1) and
the 1130.1 reading is the suspect one, being non-monotonic with its neighbours.
The unfused column predates the change and is left as measured; it would move
too, since the same three sites serve both.

Losing to the native intranode path is expected and is the same shape MI355X
shows at EP8: native writes straight into the peer, AsyncLL stages. On a single
node the staging buys nothing, because there is no RDMA leg for it to feed.

**LSA is usable on this arch but not useful.** It runs 2.7-7.4x slower than
sdma and 4.0-10.3x slower than native, which is nothing like MI355X, where the
two transports trade places (lsa 4% ahead at 64 tokens, 11% behind at 2048).
Whatever makes a peer store competitive on wave64 does not hold here. sdma
stays the default and the gap is wide enough that it is not close.

Note which end `Unroll=4` moved: -28.3% at 2048 and nothing at 64. The small
end of this transport is not store-bound, so the remaining 7.4x there is a
different problem from the 2.7x at the top.

The 512 row is from a separate six-repeat run. The three-round sweep put it at
331 us, non-monotonic between 256 and 1024; six repeats give 155.8 us with 1.2
us of spread, and the neighbouring 384 showed an 810 us spread over the same
six. This box throws an occasional large outlier that a median of three does
not absorb -- 256 and 1024 reproduced to within 0.5% across both runs, so the
rest of the column stands.

The LSA rows above could not be measured at all until recently: the transport
hung intermittently on gfx125x **under CUDA graph capture**, which is what this
benchmark uses.

The cause was a count-reuse race. LSA announced the combine epoch and only then
cleared the dispatch receive counts, so a peer that passed the epoch handshake
could publish its next dispatch count before the old round's clear erased it,
leaving dispatch waiting on a count that would never come again. SDMA clears in
`CombineSendTransfer`, before its epoch announcement, which is why it was never
affected. LSA now clears each peer's count before publishing its epoch, with
system release/acquire ordering.

Eight interleaved pairs at EP4, hidden 7168, 64 tokens, on an idle gfx1250, no
contaminated samples:

| | pass / hang |
|---|---|
| baseline | 1 / 7 |
| fixed | 8 / 0 |

Two things about that measurement are worth copying rather than the number
itself. The arms shared one host library and differed only in
`MORI_SOURCE_ROOT`, and that this actually selected different kernels was
*checked* -- an `#error` planted in the baseline tree fired when the root
pointed there and not when it pointed at the fixed tree. And the node was idle:
an earlier 55-run matrix on a busy box recorded 0 hangs for the same baseline,
because an interfering job widens the window the race needs.

Eager never hit this even though the race does not need graphs: the window is
the gap between a peer clearing the handshake and our clear landing, and eager's
host round trip between kernels is wide enough that ranks rarely sit in it.
"Eager passes" was therefore never evidence against the race.

Measure the rate before trusting any single run here. Several conclusions about
this bug were each drawn from one run and were wrong -- that the kernels were
broken on wave32, that the LSA fusion caused it, that the iteration count was a
threshold, that a lane guard fixed it, and that the race could not be the cause
because eager passed.

`v1_asyncll` has no row either -- v1 is the shmem path and does not run on this
arch.

### Tried and rejected

The first two follow from the same picture -- AsyncLL stages the payload three
times and runs nine kernels, so the obvious targets are the extra passes and the
extra launches -- and both were measured on an MI355X at EP8, six interleaved
pairs, every pair agreeing in sign. All of them are recorded because the
reasoning behind them is sound enough to be had again.

**Pipelining the send half over PE groups.** `SendTransfer` only posts SDMA
descriptors; the engine's copy is not on the stream, so splitting the send into
PE groups and alternating copy/post launches should let the engine move group 0
while the grid stages group 1. Measured +20% (2048 tokens) to +61% (64), worse
everywhere. Each `SendCopy` launch still walks every entry and copies only its
share, so half the grid idles through the scan and the fixed costs double, while
the overlap it was supposed to buy never showed up.

**Fusing SlotAssign and SendCopy behind a grid rendezvous.** One launch instead
of two, keeping both decompositions (SlotAssign is top-k parallel, SendCopy is
hidden-dim parallel) rather than interleaving per warp. Measured +12% (2048) to
+104% (64). A spin barrier waits for the slowest block, and SlotAssign's work is
uneven across blocks by nature; the kernel boundary it replaced was synchronised
by the hardware scheduler for less. Launch overhead is not the bottleneck here,
which is also why the first experiment failed -- its small-batch penalty was
load imbalance, not the extra launches.

**Fusing the cross-node count onto the payload's WQE.** `EpInterNodePutSignal`
instead of `EpInterNodePut`, so the stripe's token count rides its own QP and
`RecvTransfer` can drop both the `EpInterNodeQuiet` drain and the separate count
send. Needs per-(srcPe, qpId) count slots, because a fused flag is ordered
behind the data only on its own QP and the payload is striped across
`num_qp_per_pe` of them. Never got correct: 16 of 16 ranks disagreed at EP16,
`max|diff|` around 24 with `got_nonzero` at or near full, i.e. every output
position written but some holding the wrong value -- the signature of a wrong
offset rather than of data that has not arrived. The suspect is the count
arithmetic the new layout needs (`sum(row) - num_qp_per_pe`), which feeds the
per-PE output prefix sum. Abandoned rather than chased: even correct, the
cross-node leg is bandwidth-bound on per-PE cross-rail QPs at ~25 GB/s against
v2_ll's 64-71, so a CQE drain is not what is costing it.

**Moving the LSA transfer onto the gfx1250 TDM engine.** `benchmark/cco/ualoe_metric`
establishes that TDM reaches the xGMI ceiling (~1640 GB/s one-way) on 512 blocks
where a CU vector copy needs 8192, so a transport whose whole cost is CU-issued
peer stores looks like the case for it. Written as a global -> LDS -> peer
double-buffered copy, following `src/ops/dispatch_combine/intranode_1250x.hpp`,
which is where the non-obvious part comes from: a TDM store can target a PEER's
flat address, not only local memory. Measured on an idle gfx1250 node at EP4,
five interleaved rounds, and it is a wash or slightly negative -- combine +1.6%
at 2048, dispatch flat, and still a wash after `Unroll=4` removed the real
bottleneck. Two reasons, and the second is the general one:

* Dispatch never reached the TDM path at all. `XferBytesPerToken` is
  7168*2 + 32 + 32 + 4 = 14404, so the slot stride is 4-byte aligned and source
  and destination never share a 128 B phase.
* The same ualoe data says TDM's advantage is a *grid-width* one, and below
  about 1 MB neither transport is grid-limited -- there the CU path is often the
  faster of the two. AsyncLL's transfer hands each warp tens of KB, so it sits in
  the latency regime, not the bandwidth regime, and the extra trip through LDS is
  pure cost. TDM pays in a kernel that has other work for the freed VGPRs; a
  transfer pass has none.

**Padding the slot stride to a 128 B row.** The alignment above is real, and
fixing it (`XferBytesPerToken` rounded up to 128, with `internode_regions.py`
following) is worth ~3.4% on the dispatch half and ~1% on the pair, measured in
the same matrix. Not taken here because it changes the arena layout shared with
the v2 / v2_ll families for a gain that `Unroll=4` dwarfs; it is a reasonable
separate change.

### Reproducing

```bash
# four-way single-node comparison, writes JSON + a table under /tmp/mori_bench
MORI_ENABLE_SDMA=1 bash tools/bench_ep_asyncll_compare.sh --world-size 8

# cross-node, one arm at a time (the v2 and v1 entries take different flags)
tools/run_internode_test.sh --rank {0,1} --master-addr <ip> --ifname <nic> \
  --entry tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
  --cmd bench --kernel-type asyncll --dtype bf16 --hidden-dim 7168 --topk 8 \
  --max-tokens 128
```

Both need `MORI_ENABLE_SDMA=1` for the sdma transport; without it cco creates no
queues and every transfer is a silent no-op (the host gate catches this).
