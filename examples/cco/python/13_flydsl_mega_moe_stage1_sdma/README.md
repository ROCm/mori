# Example 13 -- MegaMoE stage1 dispatch on cco SDMA (gfx950, prototype)

aiter's `MegaMoEV2` stage1 copies every routed row to its destination rank with P2P stores from
producer CUs. Here the sender quantises the rows into a local send buffer `S` grouped by
destination rank, the copy engines ship each run with SDMA `put`, and GEMM1 reads its A rows
(and their e8m0 scales) through an index table `rowmap`. It is the stage1 counterpart of
example 11 (stage2 + combine). The stage1 outputs are bit for bit aiter's.

| file | what |
|---|---|
| `stage1_sdma.py` | `Arena` (window layout), `build_prepare` (one resident kernel: plan, route push, quantise into `S`, put the first chunks), `post_rest_chunks`, `build_post_rest`, `build_wait`, host `Stage1SdmaPrepare` |
| `gemm1_sdma.py` | GEMM1 on aiter's `do_tile`, A and scales from `[S \| R]` through `rowmap`; with `prep` its workgroup 0 puts the remaining chunks and the others wait for the chunks they need |
| `test_prepare.py` | the rows each rank receives equal aiter's quantisation, byte for byte (no aiter dispatch needed) |
| `test_dispatch.py` | `R[rowmap[r]]`, weights and srcmap equal aiter's `rx_em`, `wts_em`, `srcmap_em` row by row |
| `test_gemm1.py` | stage1 output and the final layer output equal aiter's; `--stress N` repeats a poisoned call; `--time-iters` times it; `--trace N` prints the clock stamps below; `--stall-test` runs GEMM1 twice after one prepare |
| `baseline.py` | where aiter's own prepare / stage1 / stage2 time goes |
| `test_recv_layout.py` | `recv_layout`: no plan of aiter's at all; the final layer output equals aiter's bit for bit (`--stress N` poisoned gated calls), and the whole chain is timed against aiter's prepare + stage1 |

Design in short:

* a token is sent to a rank once, however many experts it has there; its row sits at the
  position of its FIRST expert on that rank, so a landed prefix of a run is a prefix of experts;
* a token's bf16 row is read and quantised ONCE and stored to its row in every run it is in
  (a token goes to ~4.6 ranks, so the input is read once instead of ~4.6 times);
* `prepare` puts the first `--early-chunks` chunks of each run, the GEMM1 comm workgroup the
  rest (posting every chunk in `prepare` cost ~50 us of its critical path);
* a grid rendezvous separates the quant pass that fills the early chunks from the rest, so the
  first puts are not held back by the second pass racing them for bandwidth (it moved the first
  put from 177 to 106 us and the first tile's wait for data from 53 to 7 us at bs 8192);
* a GEMM1 tile waits only for the chunks its expert needs (`need`, SDMA signal counters);
* no cache invalidation per tile: a chunk starts on a 128 B line (`chunk_rows % 8 == 0`).

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/test_gemm1.py --network v4_pro --bs 4096 \
  --chunk-rows 256 --early-chunks 2 --stress 50 --time-iters 30
```

## `recv_layout`: no aiter plan at all

The route push above writes each route to its final expert-major row on the destination, and
the row numbers come from aiter's compact plan (`pair_order`, `pair_base`, `local_hist`,
`my_base`): that plan has to run every call, ~0.10-0.15 ms whatever the batch, and
`test_gemm1.py` does not time it (it reuses the tables of its reference call). With
`Stage1SdmaPrepare.run(..., recv_layout=True)` nothing of aiter's plan is needed:

* the sender puts every route of its, in no order, into a staging area of the destination
  partitioned by source (`Arena.off_stage`, 16 B each: local expert, rowmap value, weight,
  srcmap), then the counts (`off_scount`) and its flag, as before;
* `Stage1SdmaPrepare.layout()` (`build_layout`, 64 blocks, between the prepare and GEMM1) waits
  for every source, counts the staged routes per expert, pads each expert to the sort block and
  writes `rowmap`, aiter's `srcmap_em` / `wts_em` / `sorted_expert_ids` / `tile_row_base` /
  `tile_input_base` / `num_valid` / `expert_tile_end` / `max_expert_tiles`, so aiter's stage2
  runs on them unchanged. The order of the rows inside an expert is the atomics', as in
  aiter's own plan; stage2 scatters each row to its (token, k) slot, so the output does not
  depend on it;
* no runtime fanout pairs are built: `layout()` zeroes `fanout_pair_config`. aiter's own output
  is the same with and without them (`test_recv_layout.py` checks it on every run).

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/test_recv_layout.py --network flash --bs 512
```

New prepare + layout + gated GEMM1 against aiter's compact prepare + fused stage1, eager, 8x
MI355X, aiter 46cdc4c52, `--chunk-rows 256 --early-chunks 2`; every row bit-exact over 20
poisoned calls. The layout kernel takes 0.025-0.044 ms.

| tokens/rank | 256 | 512 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| GLM-5.2 | 1.20x | 1.18x | 1.19x | 1.10x | 1.05x | 1.00x |
| V4-Pro | 1.11x | 1.07x | 1.08x | 1.04x | 1.01x | 0.97x |
| V4.1-Flash | 1.31x | 1.24x | 1.07x | 1.01x | 0.98x | 0.98x |

For comparison, the route-push path WITH the compact plan timed is 0.92-1.04x over 512-16384
on the same node (`test_gemm1.py` reports 0.98-1.33x for it because it leaves the plan out).

### In a CUDA graph

`test_recv_layout.py --graph` also captures each side in a CUDA graph (SDMA: `arm` + prepare +
layout + gated GEMM1; aiter: compact prepare + fused stage1) and times replays the same way
(sync and barrier between calls), after the eager numbers in the same process. `arm()` takes a
device tensor of chunk counts, so it is captured too. Three poisoned replays, and one more after
aiter's graph has rewritten the shared tables, are checked bit for bit against aiter.

| tokens/rank | 256 | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| GLM-5.2 | 1.26x (eager 1.17x) | | 1.19x (1.13x) | 1.10x (1.06x) | 1.02x (0.99x) |
| V4-Pro | 1.16x (1.07x) | | 1.09x (1.06x) | 1.02x (1.01x) | 0.98x (0.97x) |
| V4.1-Flash | 1.40x (1.29x) | 1.30x (1.23x) | 1.12x (1.08x) | 1.03x (1.00x) | 0.98x (0.96x) |

8x MI355X, aiter 46cdc4c52, `--chunk-rows 256 --early-chunks 2`; Flash 256/512 on a second node
(the 256 point repeats to the same 1.40x). The graph removes the launch gaps of the SDMA side's
four kernels (0.02-0.09 ms); aiter's two-kernel side barely moves.

aiter's fixed-slot dispatch (no compact plan; 48 experts per rank only, i.e. V4-Pro and Flash,
on by default up to mtpr 128, `AITER_MEGA_FIXED_SLOT_MAX_MTPR` extends it) is NOT in the table
above: every point there is mtpr >= 256, where aiter runs compact. GLM-5.2 (32 experts per
rank) is always compact.

### Against the faster of aiter's two modes (V4.1-Flash)

Both aiter modes measured on the same node and run, with
`AITER_MEGA_FIXED_SLOT_MAX_MTPR=16383` for the fixed-slot side, and the SDMA side with
`--split 3 --stage-first`:

| tokens/rank | aiter compact | aiter fixed-slot | SDMA | vs the faster one |
|---|---|---|---|---|
| 256 | 0.341 | **0.255** | **0.215** | 1.19x |
| 512 | 0.380 | 0.383 | **0.271** | 1.40x |
| 1024 | **0.466** | 0.611 | **0.388** | 1.20x |
| 2048 | **0.674** | — | **0.592** | 1.14x |
| 4096 | **0.933** | — | **0.842** | 1.11x |

Milliseconds, CUDA graph, median of 30 replays, bit-exact. fixed-slot wins at 256, ties at
512 and is 32% behind compact by 1024, so compact is taken as aiter's best at 2048/4096
(fixed-slot was not measured there). The 1.40x at 256 in the first table is against compact
alone; against fixed-slot the same point is 1.19x.

### Publishing tiles in readiness order, and coalescing the quant stores (2026-11-06)

Two changes on top of the numbers above, both bit-exact, both off by default:

**`--ready-first` -- a scheduler wave instead of a per-tile gate.** GEMM1 used to gate every
tile on the SDMA signals itself (`need[source][expert]` vs the signal level, once per tile, by
`gate_npes` lanes). Now wave 0 of the comm workgroup reads each source's signal level ONCE per
round and publishes m-tiles into a ready queue in the order they become ready; a worker claims a
ticket with one `atomic_add` and reads the queue. Workers never touch the signals. Needs
`--fuse-layout` (the scheduler starts when the layout in the same kernel is done).

**`--quant-lds-store` -- one contiguous 16 B per lane.** The quantised row used to leave each
lane as TWO 16-byte pieces 32 bytes apart, so every store instruction covered half of each 128 B
line of an *uncached* (write-through) window. A wave now stages its round of 64 groups in LDS
(2304 B per wave) and reads it back so that each lane stores ONE contiguous 16 B: 1024
contiguous bytes per wave instruction, and the e8m0 scales packed 4 per dword. Same bytes, same
cache modifier. `row_bytes` (5280 for Flash) is not 128 B aligned, so this buys contiguity, not
alignment. ROCm/mori#904 measured the same change at 3.1x on its own dispatch.

Flash, 8x MI355X, CUDA graph, interleaved A/B, 3 reps, medians of 30 replays, every run checked
bit-exact. "base" is the 3-kernel path of the table above, "C1" is
`--own-tokens --fuse-layout --ready-first`:

| tokens/rank | base | C1 | C1 + `--quant-lds-store` | aiter compact | vs aiter |
|---|---|---|---|---|---|
| 256 | 0.218 | 0.222 | not measured | 0.344 | 1.55x (base) |
| 512 | 0.270 | 0.260 | not measured | 0.376 | 1.45x |
| 1024 | 0.389 | 0.354 | **0.345** | 0.463 | **1.34x** |
| 2048 | 0.586 | 0.505 | **0.485** | 0.670 | **1.38x** |
| 4096 | 0.845 | 0.811 | **0.778** | 0.910 | **1.17x** |
| 8192 | 1.541 | 1.479 | not measured | 1.536 | 1.04x |

At 256 the scheduler is a small loss (the first publish waits for the layout and for the slowest
peer's first chunk, which no reordering shortens); use `--own-tokens` alone there. The store
swizzle is what moved `prepare` at bs 4096 from 114.3 us to 73.4 us (quant pass 1: 54.3 -> 29.7).

Where the time goes at bs 2048 under C1 (`--trace`, us from the prepare kernel's start, median
over 8 ranks): prepare 74 (plan 22, staging 11, quant pass 1 36) -> early put 72 -> layout
79..115 -> scheduler sees the layout done at 120, one round costs 8 -> the peers' first chunks
are seen landing 116..135 -> **first m-tile published 140-157** -> every worker has started
within 3 us -> all 215 m-tiles published by 240 -> workers end at ~508, of which 354 us is
`do_tile` and only 5 us is queue wait. So the tile loop is compute-bound, and the front end is
gated by the slowest peer's first chunk (put -> landing is 70-85 us, of which ~28 us is the wire
time for 1.35 MB).

### What the cache operations cost

Most of the gain above came from counting `buffer_wbl2` (whole-L2 writeback) and `buffer_inv`
(whole-L2 invalidate) in the ISA rather than from algorithmic tuning. On gfx950 every
`fence_*_release` is a writeback and every acquire load is an invalidate:

| kernel | `buffer_wbl2` | `buffer_inv` | what was removed |
|---|---|---|---|
| prepare | 3 -> 1 | — | `tree_barrier(release=False)` where the data was published by atomics or by write-through stores; the two `fence_system_release` in the route push |
| GEMM1 | 0 | 4 -> 1 | the per-TILE acquire on `need`, and the per-poll acquire in the start gate |
| layout | 2 -> 0 | 4 -> 3 | both barriers' release, and the per-poll acquire in the flags spin |

The other rule that came out of this: a cross-device access is uncached, so what costs time is
the number of SERIALISED round trips, not bytes. The route push ended with one thread walking
the 8 peers (scount store + flag bump = 16 sequential trips, ~10 us); one lane per peer instead
is ~2.5 us. Packing its four 4 B stores into one 16 B store changed nothing and was dropped.

### Measured and not kept

* **Fusing the prepare into GEMM1** (`--fuse-prepare`, one launch per call): +-1% everywhere
  (bs 1024 +10 us, bs 4096 -9 us). In a CUDA graph the launch gap is already ~0, and the fused
  kernel has to add grid rendezvous of its own. The GEMM hot loop is byte-identical in the ISA
  (1178 instructions, 64 MFMA, 226 VGPR, no spills) with and without the fusion, and
  `amdgpu-disable-unclustered-high-rp-reschedule` changes nothing here.
* **A dynamic, arrival-ordered layout** (a scheduler block that composes tiles from rows as they
  land, with a per-expert ring): works and is bit-exact, but 90 us behind at bs 2048 even after
  multi-pass quant and expert-group bucketing.
* **local-first rows** (each expert's rows ordered by source, this rank first, with a per-tile
  `tsrc` so a tile waits only for the sources it holds) and **arrival-order** (the same, ordered
  by which peer's route flag arrived first): both publish the first tile 10-25 us earlier and
  both end at the same time or later, because their tiles run ~4% slower. At EP8 an expert's
  local rows are ~1/8 of a 64-row tile, so there is no local-only tile to run early.
* **Layout on dedicated blocks** (`--layout-blocks 8/16/32`, overlapping the quant) and **layout
  merged into the scheduler block**: +25 us and +90 us. The layout (32-36 us on 255 blocks) is
  already hidden under the first chunk's landing, so making it earlier buys nothing.
* **More quant blocks**: 511 blocks moves quant pass 1 from 55 to 41 us but the plan from 22 to
  30 (atomic contention on the 384 histogram counters), net zero; 767 blocks deadlocks, since
  only two 512-thread blocks fit per CU and the grid barrier needs them all resident.
* **`--plan-blocks P`** (the plan on P = ceil(bs/512) blocks only, the rest waiting): plan 22 ->
  19 but staging 15 -> 18, because the B2 rendezvous has to come back for every block.
* `--unroll 2` is -3 us and `--unroll 4` is +7 us; `--quant-skip` (pass 1 skips the tokens with
  no early row) is ~0 -- the quant is store-bound, not load-bound. Pre-building the early puts
  and leaving only the doorbell (`aggregate` + `commit`), and `HSA_USERPTR_FOR_PAGED_MEM=0`,
  are both 0.
* **`--tile-n 256`** is 29% slower at bs 2048 and gives the per-tile fixed cost: 7.6 tiles per
  worker x 46.8 us against 15.2 x 30.3 us, i.e. ~14 us per tile that does not depend on N (and
  1.39 MB of B per tile in 14 us would need 25 TB/s, so the weight stream is mostly overlapped
  with the MFMA). Fewer and bigger tiles are therefore better, which also rules out variable-M
  tail tiles.

* `--stream-group G` (prepare puts every chunk itself, a group of G chunks as soon as the quant
  pass that fills it is done; per-pass counters, no rendezvous): GLM-5.2 bs 16384 removes the
  GEMM1 head wait (64-90 us -> 6-14 us per workgroup) but prepare and layout grow by as much;
  best 3.861 ms against 3.908 ms (cr 512, G 1), ~1%.
* `--early-chunks` x `--chunk-rows` at bs 16384 is flat. Caveat: before this change FlyDSL's
  cache key did not see the `Arena` (it keys on closure scalars only), so a GEMM1 compiled for
  one chunk_rows / early_chunks could be reused for another from the same cache directory. The
  kernels now close over `repr(arena)`; sweeps run before this should be redone.

`--trace N` (in `test_recv_layout.py`) prints where in GEMM1's tile order the gates wait, in ten
slices: at bs 16384 essentially all of the wait is in the first slice; the later ones wait only
their ~4 us of gate overhead.

## Status -- read before relying on it

* Needs aiter with `MegaMoEV2`'s compact prepare (its plan tables are reused for the route
  push), FlyDSL, a gfx950 node with 8 GPUs, mori built with SDMA.
* The speedups measured are against aiter main WITHOUT ROCm/aiter#7001, eager, and on a node
  whose GPUs were shared with other jobs (so the cleanest of several runs is quoted).
  Against #7001 this is probably behind at bs 4096 and 8192 (see ROCm/mori#904, which reports the
  same for its SDMA stage1). It has not been measured. Flash (model_dim 5120, inter 2304, 384
  experts, topk 6), prepare + stage1, 8x MI355:

  | tokens/rank | 512 | 1024 | 2048 | 4096 | 8192 |
  |---|---|---|---|---|---|
  | `--chunk-rows` / `--early-chunks` | 128 / 4 | 128 / 2 | 128 / 2 | 256 / 2 | 256 / 2 |
  | vs aiter main | 1.36x | 1.17x | 1.14x | 1.06x | 1.00x |

  The gain is the dispatch overhead, which is roughly fixed, over a GEMM1 that grows with the
  token count: that overhead is 41% of the call at bs 512 and 18% at bs 8192 (`--trace`).
* One call at a time. Nothing yet protects `S` and `R` between back-to-back calls (the tests
  put a barrier between calls), and the SDMA signal / flag totals are advanced by `arm()` (a
  device tensor op, so it can be captured in a graph).
* Not done, and measured not to be worth it: GEMM1 does not do this rank's tokens first. A tile
  mixes all source ranks, so a local-first tile has to be padded -- this rank owns 1/8 of an
  expert's rows against a 64 or 128-row tile -- and the padding costs more GEMM1 time than the
  wait it removes at every token count. Smaller chunks do not help either: they move the first
  put earlier but the comm workgroup then needs 197 us instead of 56 to post them all (bs 8192).
* Not done: GEMM1 is aiter's tile, not the 8-wave 256x256 one, which alone measured 1.18-1.33x
  faster but wants contiguous A rows and pre-shuffled A scales.
* `--trace` stamps the 100 MHz clock inside both kernels (off by default): per prepare block
  [start, count done, plan done, end] and when each early put was posted; per GEMM1 workgroup
  its start, start gate, end, and per work item (gate start, gate passed, item done).
