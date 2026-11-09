# Example 13 -- MegaMoE stage1 dispatch on cco SDMA (gfx950, prototype)

aiter's `MegaMoEV2` stage1 copies every routed row to its destination rank with P2P stores from
32 producer CUs, then runs GEMM1 on the rest. Here the copy engines move the rows instead: the
sender quantises each token once into a local send buffer `S` grouped by destination rank, SDMA
`put`s ship each run to the destination's `R`, and GEMM1 (aiter's own `do_tile`) reads its A rows
and their e8m0 scales in place through an index table `rowmap`. No CU is spent on the copy, and no
aiter plan runs. aiter's stage2 + combine consume the tables unchanged; the final layer output is
bit for bit aiter's.

| file | what |
|---|---|
| `stage1_sdma.py` | `Arena` (window layout), the prepare (`make_prepare_phase` / `build_prepare`), the receive layout (`layout_phase`), the tile scheduler (`scheduler_phase`), host `Stage1SdmaPrepare` |
| `gemm1_sdma.py` | the gated GEMM1 kernel (`compile_gemm1_sdma`, `run_gemm1_sdma`, `gemm1_kwargs`) |
| `test_recv_layout.py` | the whole chain against aiter: poisoned, bit-exact checked calls (`--stress`), eager and CUDA-graph timing, `--trace` |
| `aiter_mode_bench.py` | aiter's stage1 in either of its two modes (compact / fixed-slot), for the comparison |

## How it works

**Prepare** -- one resident launch, `n_cu` workgroups of 512 threads. Workgroup 0 is the comm
workgroup; worker `w` owns tokens `w, w + NW, w + 2 NW, ...`.

1. *Plan.* A token is sent to a rank once, however many experts it has there. Its row in that
   rank's run sits at the position of its FIRST expert there (a counting sort over
   `(destination, first expert)`: per-workgroup LDS counts, one grid rendezvous, one reservation
   atomic per workgroup and cell), so a landed prefix of a run is a prefix of experts.
2. *Route staging.* Each route `(expert, row, weight, source)` is written straight into the
   destination's window, and a per-source flag is bumped -- before the quant, so the destination
   can build its layout while the sender is still quantising.
3. *Quant.* A token's bf16 row is read and quantised once (1x32 MX fp8) and stored to its row in
   every run it is in. A wave stages each round of 64 groups in LDS and stores it back so that every
   lane writes ONE contiguous 16 B: 1 KiB per store instruction into the uncached window. Pass 1
   covers the tokens with a row in an early chunk of another rank's run; a rendezvous; the comm
   workgroup `put`s the first `--early-chunks` chunks of every run, each with a remote signal.

**GEMM1** -- one resident launch. Workgroup 0 `put`s the rest of the chunks. The workers first
build the receive layout from the staged routes (expert-major `rowmap` / `srcmap` / `wts`, padded to
`sort_block_m`, aiter's tile tables), then run the tiles:

* default: a static tile order; before each tile the workgroup waits for the SDMA signal counters to
  cover the chunks its expert needs (`need[source][expert]`, published by the sender);
* `--ready-first`: wave 0 of the comm workgroup reads every source's signal level once per round and
  publishes m-tiles to a ready queue in the order they become ready; a worker claims a ticket with
  one atomic and reads the queue. No worker touches a signal.

`--no-plan` drops the counting sort: a workgroup reserves its rows of each run with one atomic and
places them in any order (no histogram, prefix sum or grid rendezvous, no `need`); the tiles then
wait for ALL of a source's chunks. That is free when a run fits in the early chunks (small batch)
and costs overlap otherwise.

`--fuse-prepare` runs the prepare inside the GEMM1 kernel (one launch per call).

Cache rule that shaped all of this: on gfx950 every release fence is a whole-L2 writeback and every
acquire a whole-L2 invalidate, and a cross-device access is uncached (its cost is the number of
serialised round trips, not bytes). Data that crosses workgroups or devices is written through
(`sc0 sc1`) or published by atomics, so the kernels need almost no fences.

## Running it

```bash
cd mori
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/test_recv_layout.py \
  --network flash --bs 2048 --ready-first --graph-only --stress 1 --time-iters 30
```

| option | |
|---|---|
| `--network` | `flash` (default) and `v4_pro` are the measured ones; `glm52` (32 experts per rank) has not been re-run on this version |
| `--bs` | tokens per rank |
| `--ready-first` | tile scheduler wave + ready queue |
| `--no-plan` | no counting sort; tiles wait for all of a source's chunks |
| `--fuse-prepare` | prepare inside the GEMM1 kernel |
| `--chunk-rows`, `--early-chunks`, `--head-rows` | SDMA chunking (default 256 rows, 2 early chunks, no short head chunk) |
| `--sbm`, `--tile-n` | override aiter's GEMM1 M / N tile (`--sbm` needs `--no-check`: aiter's stage2 is compiled for its own) |
| `--stress N` | N poisoned calls, each checked bit for bit against aiter |
| `--time-iters`, `--graph`, `--graph-only` | eager and/or CUDA-graph timing |
| `--trace N` | clock stamps of N calls: prepare phases, layout, scheduler, per-worker tile and queue-wait time |
| `--no-check`, `--seed` | |

aiter's two modes, for the comparison (the test only runs aiter's compact mode):

```bash
AITER_MEGA_FIXED_SLOT_MAX_MTPR=16383 torchrun --standalone --nproc_per_node=8 aiter_mode_bench.py --network flash --bs 256
AITER_MEGA_FIXED_SLOT_MAX_MTPR=0     torchrun --standalone --nproc_per_node=8 aiter_mode_bench.py --network flash --bs 256
```

## Results (2026-11-06 .. 11-09)

8x MI355X, EP8, CUDA graph, median of 30 replays (sync + barrier between calls), interleaved A/B
with 3 reps, every run checked bit-exact. Stage1 = quant + dispatch + GEMM1 on both sides (aiter:
compact prepare + fused stage1). "aiter" is aiter's compact mode measured in the same process, except
at bs 256 where aiter's fixed-slot mode is faster and is measured alone. Standalone runs of aiter
read 1-5% slower than in-process ones (v4_pro); the in-process number is the one used.

**V4.1-Flash** (hidden 5120, inter 2304, 384 experts, top-6)

| tokens/rank | config | SDMA | aiter | speedup |
|---|---|---|---|---|
| 256 | `--no-plan` | 0.201 | 0.265 (fixed-slot) | **1.32x** |
| 512 | `--ready-first --no-plan` | 0.253 | 0.378 | **1.49x** |
| 1024 | `--ready-first` | 0.349 | 0.465 | **1.33x** |
| 2048 | `--ready-first` | 0.474 | 0.671 | **1.42x** |
| 4096 | `--ready-first` | 0.770 | 0.912 | **1.18x** |
| 8192 | `--ready-first` | 1.386 | 1.533 | **1.11x** |

**V4-Pro** (hidden 7168, inter 3072, 384 experts, top-6); fixed-slot is slower than compact here.

| tokens/rank | config | SDMA | aiter | speedup |
|---|---|---|---|---|
| 512 | `--ready-first` | 0.390 | 0.465 | **1.19x** |
| 1024 | default | 0.514 | 0.600 | **1.17x** |
| 2048 | default | 0.739 | 0.882 | **1.19x** |
| 4096 | `--ready-first` | 1.256 | 1.393 | **1.11x** |

What each mechanism is worth (same-run A/B, end to end):

* LDS-swizzled quant stores: prepare at Flash bs 4096 114.3 -> 73.4 us (quant pass 1 54.3 -> 29.7);
  end to end -3 us (bs 256), -19 (2048), -33 (4096), -78 (8192); V4-Pro -54 us at 4096. Scales with
  the bytes staged (`~4.4 x bs` rows per rank at EP8 / top-6).
* `--ready-first`: Flash -35 us (1024), -85 (2048), -33 (4096), -62 (8192), +11 at 256 (loss);
  V4-Pro -12 (512), 0 (1024), +54 (2048, loss, not explained), -6 (4096). Its gain is the per-tile
  gate it removes, so it scales with tiles per worker; its cost (the first publish waits for the
  layout and for the slowest peer's first chunk) is fixed.
* `--no-plan`: Flash prepare at bs 256 36.7 -> 27.5 us (plan 14.7 -> 5.9); end to end -12 us (256),
  -5 (512), +23 (1024, loss: a rest chunk now gates every tile); V4-Pro 512: 0.

Where the time goes at Flash bs 2048 with `--ready-first` (`--trace`, us from the prepare's start):
prepare 74 -> early put 72 -> layout 79..115 -> peers' first chunks seen landing 116..135 -> first
m-tile published 140-157 -> every worker started within 3 us -> all published by 240 -> workers done
at ~508, of which 354 us is `do_tile` and 5 us queue wait. The tile loop is compute-bound; the front
end is gated by the slowest peer's first chunk (put -> landing 70-85 us, ~28 us of it wire time for
1.35 MB). A per-tile `--tile-n 256` vs 512 comparison gives ~14 us of every 46.8 us tile that does
not depend on N. At large batch both sides run the same `do_tile`, which bounds the speedup.

## Measured and not kept

Each was implemented, checked bit-exact and A/B'd; none is in this code any more.

* Fusing the prepare into GEMM1 as the default: +-1% (bs 1024 +10 us, 4096 -9); in a graph the launch
  gap is already ~0 and the fused kernel adds rendezvous of its own. Kept as `--fuse-prepare`.
* A dynamic, arrival-ordered layout (tiles composed from rows as they land, per-expert ring): 90 us
  behind at bs 2048.
* Local-first / arrival-ordered rows with a per-tile source set: first tile 10-25 us earlier, tiles
  ~4% slower, no end-to-end gain. Layout on dedicated workgroups (+25 us) or in the scheduler
  workgroup (+90 us): the layout is already hidden under the first chunk's landing.
* More quant workgroups (511: pass 1 -14 us, plan +8.5 us, net 0; 767 deadlocks: only two 512-thread
  workgroups fit per CU); the plan on fewer workgroups; unroll 2 / 4; skipping tokens without early
  rows in pass 1; a pre-built SDMA doorbell; `HSA_USERPTR_FOR_PAGED_MEM=0`: all ~0.
* `stream` (per-chunk ready counters instead of the rendezvous): worse at every size. One copy for
  the whole rest: 5% worse at bs 4096. Chunk-size sweeps: flat.
* Removing the acquire per tile and per poll, and the release where data was already written
  through: the cache-op counts in the ISA went `buffer_wbl2` 3 -> 1 (prepare), `buffer_inv` 4 -> 1
  (GEMM1); an acquire per tile cost ~90 us of 0.96 ms.
* `do_tile` knobs `pair_k` and `mfma_amajor=False` (much slower at bs 4096), `async_a_copy=False`
  (flat); `amdgpu-disable-unclustered-high-rp-reschedule` (ISA unchanged).

## Status -- read before relying on it

* Prototype. Stage1 only; one call at a time (the tests put a barrier between calls); uniform random
  routing. `arm()` advances the SDMA signal and flag totals per call (a device op, graph-capturable).
* Needs mori built with cco SDMA (`Sdma.put` with `remote_signal`), aiter's `MegaMoEV2`, FlyDSL, and a
  gfx950 node with 8 GPUs.
* One sporadic hang was seen (V4-Pro bs 512 `--ready-first`, the first run in a fresh FlyDSL cache;
  the same config passed on re-run) and is not explained.
