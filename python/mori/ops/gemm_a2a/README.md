# mori.ops.gemm_a2a — fp8 GEMM fused with an all-to-all

Measurements for `mori.ops.gemm_a2a`, the column-sharded all-to-all fused into
the epilogue of mori's 8-wave fp8 GEMM.

## What the operator is

Every rank holds its own `A [M, K]` and a replicated `B [N, K]`, computes the
whole `C = A @ B.T`, and sends column block `j` to rank `j`. Rank `d` ends up
with `[world*M, shard_n]` bf16, source rank `r` at rows `[r*M, (r+1)*M)`.

The shape it is aimed at is a QKV projection under Ulysses sequence
parallelism: `K` is the hidden size, `N` the QKV width, `s = N/P` the per-rank
head shard, and **`M = S/P`** — so sweeping `M` is sweeping the sequence length.

| mode | what it does |
|---|---|
| `gemm-only` | the GEMM alone, to size the ceiling |
| `split-lsa` | `gemm()`; a copy kernel reads `[M,N]` and vector-stores into the peers |
| `split-sdma` | `gemm()` into `[dst][M][shard_n]`; one SDMA push per destination |
| `split-rccl` | the same staging GEMM; then `all_to_all_single` |
| `fused-lsa` | C stored straight into the peers from the epilogue |
| `fused-sdma` | C staged per destination, pushed by the copy engine per chunk |

`split-sdma`, `split-rccl` and `fused-sdma` share a GEMM byte for byte, so the
differences between them are transport and nothing else.

## Tile

`layout.default_block_m(m, n, quant)` picks the GEMM tile, and the bench uses
it unless `--block-m` is given:

* **mxfp8: 256x256**, always. The packed A scale puts a lane's four M tiles in
  one dword and picks the byte with the MFMA's `opsel`, which is four tiles only
  at `BLOCK_M=256`.
* **ptpc: 256x256 when that gives at least two waves of tiles, else 128x256.**
  256x256 moves a third less A+B per output element; below two waves its last
  wave runs on a fraction of the CUs and that costs more than the reuse saves.
* **blockscale: 128x256.** Its per-K-block promotion doubles the accumulators
  and 256x256 spills (`gemm_ar/op.py`).

The ptpc rule was measured, not assumed — every ptpc cell below was also run at
the tile the rule did *not* pick (`sweep_inproc.py --alt-tile`):

| model | M | waves at 256x256 | picked | gemm-only 128 | gemm-only 256 | best fused 128 | best fused 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 70B | 2048 | 1.25 | **128** | **151** | 159 (+5%) | **203** | 210 (+3%) |
| 70B | 4096 | 2.50 | 256 | 264 | 257 (-2%) | 349 | **328** (-6%) |
| 70B | 8192 | 5.00 | 256 | 524 | 451 (-14%) | 640 | **578** (-10%) |
| 70B | 16384 | 10.00 | 256 | 1359 | 918 (-32%) | 1737 | **1138** (-35%) |
| 405B | 2048 | 2.25 | 256 | 618 | 502 (-19%) | 729 | **568** (-22%) |
| 405B | 4096 | 4.50 | 256 | 1157 | 899 (-22%) | 1330 | **989** (-26%) |
| 405B | 8192 | 9.00 | 256 | 2301 | 1699 (-26%) | 2645 | **1858** (-30%) |
| 405B | 16384 | 18.00 | 256 | 4576 | 3383 (-26%) | 5307 | **3678** (-31%) |

The picked tile has the faster fused path in all eight cells. 128x256 was the
ptpc default before this table existed; it was inherited from `gemm_ar`'s
blockscale path, where it is a spill limit, and under ptpc it has no such
reason.

The rule was fitted at P=8 and does not carry over to P=4 unchanged: on
MiniMax-M3's QKV at P=4 ([below](#against-aiter-6689--minimax-m3-qkv-at-p4))
128x256 has the fastest fused arm at every M, 1–7% ahead of the 256x256 the
rule picks at M=4096 and 8192.

## Conditions

8x MI355X (gfx950), one node (`crsuse2-m2m-v3-095`, 2026-10-28), nothing else
on it, fp8 e4m3 in / bf16 out, FlyDSL 0.4.0, torch 2.12.0+rocm7.14.0.

**Timing.** Every case of a table runs in one process on one communicator
(`sweep_inproc.py`). A case replays its CUDA graph back to back, 50 launches
to a batch with an event between each, until at least 3 s (8 s for the last
ptpc round and the mxfp8 rounds) and 100 launches have run; the number is the
median launch-to-launch gap, max over ranks. The GPU never waits on the host,
so there is no per-launch submit gap in the number. That makes small-M cells a
few percent faster than in this file's earlier tables, which timed each launch
with its own sync; the fused-vs-split margins moved by at most two points.

**Three rounds per configuration, interleaved.** Every configuration is measured
once per round and the rounds repeat, alternating direction, so machine drift
lands inside each configuration's own samples where the spread shows it, rather
than between the things being compared.

Spread across rounds is 0.0–3.3% on every ptpc configuration, and all 192 ptpc
samples validated against a host reference on every source rank.

> **This protocol is not optional here, and the cost of skipping it was most of
> a day.** A neighbour on the box is worth 9% on a *pure GEMM* — a kernel that
> communicates nothing — and it is not a uniform slowdown: with company, 70B's
> best chunk count measured as 2 with a 4.3% margin; alone, it is 4 with a 9.2%
> margin. The ranking changes, so a contaminated table cannot be rescued by
> scaling it. Earlier sweeps in this file's history were grouped by
> configuration rather than interleaved, and one cell measured 5153us four
> times running and 3508us four times running with a byte-identical kernel and
> an identical config; nothing in the code explained it and nothing needed to.

## Results

### PTPC — Llama-3.1-70B / Qwen2.5-72B — `N=10240 K=8192` (s=1280)

| M | S | tile | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 2048 | 16k | 128×256 | 151 | 244 | 344 | **203** `c4` | 16.6% | 40.9% |
| 4096 | 32k | 256×256 | 257 | 441 | 513 | **328** `c8` | 25.5% | 36.0% |
| 8192 | 64k | 256×256 | 451 | 813 | 930 | **578** `c4` | 28.8% | 37.8% |
| 16384 | 128k | 256×256 | 918 | 1619 | 1824 | **1138** `c8` | **29.7%** | 37.6% |

### PTPC — Llama-3.1-405B — `N=18432 K=16384` (s=2304)

| M | S | tile | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 2048 | 16k | 256×256 | 502 | 681 | 738 | **568** `c8` | 16.6% | 23.1% |
| 4096 | 32k | 256×256 | 899 | 1204 | 1303 | **989** `c8` | 17.9% | 24.1% |
| 8192 | 64k | 256×256 | 1699 | 2280 | 2457 | **1858** `c8` | 18.5% | 24.4% |
| 16384 | 128k | 256×256 | 3383 | 4694 | 4998 | **3678** `c8` | **21.6%** | 26.4% |

All modes agree to the digit on relL2 — 1.66e-3, fp8's own floor at these
operands — so the transports write identical bytes and the numbers above are
comparing the same computation.

### MXFP8 — 256x256

Same node and run as the ptpc tables, two rounds, spread 0.0–2.3%, all 80
samples validated.

| M | S | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---:|---:|---:|---|---:|---:|
| **70B** | | | | | | | |
| 2048 | 16k | 182 | 282 | 382 | **230** `c4` | 18.3% | 39.8% |
| 4096 | 32k | 289 | 465 | 560 | **369** `c4` | 20.6% | 34.1% |
| 8192 | 64k | 545 | 861 | 1001 | **651** `c4` | 24.4% | 35.0% |
| 16384 | 128k | 1121 | 1798 | 2000 | **1374** `c8` | **23.6%** | 31.3% |
| **405B** | | | | | | | |
| 2048 | 16k | 603 | 761 | 828 | **676** `c8` | 11.2% | 18.4% |
| 4096 | 32k | 1065 | 1333 | 1456 | **1174** `c8` | 11.9% | 19.3% |
| 8192 | 64k | 1989 | 2619 | 2792 | **2198** `c8` | 16.1% | 21.3% |
| 16384 | 128k | 4016 | 5332 | 5692 | **4395** `c8` | **17.6%** | 22.8% |

At the same 256x256 tile, mxfp8's GEMM is **13–22% slower** than ptpc's in
every cell (405B M=16384 gemm-only 4016 vs 3383us), and its best fused path
10–21% slower. An earlier version of this file read an mxfp8 table as "mxfp8 is
faster than ptpc at large M"; that comparison ran ptpc at 128x256 and mxfp8 at
256x256, so it was the tile, not the format. The gap is the scale loads
stalling the tile prefetch — loads retire in order, so a scale load issued
after the prefetch waits behind it. mori #857 prefetches the scales one K step
ahead in `gemm_ar`'s fused kernel; this operator's GEMM does not have that
yet, and this table is without it.

### The full mode table at one cell, `M=2048` — earlier run, 128x256

Taken on the earlier node with ptpc at 128x256; not re-measured. It is kept for
the modes the model tables do not carry (`split-lsa`, `fused-lsa`, `c2`).

| | 70B | | 405B | |
|---|---:|---:|---:|---:|
| | us | comm | us | comm |
| `gemm-only` | 180.4 | — | 686.9 | — |
| `fused-sdma --chunks 4` | **245.3** | 29.5 | 816.0 | 60.8 |
| `fused-sdma --chunks 8` | 256.2 | 36.1 | **802.0** | 37.0 |
| `fused-sdma --chunks 2` | 258.2 | 55.6 | 847.0 | 92.6 |
| `split-sdma` | 270.2 | 100.4 | 831.3 | 171.8 |
| `split-lsa` | 271.2 | 101.2 | 856.3 | 186.3 |
| `fused-lsa` | 288.2 | — | 915.0 | — |
| `split-rccl` | 298.4 | 126.5 | 890.6 | 230.2 |

"comm" is `total - gemm` **measured in the same run**. On the fused rows it is
the *unhidden tail*, not wire time — the transfer is inside the GEMM — so it is
comparable across fused configurations but not against the split ones.

## Against aiter #6689 — MiniMax-M3 QKV at P=4

aiter #6689 fuses the same operation for MiniMax-M3's QKV projection under
Ulysses SP4 (its `sp_qkv` large path): `K=6144`, and per destination either Q|K|V
(**dense**, 2304 columns, `N=9216`) or Q|K|V|IQ|IK (**indexer**, 2560 columns,
`N=10240`). aiter computes the indexer's IK once and sends it to every rank;
mori can only express that with IK's 128 weight rows replicated per destination,
so on the indexer case mori does 3.9% more GEMM work than aiter.

Measured against aiter's current head (474128f), whose shipped profile selects
its r13 kernels for every large row. aiter's side was timed through its public
plan API -- `SpQkvLarge.run` with the profile-selected parameters, i.e. its GEMM,
SDMA exchange and `quiet_ready` -- and mori's with `sweep_inproc.py --suite
minimax`. Both sides: CUDA graph replay, back to back with an event between
launches, median launch gap, max over ranks, GPUs 0-3 of one node, three
interleaved rounds (spread <= 0.4%), every result validated at 1.66e-3. These
numbers come from the review of this PR (8x MI355X, ROCm 7.14 CI image, FlyDSL
0.4.0); times in us:

| case | M | aiter r13 | mori, `GemmA2AOp` default | vs aiter | mori, best arm | vs aiter |
|---|---:|---:|---|---:|---|---:|
| dense | 2048 | 247.0 | 242.4 `128 c8` | -1.8% | **231.8** `128 c4` | -6.2% |
| dense | 4096 | 439.3 | 431.0 `256 c8` | -1.9% | **404.4** `128 c8` | -7.9% |
| dense | 8192 | 778.1 | 751.7 `256 c8` | -3.4% | **747.2** `128 c16` | -4.0% |
| indexer | 2048 | 289.6 | 259.5 `128 c8` | -10.4% | **248.6** `128 c4` | -14.2% |
| indexer | 4096 | 475.2 | 467.4 `256 c8` | -1.6% | **438.8** `128 c8` | -7.7% |
| indexer | 8192 | 870.6 | 825.4 `256 c8` | -5.2% | **814.9** `128 c16` | -6.4% |

"vs aiter" is negative where mori is faster. For scale, dense `gemm-only` is
117.5 / 195.7 / 336.0us and dense `split-rccl` 325.3 / 602.7 / 1131.3us at
M = 2048 / 4096 / 8192 (this file's own run).

mori is ahead in every cell, by 1.6-10.4% with the operator's own picks and
4.0-14.2% with the best arm, and that is with the indexer doing 3.9% more GEMM.
An earlier run here against aiter's r11 kernels, through a standalone harness
and the two configurations its profile listed per shape, agreed within about a
point. The defaults leave time on the table at P=4 -- up to 6.6% (dense and
indexer M=4096): 128x256 wins at every M (the tile rule was fitted at P=8), and
at M=2048 four chunks beat the eight that `default_chunks` picks. A
world-size-aware rule is left for a follow-up.

## Reading the tables

**Fusing wins in all eight cells, and the margin grows with sequence length
on both shapes.** 70B: 16.6% → 25.5% → 28.8% → 29.7%; 405B: 16.6% → 17.9% →
18.5% → 21.6%. At larger M there is more GEMM after the first chunk completes
for its PUT to overlap with. The 128x256 tables in this file's history had 70B
peak at S=64k and fall back at 128k; at 256x256 the curve is monotone, so that
fall-back was a property of the 128x256 kernel.

**The tile is worth more than the fusion at large M.** At 70B M=16384, moving
ptpc from 128x256 to 256x256 takes the fused path from 1737 to 1138us (-35%),
against the 29.7% that fusion itself buys over `split-sdma`. Two things move
together: the GEMM is 26–32% cheaper at large M, and with half the blocks the
epilogue's per-block release fence and completion counter cost half as much —
the fused kernel alone (`--phase-split` compute) is 3.5% above `gemm-only` at
405B M=16384 with 256x256 against 11.1% with 128x256, and 11% against 19% at
70B M=16384.

**The best chunk count is `c8` almost everywhere.** 405B wants `c8` at every M;
70B is within 1% between `c4` and `c8` at M=8192 and prefers `c4` at M=2048.
Each chunk is a run of row tiles, so at 256x256 `c8` at M=2048 is one row tile
per chunk; going further cannot start the first PUT earlier, because a chunk
cannot complete before the GEMM wave that computes it does.

**RCCL is last in every cell**, 23–41% behind the best fused path and 6–29%
behind `split-sdma`. That comparison is clean: `split-rccl` and `split-sdma`
run the same GEMM over the same bytes in the same layout. RCCL on the crsuse2
nodes is slower at small M than on the node of the 128x256 mode table below
(70B M=2048: 344 vs 298us), so the "vs rccl" column is not comparable across
the two.

**Fusing nearly eliminates the exposed transfer, and that bounds the win.** At
405B/M=16384 the communication drops 1372us → 176us, −87%, yet end to end the
gain is 21.6%. The GEMM is 95% of the fused total, so there is very little left
to hide. Of the remaining tail, about 90us is the last wave's chunk on the wire
and the rest is rank-to-rank skew in GEMM speed (`--timeline`).

**LSA loses to SDMA on this operator.** `split-lsa` reaches only ~60% of the
76.8 GB/s line rate where the copy engines reach ~75%; `fused-lsa` is the
slowest mode at both shapes. Ruled out as causes: grid size (24/48/80 blocks
within 0.6%), cache policy (`sc0|sc1` worth 1.7%), local memory bandwidth
(10.6us of an 88us gap), and the read layout — `split-lsa-staged` reads a
compacted slab instead of a strided `[M,N]`, moves 12% fewer bytes, and takes
the same time, so LSA's cost is its peer stores alone. gcnasm's independent
measurement of the same kind of store puts it at 46.1 GB/s per link against our
48.3, and its ATT capture found 99% of that time was stall. (Measured at
128x256; not re-measured.)

## Shape constraints

`validate()` rejects `n % (world_size * block_n) != 0`, which for P=8 and
block_n=256 means **N must be a multiple of 2048**. That is not an arbitrary
tile rule: a destination's column run has to be a whole number of GEMM tiles or
a tile straddles two destinations and its store cannot stay contiguous.

Checked against real QKV widths at P=8:

| model | K | N | s | validate |
|---|---:|---:|---:|---|
| Llama-3.1-8B | 4096 | 6144 | 768 | ok |
| Llama-3.1-70B | 8192 | 10240 | 1280 | ok |
| Qwen2.5-72B | 8192 | 10240 | 1280 | ok |
| Qwen3-32B | 5120 | 10240 | 1280 | ok |
| Llama-3.1-405B | 16384 | 18432 | 2304 | ok |
| Qwen3-235B-A22B | 4096 | 9216 | 1152 | **rejected** |

Qwen3-235B has 4 KV heads, so `N = (64 + 2*4) * 128 = 9216` and `9216 / 2048`
is not an integer. The deeper constraint is that it has fewer KV heads than
ranks, so Ulysses at P=8 would have to replicate KV — the layout rule is that
condition showing up in this kernel, not a separate limitation.

Head alignment is exact for the others and needs no padding: 70B's `s = 1280`
is 8 q heads + 1 k + 1 v at 128 each, which is one whole GQA group per rank.

`m` must also be a multiple of `block_m`; `default_block_m` only picks 256 when
it divides `m`.

## Reproducing

From the repository root:

```bash
# one cell (tile picked from the shape; pass --block-m 128|256 to force one)
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py \
  --mode fused-sdma --chunks 8 -m 16384 -n 18432 -k 16384 \
  --warmup 30 --iters 21 --phase-split

# the ptpc tables and the tile table; the mxfp8 table
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/sweep_inproc.py --suite models --quant ptpc \
  --alt-tile --rounds 3 --out models.jsonl
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/sweep_inproc.py --suite models --quant mxfp8 \
  --rounds 2 --out mxfp8.jsonl

# mori's side of the aiter comparison, on four GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/gemm_a2a/sweep_inproc.py --suite minimax --rounds 3 \
  --out minimax.jsonl

# the mode/chunk sweep at one shape
python benchmark/cco/flydsl/gemm_a2a/sweep_a2a.py --out sweep.jsonl
```

`sweep_inproc.py` skips the rows its `--out` already holds, so a sweep that was
stopped resumes where it was by running the same command again.

`BUILD_CCO_SDMA=ON` is required for the SDMA modes. Without it every put is a
silent no-op; that is caught here — the received slabs stay zero and validation
reports relL2 1.0 — but only because validation reads what arrived, per source
rank, rather than this rank's own block.

Launching one process per case (`sweep_models.py`, `sweep_a2a.py`) loses the
SDMA queue reclamation race (`anvil.cpp:237`) often enough that a bare loop
cannot finish a table; those drivers settle between launches and retry a lost
race rather than recording it. `sweep_inproc.py` keeps one communicator for
the whole sweep and did not hit it once in the runs behind these tables.

## Not measured

* `--quant blockscale`. `gemm_ar` found the fused margin *grew* under
  blockscale.
* `--chunks` past 8 at P=8, and `--chunks 2` anywhere but `M=2048`.
* The other validated shapes — 8B, Qwen3-32B.
* fp8 on the wire. The transfer is already 87% hidden at the large end, so this
  would help the small-M cells rather than the large ones.
* P other than 8, beyond the MiniMax-M3 comparison at P=4 — in particular, a
  tile rule fitted at P=4.
