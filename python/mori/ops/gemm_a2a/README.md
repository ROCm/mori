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
the tile the rule did *not* pick (`sweep_models.py --alt-tile`):

| model | M | waves at 256x256 | picked | gemm-only 128 | gemm-only 256 | best fused 128 | best fused 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 70B | 2048 | 1.25 | **128** | 169 | 169 (-0%) | **213** | 217 (+2%) |
| 70B | 4096 | 2.50 | 256 | 287 | 269 (-6%) | 358 | **338** (-6%) |
| 70B | 8192 | 5.00 | 256 | 532 | 459 (-14%) | 646 | **588** (-9%) |
| 70B | 16384 | 10.00 | 256 | 1372 | 911 (-34%) | 1730 | **1118** (-35%) |
| 405B | 2048 | 2.25 | 256 | 623 | 509 (-18%) | 734 | **578** (-21%) |
| 405B | 4096 | 4.50 | 256 | 1148 | 889 (-23%) | 1323 | **980** (-26%) |
| 405B | 8192 | 9.00 | 256 | 2289 | 1692 (-26%) | 2626 | **1839** (-30%) |
| 405B | 16384 | 18.00 | 256 | 4554 | 3404 (-25%) | 5269 | **3668** (-30%) |

The picked tile has the faster fused path in all eight cells. 128x256 was the
ptpc default before this table existed; it was inherited from `gemm_ar`'s
blockscale path, where it is a spill limit, and under ptpc it has no such
reason.

## Conditions

8x MI355X (gfx950), one node (`crsuse2-m2m-v3-054`, 2026-10-27), fp8 e4m3 in /
bf16 out, CUDA graph replay, median of 21 iterations after 30 warmups, max over
ranks. FlyDSL 0.2.4, torch 2.12.0+rocm7.14.0.

**Three independent launches per configuration, interleaved.** Every
configuration is measured once per round and the rounds repeat, so machine
drift lands inside each configuration's own samples where the spread shows it,
rather than between the things being compared. Every launch waits for an idle
box; any sample taken with a neighbour present is discarded, not averaged in.
Junction temperature stayed at 58–60 °C across the run.

Reported spread is 0.1–3.6% on every configuration, and all 192 samples
validated against a host reference on every source rank.

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
| 2048 | 16k | 128×256 | 169 | 254 | 352 | **213** `c4` | 16.0% | 39.4% |
| 4096 | 32k | 256×256 | 269 | 443 | 513 | **338** `c8` | 23.6% | 34.1% |
| 8192 | 64k | 256×256 | 459 | 814 | 931 | **588** `c4` | 27.8% | 36.8% |
| 16384 | 128k | 256×256 | 911 | 1613 | 1821 | **1118** `c8` | **30.7%** | 38.6% |

### PTPC — Llama-3.1-405B — `N=18432 K=16384` (s=2304)

| M | S | tile | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 2048 | 16k | 256×256 | 509 | 687 | 744 | **578** `c8` | 15.9% | 22.4% |
| 4096 | 32k | 256×256 | 889 | 1204 | 1304 | **980** `c8` | 18.6% | 24.8% |
| 8192 | 64k | 256×256 | 1692 | 2275 | 2451 | **1839** `c8` | 19.1% | 25.0% |
| 16384 | 128k | 256×256 | 3404 | 4676 | 4989 | **3668** `c8` | **21.6%** | 26.5% |

All modes agree to the digit on relL2 — 1.66e-3, fp8's own floor at these
operands — so the transports write identical bytes and the numbers above are
comparing the same computation.

### MXFP8 — earlier run, not re-measured on this node

The table below was taken on a different node with the same protocol (two
rounds, spread 0.1–1.0%), before the ptpc tile changed. Its absolute times are
not comparable with the ptpc tables above.

| M | S | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---:|---:|---:|---|---:|---:|
| **70B** | | | | | | | |
| 2048 | 16k | 197 | 294 | 322 | **249** `c4` | 15.3% | 22.8% |
| 4096 | 32k | 305 | 481 | 552 | **391** `c4` | 18.7% | 29.1% |
| 8192 | 64k | 596 | 898 | 1047 | **681** `c4` | 24.2% | 35.0% |
| 16384 | 128k | 1211 | 1916 | 2121 | **1443** `c8` | **24.7%** | 32.0% |
| **405B** | | | | | | | |
| 2048 | 16k | 669 | 826 | 872 | **733** `c8` | 11.3% | 16.0% |
| 4096 | 32k | 1151 | 1465 | 1546 | **1257** `c8` | 14.2% | 18.7% |
| 8192 | 64k | 2129 | 2788 | 2959 | **2341** `c8` | 16.1% | 20.9% |
| 16384 | 128k | 4230 | 5649 | 5988 | **4696** `c8` | **16.9%** | 21.6% |

An earlier version of this file read this table as "mxfp8 is faster than ptpc
at large M" and "fusing pays twice as much under mxfp8". Both were the tile,
not the format: that comparison ran ptpc at 128x256 and mxfp8 at 256x256. At
the **same** 256x256 tile, on one node with interleaved rounds, mxfp8's GEMM is
**15–18% slower** than ptpc's in every one of six cells (70B and 405B at
M = 2048 / 8192 / 16384; e.g. 405B M=16384 gemm-only 3921 vs 3378us), and its
best fused path 10–17% slower.

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

## Reading the tables

**Fusing wins in all eight cells, and the margin grows with sequence length
on both shapes.** 70B: 16.0% → 23.6% → 27.8% → 30.7%; 405B: 15.9% → 18.6% →
19.1% → 21.6%. At larger M there is more GEMM after the first chunk completes
for its PUT to overlap with. The 128x256 tables in this file's history had 70B
peak at S=64k and fall back at 128k; at 256x256 the curve is monotone, so that
fall-back was a property of the 128x256 kernel.

**The tile is worth more than the fusion at large M.** At 70B M=16384, moving
ptpc from 128x256 to 256x256 takes the fused path from 1730 to 1118us (-35%),
against the 30.7% that fusion itself buys over `split-sdma`. Two things move
together: the GEMM is 25–34% cheaper at large M, and with half the blocks the
epilogue's per-block release fence and completion counter cost half as much —
the fused kernel alone (`--phase-split` compute) is 2.5% above `gemm-only` at
405B M=16384 with 256x256 against 10.3% with 128x256, and 12% against 17% at
70B M=16384.

**The best chunk count is `c8` almost everywhere.** 405B wants `c8` at every M;
70B is within 1% between `c4` and `c8` at M=8192 and prefers `c4` at M=2048.
Each chunk is a run of row tiles, so at 256x256 `c8` at M=2048 is one row tile
per chunk; going further cannot start the first PUT earlier, because a chunk
cannot complete before the GEMM wave that computes it does.

**RCCL is last in every cell**, 22–39% behind the best fused path and 6–28%
behind `split-sdma`. That comparison is clean: `split-rccl` and `split-sdma`
run the same GEMM over the same bytes in the same layout. RCCL on this node is
slower at small M than on the earlier one (70B M=2048: 352 vs 298us), so the
"vs rccl" column is not comparable across the two.

**Fusing nearly eliminates the exposed transfer, and that bounds the win.** At
405B/M=16384 the communication drops 1344us → 175us, −87%, yet end to end the
gain is 21.6%. The GEMM is 93% of the fused total, so there is very little left
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

# the ptpc tables and the tile table, with the idle-box protocol
python benchmark/cco/flydsl/gemm_a2a/sweep_models.py --rounds 3 --alt-tile \
  --out models.jsonl
python benchmark/cco/flydsl/gemm_a2a/sweep_models.py --quant mxfp8 --rounds 2 \
  --out mxfp8.jsonl

# the mode/chunk sweep at one shape
python benchmark/cco/flydsl/gemm_a2a/sweep_a2a.py --out sweep.jsonl
```

`BUILD_CCO_SDMA=ON` is required for the SDMA modes. Without it every put is a
silent no-op; that is caught here — the received slabs stay zero and validation
reports relL2 1.0 — but only because validation reads what arrived, per source
rank, rather than this rank's own block.

Back-to-back launches lose the SDMA queue reclamation race (`anvil.cpp:237`)
often enough that a bare loop cannot finish a table. Both drivers settle
between launches and retry a lost race rather than recording it.

## Not measured

* `--quant blockscale`. `gemm_ar` found the fused margin *grew* under
  blockscale.
* The mxfp8 tables on the node the ptpc tables were taken on.
* `--chunks` past 8, and `--chunks 2` anywhere but `M=2048`.
* The other validated shapes — 8B, Qwen3-32B.
* fp8 on the wire. The transfer is already 87% hidden at the large end, so this
  would help the small-M cells rather than the large ones.
* Anything at P != 8.
