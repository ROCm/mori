# xGMI copy bandwidth: TDM vs CU

How much of a GPU it costs to saturate one xGMI link, and at what transfer size that cost is worth
paying.

The question is not "which transport is faster" -- at a full grid and a large payload they land
within 0.1% of each other, both at the link ceiling. It is **how much of the GPU each one has to
occupy to get there**, because whatever is left over is what a fused kernel can use for compute.

Two sweeps answer that, both built from `ualoe_bw.cpp`, which copies a payload to a peer GPU over
xGMI and reports one-way GB/s measured on the pushing side:

| sweep | script | axis | build |
|---|---|---|---|
| block sweep | `tools/blksweep.sh` | grid width, at a fixed 16 GB payload | `-DSWEEP_16 -DBLKONLY` |
| size x width matrix | `tools/uamatrix.sh` | transfer size x grid width | `-DSWEEP_MATRIX`, `MATRIX=1` |

Transports compared:

| kind | name | how it moves data |
|---|---|---|
| 0 | `copyk` | CU vector copy, `uint4` loads and stores, every thread in the block moves data |
| 1 | `tdm_write` | TDM (tensor DMA) descriptors, staged through LDS; one issuing wave per block |
| 8 | `tdm_store_cuload` | CU-staged: vector loads into LDS, TDM stores out |
| 9 | `tdmmws` | staged multi-issuer, `MWSISS` issuing waves per block |

The block is deliberately *not* normalised into equivalent hardware. The CU kernel runs 512 threads
per block all moving data; the TDM kernel runs 256 threads per block of which two waves issue
descriptors and the rest wait on a barrier. That asymmetry is the result, not a confound.

## Block sweep: how wide a grid each transport needs

16 GB payload, both transports at the same block count. Full data in
[`results/blksweep_16gb_gfx1250.csv`](results/blksweep_16gb_gfx1250.csv) (two rounds).

| blocks | CU | TDM | TDM as % of its own ceiling |
|---|---|---|---|
| 128 | 337 | 999 | 61% |
| 256 | 666 | 1582 | 96% |
| 512 | 1283 | 1641 | 99.7% |
| 2048 | 1591 | 1643 | 99.8% |
| 8192 | 1632 | 1645 | 100% |
| 16384 | 1639 | 1646 | 100% |

Both reach ~1640 GB/s, but TDM is there at 512 blocks while CU needs 8192 to come within 1% of the
same number -- a factor of 16 in grid width. In the linear region a TDM block is worth about
7.9 GB/s against 2.6 GB/s for a CU block, bought with two issuing waves instead of sixteen full ones.

`TDMms` tracks `TDM` closely and leads it slightly below 256 blocks. `TDMc2` ramps slowest because
its loads go down the vector path, and only catches up past 512 blocks.

## Matrix: where the transfer size makes that cost worth paying

Transfer size x grid width, `CUMUL=64` / `TDMMUL=32`, so a point on the axis launches **64 CU blocks
against 32 TDM blocks** -- the TDM side is running half the grid. Full data in
[`results/uamatrix_gfx1250.csv`](results/uamatrix_gfx1250.csv), 189 cells, both transports, with each
side's launched block count in its own columns rather than in a footnote.

At 8 GB, comparing the two at the same point on the axis (TDM on half the blocks):

| axis | CU blocks / TDM blocks | CU | TDM |
|---|---|---|---|
| 1 | 64 / 32 | 170 | 256 |
| 8 | 512 / 256 | 1282 | 1577 |
| 16 | 1024 / 512 | 1474 | 1636 |
| 64 | 4096 / 2048 | 1630 | 1637 |
| 256 | 16384 / 8192 | 1641 | 1641 |

Below about 1 MB neither transport is grid-limited -- widening the grid changes nothing and the two
sit within 30% of each other, TDM ahead at 1 MB (182 vs 195 GB/s at the wide end, where CU is
actually the faster of the two). Above 16 MB the picture is the one the block sweep shows: TDM is at
the ceiling on half the blocks.

## Tile size: what decides how many blocks can participate

A compiled-in tile also decides how much of the grid has anything to do. The payload is cut into
`bytes/tile` tiles, an issuing wave takes `MWSPIPE` of them per round, so a launch can only occupy
`bytes/(tile*MWSISS*MWSPIPE)` blocks no matter how wide it is. At 1 MB and an 8 KB tile that is
8 blocks, which is exactly where the 1 MB row stops responding to the grid.

`DYNTILE=1` sizes the tile per cell instead: `clamp(bytes/(blocks*MWSISS*MWSPIPE), DYNMIN, tile)`,
rounded down to a power of two. Six matrices over the same SIZE x BLOCK axes are in
[`results/tilegeom_gfx1250.csv`](results/tilegeom_gfx1250.csv), 540 cells: the single-issuer baseline
plus five multi-issuer configurations, all in one batch with identical build flags, because the tables
from earlier batches do not subtract against these -- one of them reads exactly 2x on the same cell.

Against that baseline, at 8 GB, dynamic sizing with 8 issuing waves is worth a flat **2.1x from 1 to 64
blocks** (8.3 -> 17.9 at one block, 506 -> 1061 at 64), +56% at 128, and nothing at 256, where both are
at the link ceiling and the baseline is 1.3% ahead (1579 vs 1559). The ceiling was never the thing that
moved; what moved is how few blocks are needed to reach it.

The two changes have to ship together. Eight issuing waves on a *fixed* tile is a regression at mid
sizes -- 1 MB at 256 blocks is 89 GB/s against the baseline's 181 -- because a block consumes
MWSISS*MWSPIPE = 16 tiles per round, so 1 MB feeds 8 blocks, while the single-issuer kernel takes 4 and
feeds 32. Dynamic sizing brings it to 189.

At 256 blocks, across tile policies:

| payload | fixed 8 KB | dynamic, cap 8 KB | fixed 16 KB | dynamic, cap 16 KB |
|---|---|---|---|---|
| 256 KB | 23.0 | **50.4** | 14.3 | 51.2 |
| 1 MB | 89.3 | **189.3** | 55.3 | 189.7 |
| 4 MB | 337 | **546** | 217 | 548 |
| 16 MB | 895 | **1010** | 794 | 1034 |
| 64 MB | 1318 | **1318** | 1201 | 1200 |
| 256 MB | 1474 | **1476** | 1405 | 1402 |
| 8 GB | 1559 | **1559** | 1602 | 1601 |

Dynamic sizing at an 8 KB cap is the one to use: it doubles 256 KB and 1 MB, gains 62% at 4 MB and
13% at 16 MB, and gives up nothing anywhere, at the same 128 KB of LDS. The build differs from the
fixed-8 KB one only by the switch, so that pair isolates the sizing itself.

A 16 KB tile is not worth its LDS. It costs 9% at 64 MB and 5% at 256 MB -- the sizing formula lands
exactly on 16 KB there -- and only pays above 4 GB, by 2.7%, for twice the LDS and one block per CU.

The floor stays at one row. Dropping it to 256 B, which makes the descriptor narrow its row rather
than drop rows, lets 256 KB reach 64 blocks instead of 16 and changes the bandwidth by nothing
(50.35 vs 50.37). Everything at or below ~256 KB is bounded by per-transfer fixed cost, not by tile
granularity: 64 KB at 13 GB/s is 4.9 us, which is the same order as the launch cost this path pays.
Small messages get faster by being batched into fewer launches, not by being cut differently.

## Reproduction

Both tables were re-measured on 2026-08-12 against the earlier runs, on an idle f01-2, after a
`VECADD` health check.

- Block sweep vs the 7/31 run: every point within 0.2% except 256 blocks (0.85%), which sits exactly
  on the knee where the curve is steepest.
- Matrix vs the recorded TDM table: 189 of 189 cells within 3%, worst -2.95% at 16 MB / axis 8, the
  large majority within 1%.

A single re-run cannot separate run-to-run drift from a systematic shift, so the small biases visible
in the matrix comparison (the axis-8 column reads ~1.5% low, the 64-256 MB cells at wide grids ~1-2%
high) are not attributed to anything here.

## Running it

```bash
# On a node whose GPUs are idle. Both ranks are local (GPU 0 -> GPU 1) over a socket on 127.0.0.1.
bash tools/blksweep.sh
bash tools/uamatrix.sh
```

Both refuse to start if a previous run is still alive or if the LDS preflight fails. Knobs:
`GRID`, `BASEX`, `GPUA`/`GPUB` or `GSRC`/`GDST`, `ARCH`; plus `ROUNDS`/`BLKS` for the block sweep and
`CUS`/`SZS`/`CUMUL`/`TDMMUL`/`BUDGET`/`MAXB` for the matrix. `PREFLIGHT_ONLY=1` stops after the check.

To build by hand:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx1250 ualoe_bw.cpp -o ualoe_bw
./ualoe_bw listen  -port=55637 -gpu=0 &     # omit -gpu to use every local GPU as a pair
./ualoe_bw connect 127.0.0.1 -port=55637 -gpu=1
```

## Before changing the geometry

`tools/lds_preflight.sh` checks the LDS budget of every kernel the sweeps launch. Run it after any
change to `RTD0N`, `RTD1N`, `RPIPEN`, `LDSPART`, `MWSSPAN` or `MWSPIPE`:

```bash
RTD1N=16 bash tools/lds_preflight.sh
```

It checks two different things. The first is the 320 KB per CU limit, and exceeding that is harmless:
the launch fails with an error. The second is that each per-wave LDS partition is wide enough for the
tiles the kernel puts inside it, and that one is **silent** -- shrinking the constant shrinks the
allocation but not the addressing, so the kernel hands an out-of-range LDS offset to the TDM engine
and the process wedges in a D state that outlives `kill -9`. A node was lost that way on 2026-08-11
by raising the tile to 16 KB while leaving the partition at 16 KB, which is what the command above
reproduces as a refusal.

## Comparing numbers across runs

Only within one process. Each sweep measures all of its transports back to back in a single run for
exactly this reason: `LOOP`, build flags and clock state move the absolute numbers by more than the
differences being tested. Two tables built with different `LOOP` values are not comparable even when
they came from the same source file -- the block sweep uses `LOOP=10`, the matrix derives its
iteration count from `BUDGET`, and neither is comparable to a `LOOP=50` table.
