# xGMI copy bandwidth: TDM vs CU

How much of a GPU it costs to saturate one xGMI link, measured as a function of grid width.

The question this answers is not "which transport is faster" -- at a full grid they land within 0.5%
of each other, both at the link ceiling. It is **how much of the GPU each one has to occupy to get
there**, because whatever is left over is what a fused kernel can use for compute.

## What is measured

`ualoe_bw.cpp` copies a payload from one GPU to a peer over xGMI and reports one-way GB/s measured on
the pushing side, with a barrier on each end so the timed region contains only the copy. Four
transports are compared at each block count:

| kind | name | how it moves data |
|---|---|---|
| 0 | `copyk` | CU vector copy, `uint4` loads and stores, every thread in the block moves data |
| 1 | `tdm_write` | TDM (tensor DMA) descriptors, staged through LDS; one issuing wave per block |
| 8 | `tdm_store_cuload` | CU-staged: vector loads into LDS, TDM stores out |
| 9 | `tdmmws` | staged multi-issuer, `MWSISS` issuing waves per block |

The block is deliberately *not* normalised into equivalent hardware. The CU kernel runs 512 threads
per block all moving data; the TDM kernel runs 256 threads per block of which two waves issue
descriptors and the remaining six wait on a barrier. That asymmetry is the result.

## Result (gfx1250, 16 GB payload, 2026-08-12)

Raw data in [`results/blksweep_16gb_gfx1250.csv`](results/blksweep_16gb_gfx1250.csv), both rounds.

| blocks | CU | TDM | TDM as % of its own ceiling |
|---|---|---|---|
| 128 | 337 | 999 | 61% |
| 256 | 666 | 1582 | 96% |
| 512 | 1283 | 1641 | 99.7% |
| 2048 | 1591 | 1643 | 99.8% |
| 8192 | 1632 | 1645 | 100% |
| 16384 | 1639 | 1646 | 100% |

Both transports reach ~1640 GB/s, but TDM is there at 512 blocks while CU needs 8192 to get within
1% of the same number -- a factor of 16 in grid width. In the linear region each TDM block is worth
about 7.9 GB/s against 2.6 GB/s for a CU block, and it buys that with two issuing waves instead of
sixteen full ones.

`TDMms` (multi-issuer) tracks `TDM` closely and leads it slightly below 256 blocks; `TDMc2` is the
slowest to ramp because its loads go down the vector path, and it only catches up past 512 blocks.

Reproduced from a 7/31 run of the same sweep: every point agrees within 0.2% except 256 blocks, which
differs by 0.85% and sits exactly on the knee, where the curve is steepest.

## Running it

```bash
# On a node whose GPUs are idle. Both ranks are local (GPU 0 -> GPU 1) over a socket on 127.0.0.1.
bash tools/blksweep.sh
```

`tools/blksweep.sh` refuses to start if a previous run is still alive or if the LDS preflight fails,
then compiles and runs both sides. Knobs: `GRID`, `BASEX`, `ROUNDS`, `BLKS`, `GPUA`, `GPUB`, `ARCH`.

To build by hand:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx1250 ualoe_bw.cpp -o ualoe_bw
./ualoe_bw listen  -port=55637 -gpu=0 &     # omit -gpu to use every local GPU as a pair
./ualoe_bw connect 127.0.0.1 -port=55637 -gpu=1
```

## Before changing the geometry

`tools/lds_preflight.sh` checks the LDS budget of every kernel the sweep launches. Run it after any
change to `RTD0N`, `RTD1N`, `RPIPEN`, `LDSPART`, `MWSSPAN` or `MWSPIPE`:

```bash
RTD1N=16 bash tools/lds_preflight.sh
```

It checks two different things. The first is the 320 KB per CU limit, and exceeding that is harmless:
the launch fails with an error. The second is that each per-wave LDS partition is wide enough for the
tiles the kernel puts inside it, and that one is **silent** -- shrinking the constant shrinks the
allocation but not the addressing, so the kernel hands an out-of-range LDS offset to the TDM engine
and the process wedges in a D state that outlives `kill -9`. A node was lost that way on 2026-08-11
by raising the tile to 16 KB while leaving the partition at 16 KB.

## Comparing numbers across runs

Only within one process. The sweep measures all four transports back to back inside a single run for
exactly this reason: `LOOP`, build flags and clock state move the absolute numbers by more than the
differences being tested. Two tables built with different `LOOP` values are not comparable even if
they came from the same source file.
