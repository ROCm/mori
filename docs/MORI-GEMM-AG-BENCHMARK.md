# mori.ops.gemm_ag — fp8 GEMM fused with an all-gather

Measurements for `mori.ops.gemm_ag`, the row-concatenating all-gather fused into
the epilogue of mori's 8-wave fp8 GEMM.

**BF16 performance update (2026-09-28):** the repaired pure GEMM has now been
compared directly with native BF16-input/FP32-output `torch.mm`. At
M=N=2048, K=7168, Torch takes 71.1 µs; MORI takes 131.9 µs at the default
`256x256` tile, 88.6 µs at `128x256`, and 72.0 µs at `128x128`.
See *Native BF16-to-FP32 GEMM comparison after repair* for the paired protocol.
The historical 107.2 µs Torch baseline below includes a BF16-to-FP32 output
conversion and is not the native FP32-output baseline.

**BF16 GEMM + AG update (2026-09-28):** at that same shape, the fastest
measured path is `128x128` GEMM followed by SDMA, **364.0 µs**, against native
Torch GEMM + RCCL at **397.5 µs**. Fusing SDMA takes **377.0 µs** even at its
best chunk count. See *GEMM + all-gather comparison after repair* for the full
matrix and the multi-queue completion correction included in these results.

**Follow-up optimization experiments:** single-counter election and metadata
changes did not materially improve end-to-end time. System-scope C stores
with a release-only fence reduced monolithic fused SDMA to about 365–366 µs.
A four-chunk MORI split-K pipeline showed a 1.43% gain in two final paired
repeats; a Torch small-M pipeline reached about 350 µs. These are experimental
alternatives, detailed in [the optimization report](MORI-GEMM-AG-OPT-EXPERIMENTS.md),
which also separates the lossy-wire results from the same-precision comparison.

**Headline: fusing an all-gather does not beat not fusing it, at any shape
measured here.** The best fused configuration (`fused-sdma --chunks 1 --post
lanes`) lands at 105.5us against `split-sdma`'s 104.1 at `r4 M=512`, and 217.2
against 210.5 at `M=2048`. The split paths themselves are worth having — they
beat RCCL by 10–25% — but the epilogue is not.

Getting to that statement took two rounds, and the first round's explanation was
wrong in an instructive way:

* The epilogue as first written issued a chunk's `world-1` SDMA packets back to
  back from one thread, which cost the GEMM **39–55us** — SDMA's ~2us per packet,
  reproduced here by being flat across payloads differing 8x. `--post lanes`
  issues them from `world-1` lanes instead and removes that entirely: the fused
  GEMM phase goes flat at 57–72us across every chunk count, against 105–169 for
  the serial form. See *`--post lanes`*.
* That fix also **invalidated the first round's headline claim.** With serial
  posting the drain looked like it was shrinking as chunks grew — read as the
  fusion absorbing 25–59% of the communication. It was not: the GEMM was stalled
  posting, the copy engines got a head start, and the phase split booked that as
  absorbed comm. With the posting fixed the drain stops shrinking and starts
  growing, and the real overlap at the best setting is about 5%.

What is left is small, understood, and mostly structural: ~10us of epilogue
bookkeeping against ~7.5us of genuine early-start on the wire, under a ceiling
derived in *The bound fusing runs into* that is 27% at `M=2048` and falls as the
world grows. **The lever worth pulling next is a narrower wire, not a better
epilogue.**

**A second precision was added afterwards — bf16 in, fp32 out, the one DeepSeek
V4-Pro's `wkv_gate` actually uses.** Fusing **breaks even** there rather than
losing, because the bf16 GEMM is larger and there is correspondingly more
compute to hide behind. The ported GEMM is 28% behind `torch.matmul`, which
makes the operator a net loss at the target shape; a tile sweep shows the cause
is grid occupancy and that fixing it is worth 33% — but the tile that does so
also exposes a latent correctness bug and therefore does not ship. See *BF16 in,
FP32 out* and *Tuning the bf16 GEMM*.

## What the operator is

Every rank holds its own `A [M, K]` — its shard of the tokens — and a replicated
`B [N, K]`, computes the whole `C = A @ B.T` as `[M, N]` bf16, and every rank
ends up with all `world` of them concatenated along rows: `[world*M, N]`, source
rank `r` at rows `[r*M, (r+1)*M)`.

The shape it is aimed at is the DeepSeek V4-Pro prefill context-parallel
`wkv_gate` chain: `K = 7168` is the hidden size, `N` is 2048 on the ratio-4
layers and 1024 on the ratio-128 ones, and **`M = tokens/P`** — so sweeping `M`
is sweeping the sequence length. Each CP rank scores its own tokens and then
every rank needs every token's score.

| mode | what it does |
|---|---|
| `gemm-only` | the GEMM alone into a plain tensor, to size the ceiling |
| `gemm-to-window` | the same kernel, writing this rank's `recv` slot instead |
| `split-rccl` | `gemm-only()`; then `all_gather_into_tensor` |
| `split-lsa-push` | `gemm-to-window()`; a copy kernel vector-stores into every peer |
| `split-lsa-pull` | `gemm-to-window()`; a copy kernel loads from every peer |
| `fused-lsa` | C stored straight into every peer from the epilogue |
| `split-sdma` | `gemm-to-window()`; one copy-engine push per peer |
| `fused-sdma` | C into this rank's own slot, pushed per chunk from the epilogue |

Every mode above runs **the same GEMM**. `gemm-only` and the split paths run it
byte for byte — only the C pointer differs, which is what `gemm-to-window`
isolates — so the differences between them are transport and nothing else.

## How this differs from `gemm_a2a`, structurally

All-gather shards nothing, and three simplifications follow. They are worth
stating because they are also why the *fused* side has less to work with.

1. **No staging region.** `gemm_a2a` writes `[dst][M][shard_n]` in its epilogue
   purely so a copy engine has one contiguous source range per destination, and
   reserves a second full payload of window to hold it. All-gather's payload is
   the rank's whole `[M, N]`, already contiguous and already identical for every
   destination, so the GEMM writes straight into its own `recv` slot and that
   slab is pushed unchanged.
2. **One counter per chunk, not per (destination, chunk).** A chunk completing
   arms all `world-1` pushes at once.
3. **No tile rotation.** `gemm_a2a` and `gemm_ar` walk destinations round-robin
   so the last one's link does not idle until the end of the GEMM. A broadcast
   tile belongs to everyone, so the default `block_m`-outer order already gives
   every link chunk 0 at the same moment.

The price is on the wire: a rank sends `(world-1) * M * N * 2` bytes, **`world`
times what the all-to-all sends at the same `[M, N]`**. All-gather is the
bandwidth-bound member of the family.

## Conditions

8x MI355X (gfx950), fp8 e4m3 in / bf16 out, CUDA graph replay, median of 21
iterations after 10 warmups, max over ranks. Tile is 128x256 under `--quant
ptpc` and 256x256 under `--quant mxfp8`, which needs `BLOCK_M=256` for the scale
operands. Every launch waits for an idle box; any sample taken with a neighbour
present is discarded rather than averaged in, and clock and junction temperature
are recorded next to every measurement. That protocol is `gemm_a2a`'s and its
rationale is in that file — a neighbour is worth 9% on a kernel that
communicates nothing.

## Results

`--quant ptpc`, `K = 7168`, `world = 8`. `gemm` and `comm` are the same-run
phase split (`--phase-split`): the GEMM is re-timed inside the same process
group, because "mode total minus a separate `gemm-only` run" charges the
transfer for the clock ramp between two process groups. `fused-lsa` has no
collective kernel to split against, so only its total is shown.

### wkv_gate ratio-4 — `N=2048` — 14 MiB/rank on the wire at M=512, 56 at M=2048

| mode | M=512 total | gemm | comm | M=2048 total | gemm | comm |
|---|---:|---:|---:|---:|---:|---:|
| `gemm-only` | 53.4 | — | — | 61.7 | — | — |
| `split-rccl` | 117.4 | 52.5 | 64.9 | 233.4 | 56.3 | 175.2 |
| `split-lsa-push` | **96.6** | 52.6 | 44.0 | 213.8 | 56.4 | 155.0 |
| `split-lsa-pull` | 103.4 | 52.3 | 51.0 | 217.9 | 56.1 | 158.0 |
| `split-sdma` | 105.3 | 52.8 | 50.6 | **211.2** | 56.6 | 153.9 |
| `fused-lsa` | 114.6 | — | — | 268.2 | — | — |
| `fused-sdma` `c4` `serial` | 127.2 | 105.8 | 20.8 | 235.4 | 117.2 | 115.4 |
| `fused-sdma` `c1` `lanes` | 105.5 | 57.4 | 44.7 | 217.2 | 66.6 | 146.6 |

### wkv_gate ratio-128 — `N=1024` — 7 MiB/rank at M=512, 28 at M=2048

A shape `gemm_a2a` cannot run at all: `a2a_config` requires `N` to be a
multiple of `world*block_n = 2048`.

| mode | M=512 total | gemm | comm | M=2048 total | gemm | comm |
|---|---:|---:|---:|---:|---:|---:|
| `gemm-only` | 41.4 | — | — | 53.6 | — | — |
| `split-rccl` | 89.1 | 40.2 | 45.3 | 158.0 | 52.6 | 102.8 |
| `split-lsa-push` | **67.3** | 40.6 | 26.7 | **137.8** | 53.1 | 84.7 |
| `split-lsa-pull` | 77.7 | 40.0 | 37.8 | 143.5 | 52.9 | 87.6 |
| `split-sdma` | 77.6 | 40.3 | 34.3 | 140.5 | 53.0 | 87.5 |
| `fused-lsa` | 87.6 | — | — | 154.9 | — | — |
| `fused-sdma` `c4` `serial` | 108.3 | 88.3 | 17.3 | 162.0 | 108.1 | 53.7 |

### Where the fused epilogue's time goes

`--no-put` runs the whole epilogue — the release fence, the completion counter,
the submit lock — and simply does not post, so it prices the bookkeeping apart
from the transfer. `--fence none` drops the release. Both are measured against
`--post serial`, the original epilogue.

| cell | split `gemm` | fused `gemm` | `--no-put` `gemm` | bookkeeping | the puts |
|---|---:|---:|---:|---:|---:|
| r4 @ M512 | 52.8 | 105.8 | 66.8 | +14.0 | **+39.0** |
| r4 @ M2048 | 56.6 | 117.2 | 62.3 | +5.7 | **+54.9** |
| r128 @ M512 | 40.3 | 88.3 | 42.2 | +1.9 | **+46.1** |
| r128 @ M2048 | 53.0 | 108.1 | 55.2 | +2.2 | **+52.9** |

`--fence none` moves the total by −2.5 to +0.5us — inside the noise at every
cell. **The release fence is not the cost here**, which is worth saying because
`gemm_a2a` calls it "the single most expensive line in this epilogue". The
difference is grid size: a2a's fused path at M=16384 has 9216 blocks each doing
a whole-L2 writeback, and these cells have 32 to 128.

### `--post lanes`: issuing the packets concurrently

The 39–55us above is `world-1` SDMA packets issued back to back by one thread,
at SDMA's ~2us each. `--post lanes` gives each of lanes `0..world-2` one packet.
It needs one change to make that possible: a counter per `(chunk, lane)` rather
than one per chunk, so each lane learns from **its own** atomic that its block
won and no broadcast is needed. It also drops the submit lock — `world-1` lanes
retire in lockstep, so two waves each holding part of a lock set could never
reach their release — and separates concurrent chunks by queue instead, which is
why it requires `sdma_queues >= chunks`.

`--quant ptpc`, `r4` (`N=2048 K=7168`), `us` as total / GEMM phase / drain:

| | M=512 serial | M=512 **lanes** | M=2048 serial | M=2048 **lanes** |
|---|---|---|---|---|
| `split-sdma` | 104.1 / 52.6 / 51.4 | — | 210.5 / 56.4 / 154.1 | — |
| `c1` | 113.1 / 67.4 / 45.7 | **105.5** / 57.4 / 44.7 | 224.1 / 77.1 / 145.1 | **217.2** / 66.6 / 146.6 |
| `c2` | 115.5 / 81.1 / 34.3 | 112.9 / 57.9 / 55.0 | 227.5 / 89.9 / 137.4 | 227.0 / 66.7 / 160.3 |
| `c4` | 125.9 / 105.9 / 19.7 | 128.3 / 59.8 / 64.7 | 233.5 / 116.0 / 116.2 | 243.7 / 68.5 / 172.3 |
| `c8` | 126.2 / 105.2 / 21.0 | 124.3 / 60.4 / 63.2 | 248.9 / 168.8 / 79.8 | 265.0 / 71.9 / 188.5 |

**It does exactly what it was meant to.** The fused GEMM phase goes flat: 57.4 →
60.4 across `c1..c8` at M=512 and 66.6 → 71.9 at M=2048, where serial climbs to
105.2 and 168.8. At `M=2048 c8` that is **−96.9us of GEMM time**, and the whole
39–55us penalty is gone at every chunk count. Best total improves 113.1 → 105.5
(−6.7%) and 224.1 → 217.2 (−3.1%).

**And it still does not beat `split-sdma`** — 105.5 against 104.1, and 217.2
against 210.5. Two things are now visible that the serial numbers hid:

* **The "absorbed communication" in the serial rows was largely an artifact.**
  Serial's drain shrinks with chunk count (145.1 → 79.8 at M=2048) because its
  GEMM was stalled ~100us posting, which gave the copy engines a head start that
  the phase split books as absorbed comm. With the posting fixed, the drain
  stops shrinking and starts *growing* (146.6 → 188.5). The earlier claim that
  `fused-sdma` absorbs 25–59% of the transfer does not survive this: at `c1`,
  the only configuration that wins, the real overlap is 154.1 → 146.6, about 5%.
* **SDMA's ~2us per packet does not disappear when the packets are issued
  concurrently — it moves from the issue side to the wire.** At `M=2048 c8`
  there are `8 * 7 = 56` packets of 1 MiB each, and the drain is 188.5us against
  `c1`'s 146.6 for exactly the same bytes. Chunking an all-gather multiplies the
  packet count by `world-1` per chunk, because every peer gets the same slab.

So the remaining gap at the best setting is small and fully accounted for: about
10us of epilogue bookkeeping (GEMM 66.6 against `split-sdma`'s 56.4) against
about 7.5us of genuine early-start on the wire.

### The all-gather quantisation matrix

Every mode under every quantisation, at `wkv_gate-r4 M=2048`. All validate at
relL2 1.66e-3.

| mode | ptpc | blockscale | mxfp8 |
|---|---:|---:|---:|
| `gemm-only` | 57.3 | 96.3 | 86.3 |
| `gemm-to-window` | 57.5 | 96.1 | 92.7 |
| `split-rccl` | 231.8 | 266.9 | 266.0 |
| `split-lsa-push` | 212.6 | 248.3 | 244.2 |
| `split-lsa-pull` | 217.0 | 251.0 | 248.8 |
| `fused-lsa` | 265.8 | 328.1 | 292.4 |
| `split-sdma` | 212.5 | 245.4 | 242.2 |
| `fused-sdma` `c1` | 223.6 | 259.1 | 250.7 |

All twelve `fused-sdma --post lanes` cells (three quantisations x `--chunks
{1,2,4,8}`) validate at the same relL2 as every other mode, so the counter
change that made lane-parallel posting possible is exercised across the matrix
and not only on the default.

## Reading the tables

**Writing the window costs nothing.** `gemm-to-window` is within 0.4% of
`gemm-only` at ptpc. The split paths' GEMM really is `gemm-only`'s kernel, so
"split total minus GEMM" is a clean measure of the transport.

**Every mori transport beats RCCL**, by 10–25%. That comparison is as clean as
it gets: `split-rccl` and `split-sdma` run the same GEMM over the same bytes in
the same layout, and only the collective differs.

**Push beats pull, everywhere — the opposite of `gemm_ar`'s gather leg**, which
found pull worth 5.6–10.5% over an SDMA push. The margin here is 2–29% on the
comm phase and is largest at the small shapes. The GEMM phase is equal to within
0.5us, so pull's mandatory `sc0|sc1` C store is not the cause. The likely reason
is the obvious one: a push is a fire-and-forget store and a pull is a load that
must round-trip, so pull pays fabric latency that push hides. gemm_ar's
contrary result was against a *copy-engine* push, not an LSA one.

This matters beyond the ranking, because pull is where a low-precision wire
would want its dequantize — in registers on the consumer, which is where
`gemm_ar` measured it as free against a 61.0us widen kernel. That trade is now
quantified: choosing pull for the sake of a cheap dequantize starts 2–29% behind.

**Issuing the puts cost the GEMM 39–55us, and that is fixed.** The figure was
nearly constant across four cells whose payloads differ by 8x, which ruled out
DMA bandwidth contention and pointed at a fixed per-packet cost — `gemm_ar`'s
layout gives the constant independently, SDMA at *"~2us per packet regardless of
size"*. `--post lanes` issues them concurrently and the penalty is gone; the
table in *`--post lanes`* has the before and after.

**Chunking still does not pay, but now for a different reason.** The ~2us per
packet does not vanish when the packets are issued concurrently — it moves to
the wire. Chunking an all-gather multiplies the packet count by `world-1` per
chunk, because every peer receives the same slab, so `c8` at M=2048 puts 56
packets of 1 MiB on the wire and the drain grows from 146.6us to 188.5us for
identical bytes. `--chunks 1` is the best fused setting at every cell measured.

**Which leaves the fused path at parity, not ahead.** About 10us of epilogue
bookkeeping against about 7.5us of genuine early-start. Both are small, and the
ceiling above them is 27% at M=2048 and falling with the world size.

**`fused-lsa` is the slowest mode at every cell**, 8–26% behind its own split
baseline. Its epilogue stores each tile `world` times from inside the GEMM, so
the CUs carry all `(world-1)*M*N*2` bytes of the transfer while also running the
MFMA pipeline — where the split path hands that traffic to a kernel that is
doing nothing else. This is the cost `peer_rsrcs` was added to minimise (one
scale load, one convert, one shuffle, `world` stores) and minimising it is not
enough.

## BF16 in, FP32 out — the precision the model actually uses

Everything above is the fp8 GEMM, bf16 out. The layer this operator exists for
is not that: `通算融合方向规划.md` A.4 records DeepSeek V4-Pro's `wkv_gate` going
through `linear_bf16_fp32`, and says the first implementation must keep the
original input precision and FP32 output convention rather than substituting the
existing FP8 GEMM. `--in-dtype bf16 --out-dtype {bf16,fp32}` is that path:
`mori.ops.gemm_ag._gemm_a16w16_8wave`, a port of gcnasm's
`gemm_a16w16_quad_subtile_kernel_template.hpp` at its own instantiation
`<512, 256, 256, 64, bf16, bf16, bf16, float>`.

It is a small port because **`gemm_ar/_gemm_a8w8_8wave.py` is already a FlyDSL
rendering of that same template** with fp8 operands — same 8 waves, same 2x2
half-tiles, same swapped-AB MFMA, same permlane store. Four things change: the
MFMA (`16x16x32 bf16`), `E_K = 2` instead of 1, `VEC = 8` instead of 16, and no
scales at all. The LDS keeps mori's XOR swizzle, restated in bytes
(`swizzle_row128b`), which is asserted to be the fp8 function value-for-value at
`elem_bytes = 1`.

### The FP32 store is the cheap one

The A/B swap puts **four consecutive columns** in each lane. At two bytes that is
8 B — too narrow, which is the entire reason the bf16 path pays for a
`permlane16_swap` to pair two N-tiles into 16 B. At four bytes those same four
columns are already 16 contiguous bytes, so the FP32 store is a
`buffer_store_dwordx4` with **no shuffle and no convert**. Lane group
`g = lane//16` takes `col = base_col + g*4`, and the four groups still cover 64
contiguous bytes per row.

### Accuracy

| | relL2 vs exact fp32 |
|---|---:|
| fp8 in, bf16 out | 1.66e-3 |
| bf16 in, bf16 out | 1.66e-3 |
| **bf16 in, fp32 out** | **9.9e-7** |

Three orders of magnitude, and it shows what the 1.66e-3 actually was: the *bf16
output rounding*, not the fp8 input. Keeping fp32 all the way out is worth far
more numerically than keeping fp8 out of the inputs — which is what
`linear_bf16_fp32` is for.

### Results, `K=7168 N=2048`, 8 ranks

`gemm` / `comm` are the same-run phase split. Wire bytes per rank are in the
heading; fp32 doubles them.

| mode | M512 bf16 (14 MiB) | M512 fp32 (28 MiB) | M2048 bf16 (56 MiB) | M2048 fp32 (112 MiB) |
|---|---|---|---|---|
| `gemm-only` | 125.1 | 129.0 | 132.5 | 137.6 |
| `split-rccl` | 187.9 / 123.9 / 64.0 | 230.6 / 127.8 / 100.3 | 305.5 / 129.2 / 176.3 | 462.9 / 132.4 / 327.8 |
| `split-lsa-push` | **171.5** / 124.3 / 44.0 | 213.2 / 128.7 / 81.9 | 286.6 / 129.1 / 155.8 | 444.2 / 135.0 / 307.5 |
| `split-lsa-pull` | 179.1 / 124.3 / 49.7 | 220.2 / 128.8 / 91.4 | 323.1 / 130.8 / 191.2 | 442.7 / 134.2 / 308.3 |
| `split-sdma` | 176.1 / 124.6 / 51.3 | **217.2** / 130.2 / 83.7 | **284.4** / 131.9 / 152.5 | **427.6** / 134.1 / 293.4 |
| `fused-lsa` | 202.0 | 245.8 | 344.5 | 616.4 |
| `fused-sdma` | 176.1 / 128.4 / 46.7 | 217.7 / 132.9 / 81.8 | 285.2 / 136.6 / 146.0 | 429.5 / 138.1 / 290.7 |

**The GEMM is 2.2x the fp8 one** (132.5 against 60.5 at M=2048) and that is
expected rather than a defect: `16x16x32` consumes a quarter of the K per
instruction that the fp8 `16x16x128` does. Output width barely touches it
(132.5 → 137.6 for fp32), because the store is a small part of a K=7168 GEMM.

**Fusing now breaks even instead of losing.** `fused-sdma` lands within 0.5% of
`split-sdma` at all four cells, where under fp8 it was 3–7% behind — and its
comm phase is consistently lower (46.7 vs 51.3, 81.8 vs 83.7, 146.0 vs 152.5,
290.7 vs 293.4), so ~4–9% of the transfer really is absorbed, against an epilogue
cost of ~4 µs. This is the `G₀/C₀` bound moving in the right direction for the
reason *The bound fusing runs into* gives: a 2.2x larger GEMM is 2.2x more
compute to hide behind, and it is the only lever besides `K` and `world`.

**`fused-lsa` gets much worse, and fp32 makes it dramatically worse** — 616.4
against `split-sdma`'s 427.6, 44% behind. Its epilogue pushes `world` copies of
every output element through the CUs while they are also running MFMAs, so
doubling the element width doubles exactly the traffic that was already the
problem.

### Cross-check against the plan document's FP32 baseline

A.4's row for this exact configuration (`wkv_gate-r4`, 16384 tokens over 8
ranks, FP32) is `torch.matmul` 107.2 µs + RCCL all-gather 339.2 µs = 415.0 µs.

| | A.4 (`torch.matmul` + RCCL) | this operator (bf16 MFMA + RCCL) |
|---|---:|---:|
| GEMM | 107.2 | 137.6 |
| all-gather | 339.2 | 327.8 |
| total | **415.0** | **462.9** |

**The collective leg agrees to 3.4%**, which is the check that matters — it says
the two measurements are of the same thing. **The GEMM does not: ours is 28%
slower than `torch.matmul`.** That is a real gap and it is not explained away by
the port being faithful; the kernel is simply untuned at this precision
(`BLOCK_K` is pinned to the template's 64, `xcd_swizzle` is off, `waves_per_eu`
is at its default, and no tile sweep has been run). Against mori's best transport
the total is 427.6 against A.4's 415.0, so **at this shape the operator is not
yet a win over `torch.matmul` + RCCL, and the deficit is entirely in the GEMM.**
**The cause is now known — see *Tuning the bf16 GEMM*: the template's tile
leaves three quarters of the GPU idle at this shape, and a smaller one takes the
GEMM to 91.7us and the total to 424.3 against A.4's 415.0. It does not ship,
because it also makes about a third of launches produce a wrong answer.** So
this paragraph still stands as the shipped state.

### One intermittent failure, unresolved

`fused-sdma --in-dtype bf16 --out-dtype bf16` at `M=2048` failed validation once
during the sweep and has not reproduced since: **13 subsequent runs of that exact
cell all passed**, so it stands at 1 in 14. The other 31 cells of the matrix
validated first time.

It is recorded rather than dismissed. `gemm_ar` shipped two real races behind
one-shot checks that passed repeatedly, and this path is new on three axes at
once (per-(chunk, lane) counter, lock-free lane posting, a different mainloop),
so "did not reproduce in 13 tries" is not "is not there". Anyone taking this
path to production should run the cell a few hundred times first; the
`--chunks` ladder is the obvious place for a race to hide and was not swept on
the bf16 path at all.

## Tuning the bf16 GEMM: the tile is almost the whole story

The section above closes on the bf16 GEMM being 28% behind `torch.matmul`,
called out as the largest number on the table. The cause was not the mainloop.

**The template's `256x256` tile launches 64 workgroups. An MI355X has 256 CUs.**
Three quarters of the GPU idles at the `wkv_gate` shape regardless of how good
the inner loop is, and nothing inside the loop can reach that. Measured at
`M=2048 N=2048 K=7168`, fp32 out, one GPU, against `torch.matmul` at 84.8us:

| tile | workgroups | us | TF/s |
|---|---:|---:|---:|
| `256/256` (the template's) | 64 | 176.0 | 342 |
| `256/128` | 128 | 139.6 | 431 |
| `128/256` | 128 | **134.2** | **448** |
| `128/128` | 256 | 116.5 | 516 |

`waves_per_eu` (1/2/4) and `xcd_swizzle` (0/1/4/8) are both **inert** — every
cell within 1%, which is the noise floor. The tile is the knob.

### The rule is not "maximise the grid"

| M | grid at `128/256` | us | TF/s | `torch.matmul` | ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 128.4 | 117 | 39.0 | 3.29x |
| 1024 | 64 | 128.4 | 234 | 53.1 | 2.42x |
| 2048 | 128 | 134.2 | 448 | 84.8 | 1.58x |
| 4096 | 256 | 155.2 | 775 | 122.9 | 1.26x |

At `M=4096`, `128/128` gives 512 workgroups and measures 162.8us against
`128/256`'s 256 workgroups at 155.2. Once there is at least one workgroup per
CU, the larger tile amortises better. So `pick_tile` takes **the largest tile
whose grid still covers the CUs, or the largest grid available if none does**,
which reproduces the measured best at every shape.

### Two tiles that are faster and do not ship

`128/128` is the fastest thing measured (116.5us, 516 TF/s) and is **excluded**.
It races: one wrong fp32 C in 48 single-GPU runs (relL2 1.4e-2 against the usual
9.9e-7), and under an 8-rank launch roughly half the ranks failed, a different
half each time. `128/256` is 48/48 clean on the same probe. The suspected cause
is the `wait_barrier` vmcnt thresholds, which are written in terms of
`N_LDS_STEPS_A/B` and only ever exercised at the step counts `BLOCK_N=256`
produces; at `BLOCK_N=128` both fall to 1. Unconfirmed.

`BLOCK_M=128` does not ship either, and that is the unhappy part. It is 24%
faster at the target shape and it is not reliable. Repeat counts at `M=2048`
fp32, all at `BLOCK_M=128`, in the order they were taken:

| mode | first pass | later pass |
|---|---:|---:|
| `gemm-only` | 4/4 | 2/2 |
| `gemm-to-window` | 4/4 | **1/2** |
| `split-rccl` | 3/3 | **1/2** |
| `split-lsa-push` | 4/4 | — |
| `split-sdma` | 3/3 | — |
| `split-lsa-pull` | 3/4 | — |
| `fused-sdma` | 1/3 | — |
| `fused-lsa` | **0/4** | — |

`BLOCK_M=256` has not failed once across every repeat run in this section. So
the failure rate at 128 is somewhere around a third of launches, and the "sound
/ unsound by mode" split that the first pass seemed to show **did not survive
repetition** — `gemm-to-window` and `split-rccl` looked clean at 4/4 and 3/3 and
then came back 1/2.

### What the earlier ISA audit established — and missed

The initial wait-policy sweep at `BM=128`, `gemm-only`, six independent launches
per policy, found 5/6 passing at `vmcnt(4)`, 4/6 at `vmcnt(5)`, and 6/6 at
`vmcnt(0)`. Full draining cost about 150 µs against 90–91 µs for the partial
waits. `--fence all` and `--direct-fence all` did not fix the failure.

Disassembling `835783c7` established these facts at M=N=2048, K=7168, FP32 out:

| | BM=128 | BM=256 |
|---|---:|---:|
| S2R | 16 `ds_read_b128` / iteration | 24 / iteration |
| G2S | 6 `buffer_load_dwordx4 ... lds` / iteration | 8 / iteration |
| mainloop threshold | `vmcnt(4)` | `vmcnt(6)` |
| VGPR / LDS | 152 / 96 KiB | 248 / 128 KiB |
| register spills | 0 | 0 |

S2R belongs to **lgkmcnt**, not vmcnt. Changing the combined wait/barrier asm
to a builtin barrier produced identical ISA in that experiment. Neither fact
proves that the *placement* of the waits is correct. In particular, the previous
conclusion that “the ISA clears the kernel” was too strong: it counted one
wave's instructions without checking the rendezvous of the two staggered wave
groups, and did not check the peeled tail's read-after-write dependencies.

### Two missing ordering edges

**B0 reuse in the mainloop.** Waves 4–7 execute an extra prologue barrier, so
waves 0–3 stay one barrier ahead. The barrier following `c00` in waves 0–3 meets
the barrier following the B0/A0 S2R issues in waves 4–7. On release, waves 0–3
can issue `B0@k+2` into the old B0 buffer while waves 4–7 still have B0 reads in
flight. LLVM's `lgkmcnt` waits inside `mfma.call` come *after* that rendezvous.
They protect MFMA operand use, but are too late to protect the LDS buffer from
an overwrite by another wave. The C++ template has an explicit LDS wait at this
boundary; the combined B0/A0-read port did not.

The repair waits for `lgkmcnt(0)` **before** releasing that existing barrier.
It drains A0 as well, so correctness does not depend on LLVM preserving the
relative instruction order of B0 and A0 reads. The wait has a memory clobber.
No extra workgroup barrier is introduced.

**The last K tile.** The last mainloop wait intentionally leaves final-tile
prefetches in flight. In the old tail, B0 and A0 of that final tile were read
before the remaining `vmcnt(0)`. In the saved BM=128 ISA, B0 reads start at
`0x120a4`, A0 reads at `0x12118`, and the drain is only at `0x12138`. Waiting
there cannot repair values already read from LDS. Move this drain to the
existing barrier immediately after issuing the last A1 prefetch. The next
barrier after `c10` then also rendezvous with the other group's drain, before
either group reads the final B0/A0. Barrier count and the wave staggering stay
unchanged.

There is also a threshold bound for the explicit `256x128` tile: after
`A1@k+1`, the newly issued `B0/A0/B1@k+2` account for `A+2B` instructions.
`tuned` now uses `min(2A+B, A+2B)`, which changes this tile's threshold from 5
to 4 and preserves the values at `128x256` and `256x256`. The `safe` and
`conservative` diagnostic choices remain available.

`test_gemm_ag_pipeline.py` traces the actual Python pipeline and checks G2S
completion before S2R issue, and S2R completion before buffer reuse, using
barrier **ordinals** across all eight waves. It assumes FIFO G2S completion;
it is a dependency check, not a latency simulator or a substitute for hardware
validation. The old source exposes both hazards even under that favorable FIFO
assumption. The repaired source passes all four tile geometries, K iteration
counts 2 through 6, and all three wait policies. Mutation controls remove each
repair and confirm that its corresponding hazard returns.

The repaired BM=128 ISA retains 110 mainloop `vmcnt(4)` instructions, 152 VGPRs,
96 KiB LDS, and zero spills. The kernel symbol includes `sync2` so that the old
FlyDSL disk cache cannot silently substitute a pre-repair kernel.

### Post-repair validation

The CPU dependency checks and GPU numerical checks are separate. The latter
use small integer BF16 operands and a CPU reference computed before the kernel,
so FP32 accumulation is exact and neither VMM windows nor a GPU reference GEMM
can explain a mismatch. They cover all supported input/output tile combinations,
K=128/192/256/7168, and changing inputs under graph replay.

Hardware benchmark results are recorded below after the independent-process
and transport-mode sweep completes. The shipped default remains `256x256`;
`--block-m 128 --block-n 256` selects the repaired smaller tile explicitly.

Before the synchronization repair, `pick_tile` was restricted to the template's
tile. That default is retained, and **the GEMM speedup measured in this
section does not reach the default configuration.** Both faster tiles stay
reachable through an explicit `--block-m` / `--block-n` so the measurements can
be reproduced.

One caution on reading any of this, which is the real lesson of the section: **a
single launch is not evidence.** Four separate conclusions here were reversed by
repeating a cell that had "clearly" passed or failed once — including this
one.

### BF16 split-pull publication and initial barrier state

The remaining `split-lsa-pull` failure is separate from the LDS ordering
repair. Its producer and its first cross-rank barrier both need attention:

* The unfused BF16 GEMM previously ignored `peer_uncached=True`. It now builds
  a buffer descriptor from the caller's **C pointer**, with `slab_bytes` as
  the bound (including the FP32 element width), and emits `sc0|sc1` stores.
  Every producer lane completes its stores and executes a system fence before
  returning. This path also accepts an ordinary C tensor with null CCO handles.
  The generated BM=128 ISA contains the 16 `buffer_store_dwordx4 ... sc0 sc1`
  instructions followed by `vmcnt(0)`, the block barrier, `buffer_wbl2 sc0 sc1`,
  another `vmcnt(0)`, and `buffer_inv sc0 sc1`. It still uses 152 VGPRs and
  has no register spills. The kernel symbol includes `sync2_pub2` to separate
  it from cached versions without publication.
* The benchmark initialized only `recv`, leaving the control region at the
  contents returned by `alloc_mem`. A diagnostic read found nonzero words in
  the initial arrival slots while the local epoch counters were zero. These
  stale positive flags satisfy pull's first `>= 1` wait before the producer
  finishes. A failed FP32 run had correct local C and `relL2=0.2044` on some
  peers reading the same source, consistent with one of 24 copy blocks reading
  early (`sqrt(1/24) ~= 0.2041`). The benchmark now zeros the **entire window**
  once before use. Flags remain monotonic across subsequent graph replays.

The regression suite checks both BM=128/256 and BF16/FP32 output on eight
gfx950 GPUs. It uses random small-integer inputs, an exact CPU oracle, a
different answer and poisoned output on each replay, and a rotating delayed
producer. A separate test runs the real benchmark with deliberately dirty
arrival slots and a delayed rank 0, so initialization is exercised even when
the allocator happens to return clean pages. The existing single-GPU numerical
matrix also covers `peer_uncached=True` with plain tensors and null CCO handles.

On 2026-09-28, at M=N=2048, K=7168, BN=256, five independent eight-rank
benchmark launches per configuration all validated:

| BM | FP32 output | BF16 output |
|---:|---:|---:|
| 128 | 5/5 (`relL2=9.93e-7`) | 5/5 (`relL2=1.66e-3`) |
| 256 | 5/5 (`relL2=9.93e-7`) | 5/5 (`relL2=1.66e-3`) |

These are correctness repetitions (`--warmup 5 --iters 20`), not a new
performance sweep. Unused SDMA resources were disabled for the LSA runs.
The dirty-window mutation control, which restores recv-only initialization
while retaining the producer's publication, fails with `relL2=1.0` on peers
that read the delayed producer. The unmodified regression passes.
The final checks passed 98 layout/dependency cases, 48 single-GPU numerical
cases, and five eight-rank tests (four replay configurations plus the dirty
window case).

```bash
PYTHONPATH=python .venv/bin/python -m pytest -q tests/python/cco/test_gemm_ag.py \
  -k 'bf16_pull_publishes or pull_initializes or bf16_gemm_pipeline_numerics'
```

### Native BF16-to-FP32 GEMM comparison after repair

Measured on 2026-09-28 at commit `f1397867`, after confirming all eight MI355X
GPUs were idle. Each rank uses M=N=2048, K=7168 and the same BF16 A `[M,K]`
and B `[N,K]` for all implementations. The Torch operation is
`torch.mm(a, b.T, out_dtype=torch.float32, out=c)` on
`2.10.0+rocm7.2.0.gitb6ee5fde`; it writes FP32 directly. Outputs are preallocated.
MORI uses `fuse=False, peer_uncached=False`, so these are pure GEMM times.

Every implementation is compiled and checked against a FP32 reference before
timing. Each measurement captures one GEMM in a CUDA graph, uses 100 warmups
and the median of 101 event samples, and takes the maximum across eight ranks.
Five rounds rotate implementation order; the headline is the median of those
five maxima. Compilation, validation, and host barriers are outside timing.

| Implementation | Median µs | Five-round range µs | Latency vs native Torch |
|---|---:|---:|---:|
| Torch BF16 → FP32 | **71.1** | 70.6–72.1 | baseline |
| MORI `256x256` (default) | **131.9** | 131.8–132.0 | **+85.5%** |
| MORI `128x256` | **88.6** | 88.2–88.7 | **+24.6%** |
| MORI `128x128` | **72.0** | 71.8–73.7 | **+1.3%** |

All four implementations pass on all eight ranks, with worst
`relL2=9.95e-7` against the reference. The `128x128` result is effectively
parity at the observed timing variation; it does not establish a speedup over
Torch. The default tile remains `256x256`, and these pure-GEMM measurements
do not include the pull producer's publication or any all-gather transport.

This supersedes the earlier estimate against 107.2 µs for the question of
native BF16-to-FP32 GEMM performance. That older number consists of 81.5 µs
for a BF16-output matrix multiply plus 25.6 µs to widen the result. Widening
does not recover the precision lost when the output was rounded to BF16.

The maintained [GEMM comparison](../benchmark/cco/flydsl/gemm_ag/compare_gemm.py)
runs native Torch and MORI kernels directly with the same paired protocol.
Historical per-rank records are archived separately from the source tree.

### GEMM + all-gather comparison after repair

Measured on 2026-09-28: 8x MI355X, M=N=2048, K=7168, BF16 operands and
FP32 output/wire. Each rank produces 16 MiB and sends 112 MiB to its peers.
These are end-to-end times for one GEMM followed by its complete all-gather,
including the final cross-rank synchronization. The native Torch baseline is
`torch.mm(..., out_dtype=torch.float32)` followed by RCCL
`all_gather_into_tensor`: **397.5 µs**.

| Mode | Default `256x256`, µs | `128x128`, µs |
|---|---:|---:|
| GEMM + RCCL | 462.7 | 403.7 |
| GEMM + LSA push | 444.6 | 379.0 |
| GEMM + LSA pull | 448.3 | 406.3 |
| Fused LSA, cached stores | 542.4 | 509.2 |
| Fused LSA, `peer_uncached=True` | 470.4 | 420.8 |
| **GEMM + SDMA** | **427.8** | **364.0** |
| Fused SDMA, 1 chunk | 429.0 | 377.0 |
| Fused SDMA, 2 chunks | 440.6 | 388.6 |
| Fused SDMA, 4 chunks | 462.2 | 409.9 |

SDMA uses lane-parallel posting. The split path uses one queue per peer;
fused SDMA uses one queue per chunk per peer. All fused cases retain the
default leader fence. LSA copy kernels retain their default uncached policy;
both fused-LSA store policies are shown rather than choosing only the faster
one.

**Fusion does not improve the best split result at this shape.** With the
default tile, one-chunk fused SDMA is effectively tied with split SDMA
(+0.3% measured latency); with `128x128`, it is **3.6% slower**. Two and four
chunks increase latency by **6.8% and 12.6%** respectively on the smaller tile.
Even the faster fused-LSA policy is **5.8% slower** than split LSA push at
`256x256` and **11.0% slower** at `128x128`.

The practical improvement is the smaller tile plus the split SDMA transport:
364.0 µs is **8.4% less latency than native Torch + RCCL** and **14.9% less
than the default-tile split SDMA path**. Those gains must not be reported as
fusion gains. At this shape, the 64- or 256-workgroup grid fits in one round
on 256 CUs, so chunking does not imply a long sequence of progressively ready
outputs. The measurements show additional chunks increasing total latency.

Each configuration ran five rounds with 50 warmups and the median of 101
CUDA-graph event samples per rank. Take the maximum across ranks in each
round, then the median across rounds. Configurations ran sequentially with
no other GPU processes before or after each run. Allocation, compilation,
capture, and validation are outside timing. This is a repeated measurement
within one process group per configuration, not five independent process
launches. No subtraction of a separately timed GEMM is used to claim overlap.

All 19 final configurations passed the initial numerical check and two
changed-input replays of the same captured graph. Each replay poisons recv,
changes the sign of A, and clones recv immediately after graph completion,
before constructing the reference. This catches stale results and prevents
reference-GEMM work from hiding a transfer still in flight. Every rank checks
finite output and `relL2 < 1e-5`; the worst observed replay error was
`9.94e-7`.

**Multi-queue drain correction.** The sweep exposed a protocol gap in
`kernels_sdma.py`: the fused producer assigns chunks to `chunk % queues`,
but the drain waited only for `peer % queues`. Completion of one queue does
not cover the others. The fused multi-queue drain now uses
`sdma.quiet(peer, coop=THREAD)` to wait for all of that peer's queues before
publishing arrival. Split and single-queue paths keep their existing wait.
The kernel symbol includes `drain_allq` to avoid reusing the old compiled
drain. A model executing the real kernel body failed nine multi-queue cases
before this correction; all 24 split/fused/rank/queue cases now pass. The
table uses the corrected drain for every fused multi-queue measurement;
the earlier `128x128` samples were replaced by fresh measurements.

The maintained [benchmark entry points](../benchmark/cco/flydsl/gemm_ag/README.md)
run these transport comparisons directly, including repeated graph timing
and changed-input validation. Historical raw records are archived separately.

### What it would have bought, and what it actually bought (historical)

At `BLOCK_M=128`, `gemm-only` at `M=2048` over 8 ranks under CUDA-graph replay
measures **91.7us** against the shipped tile's 137.6 — a 33% cut that would have
taken the operator from 11.5% behind the plan document's FP32 baseline to 2%
better than it, and to 8% better through `split-sdma`. Those numbers are real
and they are also unusable, for the reason above.

**What ships is the knowledge, not a speedup.** The sweep establishes three
things worth more than one tuned constant:

* the tile is worth 24–34% at these shapes and is the *only* knob that is —
  `waves_per_eu` and `xcd_swizzle` are inert to within 1%;
* the mechanism is grid occupancy, not the inner loop, which says where the
  next attempt should look (split-K, not scheduling);
* and there is a **latent correctness bug that halving `BLOCK_M` exposes**,
  which was not visible at the template's tile and is now a known, reproducible
  target. Finding it is worth the 33%.

### What is left

The gap to `torch.matmul` itself is not closed, and it widens as M falls: 1.26x
at M=4096, 1.58x at M=2048, 3.29x at M=512. The residue is the same problem one
level down — at `M=512` even the smallest sound tile is 32 workgroups on 256
CUs, and our time barely moves from M=512 to M=2048 despite four times the work,
which is the signature of a kernel waiting on the machine rather than using it.
**Split-K** is the answer there and is not implemented. After that, the store's
coalescing: a 16-lane group writes 16 different rows at one column, which the
gcnasm template's `shfl` stage exists to fix and this port does not carry.

## The bound fusing runs into

Separately from the per-packet cost above, there is a ceiling that no amount of
epilogue work can lift, and it is worth having written down because it says
where to spend effort next.

Producer-side fusion overlaps the transfer with the compute that produces it, so
the best it can do is replace `G₀ + C₀` with `max(G₀, C₀)`:

    saving ≤ min(G₀, C₀) / (G₀ + C₀)

For an all-gather the ratio inside that is fixed by the shape, not by the code:

    G₀ ∝ 2·M·N·K / FLOPS        C₀ ∝ M·N·(world−1)·2 / BW

    G₀ / C₀  ∝  K · BW / ((world−1) · FLOPS)

`M` and `N` cancel. **The only levers are `K` and `world`** — more reduction
depth per output element, or fewer peers to broadcast to. Measured: at
`r4 M=2048` the ratio is `56.6 / 153.9`, so the ceiling is 27%; at
`r4 M=512` it is `52.8 / 50.6` and the ceiling is 49%. Neither is reached, but
the first is not far above the 25% of communication that `fused-sdma` already
absorbs there.

Now do the same for the siblings, which is the useful part:

| | bytes on the wire per rank | `G₀/C₀` scales as |
|---|---|---|
| `gemm_ar` (all-reduce) | `≈ 2·M·N` (both legs, shard cancels) | `K` |
| `gemm_a2a` (all-to-all) | `M·N·(world−1)/world ≈ M·N` | `K` |
| `gemm_ag` (all-gather) | `M·N·(world−1)` | `K / (world−1)` |

Both siblings shard the payload by `world`, so their ratio is
**independent of the world size**: adding ranks adds compute and communication
in step. All-gather does not shard anything, so every extra rank adds a full
copy of the payload to the wire while adding nothing to the compute. It is the
one member of the family where **scaling out makes fusion less attractive**, and
the only one whose fused margin should be re-measured whenever `world` changes.

Two consequences follow, and they are the recommendations this file exists to
make:

* **`world < 8` is where the fused paths might come back.** At `world = 2` the
  ratio is seven times better than measured here.
* **A narrower wire moves this lever directly**, and moves it for the split
  paths too. fp8 on the wire halves `C₀`, which at `r4 M=2048` would take the
  split total from 211 to roughly 134 — a larger win than any fusion of the
  bf16 wire can offer, and it compounds with fixing the put serialisation rather
  than competing with it. That is the next thing to build.

This is the same kind of reasoning as the "post-barrier phase" account of why an
all-reduce absorbs ~37% of its communication and an EP combine absorbs ~81%,
applied one level up: that one asks what fraction of the collective is fusable,
this one asks whether there is enough compute to fuse it into.

## A correctness note worth keeping

`split-lsa-pull` shipped in a first draft with the producer's release missing,
and **it validated**: relL2 1.66e-3 under `--quant ptpc`, the same figure every
other mode reports. Under `--quant blockscale` it gave 2.05e-1. The blockscale
GEMM is 96us against ptpc's 57, so the rank skew is wider and the window the
race needs is wider with it.

The cause is specific to the pull direction and is a real constraint on it: a
pull reads a peer's HBM over xGMI, while the peer's GEMM left C dirty in its own
L2. Nothing downstream can repair that. `cco_system_fence` is
`__threadfence_system()`, which orders the *calling thread's own* prior writes —
so a fence in the copy kernel, which wrote nothing, publishes nothing. The
producer has to publish, and the producer is the GEMM: the pull path compiles it
with `peer_uncached=True`, whose epilogue stores through a buffer descriptor
with `sc0|sc1`. At this shape that costs nothing measurable (217.0us against
217.1us before the change).

The later BF16 audit above also found uninitialized arrival flags in the
benchmark. Zero the whole window before attributing a pull failure to cache
publication: both faults can let a peer read incomplete C, and the numerical
error alone does not distinguish them.

The general lesson is the one the mode matrix is built around: **a transport bug
that only one quantisation exposes is not found by spot-checking the default.**
`tests/python/cco/test_gemm_ag.py` parametrises every mode over all three.

## Shape constraints

`validate()` requires `n % block_n == 0` and `m % block_m == 0`, and that is all.
`gemm_a2a` additionally requires `n % (world_size * block_n) == 0` — **N a
multiple of 2048 at P=8** — because a destination's column shard has to be a
whole number of GEMM tiles. No column is sharded here, so that rule does not
arise.

That is not a corner case. The `wkv_gate` ratio-128 layers are `N = 1024`, which
`a2a_config` rejects at `world_size=8` and `ag_config` accepts.

| shape | K | N | `ag_config` | `a2a_config` |
|---|---:|---:|---|---|
| wkv_gate ratio-4 | 7168 | 2048 | ok | ok |
| wkv_gate ratio-128 | 7168 | 1024 | ok | **rejected** |
| Llama-3.1-70B | 8192 | 10240 | ok | ok |
| Llama-3.1-405B | 16384 | 18432 | ok | ok |

`--quant mxfp8` adds `N % 32 == 0`, `K % 128 == 0` and `BLOCK_M == 256`, all
from the ue8m0 operand format rather than from this operator.

## Reproducing

```bash
# the bf16 / fp32 path -- the model's own precision
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode split-sdma --in-dtype bf16 --out-dtype fp32 \
  -m 2048 --out-dim 2048 -k 7168 --warmup 10 --iters 21 --phase-split

# one cell
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode split-lsa-pull -m 2048 --out-dim 2048 -k 7168 \
  --warmup 30 --iters 21 --phase-split

# the model tables, with the idle-box protocol
python benchmark/cco/flydsl/gemm_ag/sweep_models.py --rounds 3 --out models.jsonl
python benchmark/cco/flydsl/gemm_ag/sweep_models.py --quant mxfp8 --rounds 2 \
  --models wkv_gate-r4 --out mxfp8.jsonl

# the knob sweep at one shape: push vs pull, unroll, cache policy, chunks
python benchmark/cco/flydsl/gemm_ag/sweep_ag.py --out sweep.jsonl

# layout arithmetic, no GPU
pytest tests/python/cco/test_gemm_ag.py -k "not validates and not deterministic"
```

`-n` is `--out-dim` here, not `-n`: `torchrun` takes `--nproc_per_node` and
argparse's abbreviation matching makes a bare `-n` ambiguous against it.

FlyDSL caches compiled kernels under `~/.flydsl/cache`, keyed on the
`@flyc.jit` **wrapper**'s name rather than the kernel body — so editing a kernel
and re-running can silently execute the previous build. While porting the bf16
GEMM that made a fixed bug look unfixed and produced a thoroughly convincing
false signal (odd `K_ITERS` bit-exact, even `K_ITERS` broken, which looks exactly
like a double-buffer parity bug and was not one). **If a kernel edit appears to
have no effect, move `~/.flydsl/cache/launch_*` aside before concluding
anything.**

`BUILD_CCO_SDMA=ON` **and** `MORI_ENABLE_SDMA=1` are required for the SDMA
modes. Without the environment variable every put is a silent no-op — the run
completes, the timing looks plausible because it is the GEMM plus a barrier, and
every peer slot keeps whatever it held. The benchmark now refuses to start the
SDMA modes without it rather than relying on validation to notice.

Back-to-back launches lose the SDMA queue reclamation race (`anvil.cpp:237`)
often enough that a bare loop cannot finish a table. Both drivers settle between
launches and retry a lost race rather than recording it.

## Not measured

* Cross-node (GDA). Everything here is intra-node LSA/SDMA over xGMI.
* **Split-K**, and the C store's coalescing. The tile sweep is done; these are
  what is left of the gap to `torch.matmul` — 1.26x at M=4096, 3.29x at M=512.
  `BLOCK_K` is still pinned to the template's 64 and was not swept: it would
  need a different LDS swizzle, since `swizzle_row128b`'s geometry is 128 bytes
  per row and `BLOCK_K=128` is 256.
* **Smaller-tile tuning after the synchronization repair.** The previous
  “ISA is clean” conclusion missed the two ordering edges described above.
  Keep correctness and performance validation separate when extending
  `pick_tile` beyond the current `256x256` default.
* The bf16 path at the `r128` shape, under `--chunks > 1`, or with
  `--post serial`. All are reachable; none were swept.
* fp8 on the wire. This is the direction where it would help most — the
  operator is bandwidth-bound by construction and a 2x narrower wire moves the
  `G₀/C₀` ratio directly — and the `pull` direction is built so the dequantize
  lands in registers on the consumer, which is where `gemm_ar` measured it as
  free against a 61.0us widen kernel on the SDMA push. Doing it is the obvious
  next step and is worth more here than any amount of epilogue tuning.
* `world != 8`. The `G₀/C₀` derivation says fusion gets *more* attractive as
  `world` falls, so 2 and 4 ranks are the regime where the fused paths might
  come back.
* `--chunks` past 8. `--post lanes` derives `sdma_queues = max(1, chunks)`, and
  `gemm_a2a` records `hsaKmtCreateQueueExt` starting to fail once a process asks
  for `world_size` queues per peer — `c8` already sits on that number and was
  seen to lose the queue-creation race once in twelve back-to-back launches. It
  succeeds with the settle the sweep drivers use, but `--chunks 1` is the best
  setting anyway.
* `--post lanes` at the `r128` shape and under the phase split at M other than
  512/2048.
* The 70B (`N=10240 K=8192`) and 405B (`N=18432 K=16384`) shapes, which
  `sweep_models.py` carries so this operator's table can be read against
  `gemm_a2a`'s on the same box. They are the high-`K` end, where the bound says
  fusion has the most room.
* `--split-gemm epilogue`, `--peer-uncached`, `--push-unroll` and
  `--copy-blocks`. All are wired into `sweep_ag.py` and none were swept; the
  numbers above are at their defaults.
