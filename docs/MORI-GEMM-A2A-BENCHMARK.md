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

## Conditions

8x MI355X (gfx950), fp8 e4m3 in / bf16 out, CUDA graph replay, median of 21
iterations after 30 warmups, max over ranks. Tile is 128x256 under `--quant
ptpc` and 256x256 under `--quant mxfp8`, which needs `BLOCK_M=256` for the
scale operands.

**Three to four independent launches per cell, interleaved.** Every
configuration is measured once per round and the rounds repeat, so machine
drift lands inside each configuration's own samples where the spread shows it,
rather than between the things being compared. Every launch waits for an idle
box; any sample taken with a neighbour present is discarded, not averaged in.
Clock and junction temperature are recorded next to every measurement.

Reported spread is 0.0–2.9% on every cell below.

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

| M | S | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---:|---:|---:|---|---:|---:|
| 2048 | 16k | 180 | 270 | 298 | **245** `c4` | 9.2% | 17.8% |
| 4096 | 32k | 300 | 471 | 552 | **417** `c4` | 11.5% | 24.5% |
| 8192 | 64k | 625 | 911 | 1146 | **766** `c8` | **15.9%** | **33.2%** |
| 16384 | 128k | 1431 | 2124 | 2304 | **1884** `c8` | 11.3% | 18.3% |

### PTPC — Llama-3.1-405B — `N=18432 K=16384` (s=2304)

| M | S | gemm-only | split-sdma | split-rccl | best fused | vs split | vs rccl |
|---:|---:|---:|---:|---:|---|---:|---:|
| 2048 | 16k | 687 | 831 | 891 | **802** `c8` | 3.5% | 9.9% |
| 4096 | 32k | 1232 | 1546 | 1633 | **1440** `c8` | 6.8% | 11.8% |
| 8192 | 64k | 2444 | 3103 | 3277 | **2866** `c8` | 7.7% | 12.5% |
| 16384 | 128k | 4880 | 6263 | 6606 | **5726** `c8` | **8.6%** | 13.3% |

All modes agree to the digit on relL2 — 1.66e-3, fp8's own floor at these
operands — so the transports write identical bytes and the numbers above are
comparing the same computation.

### MXFP8 — the same eight cells, `--quant mxfp8`

Two rounds, spread 0.1–1.0%, same protocol. `relL2` is 1.66e-3, the same as
ptpc: the operands are identical and the fp8 mantissa dominates the ue8m0
scales.

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

**Fusing pays about twice as much under mxfp8**, 11–25% against ptpc's 3.5–16%,
and **both shapes are now monotone in S** — 70B's unexplained fall-back at
S=128k is gone, so it was a property of the ptpc kernel rather than of the
slab size or the chunk count.

The split-path communication matches ptpc cell for cell (405B/M=16384: 1511us
against 1517us), which is the expected result and a useful check — the bytes,
the layout and the transport are unchanged, so only the GEMM differs between
the two tables.

Two things move, and they move in opposite directions:

* **The GEMM crosses over.** mxfp8 is 9% *slower* than ptpc at M=2048
  (197 vs 180us on 70B) and 13–15% *faster* at M=16384 (1211 vs 1431, 4230 vs
  4880). `BLOCK_M` is 256 for mxfp8 against ptpc's 128 — a shape that costs
  occupancy when there are few row tiles and buys reuse once there are many.
* **The fused path hides more of the transfer.** The unhidden tail is 50–74% of
  the split-path communication here; under ptpc at 405B/M=16384 it was 846us of
  1517 (44%) against 466 of 1511 (69%) now.

That combination is why the margin roughly doubles: the epilogue overlaps a
larger fraction of a transfer that is unchanged, against a GEMM that at the
large end is also cheaper.

### The full mode table at one cell, `M=2048`

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

**Fusing wins in all eight cells, and the margin grows with sequence length.**
405B is monotone: 3.5% → 6.8% → 7.7% → 8.6%. The mechanism is the one gcnasm
reports for its own version: at larger M each persistent workgroup owns more
tiles, so an early PUT has more remaining compute to overlap against. 70B peaks
at S=64k (15.9%) and falls back to 11.3% at 128k; that non-monotonicity is
**not explained here**, but it does not survive the change of quantisation —
the mxfp8 table is monotone at both shapes — so it is a property of the ptpc
kernel rather than of the slab size or of `--chunks` stopping at 8.

**The best chunk count moves right with size.** 70B wants `c4` at S≤32k and
`c8` at S≥64k; 405B wants `c8` throughout. Splitting further lets a
destination's early rows leave sooner, but each PUT gets smaller and below
roughly 1 MiB the per-packet cost starts dominating — so the right count is the
one that keeps the PUT above that knee while starting as early as possible.

**RCCL is last in every cell**, 10–33% behind the best fused path and 3–26%
behind `split-sdma`. That comparison is clean: `split-rccl` and `split-sdma`
run the same GEMM over the same bytes in the same layout.

**Fusing nearly eliminates the exposed transfer, and that bounds the win.** At
405B/M=16384 the communication drops 1517us → 264us, −83%, yet end to end the
gain is 8.6%. The GEMM is 78% of the fused total, so there is very little left
to hide. Past this point the lever is the GEMM, not the transport.

**LSA loses to SDMA on this operator.** `split-lsa` reaches only ~60% of the
76.8 GB/s line rate where the copy engines reach ~75%; `fused-lsa` is the
slowest mode at both shapes. Ruled out as causes: grid size (24/48/80 blocks
within 0.6%), cache policy (`sc0|sc1` worth 1.7%), local memory bandwidth
(10.6us of an 88us gap), and the read layout — `split-lsa-staged` reads a
compacted slab instead of a strided `[M,N]`, moves 12% fewer bytes, and takes
the same time, so LSA's cost is its peer stores alone. gcnasm's independent
measurement of the same kind of store puts it at 46.1 GB/s per link against our
48.3, and its ATT capture found 99% of that time was stall.

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

## Reproducing

```bash
# one cell
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py \
  --mode fused-sdma --chunks 8 -m 16384 -n 18432 -k 16384 \
  --warmup 30 --iters 21 --phase-split

# the model tables, with the idle-box protocol
python benchmark/cco/flydsl/gemm_a2a/sweep_models.py --rounds 3 --out models.jsonl
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
  blockscale, which is the direction mxfp8 went here too.
* `--chunks` past 8, and `--chunks 2` anywhere but `M=2048`.
* The other validated shapes — 8B, Qwen3-32B.
* fp8 on the wire. The transfer is already 83% hidden at the large end, so this
  would help the small-M cells rather than the large ones.
* Anything at P != 8.
