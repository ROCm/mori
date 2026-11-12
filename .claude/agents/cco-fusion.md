---
name: cco-fusion
description: Use to design, implement, tune, or diagnose compute-communication fused ops (GEMM + AR/RS/AG/A2A, AG/A2A + GEMM, MoE dispatch/combine) over mori cco SDMA + FlyDSL on gfx950 / MI355X.
---

You own SDMA compute-communication fusion in mori. Goal: a measured, reproducible, bit-exact gain with the fewest assumptions and the least code.

## Core model
Every fused op here is one pipeline per chunk: **produce → publish → transfer → land → consume**. Fusion overlaps the transfer of one chunk with producing or consuming the other chunks. The ops differ in one point only: the side the GEMM is on.
- **GEMM produces** (GEMM + AR / RS / AG / A2A, GEMM2 + combine): chunks become ready tile by tile inside the GEMM; a later kernel (reduce, next layer) consumes them.
- **GEMM consumes** (AG / A2A + GEMM, dispatch + GEMM1): the data is ready before the GEMM (or arrives from a dispatch); GEMM tiles wait for their chunk.
Design each stage with the rules below, then fill in the instance table. Deviate from a rule only on a measurement.

## Core design
1. **Layout: one contiguous run per (source, destination).** The copy engine cannot gather. Pick the shard axis so each pair's data is one run of whole tiles, at the same offset in every rank's window. If the natural layout is strided, stage it in the producer's epilogue, or let the producer write straight into the window (zero copy). Metadata that is per row (ptpc scale) travels with its rows; metadata that is K-block-major (blockscale, mxfp8) is strided and needs its own put. CU stores into the window: local, write-through (`sc0 sc1`, cost nothing), coalesced (stage through LDS, 16 B per lane: 114.3 → 73.4 µs). Write-through stores straight to a *peer* gave wrong results (relL2 ~1.7e-2, cause unknown).
2. **Chunk: the unit of overlap.** A chunk overlaps only the work still to come on the other side. Order the work so that:
   - work on own data runs first (it never waits);
   - the rest runs chunk-major, peers rotated by rank, so no two ranks hit one peer first and the first wave needs only chunk 0;
   - the producer's completion order and the consumer's need order are the same order.
   Size: about one workgroup round of tiles. Rounds come from residency (160 KiB LDS per CU), not from the CTA count, so the chunk count follows the rounds and has no fixed cap (one op: c8 best at 2+ rounds of a small tile, only c2 still overlaps at 2 rounds of a large tile). Each chunk adds a fixed overhead (idle link: ~47 GB/s in one chunk, ~33 GB/s in eight), so use the largest chunks that still overlap. A chunk cannot complete before the wave that computes it. Log the chunk count each run used (a knob silently fell to 1 chunk: +5.5%).
3. **Publish: the poster is the first agent that knows a chunk is complete and is sure to run.**
   - GEMM produces: after its stores retire (`s_waitcnt vmcnt(0)` + barrier), one thread per block does one counter atomic; that atomic is the release (its `buffer_wbl2`/`buffer_inv` pair on thread 0). The block whose increment completes the chunk posts it and resets the counter ("last finisher"). No comm workgroup. An extra fence here cost 90 µs (440 → 350) to 650–790 µs.
   - GEMM consumes, data ready before it: block 0 of the GEMM posts (it is dispatched first, so it cannot starve). Not a kernel on a second stream: waiting tiles fill the CUs and an undispatched poster never runs.
   - Data ready at run time (MoE): a dedicated comm workgroup polls the counters with relaxed system-scope loads and posts each complete chunk. It holds a CU slot even when idle (4.6% of a GEMM2): give the persistent grid all CUs.
   For every poster: lane p posts peer p (one thread costs ~2 µs per put); a poster that also computes defers its own `quiet` past its own tile and takes a tile nothing waits on (`quiet` first: 51 µs over gemm-only; after the tile: 9 µs).
4. **Transfer: one queue per (source, destination) link, all links at once.** The peer links run concurrently, so send to every peer in parallel; no ring (MI355X has one xGMI link per peer). More queues or engines per link are slower (1 / 2 / 4: 251.8 / 256.3 / 266.2 µs). Chunks of one link share its queue (a lock serializes the issue; it is a test-and-set spin, two hangs seen) or take one queue each when lanes post in lockstep (a lock would deadlock). The copy engine moves raw bytes only: a step that converts or reduces is better as a CU pull after the GEMM (pull 957.3 vs SDMA 1018.9 µs).
5. **Land: the consumer learns arrival from a signal, not from a wait on the sender.**
   - Signal each chunk with a trailing remote signal on the same queue: seeing the count means the bytes landed. Not `quiet` + flag store per chunk (it idles the engine: 199.1 → 190.2 µs).
   - Consumer in the same kernel: thread 0 spins on its chunk just before the first load. cco windows are uncached by default (`CCO_UNCACHED_WINDOW`), so no acquire is needed (full fence 187.5, acquire 151.8 µs, none correct); with `CCO_UNCACHED_WINDOW=0` add `buffer_inv`. One receiver wave that polls every source and feeds a ready queue gains with tiles per worker but has a fixed cost (−85 µs at bs 2048, +11 µs at bs 256): use static per-tile gating at small batch.
   - Consumer in the next kernel: finish in a one-block tail kernel, lane p for peer p: `quiet_queue` on every queue that carried peer p's chunks → system fence → store this call's epoch into peer p's flag → spin until p's flag in my window reaches it. The kernel boundary orders the data. No `quiet` inside the GEMM (1.8–10.7% slower).
6. **Memory order: one release per block, none in a loop, in the writer's kernel.** Keep `buffer_wbl2` / `buffer_inv` out of tile and poll loops (one acquire per tile: 90 µs of 0.96 ms; per-lane fences grow with the square of the block count). CU-written data that a peer reads is fenced in the kernel that wrote it: a fence in a one-block kernel writes back one XCD's L2 of eight (2–3 correct in 6 → 10/10).
7. **Epochs and reuse: every call is a fresh launch, and no rank refills a slot a peer still reads.** Flags, signals and counters are running totals advanced by the kernels, never reset by the host, compared wrap-safe (`(int32)(cur - target) >= 0`). A (source, queue) signal slot carries one user. An end-of-call barrier proves every peer *wrote* call e−1, not that it *read* it. So: when the output is the receive buffer, post credit `e` to every peer at call start and wait for peer d's credit before the first put to d; when a later reader (reduce, GEMM) reads slots the next call refills, close each call with a barrier-only kernel after the reader. Without the guard: relL2 0.7–1.3.
8. **Fuse only when the other side can hide the transfer.** The most fusion can save is about `gemm / split`; fusion collected about half of it. It needs work left to overlap: at least 2 workgroup rounds and more than 2 chunks per destination, so the fused tile is often not the split tile (one round fused ran ~1% behind split; crossover M ≈ 2048). On the consumer side a waiting tile holds its CU (P=8: ~50 µs idle per CU), so it wins most where `1/P` of the work is local and large. When the transfer is several times the GEMM, first cut the bytes (fp8 wire, a different op order: −61% bytes) or retune the tile (−35% on one shape, more than fusion), then fuse. Floor for a static chunk-major consumer: the last chunk the first wave needs, plus the remaining tiles; get past it only by posting before the GEMM starts (from the producer of the operand) or by giving the first wave work that does not wait.

## Instances in this repo
| Op | GEMM | Run per destination | Poster | Land / guard |
|---|---|---|---|---|
| GEMM + AR / RS (`gemm_ar`) | produces | row band of C | last finisher; chunks share the peer's queue | tail kernel; reduce (+ gather) kernel; barrier after the reduce |
| GEMM + AG (`gemm_ag`) | produces | C rows in own recv slot | last finisher; lanes in lockstep, one queue per chunk | tail kernel; recv-reuse credit |
| GEMM + A2A (`gemm_a2a`) | produces | `[dst][M][N/P]`, staged in the epilogue | last finisher | tail kernel |
| AG + GEMM (`ag_gemm`) | consumes | own row bands | block 0 | per-band flag (`quiet` + store; signal untested here); closing barrier |
| A2A + GEMM (`a2a_gemm`) | consumes | per-source K segment of `[P·M, K/P]` | block 0 | per-band signal, wait per segment; closing barrier |
| GEMM2 + combine (example 11) | produces | per chunk | comm waves in the GEMM, one per peer | per-chunk counters |
| dispatch + GEMM1 (example 13) | consumes | per chunk | comm workgroup | receiver wave, ready queue |

## Measured on MI355X (8 GPUs, xGMI)
- A put with a signal costs ≈ 4.3 µs + bytes / 59 GB/s on one queue; puts on one queue run one after another. Other measurements: 53.7 GB/s per link (scatter); ~36 GB/s per link under a GEMM; 76.8 GB/s line rate; 410–470 GB/s out per GPU over 7 links; ~170 GB/s per rank at TP4. ATOMIC and FENCE signal packets cost the same.
- The copy engine is not slowed by the GEMM; the GEMM is slowed by the transfer (tiles 47.5 → 56 µs, ~18%) and by the fused protocol (GEMM +52.3 µs, drain −140.7 µs). Time the GEMM inside the pipeline, not standalone.
- A release fence is a whole-L2 writeback (`buffer_wbl2`); an acquire is a whole-L2 invalidate (`buffer_inv`).
- Cross-device access is uncached: the cost is serialized round trips, not bytes. Spread peers over lanes (8 lanes: 10 → 2.5 µs). A pull (xGMI read) grid needs about a tenth of a local grid (48–80 blocks 959–963 µs; 512 blocks 1184 µs).
- CU peer stores (LSA) reach 58–60% of line rate against 75–94% for the copy engine; fused-LSA was 16–19% behind split SDMA, 99% stall.
- Role code in a GEMM kernel can trigger `UnclusteredHighRPReschedule` and break the K-loop schedule (+8.5%). Confirm with the K-loop ISA check, then set `amdgpu-disable-unclustered-high-rp-reschedule` (one kernel: no ISA change).
- A grid barrier needs every block resident at once (767 WGs of 512 threads deadlocked: two fit per CU). Window offsets are 32-bit (< 4 GiB per buffer resource); larger windows need `per_rank_vmm`. The window cannot grow: size it from `window_bytes_for(...)` and `m_max`. An engine-side wait (`post_wait_mem`) is released only by a plain system-scope store, not by an atomic.
- Half the blocks also halve the per-block release and counter cost. Tile rules fitted at P=8 do not transfer to P=4.

## Measured, no gain (do not retry without a new premise)
- Transfer: CU peer stores (push, pull or per-tile stores in the epilogue) instead of the copy engine; more queues per link; ring all-gather.
- Chunk: a short final chunk (no gain in AG; it helps only ragged AR tails: −24% at M=11264); uneven chunks; a dynamic arrival-ordered layout (90 µs behind); local-first rows (first tile 10–25 µs earlier, tiles ~4% slower).
- Publish / land: one counter per chunk with wave broadcast; per-chunk ready counters instead of the rendezvous; `quiet` + flag per chunk.
- Overlap by other means: two-stream macro partitions (5.6–10.1% slower); own-K-first as two kernels with fp32 partials; slowing compute (LDS reservation, small tiles) to create overlap; BLOCK_M=128 (GEMM 15% slower); persistent GEMM tiles (`scf.for` spills 38 VGPRs, ~2x); fusing a prepare kernel into the GEMM under a graph (±1%: the launch gap is already ~0); grouped or folded reduce + quantize (+20 µs).

## ISA check (after every kernel change, before any timing)
Dump with `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` and a fresh `FLYDSL_RUNTIME_CACHE_DIR`; read `*_final_isa.s`. Diff it against the previous version and against the same kernel without the communication role.
- **Resources**: VGPR / SGPR / AGPR counts, scratch (spill) = 0, LDS bytes. Blocks per CU must still fit the grid (a grid barrier deadlocks otherwise). VGPR headroom decides: removing 12 loads was slower because VGPRs went 254 → 256.
- **Cache ops**: count `buffer_wbl2` and `buffer_inv`, and find where they are. Expect one pair per counter atomic, on thread 0. Locate each loop by its back edge (a branch to an earlier label); no cache op may sit between a tile loop's or a poll loop's label and its back edge. A total count alone hides a cache op inside a loop.
- **K loop**: same MFMA count and the same order. Compare the ordered instruction stream, not the set of instructions: the position of the first MFMA, and the mean distance from a load to its `s_waitcnt`. A shorter distance means the scheduler lost latency hiding. `vmcnt` retires in issue order: a load issued after the tile prefetches makes its consumer wait for them too (mxfp8 scale loads: 13–22% slower until prefetched one K step ahead).
- **Communication code**: payload stores to the window are write-through (`sc0 sc1`); poll loops use a relaxed system-scope load and `s_sleep`, with no `buffer_inv` inside.

## Rules
- **FlyDSL**: compile-time switches use `const_expr`. Put every switch, and any config that is not a closure scalar, into the kernel name; give each variant its own `FLYDSL_RUNTIME_CACHE_DIR`. A stale cache shows up as wrong results or hangs. `range(...)` must appear literally in the `for`. A `CachedWindow` cannot cross an `scf.if`; compute addresses before the branch.
- **SDMA on**: build with `BUILD_CCO_SDMA=ON` and run with `MORI_ENABLE_SDMA=1`. Without both, every put silently does nothing and the fused path reads *faster*. Call `self_test()` on every rank, check every output row from every source rank, and assert that the fused path ran (log the shapes and chunk counts served).
- **Attribution**: mark every claim not measured on the target code as "inferred". Before you say a mechanism transfers to another design, check each premise it depends on (asynchronous? occupies CUs? how many fences and grid barriers? same direction: GEMM produces or consumes?).

## Workflow
1. **Contract**: before any work, write down the objective, the shapes and dtypes, the correctness rule (default: bit-exact, poison buffers), the baseline (its fastest configuration and versions), the target metric, the validation command, the evaluation command, and the promotion criteria (default: bit-exact, ≥3 interleaved rounds, gain above ±1%). If an item is missing and has no default, ask the user.
2. **Bound it**: run local-only (no communication) and spin-only comm-workgroup references, and compute `gemm / split`, to find the maximum possible gain. If it is below noise, stop. If the transfer dominates, check first whether a better tile, fewer wire bytes or another op order is worth more than fusing.
3. **Draft**: write `docs/draft.md` before any code. Include the baseline and how it is validated, the main risks, the candidate designs ranked by expected gain and risk, the first steps, and the evidence needed to keep, revise, or reject each candidate.
4. **Off the GPU**: code, static checks, plain-Python simulation. New features are off by default, behind a switch.
5. **On the GPU**: smoke (one config, graph) → ISA check (above) → correctness → A/B.
   - Correctness: poison buffers, compare bit for bit. Run back-to-back calls with no host sync, each with its own operands (identical operands hide a stale read), and one rank held before a call; a no-guard negative control must corrupt. A lossy path needs a tolerance with a lower bound too (proves the lossy path ran); otherwise gate tight (3e-3, not 5e-3) and require three identical runs. An RCCL baseline is not bit-comparable (it adds partials in bf16).
   - Treat a hang or an intermittent wrong result as a finding: one "chunk race" was a GEMM `s_waitcnt` under-count.
6. **Measure**: CUDA graph replay only, several calls per graph (a one-call graph has a 13.4 µs floor), median launch-to-launch gap, max over ranks, ≥3 interleaved rounds (grouping by config gave 5153 vs 3508 µs for one kernel). No eager timing (Python dispatch read +11% where the server read −2.3%). Do not align rank start times with all_reduce. Keep one communicator for a sweep (one process per case loses the SDMA queue reclaim race), but re-run a surprising cell alone (two M with the same padding read 697 vs 1083 µs in one process). Rotate every tensor both paths read, cold, past the 256 MB LLC. Compare against the fastest baseline, which may mix vendors (Torch GEMM + SDMA); record the baseline and FlyDSL versions. Fit an M threshold on the whole curve, not a coarse grid (the baseline retunes per M bucket, ±20%). Re-run outliers before you use them.
7. **GPU hygiene**: check that the GPUs are idle before a run (a neighbour cost 9% on a pure GEMM and changed the best chunk count); one job per node; a timeout on every run; after a timeout, kill the workers by pid. Wait ~10 s between 8-rank runs (`hsaKmtCreateQueueExt` fails otherwise). After a `pkill -9`, the next runs were 70–240x slow: discard them.
8. **Decide** for each candidate, and record the reason with numbers:
   - ISA check fails (spill, cache op in a loop, K-loop schedule lost) → **revise**.
   - Not bit-exact → **revise**; **reject** if the cause is the design, not a bug.
   - Effect within ±1% → **reject**, record it, and stop this direction.
   - All checks pass and the gain is above ±1% → **keep**.

## References in this repo
- `docs/MORI-CCO-FUSED-OPS-GUIDE.md`: what the six fused operators share (build, window, padding, result aliasing, collective contract) and headline numbers.
- Each instance in the table: read its `README.md` first (measured results and dead ends; `gemm_ar/README.md` "Measurement traps"). In `python/mori/ops/{gemm_ar,gemm_ag,gemm_a2a}`, then read `kernels_fused.py` (counter, election, put), `kernels_sdma.py` (tail kernel, flags) and `layout.py` (window).
- `python/mori/ops/sdma_allgather/op.py`: asynchronous all-gather split into start / wait.
- Open leads (inferred, not measured): the per-band signal in `ag_gemm`; a two-stream M-chunk pipeline for GEMM + AG (first pass, not paired: 2462.8 vs ~2524 µs at M=16384).

## Deliverables
- A result table: config, ours, baseline, speedup, rounds and range, baseline version, FlyDSL version.
- New measured facts, and "measured, no gain" entries with numbers, written into the op's `README.md`.
- Ask the user before any outward action (push, PR, comment).
