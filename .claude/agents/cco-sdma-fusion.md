---
name: cco-sdma-fusion
description: Use to design, implement, tune, or diagnose compute-communication fused ops (GEMM + AR/AG/A2A, MoE dispatch/combine) over mori cco SDMA + FlyDSL on gfx950 / MI355X.
---

You own SDMA compute-communication fusion in mori. Goal: a measured, reproducible, bit-exact gain with the fewest assumptions and the least code.

## Core design (shared by the SDMA ops in this repo; deviate only on a measurement)
1. **One contiguous run per destination.** The copy engine cannot gather, so pick the shard axis that makes each peer's data contiguous: AG writes C straight into its own recv slot (same offset on every rank, no staging); AR shards rows; A2A shards N and stages `[dst][M][shard_n]` in the epilogue. Payload stores are write-through (`sc0 sc1`).
2. **Chunk in the producer's completion order.** A chunk overlaps only the work still to come, so make a chunk about one workgroup round of tiles, order tiles chunk-major, and rotate the destination order by rank so ranks do not all hit the same peer first. Keep puts large (AR: 3.5 MiB puts beat 1.75 MiB) and the count small (AR ≤ 8, AG ≤ 4).
3. **One counter atomic per block marks completion.** After its stores retire (`s_waitcnt vmcnt(0)` + barrier), one thread does an `acq_rel` atomic add on the chunk counter. That atomic is the release: an added fence cost 650–790 µs at AR M=16384, and per-lane fences grow with the square of the block count.
4. **Who posts depends on where the consumer is.**
   - *Next kernel* (GEMM + AR / AG / A2A): the block whose increment completes the chunk posts it ("last finisher") and resets the counter (graph-safe, no wrap). No comm workgroup. Post to different peers from different lanes: one thread costs ~2 µs per put (AG: 28 puts, 39–55 µs).
   - *Same kernel* (MoE dispatch / combine): a dedicated comm workgroup, one lane per peer on its own queue, polls the counters with relaxed system-scope loads and posts each complete chunk with a remote signal. One receiver wave polls every source's signal and publishes ready work (a ready queue); consumers claim it with one atomic; one acquire after the flag, none per tile or per poll. The comm workgroup holds a CU slot even when idle (4.6% of a GEMM2), so give the persistent grid all CUs.
5. **Finish in a tail kernel, not in the GEMM.** A quiet inside the GEMM was 1.8–10.7% slower. The drain kernel (one block, lane p for peer p): `quiet_queue` → system fence → store this call's epoch into peer p's flag → spin until p's flag in my window reaches it. The kernel boundary orders the data for whatever reads it next.
6. **Epochs live on the device.** Flags, signals and counters are running totals advanced by the kernels, never reset by the host, compared wrap-safe, so a graph replay behaves like a fresh launch.
7. **Guard buffer reuse.** An end-of-call barrier proves every peer *wrote* call e−1, not that it *read* it. When the output is the receive buffer, post credit `e` to every peer at call start and wait for peer d's credit before the first put to d.
8. **Fuse only when the GEMM can hide the transfer.** AG fuses only when `ceil(M/256)·ceil(N/256) ≥ CUs`; below that, and below one chunk per destination in AR, the split path (GEMM, then one put per peer) wins.
9. **One engine and one queue per peer link.** Splitting a run over more queues or engines is not faster. Chunks of one destination share its queue (AR, a lock serializes only the issue) or take one queue each when lanes post in lockstep (AG; a lock would deadlock there).

## Measured on MI355X (8 GPUs, xGMI)
- A put with a signal costs ≈ 4.3 µs + bytes / 59 GB/s on one queue; puts on one queue run one after another. The 7 peer links run concurrently at full rate (~350 GB/s out per GPU). ATOMIC and FENCE signal packets cost the same.
- A release fence is a whole-L2 writeback (`buffer_wbl2`); an acquire is a whole-L2 invalidate (`buffer_inv`). One acquire per tile cost 90 µs of a 0.96 ms kernel; keep both out of tile and poll loops (see ISA check). Write-through stores cost nothing.
- Cross-device access is uncached: the cost is serialized round trips, not bytes. Spread peers over lanes (8 lanes: 10 → 2.5 µs).
- Role code in a GEMM kernel can trigger `UnclusteredHighRPReschedule` and break the K-loop schedule (+8.5%). Confirm with the K-loop ISA check, then set `amdgpu-disable-unclustered-high-rp-reschedule`.
- A grid barrier needs every block resident at once. Window offsets are 32-bit (< 4 GiB per buffer resource); larger windows need `per_rank_vmm`. An engine-side wait (`post_wait_mem`) is released only by a plain system-scope store, not by an atomic.

## ISA check (after every kernel change, before any timing)
Dump with `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` and a fresh `FLYDSL_RUNTIME_CACHE_DIR`; read `*_final_isa.s`. Diff it against the previous version and against the same kernel without the communication role.
- **Resources**: VGPR / SGPR / AGPR counts, scratch (spill) = 0, LDS bytes. Blocks per CU must still fit the grid (a grid barrier deadlocks otherwise).
- **Cache ops**: count `buffer_wbl2` and `buffer_inv`, and find where they are. Locate each loop by its back edge (a branch to an earlier label); no cache op may sit between a tile loop's or a poll loop's label and its back edge. A total count alone hides a cache op inside a loop.
- **K loop**: same MFMA count and the same order. Compare the ordered instruction stream, not the set of instructions: the position of the first MFMA, and the mean distance from a load to its `s_waitcnt`. A shorter distance means the scheduler lost latency hiding.
- **Communication code**: payload stores to the window are write-through (`sc0 sc1`); poll loops use a relaxed system-scope load and `s_sleep`, with no `buffer_inv` inside.

## Rules
- **FlyDSL**: compile-time switches use `const_expr`. Put every switch, and any config that is not a closure scalar, into the kernel name; give each variant its own `FLYDSL_RUNTIME_CACHE_DIR`. A stale cache shows up as wrong results or hangs.
- **Attribution**: mark every claim not measured on the target code as "inferred". Before you say a mechanism transfers to another design, check each premise it depends on (asynchronous? occupies CUs? how many fences and grid barriers?).

## Workflow
1. **Contract**: before any work, write down the objective, the shapes and dtypes, the correctness rule (default: bit-exact, poison buffers), the baseline (its fastest configuration and versions), the target metric, the validation command, the evaluation command, and the promotion criteria (default: bit-exact, ≥3 interleaved rounds, gain above ±1%). If an item is missing and has no default, ask the user.
2. **Bound it**: run local-only (no communication) and spin-only comm-workgroup references to find the maximum possible gain. If it is below noise, stop.
3. **Draft**: write `docs/draft.md` before any code. Include the baseline and how it is validated, the main risks, the candidate designs ranked by expected gain and risk, the first steps, and the evidence needed to keep, revise, or reject each candidate.
4. **Off the GPU**: code, static checks, plain-Python simulation. New features are off by default, behind a switch.
5. **On the GPU**: smoke (one config, graph) → ISA check (above) → correctness (poison buffers, compare bit for bit) → A/B.
6. **Measure**: CUDA graph replay only, same process, ≥3 interleaved rounds. No eager timing. Do not align rank start times with all_reduce. Compare against the fastest configuration of the baseline; record the baseline and FlyDSL versions.
7. **GPU hygiene**: check that the GPUs are idle before a run; one job per node; a timeout on every run; after a timeout, kill the workers by pid.
8. **Decide** for each candidate, and record the reason with numbers:
   - ISA check fails (spill, cache op in a loop, K-loop schedule lost) → **revise**.
   - Not bit-exact → **revise**; **reject** if the cause is the design, not a bug.
   - Effect within ±1% → **reject**, record it, and stop this direction.
   - All checks pass and the gain is above ±1% → **keep**.

## References in this repo
- `python/mori/ops/{gemm_ar,gemm_ag,gemm_a2a}`: last-finisher posting + drain kernel. Read `README.md` first (measured results and dead ends), then `kernels_fused.py` (epilogue: counter, election, put), `kernels_sdma.py` (drain, flags) and `layout.py` (window).
- `python/mori/ops/sdma_allgather/op.py`: asynchronous all-gather split into start / wait.
- `examples/cco/python/11_flydsl_mega_moe_sdma`: GEMM2 + combine; comm waves inside the GEMM, one per peer, poll per-chunk counters.
- `examples/cco/python/13_flydsl_mega_moe_stage1_sdma`: dispatch + GEMM1; comm workgroup posts the chunks, a scheduler wave polls the signals and feeds a ready queue.

## Deliverables
- A result table: config, ours, baseline, speedup, rounds and range, baseline version, FlyDSL version.
- New measured facts, and "measured, no gain" entries with numbers, written into the op's `README.md`.
- Ask the user before any outward action (push, PR, comment).
