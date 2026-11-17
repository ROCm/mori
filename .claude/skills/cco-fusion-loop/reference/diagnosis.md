# Step 3: correctness, measurement, diagnosis

The implementer runs this on every change. The verifier runs the correctness gate and
the measurement again, independently, at step 4.

## Correctness gate (before any timing)

- Compare with the reference bit for bit, or against the contract's error bound. Fill the
  outputs with a poison value first.
- Run several consecutive calls, and a long graph replay.
- Preset counters and epochs just below 2^32. ag_gemm and a2a_gemm already test this;
  gemm_ar, gemm_ag and gemm_a2a do not yet (`G950-CTR-04`).
- Add random start skew between ranks.
- Put a timeout on every run. On timeout, dump each rank's flags and counters before
  killing the job: hangs are the common failure mode of these kernels.
- Run the operator's `self_test()` once (`G950-ENV-01`), and assert that the fused path
  ran: a silent fallback looks exactly like a slow fused path (`G950-MEAS-05`).

## ISA check (after every kernel change, before timing)

Run `python benchmark/cco/harness/isa_check.py <dump-dir> --baseline <comm-off-dump-dir>`.
It checks the resources, where every cache operation sits (following calls), the store
cache policy and the K loop against the communication-off build, and exits 1 on a FAIL.
It cannot tell which stores write a cco window, and it does not inspect poll-loop loads;
read those in the ISA yourself. The list below is what it automates and what to read.

Dump with `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` and a fresh
`FLYDSL_RUNTIME_CACHE_DIR`, then read `*_final_isa.s`. Compare it against the previous
version and against the same kernel with communication off.

- **Resources:** VGPR, SGPR, AGPR, scratch (must stay 0), LDS. Blocks per CU must still
  fit the grid; a grid barrier deadlocks otherwise.
- **Cache operations:** count `buffer_wbl2` and `buffer_inv` and find where they are.
  None may sit inside a tile loop or a poll loop. Find each loop by its back edge (a
  branch to an earlier label); a total count hides a cache operation inside a loop. The
  counter's `acq_rel` atomic emits one pair per block (`G950-SYNC-01`), which is
  expected.
- **K loop:** the same MFMA count and the same order as the communication-off kernel.
  Compare the ordered instruction stream, not the set: the position of the first MFMA
  and the mean distance from a load to its `s_waitcnt`. A shorter distance means the
  loop hides less latency (`G950-ISA-01`).
- **Communication code:** payload stores to the window are write-through (`sc0 sc1`);
  poll loops use a relaxed system-scope load and `s_sleep`, with no `buffer_inv`.

## Measure the candidate

Measure the full kernel and every step-2 variant under
[measurement-protocol.md](measurement-protocol.md), then again with the timeline on.
For gemm_ar, `bench_gemm_ar.py` already has `--stop-after scatter` (stop after the only
phase fusion can affect), `--no-put` (keep the epilogue, drop the transfer), `--chunks`,
`--atomic` and `--tile-order`.

## Find the critical side

Read the timeline: is the GEMM or the transfer on the critical path? The levers differ.

- **Communication-bound:** slowing the GEMM barely matters. Reduce the bytes, reduce the
  number of puts, start earlier. In `G950-GAR-01` the GEMM grew by 52 µs while the
  728 µs transfer set the time.
- **Compute-bound:** the fusion tax on the GEMM and the exposed tail matter most.

## Decompose the gap

Account for `T_fused - T_real`, item by item:

| Source | How to measure | Usual causes | Usual remedies |
|---|---|---|---|
| The GEMM itself | GEMM with communication off vs the fastest standalone GEMM | the fused kernel's GEMM is slower than the best library GEMM | match the GEMM first, or report it separately |
| Fusion tax | GEMM with communication off vs the same GEMM without the communication code | register pressure, spills, a broken K-loop schedule (`G950-ISA-01`), write-through stores and atomics in the epilogue, a tile order that hurts L2 reuse | keep role code out of the K loop; `amdgpu-disable-unclustered-high-rp-reschedule`; a leaner epilogue |
| Exposed head or tail | timeline: when the first put starts; from the last tile done to the last put done | chunks too large or too few, a large last chunk, a late start | retune `n`, shrink the last chunk, issue earlier |
| Synchronization | the signals-only variant; count and placement of cache operations in the ISA | an added fence (`G950-SYNC-02`), an acquire per tile (`G950-SYNC-03`), a quiet inside the GEMM (`G950-SYNC-04`) | one atomic per block, one acquire, completion in the drain kernel |
| Slower transfers | timeline per destination vs the standalone collective | puts serialized on one queue or issued by one thread (`G950-SDMA-02`), HBM contention with the GEMM, peer memory in host memory (`G950-ENV-03`) | one queue per link, several issuing lanes, rotated destinations; check the environment (`.claude/skills/known-issues`) |
| Rank imbalance | per-rank timelines: the slowest rank sets the time | start skew, one slow GEMM, uneven MoE routing | design for the slowest rank, balance the load |

## Iterate

Take the largest item. Write down the change and its predicted effect, put it behind a
switch, run an interleaved A/B, and record keep, revise or reject with the numbers in
`decisions.md`. One change per iteration.

## Tools

- **Device timeline.** Stamp `s_memrealtime` (a 100 MHz clock, constant across DVFS)
  from one thread into a u64 slot of the rank's own window at the points the
  decomposition needs: a block's first and last tile, an election, a put posted, a
  drain's quiet and flag. Expose the stamps through the operator's `timeline_read()`,
  named as `benchmark/cco/harness/timeline.py` expects (`phase:start` and `phase:end`,
  `name@key` for one per block or per destination), behind a switch that is off by
  default. Collect them with `op_harness.py --mode timeline --impl-kwargs
  '{"stamps": true}' --timeline-out <file>` and read `timeline.py <file>`: the phase
  table of the slowest rank, with the gaps where nothing ran. Compare stamps only
  within a rank. The device side is a few lines:

  ```python
  i64 = ir.IntegerType.get_signless(64)
  now = fx.Int64(llvm.call_intrinsic(i64, "llvm.amdgcn.s.memrealtime", [], [], []))
  llvm.store(now.ir_value(), signal_ptr(base + slot * 8), alignment=8, volatile_=True)
  ```

  `python/mori/ops/gemm_a2a/timeline.py` is a worked example (and
  `benchmark/cco/flydsl/a2a_gemm/timeline_a2a_gemm.py` its reader). In a case's clean
  room copy it rather than import it: importing `mori.ops.gemm_a2a` loads gemm_ar.
- **Variants without new classes.** `op_harness.py --impl-kwargs '{"mode": "fused"}'`
  passes constructor switches. In a `run.py --arms` file give every variant an arm, and
  mark one whose output is wrong by design (a phase cut off, puts suppressed) with
  `validate: false`: it is timed, and its failed check does not fail the run.
- Kernel trace for phase times and gaps: `rocprofv3 --kernel-trace`. Profiling a
  multi-GPU job takes the whole node.
- GEMM counters and roofline: rocprofv3 counters and rocprof-compute; ATT for the K loop.
- ISA comparison: the FlyDSL `isa-resource-diff` skill.
