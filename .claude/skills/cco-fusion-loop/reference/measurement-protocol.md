# Measurement protocol

Used by the implementer for development runs and by the verifier for acceptance runs.
Each rule below once produced a confident wrong number; the gemm_ar README's
"Measurement traps" section tells those stories.

`benchmark/cco/harness/run.py` applies the launch-side rules: modes interleaved in one
process, the maximum over ranks, an A/A pair (`--aa`), one shape per invocation, the 10 s
gap, a timeout with cleanup, and `preflight.py` before the launch. Cold or hot, eager or
graph, and the M grid are chosen by the caller through the modes and `--bench-args`.
To compare two implementations, give each an arm (`--arms`) so they run interleaved in
the same launch, timed by the same code.

- **Graph replay, several calls per graph.** A single-call graph has a floor of about
  13.4 µs (`G950-MEAS-01`). Use `benchmark/cco/flydsl/gemm_ar/timing.py`.
- **Maximum over ranks** for each iteration. Do not align the ranks' start times with an
  all-reduce inside the timed region.
- **Interleaved A/B in the same process,** over several rounds. gemm_ar uses 8 calls per
  graph, 21 replays and 3 alternating rounds (`G950-MEAS-06`). Report the median and the
  spread.
- **Noise band:** the spread of A/A runs in the same session. A difference inside it is
  no difference.
- **One shape per process** (`sweep.py`, `G950-MEAS-03`).
- **Cold and hot.** A hot loop reads weights out of the 256 MB LLC; cold is what predicts
  a server for memory-bound kernels. Report both, with a cold ring longer than the
  graph's calls (`G950-MEAS-02`).
- **Eager only if the deployment is eager.** An eager benchmark charges per-launch Python
  overhead to the fused path (`G950-MEAS-04`).
- **Dense M near any threshold.** The baseline is not monotonic in M (`G950-GAR-07`), and
  a threshold fitted on a coarse grid is fitted to the gap.
- **Assert the fused path ran** (`G950-MEAS-05`), and run `self_test()` once
  (`G950-ENV-01`).
- **Record the scope** with every result: mori commit, FlyDSL and ROCm versions, node,
  image, the exact command, and the switches.
- **GPU hygiene.** Check that the node is idle before a run; one job per node; a timeout
  on every run; on timeout, kill only your own processes; about 10 s between 8-rank runs
  (`G950-ENV-05`).
