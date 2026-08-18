# SDMA Dispatch Notes

This directory documents the work done to use SDMA in MoRI's intra-node expert
parallel dispatch path.

Start here:

- [SDMA_DISPATCH_ATTEMPTS.md](./SDMA_DISPATCH_ATTEMPTS.md): technical history,
  design attempts, measurements, and conclusions.
- [EXPERIMENT_RUNBOOK.md](./EXPERIMENT_RUNBOOK.md): how to rebuild, run, profile,
  parse, and reproduce the experiments.
- [FULL_SWEEP_RUNBOOK.md](./FULL_SWEEP_RUNBOOK.md): portable BF16/FP8 full-sweep
  instructions for reproducing and comparing results on another machine.

Important current-state note:

- The workspace was intentionally moved to detached commit
  `f56a005e0776c10113986a76614c3a6d8def0677` for the latest no-return SDMA
  atomic experiment.
- On top of that commit, the local tree has a small patch adding
  `CreateAtomicAddNoReturnPacket()` and using operation `111` for SDMA
  completion atomic packets.
- The many `trace_intranode_rank*.json` files and `bench_results/` directories
  are local experiment artifacts.
