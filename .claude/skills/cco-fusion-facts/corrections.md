# Corrections

One entry per change to a fact or to a claim in an agent or skill. Newest first.

| ID | Date | What | Was | Now | Evidence |
|---|---|---|---|---|---|
| C-003 | 2026-11-16 | Per-put cost model | `cco-sdma-fusion` agent: "≈ 4.3 µs + bytes / 59 GB/s on one queue" | Unverified (U-01). Use `G950-SDMA-01` and `G950-SDMA-05`, or measure | No in-repo source; ROCm/mori#859's description fits 5.8 µs + bytes / 60.9 GB/s, which disagrees on the intercept |
| C-002 | 2026-11-16 | The agent's "Measured on MI355X" list | Numbers inline in the agent prompt | Moved here. Sourced: `G950-SDMA-02` (28 packets, 39-55 µs), `G950-SDMA-04` (3.5 vs 1.75 MiB), `G950-ISA-01` (+8.5%), `G950-SYNC-01..04`. Unsourced: U-01..U-06 | Search at `29e0c0868`; see the Unverified table in [gfx950.md](gfx950.md) |
| C-001 | 2026-11-16 | Counter reset as shared design | Agent rule 4: the last finisher "resets the counter (graph-safe, no wrap)", stated for all SDMA ops | Only gemm_ag and gemm_a2a reset (`G950-CTR-01`); gemm_ar uses monotonic counters with a modulo election (`G950-CTR-03`) | `gemm_ag/kernels_fused.py:806-811`; `gemm_ar/kernels_fused.py:1901,1975,1992` |
