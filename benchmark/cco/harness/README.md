# benchmark/cco/harness

Scripts and templates behind the `cco-fusion-loop` skill
(`.claude/skills/cco-fusion-loop`), the workflow for building fused GEMM + collective
operators. Unit tests for the scripts belong in `tests/python/cco/harness/` and run on
the CPU.

| File | Status | Used in | Purpose |
|---|---|---|---|
| `contract.example.yaml` | exists | step 0 | template for a task's contract |
| `preflight.py` | planned | every run | record the environment, refuse a run that cannot be valid |
| `gpu_run.sh` | planned | every GPU run | one GPU job: allocation, idle check, timeout, logs, cleanup of its own processes |
| `run.py` | planned | steps 1, 3, 4 | one measurement entry for every operator, writing one result schema |
| `bound.py` | planned | step 1 | split-path measurements to `T_ideal`, `T_real` and go or no-go |
| `isa_check.py` | planned | step 3 | the ISA check on a FlyDSL `*_final_isa.s` |
| `schemas/` | planned | all | JSON schemas for result, candidate and verdict files |

## Planned interfaces

### preflight.py

`python preflight.py --out <run-dir>/env.json`

Records the mori commit and build flags, `MORI_ENABLE_SDMA`, ROCm, torch, FlyDSL, the
GPU model and gfx architecture, the node, the image, the xGMI topology and the processes
on each GPU. Exits non-zero when the architecture is not the one the facts cover, when
SDMA is not built or not enabled (`G950-ENV-01`), or when a GPU is in use. Warns on the
environment issues in `.claude/skills/known-issues` (`G950-ENV-03`, `G950-ENV-04`).

### gpu_run.sh

`gpu_run.sh --timeout <s> --log <run-dir> -- <command...>`

Gets GPUs through a site hook (`CCO_FUSION_ALLOC`, a command prefix; empty means the
caller already holds the node), runs `preflight.py`, waits about 10 s after the previous
8-rank run (`G950-ENV-05`), and runs the command in its own process group with its
output logged. On timeout it signals the job to dump its flags and counters, then
terminates its own process group only. It never kills by name and never resets a GPU.

### run.py

`python run.py --contract <contract.yaml> --op gemm_ar --mode <mode> --m <M> [--set key=value ...] --out <run-dir>`

Wraps the operator's benchmark (`benchmark/cco/flydsl/<op>/bench_*.py --json-out`), one
shape per process, with interleaved rounds and the maximum over ranks. Writes
`result.json` with the command, the scope (commit, versions, node), the switches, the
code-object hashes, the correctness verdict, the per-round times, the median and the
spread. Modes follow the benchmark (`gemm-only`, `split-sdma`, `split-lsa`, `split-rccl`,
`fused-sdma`, ...) plus `concurrent`: the GEMM and the collective on two streams on
independent buffers, for the contention measurement in step 1. gemm_ar first; other
operators as their benchmarks gain the same modes.

### bound.py

`python bound.py --contract <contract.yaml> --split <result.json>... --concurrent <result.json> --t0 4,9 --n 1-16 --out <records>/bounds.md`

Per bucket: `T_split`, `T_ideal = T_serial + max(a, b)`, `T_real(n)` for the
compute-bound and the communication-bound cases, the best `n`, and go or no-go against
the contract's `min_gain_pct` and the noise band. Writes `bounds.md` and `bounds.json`.
The model is in `.claude/skills/cco-fusion-loop/reference/bounds.md`.

### isa_check.py

`python isa_check.py <kernel>_final_isa.s [--baseline <comm-off>_final_isa.s] --out isa.json`

Reports the resources (VGPR, SGPR, AGPR, scratch, LDS, occupancy from the assembler
metadata); the loops, found by their back edges; the count and placement of
`buffer_wbl2` and `buffer_inv`; window stores without `sc0 sc1`; and, against the
baseline, the K-loop MFMA count and order, the position of the first MFMA and the mean
load-to-`s_waitcnt` distance (`G950-ISA-01`). Exits non-zero on scratch use, a cache
operation inside a loop, or a changed K-loop.
