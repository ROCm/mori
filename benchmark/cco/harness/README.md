# benchmark/cco/harness

Scripts and templates behind the `cco-fusion-loop` skill
(`.claude/skills/cco-fusion-loop`), the workflow for building fused GEMM + collective
operators. Unit tests for the scripts are in `tests/python/cco/harness/` and run on
the CPU: `python -m pytest tests/python/cco/harness`.

**The GPUs are the user's.** The harness never allocates or reserves nodes. It runs where
it is started (on a GPU node, or in a container that has the GPUs), or behind a launcher
prefix the user sets in the contract (`environment.launcher`), such as a container exec
or an `srun` into an allocation the user already holds.

| File | Status | Used in | Purpose |
|---|---|---|---|
| `contract.example.yaml` | exists | step 0 | template for a task's contract |
| `preflight.py` | planned | every run | record the environment, refuse a run that cannot be valid |
| `run.py` | planned | steps 1, 3, 4 | one measurement entry for every operator: runs the benchmark on the user's GPUs with an idle check, a timeout and cleanup of its own processes, and writes one result schema |
| `bound.py` | exists | step 1 | split-path phase times to `T_ideal`, `T_real` and go or no-go |
| `bounds.example.yaml` | exists | step 1 | `bound.py` input: the gemm_ar worked example (`G950-GAR-01`) |
| `isa_check.py` | exists | steps 3, 4 | the ISA check on a FlyDSL `*_final_isa.s` |
| `schemas/` | planned | all | JSON schemas for result, candidate and verdict files |

## bound.py

`python bound.py <input.yaml|json> [--out-md bounds.md] [--out-json bounds.json]`

The input lists, per bucket, the split path's phases measured inside the pipeline (max
over ranks), each with a role: `compute` (the GEMM, `a`), `overlap` (transfer legs that
depend only on this GEMM, `b`) or `serial` (`T_serial`); optionally the whole-call
`split_us`, a `concurrent_us` contention run and the GEMM's `tiles`. Top-level keys:
`t0_us` (a number or `[low, high]`), `n_max` or `n`, `noise_band_pct`, `min_gain_pct`,
`cus`. See `bounds.example.yaml`.

Per bucket it reports `T_ideal = T_serial + max(a, b)` and, for every `n`,
`T_real = T_serial + c + m + (n - 1) * max(c, m)` with `c = a / n` and
`m = (b + (n - 1) * t0) / n` (a two-stage pipeline; its limits are the two formulas in
`.claude/skills/cco-fusion-loop/reference/bounds.md`). The decision is go when the
saving at the high `t0` and its best `n` reaches `max(min_gain_pct, noise_band_pct)` of
`T_split`, both sides have something to overlap, and the GEMM has at least one tile per
CU. Exit status 0, or 2 for unusable input.

## isa_check.py

`python isa_check.py <dump-dir-or-file> [--kernel REGEX] [--baseline <comm-off dump> [--baseline-kernel REGEX]] [--json report.json] [--max-distance-drop 0.10] [--require-write-through] [--strict]`

Per kernel, from a FlyDSL `*_final_isa.s`: resources from the `amdhsa.kernels` metadata
(VGPR, SGPR, spills, scratch, LDS); loops by their back edges, classed as poll (own
`s_sleep`), K loop (own MFMAs), outer or other; every `buffer_wbl2` / `buffer_inv` with
the loop around it, following calls the dump resolves (`@rel32` / `@gotpcrel32`
symbols); stores by cache policy; and the main K loop's MFMA count, first-MFMA position
and mean load-to-`s_waitcnt` distance. With `--baseline` it compares resources and the K
loop.

| Verdict | When |
|---|---|
| FAIL | VGPR spills or scratch; an acquire inside any loop; a release inside a K loop, a poll loop or any loop with MFMAs; against the baseline, another MFMA count or set of mnemonics, or a mean load-to-wait distance shorter by more than `--max-distance-drop` |
| WARN | SGPR spills; a release inside another loop (once per tile can be right); more VGPRs or LDS than the baseline; another MFMA order; stores that are not write-through when others are (FAIL with `--require-write-through`); a call inside a loop it cannot follow; a target other than gfx950 |

`--strict` makes every WARN a FAIL. Exit status 0 without FAIL, 1 with a FAIL, 2 when
the dump cannot be read with confidence. It does not know which stores write a cco
window, and the distance assumes in-order completion of vector-memory operations: use it
to compare two builds of one kernel.

## Planned interfaces

### preflight.py

`python preflight.py --out <run-dir>/env.json`

Records the mori commit and build flags, `MORI_ENABLE_SDMA`, ROCm, torch, FlyDSL, the
GPU model and gfx architecture, the node, the image, the xGMI topology and the processes
on each GPU. Exits non-zero when the architecture is not the one the facts cover, when
SDMA is not built or not enabled (`G950-ENV-01`), or when a GPU is in use. Warns on the
environment issues in `.claude/skills/known-issues` (`G950-ENV-03`, `G950-ENV-04`).

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

It runs each command where the session runs, or behind `environment.launcher` from the
contract, unchanged. Before an 8-rank run it checks that the GPUs are idle
(`preflight.py`) and waits about 10 s after the previous one (`G950-ENV-05`). Each command
runs in its own process group with a timeout and its output logged; on timeout it
terminates that process group only. It never allocates nodes, never kills by name and
never resets a GPU.
