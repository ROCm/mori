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
| `preflight.py` | exists | every run | record the environment, refuse a run that cannot be valid |
| `run.py` | exists | steps 1, 3, 4 | one measurement entry: modes or arms interleaved in one launch on the user's GPUs, with preflight, a timeout and cleanup of its own processes |
| `bench_driver.py` | exists | (run.py) | runs several benchmarks, each from its own file, in one torchrun launch |
| `bound.py` | exists | step 1 | split-path phase times to `T_ideal`, `T_real` and go or no-go |
| `bounds.example.yaml` | exists | step 1 | `bound.py` input: the gemm_ar worked example (`G950-GAR-01`) |
| `isa_check.py` | exists | steps 3, 4 | the ISA check on a FlyDSL `*_final_isa.s` |
| `loop_state.py` | exists | all | where a task stands (next step, budget used, done or waiting for the owner), read from its records |
| `stop_hook.py` | exists | unattended | the Stop hook that keeps a session on the loop until the task is done, needs the owner or is stopped |
| `schemas/` | exists | all | JSON schemas for `run.py`'s result, the candidate manifest, the verdict and a case's evaluation record |
| `cases/` | exists | evaluation | tasks that test the harness itself; `gemm_ar_bf16` re-implements BF16 GEMM + all-reduce in a clean room; `evaluate.py` records each run |

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
and mean load-to-`s_waitcnt` distance. FlyDSL fully unrolls the K loop when K is a
compile-time constant (the gemm_ar kernels at K=2048 have no back edge); the same metrics
are then taken over the span from the first to the last MFMA. With `--baseline` it
compares resources and the K loop.

| Verdict | When |
|---|---|
| FAIL | VGPR spills or scratch; an acquire inside any loop; a release inside a K loop, a poll loop or any loop with MFMAs; against the baseline, another MFMA count or set of mnemonics, or a mean load-to-wait distance shorter by more than `--max-distance-drop` |
| WARN | SGPR spills; a release inside another loop (once per tile can be right); more VGPRs or LDS than the baseline; another MFMA order; stores that are not write-through when others are (FAIL with `--require-write-through`); a call inside a loop it cannot follow; a target other than gfx950 |

`--strict` makes every WARN a FAIL. Exit status 0 without FAIL, 1 with a FAIL, 2 when
the dump cannot be read with confidence. It does not know which stores write a cco
window, and the distance assumes in-order completion of vector-memory operations: use it
to compare two builds of one kernel.

## preflight.py

`python preflight.py [--out env.json] [--repo <mori checkout>] [--require-arch gfx950] [--allow-busy] [--no-require-sdma]`

Never initialises HIP: it finds `mori`, `torch` and `flydsl` without importing them and
asks the GPUs through `rocminfo` and `rocm-smi`. It records the host, the repository
commit, where the three packages load from and their versions, the `BUILD_CCO_SDMA` flag
baked into the installed mori (`mori/cco/device/_build_flags.py`), the ROCm version, the
GPUs, the processes on them, the link types and `CONFIG_DMABUF_MOVE_NOTIFY`.

| Verdict | When |
|---|---|
| FAIL (exit 1) | no GPU, or a GPU of another architecture; a process on a GPU, or `rocm-smi` unreadable (unknown is not idle; `--allow-busy` makes it a WARN); mori not built with SDMA or `MORI_ENABLE_SDMA` not `1` (`G950-ENV-01`); mori not importable |
| WARN | ROCm 7.2.x (`G950-ENV-04`); `CONFIG_DMABUF_MOVE_NOTIFY` off or unknown (`G950-ENV-03`); a link that is not xGMI; mori loaded from outside `--repo`; uncommitted changes |

## run.py

`python run.py (--op gemm_ar --modes <m1,m2,...> | --arms <arms.yaml>) [--baseline <arm>] [--rounds 3] [--aa] [--bench-args "<shape and switches>"] [--nproc 8] [--timeout 1800] [--launcher "<prefix>"] [--dump-isa] [--no-require-sdma] --out <run-dir>`

An arm is a benchmark file and its arguments. `--op`/`--modes` makes one arm per mode of
a registered benchmark; `--arms` reads them from a file, each arm with its own benchmark,
so two implementations run interleaved in the same launch:

```yaml
arms:
  - name: existing
    bench: benchmark/cco/harness/cases/gemm_ar_bf16/op_harness.py
    args: --mode time --impl existing  # the existing op, by the case's alias
  - name: candidate
    bench: benchmark/cco/harness/cases/gemm_ar_bf16/op_harness.py
    args: --mode time --impl python/mori/ops/gemm_ar_v2/op.py:FusedGemmAllReduce
    time_field: max_rank_time_us      # default; the benchmark's maximum over ranks
```

One invocation is one shape (`G950-MEAS-03`): `--bench-args` sets it for every arm, and
the record keeps the benchmark's own report of it. It launches `torchrun` once, through
`bench_driver.py`, so every arm runs in the same processes, interleaved round by round;
every arm is compared with `--baseline` (default: the first), and `--aa` runs the
baseline again at the end of each round for the noise band. Times are each
run's maximum over ranks; the summary gives the median, min, max and spread per mode.

Before the launch it runs `preflight.py` and stops on a refusal, and for multi-rank runs it
waits until 10 s have passed since the previous one it started on this host
(`G950-ENV-05`). The launch runs in its own process group, under coreutils `timeout` on the
executing side and a timeout here; on timeout it terminates that group and every process
it started, including those in other sessions. It never allocates nodes.

`--launcher` is a prefix such as `docker exec <container>`: the command names the
repository by absolute path, so the repository must sit at the same path on both sides,
and cleanup on timeout is left to the `timeout` it runs there. `--dump-isa` makes FlyDSL
write every kernel's final ISA into `<run-dir>/isa` (with a fresh runtime cache) and
records each with its sha256 and the result of `isa_check.py`.

It writes `result.json` (`schemas/result.schema.json`), `env.json`, `stdout.log` and
`stderr.log`, and refuses to overwrite an existing `result.json` without `--force`. Exit
status: 0 when every run completed and validated, 1 when a run failed validation, 2 for a
usage error or a preflight refusal, 3 for a timeout or missing results.

Only `gemm_ar` is registered for `--op`. The other operator benchmarks (`gemm_ag`,
`gemm_a2a`, `ag_gemm`, `a2a_gemm`) have the same `build_parser()` / `run()` /
`RESULT_JSON` shape and report the maximum over ranks as `us`: run them through `--arms`
with `time_field: us`, or give each an entry in `OPS`.

## loop_state.py and stop_hook.py

```bash
python benchmark/cco/harness/loop_state.py <records>/<task> [--json]
```

`loop_state.py` reads a task's records directory and says where the loop stands:
`stopped` (the owner created `STOP`), `needs-owner` (`BLOCKED`, or the contract is not
approved), `done` (`report.md` exists) or `continue`, with the next step (bounds, design,
a frozen candidate, its verdict, the decision, delivery) and the rounds and GPU hours
used against the contract's budget. A spent budget is `continue` with "write the report"
as the next step.

`stop_hook.py` is the unattended driver: a Stop hook, inert unless `CCO_FUSION_RECORDS`
names the task directory, that blocks the session from stopping while the state is
`continue` and hands it the next step. It lets the session end after
`CCO_FUSION_MAX_CONTINUES` continuations (100), or when nothing under the task directory
changed across `CCO_FUSION_MAX_FLAT` continuations (3), writing `BLOCKED` with the
reason. The loop skill's "Unattended runs" has the launch command.

## schemas/

JSON Schema (2020-12) for `run.py`'s `result.json`, the implementer's candidate manifest,
the verifier's verdict and `cases/evaluate.py`'s `evaluation.json`. The templates in
`.claude/skills/cco-fusion-loop/templates/` carry every field these schemas require; the
tests check both.

## cases/

Tasks that evaluate the harness rather than an operator: a fixed contract, an interface,
a shared harness that times and checks any implementation of it, a clean-room rule with
its guard and audit, and a recorded baseline. See
[cases/gemm_ar_bf16/README.md](cases/gemm_ar_bf16/README.md).

[cases/evaluate.py](cases/evaluate.py) turns a finished run into `evaluation.json`
(`schemas/evaluation.schema.json`): the result against the contract's target, rounds, GPU
hours, the owner's prompts after the kickoff, wall time, tokens and cost, and the audit.
`--append <case>/results.md` adds the run's row to the case's results, the baseline that
later changes to the harness are compared against.
