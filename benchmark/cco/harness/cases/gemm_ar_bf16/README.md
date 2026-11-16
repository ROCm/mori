# Case gemm_ar_bf16: re-implement BF16 GEMM + all-reduce

An evaluation case for the fusion harness. Can an agent instance with the harness's
process and knowledge, but without the existing implementation's code, build a BF16
GEMM + all-reduce that runs within 5% of `GemmAllReduceOp(quant="bf16")`?

The task is in [contract.yaml](contract.yaml): 8 ranks, N=7168, K=2048, M up to 16384,
target at M=16384, correctness against the fp32 sum over ranks. The operator's interface
is in [interface.py](interface.py).

## Clean room

Nobody in the session (the main session, the implementer, the verifier) reads the
existing gemm_ar fusion, its benchmarks, its tests or its documentation:
[hidden_paths.txt](hidden_paths.txt). Not by any route (a script that opens a file, `git
show`, `inspect.getsource`, a grep over a parent directory), and not in compiled form
(FlyDSL dumps, ISA, IR, cache entries). Everything else is open, including the GEMM
building blocks (`_gemm_a8w8_8wave.py`, `_gemm_a16w16_8wave.py`), the sibling operators,
the cco device API and the `cco-fusion-facts` skill. The case measures reproduction with
the harness's knowledge, not discovery from nothing; the facts and the agents' design
rules come largely from the existing operator.

The harness's documents were written around gemm_ar, so they cite hidden files and name
hidden tools. A citation is not an invitation: take a fact or a reference as written and
do not open its source. Hidden code runs only as a black box, through this case's files,
which keep hidden paths off the command line:

| The harness's documents say | In this case |
|---|---|
| read gemm_ar's `README.md`, `kernels_fused.py`, `kernels_sdma.py`, `layout.py` (the implementer's references) | hidden; use the facts (cite the IDs), the sibling operators `gemm_ag` and `gemm_a2a`, and the cco device API |
| time with `benchmark/cco/flydsl/gemm_ar/timing.py`; one shape per process with `sweep.py` | hidden; use `run.py --arms` with [op_harness.py](op_harness.py) arms, one shape per invocation |
| measure the split path and the GEMM alone with `bench_gemm_ar.py` (`bounds.md`) | run it, do not read it: [arms.bounds.yaml](arms.bounds.yaml) |
| the existing op as the baseline | `op_harness.py --impl existing`, as in [arms.dev.yaml](arms.dev.yaml) |
| `bench_gemm_ar.py --stop-after` times a prefix of the pipeline (`diagnosis.md`) | add a switch of your own to the candidate |
| a CI ordering test in the style of `test_gemm_ar_bench.py` (step 5) | hidden; write your own: fused and split timed in one process, fused faster |
| `run.py --dump-isa` | it dumps every kernel the launch compiles, the existing op's too: dump with [arms.isa.yaml](arms.isa.yaml), the candidate alone |

`op_harness.py` times one call per CUDA graph, as the gemm_ar benchmark does. A
single-call graph costs up to 13.4 µs more than its kernels (`G950-MEAS-01`), the same for
every arm; at M ≥ 4096 a call takes 350 µs or more.

How the rule is kept:

- [settings.implementer.json](settings.implementer.json) denies the Read tool on the
  hidden paths anywhere on the filesystem (`//**/`, so the installed copies too), and
  denies outward actions. Claude Code applies a Read deny to its file tools (the Grep
  tool leaves denied files out of a directory search), to the file commands it
  recognizes in Bash (`cat`, `head`, `tail`, `sed`) and to IDE context, but not to a
  script that opens a file itself, `git show` or a shell grep over a parent directory.
  Load it for the whole session with `claude --settings <path>`.
- The same settings run [guard.py](guard.py) before every Bash call, the subagents'
  included. It refuses a command that names a hidden path or module (`python3 -c
  "open(...)"`, `git show HEAD:<path>`, `import mori.ops.gemm_ar.op`) and a recursive
  search (`grep -r`, `rg`, `git grep`, `find | xargs grep`) whose scope holds a hidden
  file, and points to the Grep tool instead.
- [audit.py](audit.py) checks afterwards whether hidden **text** reached the session.
  It scans the session's transcripts (subagents and long tool results included) for the
  lines only hidden files contain; a path or a module name is not text, so reading the
  facts or this file is fine. Each finding is one of:
  - `leak`: hidden text came back, from a hidden file, from a call that named it, or
    three lines with no source. Any leak voids the run, unless a person shows that the
    lines are a coincidence.
  - `review`: one or two lines with no source, or hidden source lines in a Python
    traceback from a black-box run that failed. A person decides.
  - `blocked`: a call named a hidden path or module and was refused.
  - `attempt`: a call named a hidden path or module, ran, and returned no hidden text.

## Files

| File | Used by | For |
|---|---|---|
| [contract.yaml](contract.yaml) | owner, main session | the task, targets, budget |
| [interface.py](interface.py) | everyone | what an implementation exposes |
| [op_harness.py](op_harness.py) | implementer (development), verifier (acceptance) | `--mode time` and `--mode check` for any implementation of the interface; `--impl existing` for the existing op |
| [existing.py](existing.py) | `op_harness.py --impl existing` | the existing op behind the interface (hidden: run it, do not read it) |
| [arms.bounds.yaml](arms.bounds.yaml) | implementer | step 1: the split path and the GEMM alone, through the existing benchmark |
| [arms.dev.yaml](arms.dev.yaml) | implementer | candidate against the existing op, interleaved |
| [arms.isa.yaml](arms.isa.yaml) | implementer, verifier | ISA dumps of the candidate alone |
| [arms.verify.yaml](arms.verify.yaml) | verifier | candidate against the existing op, plus the existing benchmark as a cross-check of the timing |
| [hidden_paths.txt](hidden_paths.txt), [settings.implementer.json](settings.implementer.json), [guard.py](guard.py), [audit.py](audit.py) | main session | the clean room |
| [kickoff.md](kickoff.md) | owner | the prompt for an unattended run |
| [results.md](results.md) | owner | one row per run of the case, from `../evaluate.py` |

## How a run goes

1. **Contract.** Copy `contract.yaml` to `<records>/<task>/contract.yaml`; the owner
   approves it.
2. **Session.** Start a fresh Claude Code session in a checkout of this branch with
   `claude --settings benchmark/cco/harness/cases/gemm_ar_bf16/settings.implementer.json`
   and give it the `cco-fusion-loop` skill, the contract and this file. GPUs come from
   the owner (a node or container, or a launcher prefix in the contract). To run it
   unattended (the loop skill's "Unattended runs"), fill in [kickoff.md](kickoff.md) and:

   ```bash
   export CCO_FUSION_RECORDS=<records>/<task>
   claude -p "$(cat <filled kickoff.md>)" \
       --settings benchmark/cco/harness/cases/gemm_ar_bf16/settings.implementer.json \
       --add-dir "$CCO_FUSION_RECORDS" --permission-mode auto --max-budget-usd <cap> \
       --output-format stream-json --verbose >"$CCO_FUSION_RECORDS/driver.jsonl" 2>&1
   ```

   The session goes on until `report.md` exists, it writes `BLOCKED` with a question for
   the owner, or the owner creates `STOP`.
3. **Bounds (step 1).** Time the split path and the GEMM alone at the target bucket,
   and the existing op with the `existing` arm of `arms.dev.yaml`:

   ```bash
   python benchmark/cco/harness/run.py \
       --arms benchmark/cco/harness/cases/gemm_ar_bf16/arms.bounds.yaml \
       --baseline split-bench --rounds 3 --bench-args "-m 16384" --out <records>/runs/<id>
   ```

   Record the existing op's own rel_l2 with `op_harness.py --mode check --impl existing`.
   Phase times of the existing pipeline at this shape are in `G950-GAR-09`.
4. **Build (steps 2 and 3).** The candidate lives in a new module outside the hidden
   paths (for example `python/mori/ops/gemm_ar_v2/`) and implements `interface.py`.
   Development timing:

   ```bash
   python benchmark/cco/harness/run.py --arms <arms.dev.yaml with --impl filled in> \
       --baseline existing --aa --bench-args "-m 16384" --out <records>/runs/<id>
   ```

   Correctness while developing:

   ```bash
   torchrun --standalone --nproc_per_node=8 \
       benchmark/cco/harness/cases/gemm_ar_bf16/op_harness.py \
       --mode check --impl <path/to/candidate.py>:<Class>
   ```

5. **Verify (step 4).** A fresh `cco-fusion-verifier` runs `--mode check` with a seed
   of its own (`--seed`, so the held-out M differ from any the implementer saw) and the
   timing with `arms.verify.yaml`, `--rounds 3 --aa`.
6. **Audit.** In the checkout the session started from:

   ```bash
   python benchmark/cco/harness/cases/gemm_ar_bf16/audit.py \
       ~/.claude/projects/<project>/<session-id>.jsonl \
       ~/.claude/projects/<project>/<session-id>/ --json <records>/audit.json
   ```

   `void` (exit status 1) ends the evaluation; `review` needs a person to read the items;
   `clean` passes. The hidden text is taken from the last commit that changed
   `hidden_paths.txt`; pass `--base <commit>` if the session started elsewhere.
7. **Record.** The verifier's verdict goes to `<records>/verdicts/<candidate>.json`, with
   the target bucket's timing against the existing op (`baseline.impl: existing`). Then:

   ```bash
   python benchmark/cco/harness/cases/evaluate.py \
       --case benchmark/cco/harness/cases/gemm_ar_bf16 --records <records> \
       --transcript ~/.claude/projects/<project>/<session-id>.jsonl \
       --transcript ~/.claude/projects/<project>/<session-id>/ \
       --audit <records>/audit.json --append benchmark/cco/harness/cases/gemm_ar_bf16/results.md
   ```

   It writes `<records>/evaluation.json` (whether the target was met at M=16384, the
   rounds from `decisions.md`, GPU hours in run.py launches, the owner's prompts after the
   kickoff, wall time, tokens and cost, the audit) and adds a row to
   [results.md](results.md), the baseline that later changes to the harness are compared
   against. `--audit-reviewed` once a person has cleared the audit's review items.

## Baseline

Measured when the case was set up (2026-11-16, `crsuse2-m2m-v3-046`, 8x MI355X, ROCm
10.0.0, torch 2.11.0+rocm10.0.0, FlyDSL 0.4.2, mori `4b87505e5` plus this case). Re-measure
at step 1 of every run: the image, ROCm, FlyDSL and the node change.

| M | existing op (op_harness) | existing benchmark, same config | split-sdma | GEMM only |
|---:|---:|---:|---:|---:|
| 16384 | 1169.5 µs | 1168.8 µs | 1443.5 µs | 394.0 µs |
| 8192 | 627.6 µs | 627.1 µs | | |
| 4096 | 359.4 µs | 357.5 µs | | |

- Times are the median of interleaved rounds, each the maximum over ranks; spread across
  rounds 0.05-0.6%, A/A band 0.04% at M=16384. The split and GEMM-only columns come from
  a separate launch on the same node an hour earlier.
- `op_harness` and the existing benchmark agree within 0.06-0.5% on the same op
  configuration, so the two time the op the same way. One SDMA queue per peer (the op's
  default) and eight time the same at every M above.
- The existing op's rel_l2 against the fp32 reference is 2.35e-3 (2.38e-3 at M=1) under
  `--mode check`, on M = 16384, 1, 10657 and 6309 (seed 1), eager and replayed; the
  tolerance is 6e-3. It has no counter hook, so its wrap test is "not covered".
- The target at M=16384 is therefore about 1228 µs, judged on the existing op measured in
  the same launch as the candidate.
