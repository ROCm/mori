# Case gemm_ar_bf16: re-implement BF16 GEMM + all-reduce

An evaluation case for the fusion harness. Can an agent instance with the harness's
process and knowledge, but without the existing implementation's code, build a BF16
GEMM + all-reduce that runs within 5% of `GemmAllReduceOp(quant="bf16")`?

The task is in [contract.yaml](contract.yaml): 8 ranks, N=7168, K=2048, M up to 16384,
target at M=16384, correctness against the fp32 sum over ranks. The operator's interface
is in [interface.py](interface.py).

## Clean room

The implementer may not read the existing gemm_ar fusion, its benchmark, its tests or
its documentation: [hidden_paths.txt](hidden_paths.txt). It may read everything else,
including the GEMM building blocks (`_gemm_a8w8_8wave.py`, `_gemm_a16w16_8wave.py`), the
sibling operators, the cco device API and the `cco-fusion-facts` skill. The case measures
reproduction with the harness's knowledge, not discovery from nothing; the facts and the
agent's design rules come largely from the existing operator.

How the rule is kept:

- [settings.implementer.json](settings.implementer.json) denies the Read tool on the
  hidden paths and denies outward actions. Load it for the whole session with
  `claude --settings <path>`. A shell command can still read a hidden file, so this is a
  guard, not the enforcement.
- [audit.py](audit.py) is the enforcement. After the run, it scans the session's
  transcripts, subagents included, for tool calls that name a hidden path (**access**)
  and tool results that contain one (**exposure**). Any access voids the run; review
  every exposure.

Running the existing operator as a black-box baseline is allowed; that is what
`existing.py` is for.

## Files

| File | Used by | For |
|---|---|---|
| [contract.yaml](contract.yaml) | owner, main session | the task, targets, budget |
| [interface.py](interface.py) | everyone | what an implementation exposes |
| [op_harness.py](op_harness.py) | implementer (development), verifier (acceptance) | `--mode time` and `--mode check` for any implementation of the interface |
| [existing.py](existing.py) | verifier; run as a baseline by anyone | the existing op behind the interface |
| [arms.dev.yaml](arms.dev.yaml) | implementer | candidate against the existing op, interleaved |
| [arms.verify.yaml](arms.verify.yaml) | verifier | the same, plus the existing benchmark as a cross-check of the timing |
| [hidden_paths.txt](hidden_paths.txt), [settings.implementer.json](settings.implementer.json), [audit.py](audit.py) | main session | the clean room |

## How a run goes

1. **Contract.** Copy `contract.yaml` to `<records>/<task>/contract.yaml`; the owner
   approves it.
2. **Session.** Start a fresh Claude Code session in a checkout of this branch with
   `claude --settings benchmark/cco/harness/cases/gemm_ar_bf16/settings.implementer.json`
   and give it the `cco-fusion-loop` skill, the contract and this file. GPUs come from
   the owner (a node or container, or a launcher prefix in the contract).
3. **Bounds (step 1).** Time the existing op and the split path at the target bucket and
   record the existing op's own rel_l2 under `--mode check`. Phase times of the existing
   pipeline at this shape are in `G950-GAR-09`.
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
6. **Audit.** `python benchmark/cco/harness/cases/gemm_ar_bf16/audit.py
   ~/.claude/projects/<project>/<session-id>.jsonl ~/.claude/projects/<project>/<session-id>/`
   must report no access.
7. **Report.** Whether the target was met at M=16384, the M sweep, rounds and GPU hours
   used, how often the owner stepped in, and the audit result. This run's numbers are the
   harness's first evaluation baseline.

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
