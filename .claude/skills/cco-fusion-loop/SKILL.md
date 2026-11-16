---
name: cco-fusion-loop
description: >-
  Drive the compute-communication fusion loop for a mori fused GEMM +
  collective operator (GEMM + AR/RS/AG/A2A, AG or A2A + GEMM, MoE
  dispatch/combine over cco SDMA or LSA with FlyDSL on gfx950 / MI355X): write
  the contract, measure the split baseline and the fusion bounds, decide go or
  no-go, have the cco-sdma-fusion agent design and build candidates, have the
  cco-fusion-verifier agent verify a frozen candidate, then accept or route
  back. Use when the user asks to fuse, port, tune or evaluate such an operator
  for a model or a shape, or asks whether fusing is worth it. The main session
  runs this loop and does not write kernel code itself.
---

# cco-fusion-loop

The main session runs this loop. It owns the contract, the records and every decision,
and it is the only one that talks to the user. It delegates building to the
`cco-sdma-fusion` agent and acceptance measurements to the `cco-fusion-verifier` agent.

## Roles

| Role | Who | Does | Does not |
|---|---|---|---|
| Owner | the user | approves the contract, any change to it, and every outward action | |
| Orchestrator | the main session, with this skill | contract, bounds, briefs, routing, records, final report | write kernel code; certify a candidate |
| Implementer | `cco-sdma-fusion` agent | designs, builds, debugs, self-checks, freezes candidates | judge its own candidate; change the contract |
| Verifier | `cco-fusion-verifier` agent: a fresh context, on a different model from the implementer | checks a frozen candidate on held-out shapes and times it against the baseline | edit code; tune; reuse the implementer's numbers |

Subagents cannot ask the user anything. A subagent that lacks an input returns the
list of what is missing, and the orchestrator asks the owner.

The two agents pin different models (`opus` for the implementer, `sonnet` for the
verifier). If you change one, keep them different.

## Before starting

- **Branch.** Work on a branch based on current main; the benchmarks and tests this
  loop calls live there.
- **Records.** One directory per task, outside Git:
  `${CCO_FUSION_RECORDS:-$HOME/.mori-cco-fusion}/<task>/`. Layout in [Records](#records).
- **Facts.** Numbers come from the `cco-fusion-facts` skill, cited by ID. Anything not
  there is measured in this task or labelled `inferred`.
- **FlyDSL skills.** `isa-resource-diff` and the other FlyDSL skills live in the FlyDSL
  repository. If they are not visible, link them into `~/.claude/skills/`.
- **GPUs.** Follow the site's allocation rules and never use a GPU another job holds.
  Multi-GPU jobs and profiling take the whole node. One job per node, a timeout on
  every run, and about 10 s between 8-rank runs (`G950-ENV-05`).
- **Architecture.** This loop and its facts are for gfx950. On any other architecture,
  stop and tell the owner; the facts and the ISA rules do not carry over.

## The loop

| Step | Who | Output | Leave when |
|---|---|---|---|
| 0. Contract | orchestrator with the owner | `contract.yaml` | the owner approves it |
| 1. Baseline and bounds | orchestrator; measurements may be delegated to the implementer | `bounds.md` | go or no-go is decided |
| 2. Design | implementer | `design/<candidate>.md` | predictions and switches are written down |
| 3. Build, check, diagnose | implementer | development runs, candidate manifest | a candidate is frozen |
| 4. Verify, then accept or route | verifier, then orchestrator | `verdict.json`, a row in `decisions.md` | accepted, routed back, or stopped |
| 5. Deliver | orchestrator, implementer | operator, dispatch rule, tests, `report.md` | the owner approves outward actions |

### Step 0. Contract

Copy `benchmark/cco/harness/contract.example.yaml` to `<records>/contract.yaml` and fill
it in with the owner: workload and shapes, baseline, correctness rule, protocol,
targets, budget and stop rule, environment. If an item has no value and no default,
ask the owner; do not guess. The owner approves the contract. Any later change to
shapes, baseline, correctness rule, targets, budget or protocol needs approval again.

### Step 1. Baseline and bounds

Measure the fastest GEMM, the fastest collective, the split path end to end with its
in-pipeline phase times, and the GEMM and collective running concurrently. Compute
`T_ideal` and `T_real` per bucket (with an `n` sweep), estimate the other levers
(a narrower wire, another collective, a faster GEMM), and decide go or no-go. Method and
a worked example: [reference/bounds.md](reference/bounds.md). Write `bounds.md`.

### Step 2. Design

Brief the implementer with the contract path, `bounds.md`, the fact IDs that apply and
the records path. It returns one design note per candidate
([templates/design-note.md](templates/design-note.md)), following
[reference/design-checklist.md](reference/design-checklist.md), each with a predicted
time per bucket. Rank the candidates by expected gain and risk and choose the next one.

### Step 3. Build, check, diagnose

The implementer builds the chosen candidate behind switches, passes its own
correctness gate and ISA check, measures the variants, decomposes the gap and iterates
one change at a time ([reference/diagnosis.md](reference/diagnosis.md)). When it
considers the candidate ready, it freezes it: a commit plus a candidate manifest
([templates/candidate.example.json](templates/candidate.example.json)). Its numbers are
development numbers, not acceptance numbers.

### Step 4. Verify, then accept or route

Start a fresh `cco-fusion-verifier` with three paths only: the contract, the candidate
manifest and the records directory. Do not pass the implementer's conclusions. The
verifier checks out the frozen commit, checks the code-object hashes, runs the
correctness gate on held-out shapes it draws itself, measures the candidate and the
baseline under [reference/measurement-protocol.md](reference/measurement-protocol.md),
and writes `verdict.json` ([templates/verdict.example.json](templates/verdict.example.json)).

Accept the candidate when all of these hold:

1. Correctness and robustness pass on the frozen build.
2. In every target bucket it beats the fastest split path, the stream-level overlap and
   any existing fused implementation (where they exist) by more than the noise band.
3. `η_real` meets the contract's target, where
   `η = (T_split - T_fused) / (T_split - T_bound)`, computed for `T_ideal` and `T_real`.
   `η_ideal` shows what a different structure could still give. Explain any `η` above 1
   before believing it.
4. The gain is attributed to the GEMM, the wire format and the overlap separately.
5. No other bucket regresses; the crossover M between fused and split is recorded as the
   dispatch rule.

Otherwise route by the finding:

| Finding | Next |
|---|---|
| The gap is mostly fusion tax, synchronization or issuing | step 3, same design |
| The gap is structural: the exposed tail follows from the dependencies, the consumer is in the wrong place, or the transport does not fit | step 2, another design |
| The measurements contradict the model: more contention than assumed, or the baseline changed | step 1, redo the bounds |
| `η_real` near 1 but `η_ideal` low | step 2 (a structure that overlaps more) or step 1 (another lever, such as a narrower wire) |
| A stop rule fired | stop: record the negative result and the crossover |

Every pass through step 4 adds a row to `decisions.md`
([templates/records.md](templates/records.md)).

### Step 5. Deliver

- The operator API and its per-bucket dispatch rule (fused or split, from the crossover).
- CI tests: correctness, counter wrap, an assertion that the fused path ran, and one
  same-process ordering (fused faster than split) in the style of
  `tests/python/cco/test_gemm_ar_bench.py`. Absolute times are recorded, not asserted.
- `report.md` ([templates/records.md](templates/records.md)).
- New measured facts and "measured, no gain" entries in the operator README, and new or
  corrected facts in `cco-fusion-facts`.
- Push, pull request and review comments only after the owner approves them.

## Stop rules

Stop and report when any of these holds:

- step 1 says no-go;
- the contract's number of rounds passes with no gain above the noise band;
- the budget (rounds or GPU hours) is spent;
- the remaining gap to `T_real` is below the noise band;
- a permission, safety or environment problem needs the owner.

Report the best verified result, the negative results with their numbers and retry
conditions, and the crossover. Do not describe a stopped run as close to the bound
unless the verifier's numbers show it.

## Records

```text
<records>/<task>/
  contract.yaml            approved by the owner
  bounds.md                step 1
  design/<candidate>.md    step 2, one per candidate
  runs/<run-id>/           result.json, timeline, ISA summary, log (development and verification)
  candidates/<id>.json     frozen candidate manifests
  verdicts/<id>.json       verifier output
  decisions.md             one row per pass through step 4
  report.md                the final table
```

## Per-run settings

[settings.example.json](settings.example.json) denies outward actions and commands that
touch other tenants (GPU resets, `pkill`). Load it for a run with
`claude --settings <path>`; do not merge it into the project-wide settings.

## Rules

- Correctness before timing.
- Every number carries its scope: mori commit, FlyDSL and ROCm versions, node, shape,
  dtype, protocol.
- Predict, then measure. A result better than the bound is a bug or a wrong model until
  shown otherwise.
- One change per iteration, behind a switch, recorded.
- The author does not certify; the verifier does not tune.
- Negative results are recorded with their numbers and retry conditions.
