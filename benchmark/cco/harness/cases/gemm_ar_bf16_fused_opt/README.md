# Case gemm_ar_bf16_fused_opt: bring the fused candidate to the existing op

An evaluation case for optimization tasks (`kind: optimize`). The gemm_ar_bf16 case's first
run left two candidates. v2-stream runs the GEMM as eight chunk launches with the scatter
behind them and was accepted at the existing op's speed. v2-fused keeps the structure the
facts suggest for the existing op, one GEMM launch whose epilogue posts each (destination,
chunk) from the block that completes it, but measured 1208.2 µs against 1164.4 at
M=16384 (development runs). Can a session with the harness's process, its facts and
v2-fused's records bring v2-fused to the existing op's time without leaving its structure?

The task is in [contract.yaml](contract.yaml): start from `GemmArV2Fused` at commit
`1422e4b88`; keep the structure; be no slower than `GemmAllReduceOp(quant="bf16")` at
M=16384 in the same launch, with no regression at 8192 and 4096. The interface,
`op_harness.py`, the existing op's adapter and the correctness suite are
[gemm_ar_bf16](../gemm_ar_bf16/README.md)'s.

## Starting point

- **Code:** `python/mori/ops/gemm_ar_v2/` at `1422e4b88` on branch
  `xiangch/gemm-ar-bf16-eval-1`, merged onto the harness commit the run uses. The module
  carries both candidates: `GemmArV2` (stream) and `GemmArV2Fused` (`mode="fused"`), with
  switches for diagnosis (`stop_after`, `emit_put`, `wait`, `ctr_order`, `chunks`).
- **Records:** the first run's task directory: `design/v2-fused.md` (the design and its
  prediction, 1158 µs), `runs/ep2/` (development timings and the diagnostic variants),
  `decisions.md` (row dev-2: rejected, cause not decomposed) and `report.md` (the retry
  conditions: one SDMA queue per peer, release-only counter atomics, a per-tile breakdown).
- **What was measured** (development, M=16384): fused 1208.2 against existing 1164.4 and
  stream 1164.5; the call up to the scatter drain 648.7 for fused against 603.1 for
  stream; the fused GEMM with its counters but no puts, plus the drain, 411.6 against
  383.3 for one communication-off launch. Re-measure before relying on any of these.

## Clean room: the black box

The gemm_ar_bf16 case's clean room holds unchanged: its hidden paths, its settings (Read
denies and `guard.py`) and its audit. This case adds one permission, observing the
existing op as a black box:

| Allowed | Not allowed |
|---|---|
| running it through the case's files (`op_harness.py --impl existing`, the arms files) | its source, and the hidden benchmarks' and tests' source |
| its kernel trace: `op_harness.py --mode trace --impl existing`, the kernel names, starts and durations of every call on every rank | its ISA, IR, FlyDSL dumps or cache entries, or disassembling its code objects |
| phase times derived from that trace, for the gap ledger | dumping ISA from a launch that runs it (`run.py --dump-isa` with the existing arm) |

The candidate is the session's own: its code, its ISA, its timeline stamps and its traces
are open.

## Files

| File | Used by | For |
|---|---|---|
| [contract.yaml](contract.yaml) | owner, main session | the task, the start point, the target, the budget |
| [arms.dev.yaml](arms.dev.yaml) | implementer | the candidate (`CANDIDATE`) against the existing op and v2-stream, interleaved |
| [arms.verify.yaml](arms.verify.yaml) | verifier | the same, plus the existing benchmark as a cross-check of the timing |
| [kickoff.md](kickoff.md) | owner | the prompt for an unattended run |
| [results.md](results.md) | owner | one row per run, from `../evaluate.py` |

## How a run goes

1. **Contract.** Copy `contract.yaml` to `<records>/<task>/contract.yaml` and fill in
   `start_from.records`; the owner approves it.
2. **Checkout and session.** A branch from the harness commit with `1422e4b88`'s branch
   merged. Start the session with
   `claude --settings benchmark/cco/harness/cases/gemm_ar_bf16/settings.implementer.json`,
   or unattended from [kickoff.md](kickoff.md) as gemm_ar_bf16's README shows.
3. **Re-baseline and gap ledger (step 1).** Time the three arms at every M:

   ```bash
   CANDIDATE=mori.ops.gemm_ar_v2.op:GemmArV2Fused python benchmark/cco/harness/run.py \
       --arms benchmark/cco/harness/cases/gemm_ar_bf16_fused_opt/arms.dev.yaml \
       --baseline existing --aa --rounds 3 --bench-args "-m 16384" --out <records>/runs/<id>
   ```

   Then trace both sides and set their phase tables next to each other:

   ```bash
   torchrun --standalone --nproc_per_node=8 \
       benchmark/cco/harness/cases/gemm_ar_bf16/op_harness.py --mode trace \
       --impl existing --timeline-out <records>/runs/<id>/existing.trace.json
   python benchmark/cco/harness/timeline.py <records>/runs/<id>/existing.trace.json
   ```

   and the same with `--impl mori.ops.gemm_ar_v2.op:GemmArV2Fused`. The trace shows
   kernels, not the copy engines; what happens inside the fused GEMM (when each chunk is
   posted) needs the candidate's own stamps (`--mode timeline`, see the loop skill's
   `reference/diagnosis.md`). `bounds.md` holds the ledger: the gap to the existing op,
   split by phase, largest first.
4. **Rounds (steps 2 and 3).** `hypotheses.md`, then one change per round behind a switch,
   timed with `arms.dev.yaml` plus variant arms of your own (`--impl-kwargs`,
   `validate: false` for cut-off variants), and frozen when it pays.
5. **Verify (step 4).** A fresh `cco-fusion-verifier`: `op_harness.py --mode check` with a
   seed of its own, and `arms.verify.yaml` with `--rounds 3 --aa` at 16384, 8192 and
   4096.
6. **Audit and record.** `audit.py` as in gemm_ar_bf16 (step 6), then:

   ```bash
   python benchmark/cco/harness/cases/evaluate.py \
       --case benchmark/cco/harness/cases/gemm_ar_bf16_fused_opt --records <records> \
       --transcript <session>.jsonl --transcript <session dir> --audit <records>/audit.json \
       --append benchmark/cco/harness/cases/gemm_ar_bf16_fused_opt/results.md
   ```
