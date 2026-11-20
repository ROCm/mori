You are the main session of a cco-fusion-loop run of the gemm_ar_bf16_fused_opt evaluation
case, an optimization task. Fill in the placeholders, then pass this file as the prompt.

- The task directory is <records>/<task>; its contract.yaml (kind: optimize) is approved.
  Keep every record there.
- Read benchmark/cco/harness/cases/gemm_ar_bf16_fused_opt/README.md first, then the
  gemm_ar_bf16 case's README for the clean room it shares. Keep both: open no path in
  hidden_paths.txt by any route; observe the existing operator only as the black box the
  README allows. Give both READMEs to every subagent with its task.
- Start from the candidate in start_from: its code is in this checkout
  (python/mori/ops/gemm_ar_v2/), its records in <the first run's task directory>.
- Run the cco-fusion-loop skill's optimization steps: re-baseline and the gap ledger,
  hypotheses, one change per round (the cco-sdma-fusion agent), verification (a fresh
  cco-fusion-verifier), decisions, and the report with the trajectory.
- GPUs: <where GPU commands run, e.g. "on this machine", or "through <your site's
  launcher> on <node>, with the records mirrored there">. Do not reserve or release nodes
  unless this prompt says how.
- When you need the owner (GPUs gone, a decision, a permission), write the question to
  <records>/<task>/BLOCKED and end the turn.
- When the loop ends, by meeting the target or by a stop rule, write
  <records>/<task>/report.md.
