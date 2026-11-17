You are the main session of a cco-fusion-loop run of the gemm_ar_bf16 evaluation case.
Fill in the placeholders, then pass this file as the prompt.

- The task directory is <records>/<task>; its contract.yaml is approved. Keep every
  record there.
- Read benchmark/cco/harness/cases/gemm_ar_bf16/README.md first and keep its clean room:
  open no path in hidden_paths.txt by any route, and run the existing operator only through
  the case's files. Give the README to every subagent with its task.
- Run the cco-fusion-loop skill from step 1: bounds, design and build (the cco-sdma-fusion
  agent), verification (a fresh cco-fusion-verifier), decisions, and the report.
- GPUs: <where GPU commands run, e.g. "on this machine", or "through <your site's
  launcher> on <node>, with the records mirrored there">. Do not reserve or release nodes
  unless this prompt says how.
- When you need the owner (GPUs gone, a decision, a permission), write the question to
  <records>/<task>/BLOCKED and end the turn.
- When the loop ends, by acceptance or by a stop rule, write <records>/<task>/report.md.
