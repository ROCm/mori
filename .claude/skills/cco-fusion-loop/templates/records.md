# Record templates

## decisions.md

One row per pass through step 4 (and per development iteration, if the implementer keeps
its own log). Numbers are the median and spread, maximum over ranks, with the run IDs
that produced them.

| Round | Candidate | Hypothesis | Change | Predicted | Measured | Decision | Reason | Runs |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | keep / revise / reject / stop | | |

## report.md

### Result

| Bucket | Fused (µs) | Best baseline (µs) | Which baseline | Speedup | η_ideal | η_real | Rounds, spread |
|---|---:|---:|---|---:|---:|---:|---|
| | | | | | | | |

### Attribution
Per bucket: what the GEMM, the wire format and the overlap each contributed.

### Dispatch rule
The crossover M between fused and split, and the rule the operator uses.

### Negative results
What was measured and did not help, with numbers and the condition under which it is
worth retrying.

### Scope
mori commit, FlyDSL and ROCm versions, torch, image, node, protocol, and the verdict IDs
the numbers come from.
