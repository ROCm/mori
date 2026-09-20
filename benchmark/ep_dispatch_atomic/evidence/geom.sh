#!/bin/bash
# Atomics on the shared offTokOff word = blocks x npes, independent of warps.
# So holding TOTAL WARPS fixed while trading blocks for warps-per-block changes
# the contention and (almost) nothing else. If contention is really costing us,
# fewer/fatter blocks must win.
set -u
cd /app/mori/tests/python/ops/dispatch_combine_v2
unset MORI_JIT_EXTRA_FLAGS
export LD_LIBRARY_PATH=/tmp/hiplink:${LD_LIBRARY_PATH:-}
export HIDDEN=7168 TOPK=6 EPR=96 DISP=fp4 SWEEP=512 ITERS=50 WARMUP=10 MODES=graph JSON=0
for g in "64 8" "32 16" "16 32" "128 4" "32 8" "16 16"; do
  set -- $g
  echo -n "blocks=$1 warps=$2 (warps_total=$(($1*$2)) atomics=$(($1*4)))  "
  DBN=$1 DWPB=$2 timeout 600 torchrun --standalone --nproc_per_node=4 bench_ep.py 2>&1 \
    | grep -E "^  ct=|FAIL|rror" | head -2 | tr '\n' ' '
  echo
done
