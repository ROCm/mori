#!/usr/bin/env bash
# Run the EP8 dispatch/combine bench inside the rocm/mori:ci container.
# Usage: ./run_bench.sh [extra env ...]   e.g.  MODE=eager SWEEP=128 ./run_bench.sh
set -euo pipefail
DIR=/home/jiahzhou/workspace/mori/tests/python/ops/dispatch_combine_v2
ENVS=""
for kv in MODE SWEEP ITERS WARMUP HIDDEN TOPK EPR DISP_BLOCK COMB_BLOCK WARP_NUM \
          UNROLL S3_CACHE STDMOE DTYPE QUANT; do
  v="${!kv:-}"; [ -n "$v" ] && ENVS="$ENVS $kv=$v"
done
docker exec -w "$DIR" mori_cco_test bash -lc \
  "rm -rf /root/.flydsl/cache/* 2>/dev/null; $ENVS timeout ${TIMEOUT:-600} \
   torchrun --standalone --nproc_per_node=${NP:-8} bench_dispatch_combine.py 2>&1 | \
   grep -v -E 'Gloo|OMP_NUM_THREADS|run.py:774|^\\*+$|^W[0-9]'"
