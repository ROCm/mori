#!/usr/bin/env bash
# GEMM+AR fusion at DeepSeek-V4.1-Flash's wo_b shapes, against the V4-Pro shape
# the original work targeted.
#
#   wo_b = RowParallelLinear(o_groups*o_lora_rank -> hidden), so per rank
#     V4.1-Flash  o_groups=8  o_lora=1024  hidden=5120 -> K=8192/tp, N=5120
#     V4-Pro      o_groups=16 o_lora=1024  hidden=7168 -> K=16384/tp, N=7168
#
# world_size must equal tp: the all-reduce is over exactly that many ranks.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-/workspace/dsv41/sweep_dsv41.jsonl}
: > "$OUT"

run() {   # world n k m mode label
  local w=$1 n=$2 k=$3 m=$4 mode=$5 label=$6
  echo "### $label  w=$w n=$n k=$k m=$m mode=$mode" >&2
  MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \
    timeout 900 torchrun --standalone --nproc_per_node="$w" bench_gemm_ar.py \
      --mode "$mode" -m "$m" -n "$n" -k "$k" --quant blockscale 2>/dev/null \
    | grep '^RESULT_JSON' \
    | LABEL="$label" python3 -c "
import json, os, sys
for line in sys.stdin:
    d = json.loads(line.split(' ', 1)[1])
    d['label'] = os.environ['LABEL']
    print(json.dumps(d))
" >> "$OUT" || echo "  (failed)" >&2
}

MODES="gemm-only split-sdma split-lsa fused-sdma"

for m in 16384 8192 4096; do
  for mode in $MODES; do
    run 8 5120 1024 "$m" "$mode" "v41-tp8"
    run 4 5120 2048 "$m" "$mode" "v41-tp4"
    run 8 7168 2048 "$m" "$mode" "pro-tp8"
  done
done
echo "done -> $OUT" >&2
