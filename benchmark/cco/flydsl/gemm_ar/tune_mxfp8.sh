#!/usr/bin/env bash
# Sweep one knob of the mxfp8 GEMM at DeepSeek-V4.1-Flash's wo_b shapes.
#
#   REPS=3 ./tune_mxfp8.sh "--block-m 128" "--block-m 256"
#
# Every cell is repeated and the minimum kept: contention on this box only ever
# adds time, and a single shot has been seen to read 205.6us and 86.0us on
# consecutive runs of the same configuration. The spread is printed so a
# contaminated cell is visible rather than silently averaged in.
set -uo pipefail
cd "$(dirname "$0")"

REPS=${REPS:-3}
QUANT=${QUANT:-mxfp8}
NPROC=${NPROC:-4}
MS=${MS:-"4096 8192 12288 16384"}
KS=${KS:-"2048 1024"}

occupancy() {
  rocm-smi --showmeminfo vram --csv 2>/dev/null |
    awk -F, 'NR>1&&$1!=""{printf "%.0f ",$3/1073741824}'
}

echo "occupancy before: $(occupancy)" >&2

for K in $KS; do
  echo ""
  echo "=== N=5120 K=$K  quant=$QUANT  world=$NPROC  (min of $REPS, spread in parens) ==="
  printf "%7s" "M"
  for cfg in "$@"; do printf " %22s" "$cfg"; done
  echo ""
  for M in $MS; do
    printf "%7s" "$M"
    for cfg in "$@"; do
      best=""; worst=""
      for _ in $(seq "$REPS"); do
        v=$(MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \
            timeout 600 torchrun --standalone --nproc_per_node="$NPROC" \
              bench_gemm_ar.py --mode gemm-only --quant "$QUANT" \
              -m "$M" -n 5120 -k "$K" --warmup 5 --iters 31 $cfg 2>/dev/null |
            grep -oP 'max_rank_time=\K[0-9.]+')
        [ -z "$v" ] && continue
        best=$(python3 -c "print(min($v, ${best:-1e9}))")
        worst=$(python3 -c "print(max($v, ${worst:-0}))")
      done
      if [ -z "$best" ]; then
        printf " %22s" "FAIL"
      else
        sp=$(python3 -c "print(f'{($worst-$best)/$best*100:.0f}')")
        printf " %15sus (%3s%%)" "$(printf '%.1f' "$best")" "$sp"
      fi
    done
    echo ""
  done
done

echo "" >&2
echo "occupancy after: $(occupancy)" >&2
