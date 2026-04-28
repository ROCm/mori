#!/usr/bin/env bash
# tools/bench_plan_a_sweep.sh
# -----------------------------------------------------------------------------
# Sweep Plan A v2 CU budget / R:A split for Test 6 (multi-stage overlap) @ 256MB.
#
# Usage (inside ROCm container):
#   cd /home/fizhang/test/mori
#   bash tools/bench_plan_a_sweep.sh
#
# Env overrides:
#   VARIANTS     default: "80:24 96:32 112:40 128:48"
#                format: "<MORI_PLAN_A_CU>:<MORI_PLAN_A_NR> ..."
#   ITERATIONS   default: 50
#   WARMUP       default: 10
#   SIZE_MB      default: 256
#   NUM_STAGES   default: 4
#   PIPELINE_CU  default: 160
#   SKIP_PULL    default: 0
#   SKIP_BUILD   default: 0
#
# Build command must remain:
#   BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${VARIANTS:=80:24 96:32 112:40 128:48}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=50}"
: "${WARMUP:=10}"
: "${PIPELINE_CU:=160}"
: "${SKIP_PULL:=0}"
: "${SKIP_BUILD:=0}"
ELEMS=$(( SIZE_MB * 1024 * 1024 / 4 ))

cd "$REPO"
[ -f "pyproject.toml" ] || { echo "ERROR: not mori repo root: $REPO"; exit 1; }

echo "==================== [preflight] ===================="
hostname
id -un
pwd
echo "VARIANTS=$VARIANTS"
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ITERATIONS=$ITERATIONS WARMUP=$WARMUP PIPELINE_CU=$PIPELINE_CU"
command -v python3 >/dev/null || { echo "MISSING: python3"; exit 1; }
command -v cmake >/dev/null || { echo "MISSING: cmake"; exit 1; }
command -v hipcc >/dev/null || { echo "MISSING: hipcc"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'HIP not available'; print('devices:', torch.cuda.device_count())"
git rev-parse --abbrev-ref HEAD
git log -1 --oneline

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo
  echo "==================== [pip install] ===================="
  BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
fi

LOG="/tmp/plan_a_sweep_$(date +%s).log"
echo
echo "==================== [run] LOG=$LOG ===================="

run_variant() {
  local label="$1" cu="$2" nr="$3"
  echo
  echo "========== $label =========="
  echo "ENV: MORI_PIPELINE_CU=$PIPELINE_CU MORI_DIRECT_OUTPUT=1 MORI_PLAN_A_CU=$cu MORI_PLAN_A_NR=$nr"
  env MORI_PIPELINE_CU="$PIPELINE_CU" \
      MORI_DIRECT_OUTPUT=1 \
      MORI_PLAN_A_CU="$cu" \
      MORI_PLAN_A_NR="$nr" \
      python3 tests/python/ccl/test_allreduce.py \
        --num-stages "$NUM_STAGES" \
        --elems "$ELEMS" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" 2>&1
}

{
  echo "========== HEAD =========="
  git log -1 --oneline
  for spec in $VARIANTS; do
    cu="${spec%%:*}"
    nr="${spec##*:}"
    run_variant "PLAN_A_CU${cu}_NR${nr}" "$cu" "$nr"
  done
} | tee "$LOG"

echo
echo "################################################################"
echo "## PLAN A v2 SWEEP SUMMARY (auto-extracted from $LOG)"
echo "################################################################"

awk -v size="$SIZE_MB" '
  /^========== PLAN_A_/ {
    gsub(/=/, ""); gsub(/^ +| +$/, "");
    label = $0
  }
  /Plan A active/ {
    active[label] = $0
  }
  $0 ~ ("^[[:space:]]*" size " MB[[:space:]]*\\|") {
    # Table 1 / 2 / 3 / 4 have same first field; infer by nearby header.
    if (section == "wall") wall[label] = $0
    else if (section == "slow") slow[label] = $0
    else if (section == "ar") ar[label] = $0
    else if (section == "gemm") gemm[label] = $0
  }
  /Table 1: Overlap Wall Time/ { section="wall" }
  /Table 2: GEMM Slowdown/ { section="slow" }
  /Table 3: Sequential AllReduce Time/ { section="ar" }
  /Table 4: Sequential GEMM Time/ { section="gemm" }
  END {
    printf "%-22s | %-74s\n", "label", "Plan A active"
    for (l in active) printf "%-22s | %s\n", l, active[l]
    print ""
    print "---- wall ----"
    for (l in wall) printf "%-22s %s\n", l, wall[l]
    print ""
    print "---- GEMM slowdown ----"
    for (l in slow) printf "%-22s %s\n", l, slow[l]
    print ""
    print "---- seq_ar ----"
    for (l in ar) printf "%-22s %s\n", l, ar[l]
  }
' "$LOG"

echo
echo "---- correctness / failures ----"
grep -E "All Tests PASSED|FAILED|Traceback|STUCK|BUG:" "$LOG" | tail -40 || true

echo
echo "LOG: $LOG"
