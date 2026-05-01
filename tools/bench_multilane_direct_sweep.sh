#!/usr/bin/env bash
# Usage:
#   bash tools/bench_multilane_direct_sweep.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 NUM_STAGES=4 CONTINUOUS_ITERS=100 \
#   VARIANTS="ML_3F3R_B8:3F3R:8 ML_6F6R_B8:6F6R:8 ML_3F3R_B16:3F3R:16 ML_6F6R_B16:6F6R:16" \
#   CASE_TIMEOUT_SEC=900 SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_multilane_direct_sweep.sh

set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=1}"
: "${WARMUP:=1}"
: "${CONTINUOUS_ITERS:=100}"
: "${PIPELINE_CU:=224}"
: "${PIPELINE_CHUNKS:=4}"
: "${VARIANTS:=ML_3F3R_B8:3F3R:8 ML_6F6R_B8:6F6R:8 ML_3F3R_B16:3F3R:16 ML_6F6R_B16:6F6R:16}"
: "${CASE_TIMEOUT_SEC:=900}"
: "${SKIP_PULL:=0}"
: "${SKIP_BUILD:=0}"

ELEMS=$(( SIZE_MB * 1024 * 1024 / 4 ))

cd "$REPO"
[ -f "setup.py" ] || [ -f "pyproject.toml" ] || {
  echo "ERROR: REPO=$REPO does not look like mori root"; exit 1;
}

echo "==================== [preflight] ===================="
hostname
id -un
pwd
command -v python3 >/dev/null || { echo "MISSING: python3"; exit 1; }
command -v git >/dev/null || { echo "MISSING: git"; exit 1; }
command -v cmake >/dev/null || { echo "MISSING: cmake"; exit 1; }
command -v hipcc >/dev/null || { echo "MISSING: hipcc"; exit 1; }
command -v timeout >/dev/null || { echo "MISSING: timeout"; exit 1; }
FREE_KB=$(df -Pk /tmp "$REPO" | awk 'NR>1 {if (min=="" || $4<min) min=$4} END {print min+0}')
if [ "$FREE_KB" -lt $((20 * 1024 * 1024)) ]; then
  echo "ERROR: low disk space; min free across /tmp and repo is ${FREE_KB} KB (<20GB)"
  df -h /tmp "$REPO"
  exit 1
fi
cmake --version | head -1
hipcc --version 2>/dev/null | head -2 || true
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "HIP not available"
print("devices:", torch.cuda.device_count())
PY
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git status --short || true
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ELEMS=$ELEMS CONTINUOUS_ITERS=$CONTINUOUS_ITERS PIPELINE_CU=$PIPELINE_CU PIPELINE_CHUNKS=$PIPELINE_CHUNKS"
echo "VARIANTS=$VARIANTS"

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
  git log -1 --oneline
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo
  echo "==================== [pip install] ===================="
  pip install -e .
fi

LOG="/tmp/perf_multilane_direct_sweep_$(date +%s).log"

run_case() {
  local label="$1"
  shift
  echo
  echo "========== $label =========="
  echo "ENV: $*"
  timeout --signal=TERM "$CASE_TIMEOUT_SEC" env "$@" python3 tests/python/ccl/test_allreduce.py \
    --num-stages "$NUM_STAGES" \
    --elems "$ELEMS" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --continuous-iters "$CONTINUOUS_ITERS" \
    --continuous-phase-iter 5 \
    --continuous-phase-stage 0 2>&1
}

{
  echo "========== HEAD =========="
  git log -1 --oneline
  run_case "BASELINE" \
    MORI_CONTINUOUS_PREP=0 \
    MORI_PIPELINE_CU="$PIPELINE_CU" \
    MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"

  for spec in $VARIANTS; do
    IFS=: read -r label lanes bpl <<<"$spec"
    run_case "$label" \
      MORI_CONTINUOUS_PREP=0 \
      MORI_MULTILANE_DIRECT=1 \
      MORI_MULTILANE_DIRECT_LANES="$lanes" \
      MORI_MULTILANE_BLOCKS_PER_LANE="$bpl" \
      MORI_PIPELINE_CU="$PIPELINE_CU" \
      MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"
  done
} | tee "$LOG"

echo
echo "################################################################"
echo "## MULTILANE DIRECT SWEEP SUMMARY (auto-extracted from $LOG)"
echo "################################################################"

awk -v size="$SIZE_MB" '
  /^========== [A-Z0-9_]+ ==========$/ {
    label = $0
    gsub(/=/, "", label)
    gsub(/^ +| +$/, "", label)
  }
  /Table 1: Overlap Wall Time/ { table = "wall_ms"; next }
  /Table 2: GEMM Slowdown/ { table = "slowdown"; next }
  /Table 3: Sequential AllReduce Time/ { table = "seq_ar_ms"; next }
  /Table 4: Sequential GEMM Time/ { table = "seq_gemm_ms"; next }
  $1 == size && $2 == "MB" && $3 == "|" {
    printf "%-18s %-12s %s\n", label, table, $0
    if (table == "wall_ms") {
      copy = $4 + 0.0
      nocopy = $6 + 0.0
      rccl = $8 + 0.0
      printf "%-18s %-12s copy_gap=%.3f ms no_copy_gap=%.3f ms copy_penalty=%.3f ms\n",
             label, "derived", copy - rccl, nocopy - rccl, copy - nocopy
    }
  }
' "$LOG"

echo
echo "---- raw key lines ----"
grep -E "MORI_MULTILANE_DIRECT|Table 1:|copy vs RCCL|All Tests PASSED|FAILED|Timeout|STUCK" "$LOG" || true
echo
echo "LOG: $LOG"
