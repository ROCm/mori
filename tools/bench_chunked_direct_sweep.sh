#!/usr/bin/env bash
# Usage:
#   bash tools/bench_chunked_direct_sweep.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 NUM_STAGES=4 CONTINUOUS_ITERS=100 \
#   VARIANTS="CD_CU224_CH2:224:2 CD_CU224_CH4:224:4 CD_CU160_CH4:160:4 CD_CU112_CH4:112:4 CD_CU224_CH8:224:8" \
#   CASE_TIMEOUT_SEC=900 SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_chunked_direct_sweep.sh
#
# Runs: preflight -> optional git pull -> optional build -> baseline ->
# chunked-direct sweep -> auto-extracted COPY-vs-RCCL table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=1}"
: "${WARMUP:=1}"
: "${CONTINUOUS_ITERS:=100}"
: "${BASELINE_CU:=224}"
: "${BASELINE_CHUNKS:=4}"
: "${VARIANTS:=CD_CU224_CH2:224:2 CD_CU224_CH4:224:4 CD_CU160_CH4:160:4 CD_CU112_CH4:112:4 CD_CU224_CH8:224:8}"
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
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ELEMS=$ELEMS CONTINUOUS_ITERS=$CONTINUOUS_ITERS BASELINE_CU=$BASELINE_CU BASELINE_CHUNKS=$BASELINE_CHUNKS"
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

LOG="/tmp/perf_chunked_direct_sweep_$(date +%s).log"

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
    MORI_PIPELINE_CU="$BASELINE_CU" \
    MORI_PIPELINE_CHUNKS="$BASELINE_CHUNKS"

  for spec in $VARIANTS; do
    IFS=: read -r label cu chunks <<<"$spec"
    run_case "$label" \
      MORI_CONTINUOUS_PREP=0 \
      MORI_CHUNKED_DIRECT=1 \
      MORI_PIPELINE_CU="$cu" \
      MORI_PIPELINE_CHUNKS="$chunks"
  done
} | tee "$LOG"

echo
echo "################################################################"
echo "## CHUNKED DIRECT SWEEP SUMMARY (auto-extracted from $LOG)"
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
grep -E "MORI_CHUNKED_DIRECT|Table 1:|copy vs RCCL|All Tests PASSED|FAILED|Timeout|STUCK" "$LOG" || true
echo
echo "LOG: $LOG"
