#!/usr/bin/env bash
# Usage:
#   bash tools/bench_cadence_baseline.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 NUM_STAGES=4 CONTINUOUS_ITERS=100 \
#   TIMELINE_SAMPLES=8 PIPELINE_CU=224 PIPELINE_CHUNKS=4 \
#   CASE_TIMEOUT_SEC=900 SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_cadence_baseline.sh
#
# Runs: preflight -> optional git pull -> optional build -> baseline continuous
# timeline + phase/copy timing. Extracts AR service/backlog/copy evidence for
# the next new-algorithm implementation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=1}"
: "${WARMUP:=1}"
: "${CONTINUOUS_ITERS:=100}"
: "${TIMELINE_SAMPLES:=8}"
: "${PIPELINE_CU:=224}"
: "${PIPELINE_CHUNKS:=4}"
: "${PHASE_ITER:=5}"
: "${PHASE_STAGE:=0}"
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
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ELEMS=$ELEMS CONTINUOUS_ITERS=$CONTINUOUS_ITERS TIMELINE_SAMPLES=$TIMELINE_SAMPLES PIPELINE_CU=$PIPELINE_CU PIPELINE_CHUNKS=$PIPELINE_CHUNKS"

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

LOG="/tmp/perf_cadence_baseline_$(date +%s).log"

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
    --continuous-timeline-samples "$TIMELINE_SAMPLES" \
    --continuous-phase-iter "$PHASE_ITER" \
    --continuous-phase-stage "$PHASE_STAGE" \
    --timeline 2>&1
}

{
  echo "========== HEAD =========="
  git log -1 --oneline
  run_case "BASELINE_TIMELINE_PHASE" \
    MORI_CONTINUOUS_PREP=0 \
    MORI_PIPELINE_CU="$PIPELINE_CU" \
    MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"
} | tee "$LOG"

echo
echo "################################################################"
echo "## CADENCE SUMMARY (auto-extracted from $LOG)"
echo "################################################################"

awk '
  /Table 1: Overlap Wall Time/ { table = "wall_ms"; next }
  /Table 2: GEMM Slowdown/ { table = "slowdown"; next }
  /Table 3: Sequential AllReduce Time/ { table = "seq_ar_ms"; next }
  /Table 4: Sequential GEMM Time/ { table = "seq_gemm_ms"; next }
  $1 == "256" && $2 == "MB" && $3 == "|" {
    printf "%-12s %s\n", table, $0
  }
  /Continuous Timeline/ { in_timeline = 1; next }
  in_timeline && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 == "|" {
    ar_dur = $11 + 0.0
    gap = $13 + 0.0
    if (ar_dur > max_ar) max_ar = ar_dur
    if (gap > max_gap) max_gap = gap
    sum_ar += ar_dur
    n_ar += 1
  }
  /Copy-path/ { copy_section = 1 }
  copy_section && /host-side hipMemcpyAsync/ { host_us = $(NF-1) }
  copy_section && /gpu-side copy kernel wall/ { copy_ms = $(NF-1); copy_section = 0 }
  END {
    if (n_ar > 0) {
      printf "timeline     samples=%d avg_ar_dur=%.3f ms max_ar_dur=%.3f ms max_ar_gemm_gap=%.3f ms\n",
             n_ar, sum_ar / n_ar, max_ar, max_gap
    }
    if (copy_ms != "") {
      printf "copy_timing  host_us=%s gpu_copy_ms=%s\n", host_us, copy_ms
    }
  }
' "$LOG"

echo
echo "---- raw key lines ----"
grep -E "Table 1:|copy vs RCCL|Continuous Timeline|AR-GEMM gap|Copy-path|host-side hipMemcpy|gpu-side copy|decoded block0 phases|decoded first R/compute block phases|AR\\[[0-3]\\] duration" "$LOG" || true
echo
echo "LOG: $LOG"
