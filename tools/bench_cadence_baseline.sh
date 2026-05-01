#!/usr/bin/env bash
# Usage:
#   bash tools/bench_cadence_baseline.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 NUM_STAGES=4 CONTINUOUS_ITERS=100 \
#   TIMELINE_SAMPLES=8 PIPELINE_CU=224 PIPELINE_CHUNKS=4 \
#   CASE_TIMEOUT_SEC=900 SKIP_PULL=0 SKIP_BUILD=0 \
#   EXTRACT_ONLY_LOG=/tmp/existing.log bash tools/bench_cadence_baseline.sh
#
# Runs: preflight -> optional git pull -> optional build -> baseline continuous
# timeline + phase/copy timing. Extracts AR service/backlog/copy evidence for
# the next new-algorithm implementation.

set -euo pipefail
ulimit -c 0 || true

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
: "${EXTRACT_ONLY_LOG:=}"

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
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ELEMS=$ELEMS CONTINUOUS_ITERS=$CONTINUOUS_ITERS TIMELINE_SAMPLES=$TIMELINE_SAMPLES PIPELINE_CU=$PIPELINE_CU PIPELINE_CHUNKS=$PIPELINE_CHUNKS"

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
  git log -1 --oneline
fi

if [ "$SKIP_BUILD" != "1" ] && [ -z "$EXTRACT_ONLY_LOG" ]; then
  echo
  echo "==================== [pip install] ===================="
  pip install -e .
fi

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

if [ -n "$EXTRACT_ONLY_LOG" ]; then
  LOG="$EXTRACT_ONLY_LOG"
  [ -f "$LOG" ] || { echo "ERROR: EXTRACT_ONLY_LOG=$LOG not found"; exit 1; }
  echo "==================== [extract-only] LOG=$LOG ===================="
else
  LOG="/tmp/perf_cadence_baseline_$(date +%s).log"
  {
    echo "========== HEAD =========="
    git log -1 --oneline
    run_case "BASELINE_TIMELINE_PHASE" \
      MORI_CONTINUOUS_PREP=0 \
      MORI_PIPELINE_CU="$PIPELINE_CU" \
      MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"
  } | tee "$LOG"
fi

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
  /Continuous Timeline \[/ {
    timeline_label = $0
    sub(/^.*\[/, "", timeline_label)
    sub(/\].*$/, "", timeline_label)
    in_timeline = 1
    next
  }
  in_timeline && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 == "|" {
    # Fields after awk splitting:
    # iter stage | GEMM_st GEMM_end GEMM_dur | AR_st AR_end AR_dur | AR-GEMM_gap
    #   $1   $2  $3   $4      $5       $6   $7  $8    $9    $10  $11    $12
    key = timeline_label
    ar_dur = $10 + 0.0
    gap = $12 + 0.0
    if (!(key in n_ar) || ar_dur > max_ar[key]) max_ar[key] = ar_dur
    if (!(key in n_ar) || gap > max_gap[key]) max_gap[key] = gap
    sum_ar[key] += ar_dur
    n_ar[key] += 1
  }
  /Copy-path/ { copy_section = 1 }
  copy_section && /host-side hipMemcpyAsync/ { host_us = $(NF-1) }
  copy_section && /gpu-side copy kernel wall/ { copy_ms = $(NF-1); copy_section = 0 }
  END {
    for (key in n_ar) {
      printf "timeline     %-20s samples=%d avg_ar_dur=%.3f ms max_ar_dur=%.3f ms max_ar_gemm_gap=%.3f ms\n",
             key, n_ar[key], sum_ar[key] / n_ar[key], max_ar[key], max_gap[key]
    }
    if (copy_ms != "") {
      printf "copy_timing  host_us=%s gpu_copy_ms=%s\n", host_us, copy_ms
    }
  }
' "$LOG"

echo
echo "---- raw key lines ----"
grep -E "Table 1:|copy vs RCCL|Continuous Timeline|AR-GEMM gap|^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+\\||Copy-path|host-side hipMemcpy|gpu-side copy|decoded block0 phases|decoded first R/compute block phases|AR\\[[0-3]\\] duration" "$LOG" || true
echo
echo "LOG: $LOG"
