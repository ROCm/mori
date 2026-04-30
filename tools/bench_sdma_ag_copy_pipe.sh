#!/usr/bin/env bash
# Usage:
#   bash tools/bench_sdma_ag_copy_pipe.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 NUM_STAGES=4 CONTINUOUS_ITERS=100 \
#   PIPELINE_CU=224 PIPELINE_CHUNKS=4 RUN_PIPE=0 PIPE_NRS="112 144 176 200" \
#   RUN_PHASE_TIMING=1 PHASE_ITERATIONS=20 PHASE_WARMUP=5 \
#   CASE_TIMEOUT_SEC=900 SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_sdma_ag_copy_pipe.sh
#
# Runs: preflight -> optional git pull -> optional build -> baseline COPY/RCCL.
# RUN_PIPE=1 also runs the deprecated MORI_SDMA_AG_COPY_PIPE nR sweep with
# MORI_ALLOW_FAILED_COPY_PIPE=1. That path is off by default after Entry 56
# showed it still hangs in K1 scatter chunk 3.
# Output: /tmp/perf_sdma_ag_copy_pipe_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=1}"
: "${WARMUP:=1}"
: "${CONTINUOUS_ITERS:=100}"
: "${PIPELINE_CU:=224}"
: "${PIPELINE_CHUNKS:=4}"
: "${RUN_PIPE:=0}"
: "${PIPE_NRS:=112 144 176 200}"
: "${RUN_PHASE_TIMING:=1}"
: "${PHASE_ITERATIONS:=20}"
: "${PHASE_WARMUP:=5}"
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
rocm-smi --showproductname 2>&1 | head -8 || true
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git status --short || true
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ELEMS=$ELEMS CONTINUOUS_ITERS=$CONTINUOUS_ITERS PIPELINE_CU=$PIPELINE_CU PIPELINE_CHUNKS=$PIPELINE_CHUNKS RUN_PIPE=$RUN_PIPE PIPE_NRS=$PIPE_NRS RUN_PHASE_TIMING=$RUN_PHASE_TIMING PHASE_ITERATIONS=$PHASE_ITERATIONS PHASE_WARMUP=$PHASE_WARMUP CASE_TIMEOUT_SEC=$CASE_TIMEOUT_SEC"

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

LOG="/tmp/perf_sdma_ag_copy_pipe_$(date +%s).log"

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

run_phase_case() {
  local label="$1"
  shift
  echo
  echo "========== $label =========="
  echo "ENV: $*"
  timeout --signal=TERM "$CASE_TIMEOUT_SEC" env "$@" python3 tests/python/ccl/test_allreduce.py \
    --num-stages "$NUM_STAGES" \
    --elems "$ELEMS" \
    --iterations "$PHASE_ITERATIONS" \
    --warmup "$PHASE_WARMUP" \
    --ar-phase-timing 2>&1
}

{
  echo "========== HEAD =========="
  git log -1 --oneline
  echo
  run_case "BASELINE" \
    MORI_CONTINUOUS_PREP=0 \
    MORI_PIPELINE_CU="$PIPELINE_CU" \
    MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"

  if [ "$RUN_PIPE" = "1" ]; then
    for nr in $PIPE_NRS; do
      run_case "PIPE_NR_${nr}" \
        MORI_CONTINUOUS_PREP=0 \
        MORI_SDMA_AG_COPY_PIPE=1 \
        MORI_ALLOW_FAILED_COPY_PIPE=1 \
        MORI_SDMA_AG_COPY_NR="$nr" \
        MORI_PIPELINE_CU="$PIPELINE_CU" \
        MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS"
    done
  else
    echo
    echo "========== PIPE_DISABLED =========="
    echo "MORI_SDMA_AG_COPY_PIPE is skipped by default: Entry 56 shows K1 chunk=3 signal stuck at 3/4."
    echo "Set RUN_PIPE=1 only for targeted debugging."
  fi

  if [ "$RUN_PHASE_TIMING" = "1" ]; then
    run_phase_case "BASELINE_PHASE_STAGE0" \
      MORI_CONTINUOUS_PREP=0 \
      MORI_PIPELINE_CU="$PIPELINE_CU" \
      MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS" \
      MORI_PHASE_TARGET_STAGE=0
  fi
} | tee "$LOG"

echo
echo "################################################################"
echo "## COPY VS RCCL SUMMARY (auto-extracted from $LOG)"
echo "################################################################"

awk -v size="$SIZE_MB" '
  /Table 1: Overlap Wall Time/ { table = "wall_ms"; next }
  /Table 2: GEMM Slowdown/ { table = "slowdown"; next }
  /Table 3: Sequential AllReduce Time/ { table = "seq_ar_ms"; next }
  /Table 4: Sequential GEMM Time/ { table = "seq_gemm_ms"; next }
  /^========== [A-Z0-9_]+ ==========$/ {
    label = $0
    gsub(/=/, "", label)
    gsub(/^ +| +$/, "", label)
  }
  $1 == size && $2 == "MB" && $3 == "|" {
    printf "%-16s %-12s %s\n", label, table, $0
    if (table == "wall_ms") {
      copy = $4 + 0.0
      nocopy = $6 + 0.0
      rccl = $8 + 0.0
      printf "%-16s %-12s copy_gap=%.3f ms no_copy_gap=%.3f ms copy_penalty=%.3f ms\n",
             label, "derived", copy - rccl, nocopy - rccl, copy - nocopy
    }
  }
' "$LOG"

echo
echo "---- raw phase/copy lines ----"
grep -E "MORI_SDMA_AG_COPY_PIPE|Table 1:|copy vs RCCL|Copy-path|host-side hipMemcpy|gpu-side copy|AR\\[[0-3]\\] duration|median wall" "$LOG" || true

echo
echo "LOG: $LOG"
