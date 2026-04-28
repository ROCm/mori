#!/usr/bin/env bash
# tools/bench_plan_a.sh
# -----------------------------------------------------------------------------
# Plan A (PipelinedXGMIPullKernel) bench on Test 6 (multi-stage overlap)
# @ 256 MB. See perf_history Entry 19.
#
# Runs BASELINE (SDMA copy / no-copy / RCCL) + Plan A (DIRECT) wall,
# + BASELINE vs Plan A AR[0] phase timing, extracts comparison table.
#
# Usage (inside ROCm container):
#   cd /home/fizhang/test/mori
#   bash tools/bench_plan_a.sh
#
# Overridable env vars (optional):
#   REPO         (default: auto-detect)
#   SIZE_MB      (default: 256)
#   NUM_STAGES   (default: 4)
#   ITERATIONS   (default: 100)
#   WARMUP       (default: 20)
#   PIPELINE_CU  (default: 160)
#   SKIP_PULL    (default: 0; set 1 to skip git pull)
#   SKIP_BUILD   (default: 0; set 1 to skip rebuild)
#
# Build command (correct per user 2026-04-24):
#   BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
# NOT `pip install -e . --no-build-isolation` — that bypasses build-time
# dep install and fails with `ModuleNotFoundError: No module named 'pybind11'`
# because pybind11 is a build-only dep (pyproject.toml build-system.requires).
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=100}"
: "${WARMUP:=20}"
: "${PIPELINE_CU:=160}"
: "${SKIP_PULL:=0}"
: "${SKIP_BUILD:=0}"
ELEMS=$(( SIZE_MB * 1024 * 1024 / 4 ))   # uint32 = 4 bytes

cd "$REPO"
[ -f "setup.py" ] || [ -f "pyproject.toml" ] || {
  echo "ERROR: REPO=$REPO does not look like mori root"; exit 1;
}

echo "==================== [preflight] ===================="
hostname
id -un
pwd
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ITERATIONS=$ITERATIONS WARMUP=$WARMUP PIPELINE_CU=$PIPELINE_CU ELEMS=$ELEMS"

command -v python3 >/dev/null || { echo "MISSING: python3"; exit 1; }
command -v cmake   >/dev/null || { echo "MISSING: cmake"; exit 1; }
command -v hipcc   >/dev/null || { echo "MISSING: hipcc"; exit 1; }
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

LOG="/tmp/plan_a_$(date +%s).log"
echo
echo "==================== [run] LOG=$LOG ===================="

run_variant() {
  local label="$1" envs="$2"
  echo
  echo "========== $label =========="
  echo "ENV: MORI_PIPELINE_CU=$PIPELINE_CU $envs"
  # shellcheck disable=SC2086
  env MORI_PIPELINE_CU="$PIPELINE_CU" $envs python3 tests/python/ccl/test_allreduce.py \
    --num-stages "$NUM_STAGES" \
    --elems "$ELEMS" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" 2>&1
}

run_variant_extra() {
  local label="$1" envs="$2"
  shift 2
  echo
  echo "========== $label =========="
  echo "ENV: MORI_PIPELINE_CU=$PIPELINE_CU $envs"
  echo "extra_args: $*"
  # shellcheck disable=SC2086
  env MORI_PIPELINE_CU="$PIPELINE_CU" $envs python3 tests/python/ccl/test_allreduce.py \
    --num-stages "$NUM_STAGES" \
    --elems "$ELEMS" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    "$@" 2>&1
}

{
  echo "========== HEAD =========="
  git log -1 --oneline

  DIR_ENV="MORI_DIRECT_OUTPUT=1"

  run_variant "BASELINE" ""
  run_variant "PLAN_A"   "$DIR_ENV"

  run_variant_extra "BASELINE_AR0_PHASE" "MORI_PHASE_TARGET_STAGE=0" --ar-phase-timing
  run_variant_extra "PLAN_A_AR0_PHASE"   "$DIR_ENV MORI_PHASE_TARGET_STAGE=0" --ar-phase-timing
} | tee "$LOG"

echo
echo "################################################################"
echo "## COMPARE TABLE (auto-extracted from $LOG)"
echo "################################################################"

SZ_PAT="^[[:space:]]*${SIZE_MB} MB \\|"

echo
echo "---- [1] wall ms @ ${SIZE_MB}MB (all labels) ----"
awk -v pat="$SZ_PAT" '
  /^========== [A-Z_0-9]+ ==========$/ {
    gsub(/=/, ""); gsub(/^ +| +$/, "");
    lbl = $0
  }
  $0 ~ pat { printf "  %-25s %s\n", lbl, $0 }
' "$LOG"

# Phase timing patterns (matches baseline AND Plan A slot names)
PHASE_RE="(\[[0-9]\] total|entry.*scatter_done|scatter.*compute-wait|compute-wait.*barrier|barrier.*AG-submit|AG-submit.*AG-wait-done|ag_sync signaled|AG-wait-done.*exit|reduce-done|cb-exit|ag-ready|pull-done|ag-exit|AG-pull|host-side hipMemcpy|gpu-side copy|per-chunk)"

echo
echo "---- [2] BASELINE AR[0] phase ----"
awk '/========== BASELINE_AR0_PHASE ==========/,/========== PLAN_A_AR0_PHASE ==========/' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "---- [3] PLAN_A AR[0] phase (KEY) ----"
echo "     Expectations: no AG-submit/AG-wait-done (SDMA gone);"
echo "                   instead ag_sync signaled + reduce-done + cb-exit;"
echo "                   no external hipMemcpy (Plan A writes user_output in-kernel)."
awk '/========== PLAN_A_AR0_PHASE ==========/,0' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "---- [4] Plan A active log ----"
grep "Plan A active" "$LOG" | head -8 || \
  echo "(Plan A announce not found — kernel may not have been triggered; check MORI_DIRECT_OUTPUT propagation)"

echo
echo "---- [5] Correctness check (all Test variants) ----"
grep -E "Test [0-9]+.*(PASSED|FAILED)|correctness (PASSED|FAILED)|BUG:" "$LOG" | head -20 || \
  echo "(no explicit PASSED/FAILED lines found)"

echo
echo "LOG: $LOG"
