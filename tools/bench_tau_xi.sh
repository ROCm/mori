#!/usr/bin/env bash
# tools/bench_tau_xi.sh
# -----------------------------------------------------------------------------
# A/B benchmark for direction ν (MORI_SKIP_ITER_SYNC) on Test 6
# (multi-stage pipelined GEMM+AR overlap) @ 256MB.
#
# History:
#   - v1..v6: τ/ξ/ν/τ''/E'' all FAILED, reverted. See perf_history
#     Entries 14-17.
#   - v7: baseline-only.
#   - v8 (current): DIRECT (MORI_DIRECT_OUTPUT=1) — plan A per
#     perf_history Entry 18. CU XGMI AG + direct write user_output,
#     skips SDMA AG and external hipMemcpyAsync. Measured CU XGMI BW
#     = 370 GB/s at 16 blocks/peer → expected AR[3] total ~0.8 ms
#     (vs 1.3 ms baseline), ~0.5 ms/AR saving, wall 7.79 → ~6.5-6.7 ms
#     (superior to RCCL 7.42).
#
# Usage (inside ROCm container):
#   cd /home/fizhang/test/mori
#   bash tools/bench_tau_xi.sh
#
# Overridable env vars (optional):
#   REPO         (default: $(pwd))
#   SIZE_MB      (default: 256)
#   NUM_STAGES   (default: 4)
#   ITERATIONS   (default: 100)
#   WARMUP       (default: 20)
#   PIPELINE_CU  (default: 160)
#   SKIP_PULL    (default: 0; set 1 to skip git pull)
#   SKIP_BUILD   (default: 0; set 1 to skip pip install -e .)
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
rocm-smi --showproductname 2>&1 | grep -E "GPU\[.\].*(Card Series|GFX)" | head -4 || true
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
  pip install -e .
fi

LOG="/tmp/perf_$(date +%s).log"
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
  run_variant "DIRECT"   "$DIR_ENV"

  run_variant_extra "BASELINE_TIMELINE"  "" --timeline
  run_variant_extra "DIRECT_TIMELINE"    "$DIR_ENV" --timeline

  run_variant_extra "BASELINE_AR0_PHASE" "MORI_PHASE_TARGET_STAGE=0" --ar-phase-timing
  run_variant_extra "DIRECT_AR0_PHASE"   "$DIR_ENV MORI_PHASE_TARGET_STAGE=0" --ar-phase-timing

  run_variant_extra "BASELINE_AR3_PHASE" "MORI_PHASE_TARGET_STAGE=3" --ar-phase-timing
  run_variant_extra "DIRECT_AR3_PHASE"   "$DIR_ENV MORI_PHASE_TARGET_STAGE=3" --ar-phase-timing
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

# Phase breakdown extraction: use ".*" to span the unicode arrow character
# (→ is 3 bytes in UTF-8; a single "." only matches 1 byte, so earlier
# patterns like "entry.scatter_done" silently failed to match).
PHASE_RE="(\[[0-9]\] total|entry.*scatter_done|scatter.*compute-wait|compute-wait.*barrier|barrier.*AG-submit|AG-submit.*AG-wait-done|AG-wait-done.*exit|reduce-done|host-side hipMemcpy|gpu-side copy)"

echo
echo "---- [2] per-stage median (BASELINE_TIMELINE) ----"
awk '/========== BASELINE_TIMELINE ==========/,/========== DIRECT_TIMELINE ==========/' "$LOG" \
  | grep -E "stage \|| *[0-3] \||median wall|AR\[[0-3]\] duration|AR\[.\]-GEMM" || true

echo
echo "---- [3] per-stage median (DIRECT_TIMELINE) ----"
awk '/========== DIRECT_TIMELINE ==========/,/========== BASELINE_AR0_PHASE ==========/' "$LOG" \
  | grep -E "stage \|| *[0-3] \||median wall|AR\[[0-3]\] duration|AR\[.\]-GEMM" || true

echo
echo "---- [4] BASELINE AR[0] phase ----"
awk '/========== BASELINE_AR0_PHASE ==========/,/========== DIRECT_AR0_PHASE ==========/' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "---- [5] DIRECT AR[0] phase ----"
awk '/========== DIRECT_AR0_PHASE ==========/,/========== BASELINE_AR3_PHASE ==========/' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "---- [6] BASELINE AR[3] phase ----"
awk '/========== BASELINE_AR3_PHASE ==========/,/========== DIRECT_AR3_PHASE ==========/' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "---- [7] DIRECT AR[3] phase (KEY — expect ~0.8 ms total, no hipMemcpy) ----"
awk '/========== DIRECT_AR3_PHASE ==========/,0' "$LOG" \
  | grep -E "$PHASE_RE" || true

echo
echo "LOG: $LOG"
