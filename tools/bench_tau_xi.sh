#!/usr/bin/env bash
# tools/bench_tau_xi.sh
# -----------------------------------------------------------------------------
# A/B benchmark for direction τ (MORI_KEEP_HBM_HOT) and ξ (MORI_AR_WARMUP) on
# Test 6 (multi-stage pipelined GEMM+AR overlap) @ 256MB.
#
# Design goals:
#   - one command:  bash tools/bench_tau_xi.sh
#   - preflight: hostname / cmake / python3 / torch.cuda before any build
#   - fail-fast: set -euo pipefail; bad env => abort, not "empty OFF section"
#   - isolated: script runs as its own bash process; set -e does NOT kill
#               the interactive shell that invoked it (see relentless-perf R13)
#   - self-contained: args hardcoded correctly (this is where --iters vs
#                     --iterations, --sweep vs --elems mistakes get fixed once)
#   - tee to a single log; inline awk extracts the comparison table
#
# Usage (inside ROCm container, after "cd /home/fizhang/test/mori"):
#   bash tools/bench_tau_xi.sh
#
# Overridable env vars (optional):
#   REPO         (default: current directory)
#   SIZE_MB      (default: 256)   -> AR buffer size per PE
#   NUM_STAGES   (default: 4)
#   ITERATIONS   (default: 100)
#   WARMUP       (default: 20)
#   PIPELINE_CU  (default: 160)
#   SKIP_PULL    (default: 0; set 1 to skip `git pull`)
#   SKIP_BUILD   (default: 0; set 1 to skip `pip install -e .`)
# -----------------------------------------------------------------------------
set -euo pipefail

# Default REPO = this script's parent's parent (= repo root),
# so user can run `bash tools/bench_tau_xi.sh` or an absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=100}"
: "${WARMUP:=20}"
: "${PIPELINE_CU:=160}"
: "${SKIP_PULL:=0}"
: "${SKIP_BUILD:=0}"
# uint32 = 4 bytes; elements per PE for a given size in MiB:
ELEMS=$(( SIZE_MB * 1024 * 1024 / 4 ))

cd "$REPO"

# Sanity: we should be in the mori repo root
[ -f "setup.py" ] || [ -f "pyproject.toml" ] || {
  echo "ERROR: REPO=$REPO does not look like mori root (no setup.py/pyproject.toml)."
  echo "Run from repo root, or set REPO=/path/to/mori"
  exit 1
}

echo "==================== [preflight] ===================="
hostname
id -un
pwd
echo "SIZE_MB=$SIZE_MB NUM_STAGES=$NUM_STAGES ITERATIONS=$ITERATIONS WARMUP=$WARMUP PIPELINE_CU=$PIPELINE_CU ELEMS=$ELEMS"

# Build deps (we use --no-build-isolation which skips pyproject.toml auto-fetch,
# so we must pre-install cmake / ninja / pybind11 ourselves).
BUILD_PIPS=()
command -v cmake >/dev/null 2>&1 || BUILD_PIPS+=("cmake>=3.20")
command -v ninja >/dev/null 2>&1 || BUILD_PIPS+=("ninja")
python3 -c "import pybind11" 2>/dev/null || BUILD_PIPS+=("pybind11")
if [ "${#BUILD_PIPS[@]}" -gt 0 ]; then
  echo "installing missing build deps: ${BUILD_PIPS[*]}"
  pip install "${BUILD_PIPS[@]}" >/dev/null
fi
command -v cmake   >/dev/null || { echo "MISSING: cmake";   exit 1; }
command -v ninja   >/dev/null || { echo "MISSING: ninja";   exit 1; }
command -v python3 >/dev/null || { echo "MISSING: python3"; exit 1; }
python3 -c "import pybind11; print('pybind11:', pybind11.__version__)"
cmake --version | head -1
ninja  --version

# rocm-smi / HIP / GPU presence
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
  pip install -e . --no-build-isolation
fi

LOG="/tmp/perf_tau_xi_$(date +%s).log"
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

  run_variant "OFF"     ""
  run_variant "TAU"     "MORI_KEEP_HBM_HOT=1"
  run_variant "XI"      "MORI_AR_WARMUP=1"
  run_variant "TAU_XI"  "MORI_KEEP_HBM_HOT=1 MORI_AR_WARMUP=1"

  run_variant_extra "TAU_XI_TIMELINE" "MORI_KEEP_HBM_HOT=1 MORI_AR_WARMUP=1" --timeline

  run_variant_extra "AR0_PHASE_TAU_XI" \
    "MORI_KEEP_HBM_HOT=1 MORI_AR_WARMUP=1 MORI_PHASE_TARGET_STAGE=0" \
    --ar-phase-timing

  run_variant_extra "AR3_PHASE_TAU_XI" \
    "MORI_KEEP_HBM_HOT=1 MORI_AR_WARMUP=1 MORI_PHASE_TARGET_STAGE=3" \
    --ar-phase-timing
} | tee "$LOG"

echo
echo "################################################################"
echo "## COMPARE TABLE (auto-extracted from $LOG)"
echo "################################################################"

# grep-style pattern used by awk to locate the "${SIZE_MB} MB |" summary row
SZ_PAT="^[[:space:]]*${SIZE_MB} MB \\|"

echo
echo "---- [1] wall ms @ ${SIZE_MB}MB (all labels) ----"
awk -v pat="$SZ_PAT" '
  /^========== [A-Z_0-9]+ ==========$/ {
    gsub(/=/, ""); gsub(/^ +| +$/, "");
    lbl = $0
  }
  $0 ~ pat { printf "  %-22s %s\n", lbl, $0 }
' "$LOG"

echo
echo "---- [2] per-stage (TAU_XI_TIMELINE) ----"
awk '/========== TAU_XI_TIMELINE/,/========== AR0_PHASE_TAU_XI/' "$LOG" \
  | grep -E "stage \|| *[0-3] \||median wall|AR\[[0-3]\] duration|AR\[.\]-GEMM" || true

echo
echo "---- [3] AR[0] phase (TAU_XI) ----"
awk '/========== AR0_PHASE_TAU_XI/,/========== AR3_PHASE_TAU_XI/' "$LOG" \
  | grep -E "AR\[0\] total|entry.scatter_done|scatter.compute-wait|compute-wait.barrier|barrier.AG-submit|AG-submit.AG-wait-done|reduce-done|host-side hipMemcpy|gpu-side copy" || true

echo
echo "---- [4] AR[3] phase (TAU_XI, key target) ----"
awk '/========== AR3_PHASE_TAU_XI/,0' "$LOG" \
  | grep -E "AR\[3\] total|entry.scatter_done|scatter.compute-wait|compute-wait.barrier|barrier.AG-submit|AG-submit.AG-wait-done|reduce-done|host-side hipMemcpy|gpu-side copy" || true

echo
echo "LOG: $LOG"
