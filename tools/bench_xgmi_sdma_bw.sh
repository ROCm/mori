#!/usr/bin/env bash
# tools/bench_xgmi_sdma_bw.sh
# -----------------------------------------------------------------------------
# Step-1/Step-2 of the hybrid AR design (perf_history Entry 18 planning).
# Measures two physical quantities needed to decide between plan A and plan B:
#
#   1. CU XGMI load BW — how fast can a kernel read data from N-1 peer GPUs'
#      HBM concurrently via P2P pointers. Drives plan A (CU AG phase).
#
#   2. SDMA self-copy BW (multi-queue) — how fast can SDMA move 256 MB of
#      local HBM to local HBM using multiple queues in parallel. Drives
#      plan B (SDMA copy at the end of AR).
#
# Both tests are run under MPI (1 process per GPU, 8 processes total) so
# MORI shmem has peer pointers set up.
#
# Usage (inside ROCm container, 8 GPUs visible):
#   bash tools/bench_xgmi_sdma_bw.sh
#
# Overridable env:
#   REPO         (default: repo root from script dir)
#   SKIP_PULL    (default: 0)
#   SKIP_BUILD   (default: 0)
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SKIP_PULL:=0}"
: "${SKIP_BUILD:=0}"

cd "$REPO"
[ -f "setup.py" ] || [ -f "pyproject.toml" ] || {
  echo "ERROR: REPO=$REPO does not look like mori root"; exit 1;
}

echo "==================== [preflight] ===================="
hostname
id -un
pwd
rocm-smi --showproductname 2>&1 | grep -E "GPU\[.\].*(Card Series|GFX)" | head -4 || true

command -v mpirun >/dev/null 2>&1 || { echo "MISSING: mpirun (need openmpi)"; exit 1; }
mpirun --version 2>&1 | head -1

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
  git log -1 --oneline
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo
  echo "==================== [build examples] ===================="
  # Rebuild mori with BUILD_EXAMPLES=ON (default in CMakeLists but OFF in pip)
  BUILD_DIR="$REPO/build"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake .. \
    -G Ninja \
    -DUSE_ROCM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS=gfx950 \
    -DBUILD_EXAMPLES=ON \
    -DWITH_MPI=ON 2>&1 | tail -20
  ninja sdma_self_copy_test cu_xgmi_bench 2>&1 | tail -20
  cd "$REPO"
fi

SDMA_BIN="$REPO/build/examples/sdma_self_copy_test"
XGMI_BIN="$REPO/build/examples/cu_xgmi_bench"

[ -x "$SDMA_BIN" ] || { echo "MISSING: $SDMA_BIN"; exit 1; }
[ -x "$XGMI_BIN" ] || { echo "MISSING: $XGMI_BIN"; exit 1; }

LOG=/tmp/xgmi_sdma_bw_$(date +%s).log

echo
echo "==================== [run sdma_self_copy_test] ===================="
echo "LOG -> $LOG"
{
  echo "##########################################"
  echo "## SDMA self-copy + hipMemcpy D2D BW"
  echo "## multi-q SDMA (64 queues per thread), sweep sizes 1-256 MB"
  echo "##########################################"
  echo
  mpirun -n 8 --allow-run-as-root \
    -x HIP_VISIBLE_DEVICES \
    "$SDMA_BIN" 2>&1

  echo
  echo "##########################################"
  echo "## CU XGMI multi-peer read BW"
  echo "## sweep bytes-per-peer 1-64 MB, blocks/peer 1-32"
  echo "##########################################"
  echo
  mpirun -n 8 --allow-run-as-root \
    -x HIP_VISIBLE_DEVICES \
    "$XGMI_BIN" 2>&1
} | tee "$LOG"

echo
echo "################################################################"
echo "## SUMMARY (auto-extracted)"
echo "################################################################"
echo
echo "--- SDMA self-copy @ 256 MB (plan B viability) ---"
grep -E "^\s*256\.[0-9] " "$LOG" | head -2 || true

echo
echo "--- CU XGMI load @ 32 MB/peer (AR per-peer shard, plan A viability) ---"
awk '/per-peer bytes = 32 MB/,/per-peer bytes = 64 MB/' "$LOG" | head -8 || true

echo
echo "LOG: $LOG"
