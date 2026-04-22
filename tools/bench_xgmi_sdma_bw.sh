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
  echo "==================== [pip install with BUILD_EXAMPLES=ON] ===================="
  # Re-use the existing pip build infra; just flip BUILD_EXAMPLES to ON so
  # the example binaries (sdma_self_copy_test, cu_xgmi_bench) get produced
  # alongside the python module. No raw cmake tricks.
  BUILD_EXAMPLES=ON pip install -e .
fi

# Locate example binaries — pip/setup.py puts them inside the build tree used
# by setuptools. Resolve by glob.
SDMA_BIN="$(find "$REPO" -path "*/build*" -name sdma_self_copy_test -type f 2>/dev/null | head -1)"
XGMI_BIN="$(find "$REPO" -path "*/build*" -name cu_xgmi_bench       -type f 2>/dev/null | head -1)"

echo "SDMA_BIN = ${SDMA_BIN:-NOT_FOUND}"
echo "XGMI_BIN = ${XGMI_BIN:-NOT_FOUND}"
[ -x "$XGMI_BIN" ] || { echo "MISSING: $XGMI_BIN (cu_xgmi_bench)"; exit 1; }

LOG=/tmp/xgmi_sdma_bw_$(date +%s).log

echo
echo "==================== [run] LOG -> $LOG ===================="
{
  echo "##########################################"
  echo "## CU XGMI multi-peer read BW"
  echo "## sweep bytes-per-peer 1-64 MB, blocks/peer 1-32"
  echo "##########################################"
  echo
  mpirun -n 8 --allow-run-as-root "$XGMI_BIN" 2>&1

  # sdma_self_copy_test has a pre-existing bug that faults on multi-GPU
  # (not our concern for this decision — SDMA self-copy BW is only plan B
  # data, plan A depends only on CU XGMI BW).
  if [ -x "$SDMA_BIN" ]; then
    echo
    echo "##########################################"
    echo "## SDMA self-copy BW (may fault on multi-GPU — existing bug)"
    echo "##########################################"
    echo
    mpirun -n 8 --allow-run-as-root "$SDMA_BIN" 2>&1 || echo "(sdma_self_copy_test faulted; skipped)"
  fi
} | tee "$LOG"

echo
echo "################################################################"
echo "## SUMMARY (auto-extracted)"
echo "################################################################"
echo
echo "--- CU XGMI load @ 32 MB/peer (AR per-peer shard, plan A viability) ---"
awk '/per-peer bytes = 32 MB/,/per-peer bytes = 64 MB/' "$LOG" | grep -v "per-peer bytes = 64" || true

echo
echo "--- CU XGMI load @ 64 MB/peer ---"
awk '/per-peer bytes = 64 MB/,0' "$LOG" | head -8 || true

echo
echo "--- SDMA self-copy @ 256 MB (plan B, may be missing if fault) ---"
grep -E "^\s+256\.[0-9]\s" "$LOG" | head -2 || echo "(no data)"

echo
echo "LOG: $LOG"
