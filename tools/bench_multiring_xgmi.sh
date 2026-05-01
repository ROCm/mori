#!/usr/bin/env bash
# Usage:
#   bash tools/bench_multiring_xgmi.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori MORI_MR_LANES="1F0R,3F0R,3F3R,4F4R,6F6R" \
#   MORI_MR_MB_PER_LANE=16 MORI_MR_BLOCKS_PER_LANE=8 SKIP_PULL=0 SKIP_BUILD=0 \
#   bash tools/bench_multiring_xgmi.sh

set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${MORI_MR_LANES:=1F0R,3F0R,3F3R,4F4R,6F6R}"
: "${MORI_MR_MB_PER_LANE:=16}"
: "${MORI_MR_BLOCKS_PER_LANE:=8}"
: "${MORI_MR_WARMUP:=5}"
: "${MORI_MR_ITERS:=20}"
: "${NPROCS:=8}"
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
command -v python3 >/dev/null || { echo "MISSING: python3"; exit 1; }
command -v cmake >/dev/null || { echo "MISSING: cmake"; exit 1; }
command -v hipcc >/dev/null || { echo "MISSING: hipcc"; exit 1; }
command -v mpirun >/dev/null || { echo "MISSING: mpirun"; exit 1; }
FREE_KB=$(df -Pk /tmp "$REPO" | awk 'NR>1 {if (min=="" || $4<min) min=$4} END {print min+0}')
if [ "$FREE_KB" -lt $((20 * 1024 * 1024)) ]; then
  echo "ERROR: low disk space; min free across /tmp and repo is ${FREE_KB} KB (<20GB)"
  df -h /tmp "$REPO"
  exit 1
fi
cmake --version | head -1
hipcc --version 2>/dev/null | head -2 || true
mpirun --version 2>&1 | head -1
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "HIP not available"
print("devices:", torch.cuda.device_count())
PY
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
echo "NPROCS=$NPROCS MORI_MR_LANES=$MORI_MR_LANES MORI_MR_MB_PER_LANE=$MORI_MR_MB_PER_LANE MORI_MR_BLOCKS_PER_LANE=$MORI_MR_BLOCKS_PER_LANE"

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
  git log -1 --oneline
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo
  echo "==================== [pip install] ===================="
  BUILD_EXAMPLES=ON pip install -e .
fi

EXE=""
for p in build/examples/multiring_xgmi_bench build/multiring_xgmi_bench multiring_xgmi_bench; do
  if [ -x "$p" ]; then
    EXE="$p"
    break
  fi
done
[ -n "$EXE" ] || { echo "ERROR: multiring_xgmi_bench executable not found after build"; exit 1; }

LOG="/tmp/perf_multiring_xgmi_$(date +%s).log"
echo
echo "==================== [run] EXE=$EXE LOG=$LOG ===================="
env MORI_MR_LANES="$MORI_MR_LANES" \
    MORI_MR_MB_PER_LANE="$MORI_MR_MB_PER_LANE" \
    MORI_MR_BLOCKS_PER_LANE="$MORI_MR_BLOCKS_PER_LANE" \
    MORI_MR_WARMUP="$MORI_MR_WARMUP" \
    MORI_MR_ITERS="$MORI_MR_ITERS" \
    mpirun -np "$NPROCS" --allow-run-as-root "$EXE" 2>&1 | tee "$LOG"

echo
echo "################################################################"
echo "## MULTI-RING XGMI SUMMARY (auto-extracted from $LOG)"
echo "################################################################"
awk '
  /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+F\/[[:space:]]*[0-9]+[[:space:]]+R/ {
    print $0
  }
' "$LOG"
echo "LOG: $LOG"
