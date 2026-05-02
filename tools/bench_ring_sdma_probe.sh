#!/usr/bin/env bash
# Usage:
#   bash tools/bench_ring_sdma_probe.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 CASE_TIMEOUT_SEC=300 \
#   PROBE_WAIT=1 PROBE_MATRIX=1 SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_ring_sdma_probe.sh
#
# Runs only the copy-to-user correctness path with MORI_RING_SHARD_SDMA_PROBE=1
# to isolate SDMA submit vs signal wait. The probe is not expected to pass
# allreduce correctness. PROBE_WAIT=0 checks submit only; PROBE_WAIT=1 also
# checks signal delivery using current signal + 1.

set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${CASE_TIMEOUT_SEC:=300}"
: "${PROBE_WAIT:=1}"
: "${PROBE_MATRIX:=1}"
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
FREE_KB=$(df -Pk /tmp "$REPO" | awk 'NR>1 {if (min=="" || $4<min) min=$4} END {print min+0}')
if [ "$FREE_KB" -lt $((20 * 1024 * 1024)) ]; then
  echo "ERROR: low disk space; min free across /tmp and repo is ${FREE_KB} KB (<20GB)"
  df -h /tmp "$REPO"
  exit 1
fi
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "HIP not available"
print("devices:", torch.cuda.device_count())
PY
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
echo "SIZE_MB=$SIZE_MB ELEMS=$ELEMS PROBE_WAIT=$PROBE_WAIT PROBE_MATRIX=$PROBE_MATRIX CASE_TIMEOUT_SEC=$CASE_TIMEOUT_SEC"

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

LOG="/tmp/perf_ring_sdma_probe_$(date +%s).log"
echo
echo "==================== [run] LOG=$LOG ===================="
run_one() {
  local label="$1" phase="$2" round="$3"
  echo
  echo "========== $label =========="
  set +e
  timeout --signal=TERM "$CASE_TIMEOUT_SEC" env \
    MORI_RING_SHARD_DIRECT=1 \
    MORI_RING_SHARD_SDMA_PROBE=1 \
    MORI_RING_SHARD_CU_DEBUG=0 \
    MORI_RING_SHARD_SDMA_PROBE_WAIT="$PROBE_WAIT" \
    MORI_RING_SHARD_SDMA_PROBE_PHASE="$phase" \
    MORI_RING_SHARD_SDMA_PROBE_ROUND="$round" \
    python3 tests/python/ccl/test_allreduce.py \
      --elems "$ELEMS" \
      --iterations 1 \
      --warmup 1 2>&1
  local rc=$?
  set -e
  echo "========== ${label}_EXIT rc=$rc =========="
  return 0
}

{
  if [ "$PROBE_MATRIX" = "1" ]; then
    for phase in 0 1; do
      for round in 0 1 2 3 4 5 6; do
        label=$([ "$phase" = "0" ] && echo "RS_${round}" || echo "AG_${round}")
        run_one "$label" "$phase" "$round"
      done
    done
  else
    run_one "PROBE" "${MORI_RING_SHARD_SDMA_PROBE_PHASE:-0}" \
      "${MORI_RING_SHARD_SDMA_PROBE_ROUND:-0}"
  fi
} | tee "$LOG"

echo
echo "################################################################"
echo "## RING SDMA PROBE SUMMARY (auto-extracted from $LOG)"
echo "################################################################"
grep -E "========== (RS|AG|PROBE)|RING_SDMA_PROBE|FAILED|PASSED|ProcessRaisedException|Timeout|STUCK|_EXIT" "$LOG" || true
echo "LOG: $LOG"

echo "NOTE: ring_sdma_probe does not compute allreduce; correctness failure is expected."
exit 0
