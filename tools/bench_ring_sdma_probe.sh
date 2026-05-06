#!/usr/bin/env bash
# Usage:
#   bash tools/bench_ring_sdma_probe.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori SIZE_MB=256 CASE_TIMEOUT_SEC=300 \
#   PROBE_WAIT=1 PROBE_MATRIX=0 PROBE_PHASE=1 PROBE_ROUND=6 REPEAT=1 \
#   SKIP_PULL=0 SKIP_BUILD=0 bash tools/bench_ring_sdma_probe.sh
#
# Runs only the copy-to-user correctness path with MORI_RING_SHARD_SDMA_PROBE=1
# to isolate SDMA submit vs signal wait. The probe is not expected to pass
# allreduce correctness. PROBE_WAIT=0 checks submit only; PROBE_WAIT=1 also
# checks signal delivery using generation-based expected values.

set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${SIZE_MB:=256}"
: "${CASE_TIMEOUT_SEC:=60}"
: "${PROBE_WAIT:=1}"
: "${PROBE_MATRIX:=0}"
: "${PROBE_PHASE:=1}"
: "${PROBE_ROUND:=6}"
: "${REPEAT:=1}"
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
command -v pip3 >/dev/null || { echo "MISSING: pip3"; exit 1; }
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
echo "SIZE_MB=$SIZE_MB ELEMS=$ELEMS PROBE_WAIT=$PROBE_WAIT PROBE_MATRIX=$PROBE_MATRIX PROBE_PHASE=$PROBE_PHASE PROBE_ROUND=$PROBE_ROUND REPEAT=$REPEAT CASE_TIMEOUT_SEC=$CASE_TIMEOUT_SEC"
echo "PROBE_SCRIPT_VERSION=pre_submit_wait_v2"
if grep -q "version=pre_submit_wait_v2" include/mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp &&
   grep -q "wait target qId=0 base=.*before=.*expected" include/mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp; then
  echo "SOURCE_FEATURE: probe version=pre_submit_wait_v2 uses generation expected and prints signal before/target"
else
  echo "ERROR: source is missing probe version=pre_submit_wait_v2 generation wait-target instrumentation"
  exit 1
fi

if [ "$SKIP_PULL" != "1" ]; then
  echo
  echo "==================== [git pull] ===================="
  git pull origin sdma-test
  git log -1 --oneline
fi

if [ "$SKIP_BUILD" != "1" ]; then
  echo
  echo "==================== [pip install] ===================="
  BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
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
    --warmup 1 \
    --ring-sdma-probe-only 2>&1
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
    for rep in $(seq 1 "$REPEAT"); do
      label=$([ "$PROBE_PHASE" = "0" ] && echo "RS_${PROBE_ROUND}_REP_${rep}" || echo "AG_${PROBE_ROUND}_REP_${rep}")
      run_one "$label" "$PROBE_PHASE" "$PROBE_ROUND"
    done
  fi
} | tee "$LOG"

echo
echo "################################################################"
echo "## RING SDMA PROBE SUMMARY (auto-extracted from $LOG)"
echo "################################################################"

awk -v wait="$PROBE_WAIT" '
  /^========== .*_EXIT rc=/ {
    label = $2
    sub(/_EXIT$/, "", label)
    rc = $3
    sub(/^rc=/, "", rc)
    exit_rc[label] = rc + 0
    next
  }
  /^========== (RS|AG|PROBE)/ {
    label = $2
    current = label
    if (!(label in seen)) {
      seen[label] = 1
      order[++n] = label
    }
    next
  }
  /RING_SDMA_PROBE after put/ { after_put[current]++; next }
  /RING_SDMA_PROBE version=pre_submit_wait_v2/ { version_seen[current]++; next }
  /RING_SDMA_PROBE wait target qId=0 .*expected=/ { wait_target[current]++; next }
  /RING_SDMA_PROBE skip wait/ { skipped[current]++; next }
  /RING_SDMA_PROBE done/ { done[current]++; next }
  /\[STUCK\] PE .* RING_SDMA_PROBE/ { stuck[current]++; next }
  END {
    printf "%-8s %8s %8s %8s %8s %8s %8s %8s %s\n", "label", "version", "target", "after", "done", "skip", "stuck", "rc", "status"
    bad = 0
    first_bad = ""
    for (i = 1; i <= n; ++i) {
      label = order[i]
      rc = (label in exit_rc) ? exit_rc[label] : -999
      pass = (rc == 0 && stuck[label] == 0 && version_seen[label] >= 8 && wait_target[label] >= 8)
      if (wait == 1) {
        pass = pass && done[label] >= 8
      } else {
        pass = pass && skipped[label] >= 8
      }
      status = pass ? "PASS" : "FAIL"
      if (!pass) {
        bad++
        if (first_bad == "") first_bad = label
      }
      printf "%-8s %8d %8d %8d %8d %8d %8d %8d %s\n",
             label, version_seen[label], wait_target[label], after_put[label], done[label], skipped[label],
             stuck[label], rc, status
    }
    if (bad == 0 && n > 0) {
      print "NEXT_DECISION: all probe labels passed; run full fused ring with tools/bench_sdma_ag_copy_pipe.sh."
    } else {
      printf "NEXT_DECISION: fix first failing probe label=%s before running full fused ring.\n", first_bad
    }
  }
' "$LOG"

if ! grep -q "RING_SDMA_PROBE version=pre_submit_wait_v2" "$LOG" ||
   ! grep -q "RING_SDMA_PROBE wait target qId=0 .*expected=" "$LOG"; then
  echo
  echo "BUILD_MISMATCH: runtime log has no probe version=pre_submit_wait_v2 and pre-submit wait-target markers."
  echo "BUILD_MISMATCH: rerun without SKIP_BUILD=1, and make sure this script was pulled/applied before running."
fi

echo
echo "---- raw probe lines ----"
grep -E "========== (RS|AG|PROBE)|RING_SDMA_PROBE|Ring SDMA probe|FAILED|PASSED|ProcessRaisedException|Timeout|STUCK|_EXIT" "$LOG" || true
echo "LOG: $LOG"

echo "NOTE: ring_sdma_probe does not compute allreduce; correctness failure is expected."
exit 0
