#!/usr/bin/env bash
# Usage:
#   bash tools/bench_baseline_multistage_sweep.sh
#
# Env overrides:
#   REPO=/home/fizhang/test/mori NUM_STAGES=4 ITERATIONS=100 WARMUP=20 \
#   PIPELINE_CU=224 PIPELINE_CHUNKS=4 CONTINUOUS_ITERS=0 SKIP_PULL=0 SKIP_BUILD=0 \
#   bash tools/bench_baseline_multistage_sweep.sh
#
# Runs baseline multi-stage size sweep and prints 2..256MB comparison tables for:
# SDMA copy, SDMA no-copy, RCCL, copy-vs-RCCL, no-copy-vs-RCCL, copy penalty.

set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${NUM_STAGES:=4}"
: "${ITERATIONS:=100}"
: "${WARMUP:=20}"
: "${PIPELINE_CU:=224}"
: "${PIPELINE_CHUNKS:=4}"
: "${CONTINUOUS_ITERS:=0}"
: "${CASE_TIMEOUT_SEC:=1800}"
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
echo "NUM_STAGES=$NUM_STAGES ITERATIONS=$ITERATIONS WARMUP=$WARMUP PIPELINE_CU=$PIPELINE_CU PIPELINE_CHUNKS=$PIPELINE_CHUNKS CONTINUOUS_ITERS=$CONTINUOUS_ITERS CASE_TIMEOUT_SEC=$CASE_TIMEOUT_SEC"

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

LOG="/tmp/perf_baseline_multistage_sweep_$(date +%s).log"
echo
echo "==================== [baseline sweep] LOG=$LOG ===================="

RUN_ARGS=(
  --num-stages "$NUM_STAGES"
  --elems 67108864
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --sweep
)
if [ "$CONTINUOUS_ITERS" -gt 0 ]; then
  RUN_ARGS+=(--continuous-iters "$CONTINUOUS_ITERS")
fi

timeout --signal=TERM "$CASE_TIMEOUT_SEC" env \
  MORI_CONTINUOUS_PREP=0 \
  MORI_PIPELINE_CU="$PIPELINE_CU" \
  MORI_PIPELINE_CHUNKS="$PIPELINE_CHUNKS" \
  python3 tests/python/ccl/test_allreduce.py "${RUN_ARGS[@]}" 2>&1 | tee "$LOG"

echo
echo "################################################################"
echo "## BASELINE MULTI-STAGE SWEEP SUMMARY (2..256MB)"
echo "## Extracted from $LOG"
echo "################################################################"

awk '
  /Table 1: Overlap Wall Time/ { table="wall"; next }
  /Table 2: GEMM Slowdown/ { table="slowdown"; next }
  /Table 3: Sequential AllReduce Time/ { table="seq_ar"; next }
  /Table 4: Sequential GEMM Time/ { table="seq_gemm"; next }
  $1 ~ /^(2|4|8|16|32|64|128|256)$/ && $2 == "MB" && $3 == "|" {
    size=$1 + 0
    copy=$4 + 0.0
    nocopy=$6 + 0.0
    rccl=$8 + 0.0
    data[table, size, "copy"] = copy
    data[table, size, "nocopy"] = nocopy
    data[table, size, "rccl"] = rccl
    seen[size] = 1
  }
  END {
    split("2 4 8 16 32 64 128 256", sizes, " ")
    print ""
    print "### Wall ms / Gap"
    printf "%8s %12s %14s %12s %14s %16s %14s\n", "MB", "SDMA copy", "SDMA no-copy", "RCCL", "copy-RCCL", "no-copy-RCCL", "copy-penalty"
    for (i=1; i<=8; ++i) {
      s=sizes[i]+0
      if (!seen[s]) continue
      copy=data["wall",s,"copy"]; nocopy=data["wall",s,"nocopy"]; rccl=data["wall",s,"rccl"]
      printf "%8d %12.3f %14.3f %12.3f %+14.3f %+16.3f %+14.3f\n", s, copy, nocopy, rccl, copy-rccl, nocopy-rccl, copy-nocopy
    }
    print ""
    print "### Sequential AllReduce ms"
    printf "%8s %12s %14s %12s\n", "MB", "SDMA copy", "SDMA no-copy", "RCCL"
    for (i=1; i<=8; ++i) {
      s=sizes[i]+0
      if (!seen[s]) continue
      printf "%8d %12.3f %14.3f %12.3f\n", s, data["seq_ar",s,"copy"], data["seq_ar",s,"nocopy"], data["seq_ar",s,"rccl"]
    }
    print ""
    print "### GEMM Slowdown"
    printf "%8s %12s %14s %12s\n", "MB", "SDMA copy", "SDMA no-copy", "RCCL"
    for (i=1; i<=8; ++i) {
      s=sizes[i]+0
      if (!seen[s]) continue
      printf "%8d %12.3f %14.3f %12.3f\n", s, data["slowdown",s,"copy"], data["slowdown",s,"nocopy"], data["slowdown",s,"rccl"]
    }
  }
' "$LOG"

echo
echo "LOG: $LOG"
