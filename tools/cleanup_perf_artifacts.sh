#!/usr/bin/env bash
# Usage:
#   bash tools/cleanup_perf_artifacts.sh
#
# Env overrides:
#   CLEAN_TMP_LOGS=1 CLEAN_GPU_CORES=1 CLEAN_PY_CACHE=1 CLEAN_BUILD=0 \
#   bash tools/cleanup_perf_artifacts.sh
#
# Removes generated benchmark/core artifacts that commonly fill the remote
# test container. It prints disk usage before and after.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${CLEAN_TMP_LOGS:=1}"
: "${CLEAN_GPU_CORES:=1}"
: "${CLEAN_PY_CACHE:=1}"
: "${CLEAN_BUILD:=0}"

cd "$REPO"

echo "==================== [disk before] ===================="
df -h /tmp "$REPO" || true

if [ "$CLEAN_GPU_CORES" = "1" ]; then
  echo
  echo "==================== [clean gpu/core dumps] ===================="
  rm -fv gpucore.* core core.* 2>/dev/null || true
  rm -fv /tmp/gpucore.* /tmp/core /tmp/core.* 2>/dev/null || true
fi

if [ "$CLEAN_TMP_LOGS" = "1" ]; then
  echo
  echo "==================== [clean perf logs] ===================="
  rm -fv /tmp/perf_*.log /tmp/*perf*.log 2>/dev/null || true
fi

if [ "$CLEAN_PY_CACHE" = "1" ]; then
  echo
  echo "==================== [clean python caches] ===================="
  rm -rfv .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
  find . -type d -name __pycache__ -prune -exec rm -rfv {} + 2>/dev/null || true
fi

if [ "$CLEAN_BUILD" = "1" ]; then
  echo
  echo "==================== [clean build dirs] ===================="
  rm -rfv build dist *.egg-info 2>/dev/null || true
fi

echo
echo "==================== [disk after] ===================="
df -h /tmp "$REPO" || true
