#!/bin/bash
# Best-effort VRAM leak self-heal for the shared CI runner.
#
# Even with graceful container teardown (ci_stop.sh) + tini reaping (ci_run.sh
# --init), a rank that was hard-killed mid-HIP-operation can leave an
# unreclaimable KFD context that holds VRAM with no owning process. Because our
# runner is long-lived (not an ephemeral pod), that leaked VRAM persists and
# starves the next job. This script detects GPUs still holding VRAM after a job
# and, as a last resort, issues a per-GPU `rocm-smi --gpureset` scoped to ONLY
# the leaked cards so we never disturb a device another job might be using.
#
# Intentionally best-effort: never fail the job. `set -e` is NOT used.

set -uo pipefail

# VRAM% at or above this is considered "leaked / not clean" after teardown.
THRESHOLD="${VRAM_CLEAR_THRESHOLD:-6}"

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "[vram] rocm-smi not available; skipping VRAM check."
  exit 0
fi

# --gpureset needs root. Use a non-interactive sudo only when we are not already
# root and passwordless sudo is available; otherwise fall through (best-effort).
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
  fi
fi

# Print indices of GPUs whose "GPU Memory Allocated (VRAM%)" is >= THRESHOLD,
# one per line. Matches ONLY the allocation line (not Read/Write Activity).
# Uses grep/sort only so it works on mawk-only hosts.
get_dirty_gpu_indices() {
  timeout 30 rocm-smi --showmemuse 2>/dev/null \
    | grep -E "GPU Memory Allocated \(VRAM%\): ([${THRESHOLD}-9]|[1-9][0-9]|100)\b" \
    | grep -oE 'GPU\[[0-9]+\]' \
    | grep -oE '[0-9]+' \
    | sort -un
}

echo "=== [vram] post-job VRAM status ==="
timeout 30 rocm-smi --showmemuse 2>&1 | grep -E "GPU Memory Allocated \(VRAM%\)" || true

mapfile -t DIRTY < <(get_dirty_gpu_indices)

if [ "${#DIRTY[@]}" -eq 0 ]; then
  echo "[vram] all GPUs clean (< ${THRESHOLD}% VRAM). Nothing to do."
  exit 0
fi

echo "[vram] leaked GPUs (>= ${THRESHOLD}% VRAM with no owner): ${DIRTY[*]}"
echo "[vram] attempting scoped last-resort reset on those GPUs only..."

for idx in "${DIRTY[@]}"; do
  echo "[vram] ${SUDO} rocm-smi -d ${idx} --gpureset"
  timeout 60 ${SUDO} rocm-smi -d "${idx}" --gpureset 2>&1 || echo "[vram] gpureset on GPU ${idx} failed (best-effort)"
done

# Report final state for visibility; do not fail on residual leak.
echo "=== [vram] VRAM status after reset ==="
timeout 30 rocm-smi --showmemuse 2>&1 | grep -E "GPU Memory Allocated \(VRAM%\)" || true

mapfile -t STILL_DIRTY < <(get_dirty_gpu_indices)
if [ "${#STILL_DIRTY[@]}" -ne 0 ]; then
  echo "[vram] WARNING: GPUs still not clean after reset: ${STILL_DIRTY[*]}"
fi

exit 0
