#!/bin/bash
# Gracefully tear down a mori CI container at the end of a job.
#
# Why this exists: the CI jobs launch a long-lived container (via ci_run.sh with
# `sleep infinity`) and historically tore it down with `docker rm -f`, which
# SIGKILLs every process inside — including GPU test ranks that may still be
# mid-HIP-operation. A SIGKILL'd rank leaves an unreclaimable KFD context that
# leaks VRAM with no owning process (see ROCm/aiter#2061). This is amplified by
# `cancel-in-progress` cancellations and per-test `timeout -k ... SIGKILL`.
#
# This script signals the GPU processes to shut down cleanly first, then relies
# on tini (docker run --init, see ci_run.sh) to forward SIGTERM to the process
# group and reap children, so HIP contexts are released before the container dies.
#
# Intentionally best-effort: a teardown problem must never turn a green run red.
# `set -e` is deliberately NOT used.

set -uo pipefail

CONTAINER="${1:?usage: ci_stop.sh CONTAINER}"
CT="${CONTAINER_RUNTIME:-docker}"
STOP_TIMEOUT="${DOCKER_STOP_TIMEOUT:-60}"

if ! command -v "$CT" >/dev/null 2>&1; then
  echo "[ci_stop] '$CT' not available; nothing to tear down."
  exit 0
fi

if ! "$CT" inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "[ci_stop] container '$CONTAINER' not found; nothing to tear down."
  exit 0
fi

echo "=== [ci_stop] gracefully stopping '$CONTAINER' ==="

# Step 1: ask the GPU workloads inside the container to stop cleanly. SIGINT
# lets MPI ranks / torchrun workers / pytest run their normal teardown (HIP
# context destruction) so VRAM is released. Best-effort; the container may
# already be idle between steps.
echo "[ci_stop] signalling GPU processes inside the container (SIGINT)..."
"$CT" exec "$CONTAINER" bash -c '
  pkill -INT -f "mpirun|orted|torchrun|[[:space:]]python[0-9.]*[[:space:]]|/build/tests/|/build/examples/|/build/benchmark/|pytest" 2>/dev/null || true
' 2>/dev/null || true

# Give processes a few seconds to unwind before the harder stop.
sleep 5

# Step 2: `docker stop` with a generous grace period. tini (--init) forwards
# this SIGTERM to the process group and reaps children, so remaining GPU
# contexts are torn down cleanly instead of being SIGKILLed.
echo "[ci_stop] $CT stop --time ${STOP_TIMEOUT} ${CONTAINER}..."
"$CT" stop --time "$STOP_TIMEOUT" "$CONTAINER" 2>/dev/null || true

# Step 3: remove the container so the name is free for the next job.
"$CT" rm -f "$CONTAINER" 2>/dev/null || true

echo "=== [ci_stop] teardown of '$CONTAINER' complete ==="
exit 0
