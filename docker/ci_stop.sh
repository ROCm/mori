#!/bin/bash
# Gracefully tear down a mori CI container so GPU ranks release their HIP/KFD
# contexts (and thus VRAM) instead of being SIGKILLed by `docker rm -f`.
# Best-effort: never fails the job (`set -e` intentionally not used).

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

# Ask GPU workloads to unwind cleanly (SIGINT), then let tini (--init) forward
# SIGTERM from `docker stop` to the process group and reap children.
"$CT" exec "$CONTAINER" bash -c '
  pkill -INT -f "mpirun|orted|torchrun|[[:space:]]python[0-9.]*[[:space:]]|/build/tests/|/build/examples/|/build/benchmark/|pytest" 2>/dev/null || true
' 2>/dev/null || true
# Wait for the interrupted workload to exit, then stop. PID 1 (sleep infinity)
# ignores SIGTERM, so a plain stop always waited out the whole timeout.
# Zombies (stat Z) do not count: without --init nothing reaps them.
deadline=$((SECONDS + STOP_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  left=$(timeout 10 "$CT" exec "$CONTAINER" ps -eo pid=,stat=,args= 2>/dev/null |
    awk '$1 != 1 && $2 !~ /^Z/ && !/sleep infinity/ && !/ps -eo/' | wc -l)
  [ "${left:-0}" -eq 0 ] && break
  sleep 1
done

"$CT" stop --timeout 5 "$CONTAINER" 2>/dev/null || true
"$CT" rm -f "$CONTAINER" 2>/dev/null || true

echo "=== [ci_stop] teardown of '$CONTAINER' complete ==="
exit 0
