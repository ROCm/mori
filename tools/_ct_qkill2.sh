#!/usr/bin/env bash
# Kill the leftovers from the HOST by pid.
#
# pkill inside the container did not reach them: the survivors are the bench itself, and the
# torch.distributed children hold the GPU whether or not the shell that launched them is gone.
# The host sees the container's processes through its own pid namespace, so kill -9 by pid works
# where a pattern match inside the container did not.
set -uo pipefail
for _ in 1 2 3; do
  pids=$(pgrep -f 'bench_dispatch_combine|spawn_main|ep_test.sh' || true)
  [ -z "$pids" ] && break
  echo "killing: $pids"
  kill -9 $pids 2>/dev/null || true
  sleep 3
done
echo "--- still alive ---"
pgrep -af 'bench_dispatch_combine|ep_test.sh' | head -10 || echo "(nothing)"
echo "--- vram ---"
rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -3
echo "QKILL2_DONE"
