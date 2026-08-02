#!/usr/bin/env bash
# Kill everything a previous run left behind, in the container and on the host.
#
# `timeout N bash -s` kills only the bash it started. A `docker exec` under it, and the ep_test.sh
# loop inside the container, both survive -- so a sweep that hits the remote timeout keeps running
# for as long as its own loop takes, and the NEXT sweep shares the GPU with it. That is what turned
# a 12-row script into two hours of nothing: two ep_test.sh loops alternately pkill-ing each
# other's bench. Run this before any timing run, and after any run that had to be cut short.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
docker exec "$CTR" bash -lc "pkill -9 -f ep_test.sh; pkill -9 -f bench_dispatch_combine; pkill -9 -f spawn_main; true"
pkill -9 -f 'docker exec' 2>/dev/null; true
sleep 5
echo "--- still alive ---"
pgrep -af 'bench_dispatch_combine|ep_test.sh' | head -10 || echo "(nothing)"
echo "--- vram ---"
rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -3
echo "QKILL_DONE"
