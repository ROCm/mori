#!/usr/bin/env bash
# Where the node is: container, wedged pids, VRAM. Cheap enough to run between everything.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
docker ps -a --filter "name=$CTR" --format '{{.Names}} | {{.Status}}'
echo "--- pids ---"
pgrep -af 'bench_dispatch_combine|ep_test.sh' | head -6 || echo "(nothing)"
echo "--- kfd pids ---"
timeout 20 rocm-smi --showpids 2>&1 | grep -E "python3|inb-node|No KFD|PID" | head -8 || true
echo "--- vram ---"
timeout 20 rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -4
echo "QSTATE_DONE"
