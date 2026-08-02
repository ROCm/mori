#!/usr/bin/env bash
# Bring the container back after a reset and prove the GPUs are idle before anything is timed.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
timeout 120 docker start "$CTR" 2>&1 | tail -2
sleep 5
docker exec "$CTR" bash -lc 'HN=$(hostname); grep -q "$HN" /etc/hosts || echo "127.0.0.1 $HN" >> /etc/hosts; echo hosts_ok'
docker ps --filter "name=$CTR" --format '{{.Names}} | {{.Status}}'
timeout 20 rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'VRAM Total Used Memory \(B\): \K[0-9]+' | sort -rn | head -4
echo "QSTART_DONE"
