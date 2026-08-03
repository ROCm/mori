#!/usr/bin/env bash
# What is on f01-1: cards, images, whether an EPV2-shaped container can be stood up here.
set -uo pipefail
echo "== cards =="
ls /sys/class/kfd/kfd/topology/nodes 2>/dev/null | head
for n in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  gid=$(grep -m1 '^gfx_target_version' "$n" 2>/dev/null | awk '{print $2}')
  simd=$(grep -m1 '^simd_count' "$n" 2>/dev/null | awk '{print $2}')
  [ "${simd:-0}" != "0" ] && echo "node $(dirname $n | xargs basename) gfx=$gid simd=$simd"
done | head -12
echo "== images =="
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | head -20
echo "== running containers =="
docker ps --format '{{.Names}} {{.Image}}' | head -10
echo "== free vram =="
docker exec gfx1250_attn_res bash -lc 'rocm-smi --showmeminfo vram 2>/dev/null | grep -oP "VRAM Total Used Memory \(B\): \K[0-9]+" | sort -rn | head -8' 2>&1 | head -10
echo "== who is busy =="
ps -eo pid,user,pcpu,etime,args --sort=-pcpu | head -6
echo "F1PROBE_DONE"
