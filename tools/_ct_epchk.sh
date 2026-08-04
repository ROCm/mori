#!/usr/bin/env bash
# Run the standalone xGMI bandwidth check and throw the container away. Nothing here touches mori,
# which is the entire point: when dispatch reads 18% below its historical figure, this says whether
# the node still delivers its baseline bandwidth at all.
#
# It answers a sharper question than epsim does. The four checks split into two groups with very
# different repeatability (tools/epimg/README.md:121-125):
#
#   CU/TDM copy 1way   two processes, point to point, +-0.1% across runs   <- sensitive
#   a2a grid=512       forks 4 ranks and synchronises them, +-1.8%         <- carries scheduling noise
#
# So copy low  => link or hardware really degraded.
#    copy fine + a2a low => the loss is in multi-rank synchronisation, not bandwidth -- which is the
#    shape seen on 2026-08-04 (collectives -18%, single-card copy_ unchanged, epsim -15%).
# Do not read a ~2% a2a deviation as a regression; README says so explicitly.
#
# Baseline (same node, compiled-in BLKMUL=64 WTH=512 TWBLK=32 TWTH=256, tile 256x8, pipe 4):
#   a2a 1627.6 / CU copy 1641.0 / TDM copy 1637.3 / TDM store 1643.0, tolerance +-5%.
set -uo pipefail
CT="${CT:-epchk_now}"
IMG="${IMG:-}"

# Prefer whatever is already local -- pulling the private repo needs a login this script should not
# assume, and the local build tag and the registry tag are the same image.
if [ -z "$IMG" ]; then
  for cand in mori-epcheck:gfx1250-20260731 rocm/aigmodels-private:epcheck-gfx1250-20260731; do
    docker image inspect "$cand" >/dev/null 2>&1 && { IMG="$cand"; break; }
  done
fi
[ -n "$IMG" ] || { echo "ABORT: no epcheck image locally"; docker images --format '  {{.Repository}}:{{.Tag}}' | grep -i epcheck; exit 1; }
echo "image: $IMG"

# Same flags as tools/epimg/README.md:82-87. --ipc=host and --shm-size matter: tdma2a forks 4 ranks
# that share GPU memory through IPC handles, and ualoe_bw rendezvous over loopback between two
# processes. Getting them wrong yields a container that starts but cannot use the GPUs.
docker rm -f "$CT" >/dev/null 2>&1
docker run -d --name "$CT" \
  --device /dev/kfd --device /dev/dri \
  --ipc=host --network host --privileged \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
  --group-add video --shm-size 64g \
  "$IMG" >/dev/null || { echo "ABORT: docker run failed"; exit 1; }

echo "=== environment at start ==="
date -u '+  utc %F %T'
uptime | sed 's/^/  /'
ps -eo pcpu,pid,etime,comm --sort=-pcpu --no-headers 2>/dev/null | head -4 | sed 's/^/  cpu /'
rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "Used" | sed 's/^/  /'

echo "=== epcheck ==="
docker exec "$CT" /opt/epcheck/epcheck.sh 2>&1 | sed 's/^/  /'
echo "  exit=$?"

docker rm -f "$CT" >/dev/null 2>&1
echo "EPCHK_DONE"
