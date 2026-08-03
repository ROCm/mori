#!/usr/bin/env bash
# Recreate MORI-F1 against the CURRENT /dev after a host reboot.
#
# The container came back seeing 16 render nodes against the host's 32 and failing libdrm auth from
# GPU[2] on, with torch counting 0 devices. Its device list is the /dev/dri DIRECTORY, resolved when
# the container was created, so a reboot that renumbers the nodes leaves it holding a stale set and
# no amount of `docker start` re-resolves it.
#
# ORDER MATTERS. /root/mori_tdm and the JIT cache live in the container's writable layer, not a
# volume, so they are lost with the container. Every step below is checked before the next one runs
# and the old container is only renamed, never removed: if the new one comes up wrong the old layer
# is still there to commit again.
set -uo pipefail
C=${C:-MORI-F1}
IMG="${IMG:-}"
OLD="${C}-preboot-$(date -u +%H%M%S)"
# SNAPONLY=1 skips straight to running an image that was already committed, for the case where the
# commit worked and only the run was wrong.
SNAPONLY="${SNAPONLY:-0}"

echo "HOST_RENDER=$(ls /dev/dri/renderD* 2>/dev/null | wc -l) HOST_CARD=$(ls /dev/dri/card* 2>/dev/null | wc -l)"

if [ "$SNAPONLY" = 0 ]; then
  IMG=mori-f1-snap:$(date -u +%Y%m%d-%H%M%S)
  echo "-- commit --"
  docker commit "$C" "$IMG" >/dev/null || { echo "ABORT: commit failed"; exit 1; }
  docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: snapshot image missing"; exit 1; }
  echo "committed $IMG"

  echo "-- stop and rename, NOT remove --"
  docker stop -t 20 "$C" >/dev/null 2>&1
  docker rename "$C" "$OLD" || { echo "ABORT: rename failed, old container still named $C"; exit 1; }
  echo "old container kept as $OLD"
else
  [ -n "$IMG" ] || { echo "ABORT: SNAPONLY=1 needs IMG="; exit 1; }
  docker rm -f "$C" >/dev/null 2>&1
  echo "reusing $IMG"
fi

echo "-- run --"
# Same flags docker inspect reported for the original: privileged, host net and ipc, 64 GiB shm,
# video group, SYS_PTRACE, seccomp and label off, in /app/ATOM.
#
# --entrypoint is set explicitly and NO command is passed. The snapshot inherits the original's
# Entrypoint=[sleep], so passing `sleep infinity` as the command appends to it and the container
# tries to run `sleep sleep infinity`, exits instantly, and takes the run down with it -- which is
# what happened the first time this script ran.
docker run -d --name "$C" \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host --privileged \
  --shm-size=64g --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -w /app/ATOM --entrypoint sleep "$IMG" infinity >/dev/null || { echo "ABORT: run failed"; exit 1; }
sleep 4
docker ps --format '{{.Names}} {{.Status}}' | grep -q "^$C " || {
  echo "ABORT: $C is not running"; docker logs "$C" 2>&1 | tail -5; exit 1; }

echo "-- verify --"
docker exec "$C" bash -lc '
  echo "C_RENDER=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)"
  rocm-smi --showid 2>&1 | grep -ciE "^GPU\[" | sed "s/^/SMI_GPUS=/"
  rocm-smi --showid 2>&1 | grep -iE "error|libdrm" | head -3
  python3 -c "import torch;print(\"TORCH_GPUS=\"+str(torch.cuda.device_count()))" 2>&1 | tail -2
  ls -d /root/mori_tdm >/dev/null 2>&1 && cd /root/mori_tdm && git log --oneline -1 | sed "s/^/REPO_HEAD=/"
  ls ~/.mori* /root/.cache/mori* -d 2>/dev/null | head -3
'
echo "RECREATE_DONE"
