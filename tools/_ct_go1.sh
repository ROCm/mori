#!/usr/bin/env bash
# One detached pass over everything the quantised combine question needs, ordered so that a node
# that wedges partway still leaves the most answers behind.
#
# THE ORDERING RULE. This node has stopped answering ssh three times, and both times the step was
# identifiable it was a four-rank peer-read run -- the single-GPU quantise bench and the hipcc
# builds have never done it. So every single-GPU measurement runs BEFORE any four-rank one, even
# though the four-rank ones are more interesting: a wedge costs 15 to 60 minutes, and losing the
# cheap half of the answer to it twice is how a night goes.
#
# S1 container health   -- the devices go stale across a host reboot; recreate rather than guess
# S2 q_micro smoke      -- one GPU, seconds. If THIS wedges the node the hypothesis above is wrong
# S3 q_micro full       -- the quantise pass priced against its 51us floor, all modes and both
#                          allocators. One GPU.
# S4 g_micro smoke      -- four ranks, tiny. The first thing that can wedge the node
# S5 g_micro full       -- the scale-broadcast strategies, including the wave-uniform one that says
#                          whether the register prefetch is solving a self-inflicted problem
# Everything prints a marker and a timestamp so a truncated log still says where it stopped.
set -uo pipefail
C=${C:-MORI-F1}
SRC=${SRC:-/root/mori_tdm}
STAGES="${STAGES:-1 2 3 4 5}"
mark() { echo "=== $* @ $(date -u +%H:%M:%S) ==="; }

for s in $STAGES; do
case $s in

1)
mark S1 container
if ! docker exec "$C" bash -lc 'python3 -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==4 else 1)"' 2>/dev/null; then
  echo "S1: container does not see 4 GPUs, recreating from a commit"
  IMG=mori-f1-snap:$(date -u +%Y%m%d-%H%M%S)
  docker commit "$C" "$IMG" >/dev/null || { echo "S1 ABORT: commit failed"; exit 1; }
  docker stop -t 20 "$C" >/dev/null 2>&1
  docker rename "$C" "${C}-old-$(date -u +%H%M%S)" || { echo "S1 ABORT: rename failed"; exit 1; }
  # --entrypoint sleep with no command: the snapshot inherits Entrypoint=[sleep], so passing
  # `sleep infinity` would run `sleep sleep infinity` and exit at once.
  docker run -d --name "$C" --device=/dev/kfd --device=/dev/dri \
    --network=host --ipc=host --privileged --shm-size=64g --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
    -w /app/ATOM --entrypoint sleep "$IMG" infinity >/dev/null || { echo "S1 ABORT: run failed"; exit 1; }
  sleep 5
fi
docker exec "$C" bash -lc 'python3 -c "import torch;print(\"S1 TORCH_GPUS=\"+str(torch.cuda.device_count()))"'
docker exec "$C" bash -lc "cd $SRC && git fetch --quiet origin && git reset --hard origin/tdm-dispatch 2>&1 | tail -1"
;;

2)
mark S2 q_micro smoke
docker exec "$C" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
  -I $SRC/include -I $SRC $SRC/tools/q_micro.cc -o /tmp/q_base 2>&1 | grep -E 'error' | head -8; echo BUILD_RC=\$?"
docker exec "$C" bash -lc "Q_TOK=512 Q_ITERS=3 Q_GRIDS=256 Q_WPB=8 Q_MODES=0 timeout 300 /tmp/q_base"
;;

3)
mark S3 q_micro full
docker exec "$C" bash -lc "Q_MODES='0 1 2 3 4' Q_UNC=2 Q_GRIDS='256 512 1024' Q_WPB='8 16' timeout 900 /tmp/q_base"
mark S3b q_micro cached allocator
docker exec "$C" bash -lc "Q_MODES='0 2' Q_UNC=0 Q_GRIDS='256 512 1024' Q_WPB='8 16' timeout 900 /tmp/q_base"
for d in 1 4 14 28; do
  mark "S3c QSTGU=$d"
  docker exec "$C" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
    -I $SRC/include -I $SRC -DMORI_COMB_QSTGU=$d $SRC/tools/q_micro.cc -o /tmp/q_s$d 2>&1 | grep -E 'error' | head -4"
  docker exec "$C" bash -lc "Q_MODES=0 Q_UNC=2 Q_GRIDS='256 512 1024' Q_WPB='8 16' timeout 900 /tmp/q_s$d"
done
for w in 1 2; do
  mark "S3d QSCW=$w"
  docker exec "$C" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
    -I $SRC/include -I $SRC -DMORI_COMB_QSCW=$w $SRC/tools/q_micro.cc -o /tmp/q_w$w 2>&1 | grep -E 'error' | head -4"
  docker exec "$C" bash -lc "Q_MODES=0 Q_UNC=2 Q_CHECK=$([ $w = 2 ] && echo 0 || echo 1) Q_GRIDS='256 512' Q_WPB='8 16' timeout 900 /tmp/q_w$w"
done
;;

4)
mark S4 g_micro smoke FOUR RANKS
docker exec "$C" bash -lc "/opt/rocm/bin/hipcc --offload-arch=gfx1250 -std=c++17 -O3 \
  -I $SRC/include -I $SRC $SRC/tools/g_micro.cc -o /tmp/g_micro 2>&1 | grep -E 'error' | head -8; echo BUILD_RC=\$?"
docker exec "$C" bash -lc "G_TOK=256 G_ITERS=3 G_GRIDS=64 G_WPB=8 G_CHUNKS=1792 G_BYTES=1 G_DEQ=5 timeout 300 /tmp/g_micro"
;;

5)
mark S5 g_micro full
for d in 5 6 1 2 3; do
  mark "S5 DEQ=$d"
  docker exec "$C" bash -lc "G_BYTES=1 G_PIPE=0 G_DEQ=$d G_GRIDS='64 128 256' G_WPB='8 16' \
    G_CHUNKS='1792 3584' G_TOK=4096 G_ITERS=20 timeout 900 /tmp/g_micro"
done
mark "S5 bf16"
docker exec "$C" bash -lc "G_BYTES=2 G_PIPE=0 G_DEQ=0 G_GRIDS='64 128 256' G_WPB='8 16' \
  G_CHUNKS='1792 3584' G_TOK=4096 G_ITERS=20 timeout 900 /tmp/g_micro"
;;

esac
done
mark GO1_DONE
