#!/bin/bash
set -euo pipefail
set -x

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your setup
# ---------------------------------------------------------------------------
REMOTE_HOST="banff-ccs-aus-p20-14"
LOCAL_CONTAINER="pemeliya-jax191"
REMOTE_CONTAINER="${LOCAL_CONTAINER}"

WORLD_SIZE=2
LOCAL_RANK=0
REMOTE_RANK=1
NUM_GPUS_PER_PROCESS=2

#$TEST $pid $NUM_PROCS $NUM_GPUS_PER_PROCESS

# NFS-mounted path *inside* the container where the repo / build lives
NFS_WORK_DIR="${NFS_WORK_DIR:-/data/mori}"
PROGRAM="${PROGRAM:-build/examples/allgather_test}"
CHUNK_BYTES="${CHUNK_BYTES:-1048576}"

# ---------------------------------------------------------------------------
# Environment variables forwarded into the containers
# ---------------------------------------------------------------------------
ENV_VARS=(
  -e HSA_NO_SCRATCH_RECLAIM=1
  -e HIP_FORCE_DEV_KERNARG=1
  -e MORI_SHMEM_MODE=STATIC_HEAP
  -e MORI_SHMEM_HEAP_SIZE=5G
  -e MORI_KERNEL_DIR="${NFS_WORK_DIR}/build/lib/gfx942_mlx5"
  -e MORI_APP_LOG_LEVEL=DEBUG
  -e MORI_SHMEM_LOG_LEVEL=DEBUG
  -e MORI_CORE_LOG_LEVEL=DEBUG
  -e MORI_OPS_LOG_LEVEL=DEBUG
  -e MORI_DISABLE_P2P=0
  -e MORI_ENABLE_SDMA=0
  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1
)

for v in ${EXTRA_ENV:-}; do
  ENV_VARS+=(-e "$v")
done

rm -f ${NFS_WORK_DIR}/allgather_test_uid.bin
# ---------------------------------------------------------------------------
# Build the inner commands: <rank> <world_size> [chunk_bytes]
# ---------------------------------------------------------------------------
LOCAL_CMD="cd ${NFS_WORK_DIR} && ./${PROGRAM} ${LOCAL_RANK} ${WORLD_SIZE} ${NUM_GPUS_PER_PROCESS} ${CHUNK_BYTES}"
REMOTE_CMD="cd ${NFS_WORK_DIR} && ./${PROGRAM} ${REMOTE_RANK} ${WORLD_SIZE} ${NUM_GPUS_PER_PROCESS} ${CHUNK_BYTES}"

echo "============================================================"
echo " Local host  : $(hostname)  container: ${LOCAL_CONTAINER}"
echo " Remote host : ${REMOTE_HOST}  container: ${REMOTE_CONTAINER}"
echo " Program     : ${PROGRAM}  world_size=${WORLD_SIZE}  chunk=${CHUNK_BYTES}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch rank 0 locally (background) — must start first to create uid file
# ---------------------------------------------------------------------------
echo "[rank ${LOCAL_RANK}]  Starting on local container ${LOCAL_CONTAINER} ..."
docker exec "${ENV_VARS[@]}" "${LOCAL_CONTAINER}" \
  bash -c "${LOCAL_CMD}" 2>&1 | sed "s/^/[rank ${LOCAL_RANK}] /" &
LOCAL_PID=$!

# Small delay to let rank 0 write the uid file before rank 1 starts polling
sleep 1

# ---------------------------------------------------------------------------
# Launch rank 1 on remote host via SSH (background)
# ---------------------------------------------------------------------------
DOCKER_CMD="docker exec"
for e in "${ENV_VARS[@]}"; do
  DOCKER_CMD+=" $e"
done
DOCKER_CMD+=" ${REMOTE_CONTAINER} bash -c '${REMOTE_CMD}'"

echo "[rank ${REMOTE_RANK}] Starting on ${REMOTE_HOST} container ${REMOTE_CONTAINER} ..."
ssh -i ~/.ssh/id_rsa_scp "${REMOTE_HOST}" "${DOCKER_CMD}" 2>&1 \
  | sed "s/^/[rank ${REMOTE_RANK}] /" &
REMOTE_PID=$!

# ---------------------------------------------------------------------------
# Wait for both and report exit codes
# ---------------------------------------------------------------------------
LOCAL_RC=0;  wait "${LOCAL_PID}"  || LOCAL_RC=$?
REMOTE_RC=0; wait "${REMOTE_PID}" || REMOTE_RC=$?

echo "============================================================"
echo " Rank ${LOCAL_RANK}  (local)  exit code: ${LOCAL_RC}"
echo " Rank ${REMOTE_RANK} (remote) exit code: ${REMOTE_RC}"
echo "============================================================"

if [[ ${LOCAL_RC} -ne 0 || ${REMOTE_RC} -ne 0 ]]; then
  exit 1
fi
