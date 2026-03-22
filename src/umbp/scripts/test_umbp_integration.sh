#!/usr/bin/env bash
set -e

# UMBP Integration Test — one-click, non-interactive.
# Launches docker container, rebuilds mori with UMBP, and runs hicache benchmarks.
#
# Usage:
#   bash test_umbp_integration.sh [branch]
#   bash test_umbp_integration.sh feat_ump_dist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORI_BRANCH="${1:-main}"
IMAGE_NAME=rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-20260316-hicache

echo "=== UMBP Integration Test ==="
echo "Image: ${IMAGE_NAME}"

sudo docker run -i \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -w /apps/ditian12/mori \
    --env HUGGINGFACE_HUB_CACHE=/models \
    --env MODELSCOPE_CACHE=/models \
    -v /apps/data/models/:/models \
    -v /nfsdata/DeepSeek-R1:/nfsdata/DeepSeek-R1 \
    -v /apps:/apps \
    -v /home/ditian12:/home/ditian12 \
    -v /it-share:/it-share \
    -v /usr/sbin/nicctl:/usr/sbin/nicctl \
    --shm-size 32G \
    ${IMAGE_NAME} /bin/bash "${SCRIPT_DIR}/test_umbp_inner.sh" "${MORI_BRANCH}"
