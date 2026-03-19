#!/bin/bash
set -e

IMAGE_NAME="${1:-rocm/mori:benchmark}"

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --group-add video \
  --group-add render \
  "$IMAGE_NAME" \
  bash
