#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
TEST_IMAGE=${TEST_IMAGE:-mori-umbp-sglang:tp4-20261008}
ACTION=${1:-help}

usage() {
  cat <<'EOF'
Usage: bash examples/umbp/sglang/run_demo.sh ACTION [RUN_NAME]
  build          Build from the pinned public base (no GPU/model needed)
  check          CPU runtime check; also check MODEL_HOST_PATH if supplied
  check-hardware Check host GPU/sysfs/dmesg visibility without running kernels
  preview        Show the complete server configuration without starting it
  serve          Run the TP4 server until timeout/Ctrl-C
  smoke          Start server, send two requests, save results and stop
  fill-replay    Start server, generate new inputs, require DRAM restore and stop

Environment:
  TEST_IMAGE       Output/runtime image tag
  BASE_IMAGE       Optional equivalent local base for offline build
  MODEL_HOST_PATH  Absolute DeepSeek V4 Pro directory (required for model runs)
  RUN_HOST_PATH    Absolute output parent directory (required for model runs)
  PORT / NCCL_PORT Defaults: 18000 / 19500
  IDLE_SECONDS     Defaults: 300 (allowed: 60..1799)
  RUN_TIMEOUT      Defaults: 3600 seconds after the idle gate
EOF
}

case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  build)
    BUILD_ARGS=()
    if [[ -n ${BASE_IMAGE:-} ]]; then
      BUILD_ARGS+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
    fi
    # A small tar context also works with Docker's legacy builder, which does
    # not support Dockerfile-specific ignore files.
    tar -C "$REPO_DIR" --exclude=__pycache__ --exclude='*.pyc' --exclude='*.pyo' \
      -cf - examples/umbp/sglang examples/umbp/sglang_linker_smoke.py |
      docker build "${BUILD_ARGS[@]}" -t "$TEST_IMAGE" \
        -f examples/umbp/sglang/Dockerfile -
    exit 0
    ;;
  check|check-hardware|preview|serve|smoke|fill-replay) ;;
  *) usage >&2; exit 2 ;;
esac

DOCKER_ARGS=(run --rm --init)
ENTRY_ARGS=("$ACTION")
if [[ -n ${MODEL_HOST_PATH:-} ]]; then
  [[ $MODEL_HOST_PATH == /* && -d $MODEL_HOST_PATH ]] || {
    printf 'MODEL_HOST_PATH must be an existing absolute directory\n' >&2; exit 2;
  }
  # --mount also rejects missing source paths, unlike -v.
  DOCKER_ARGS+=(--mount "type=bind,src=$MODEL_HOST_PATH,dst=/model/DeepSeek-V4-Pro,readonly")
  ENTRY_ARGS+=(--model /model/DeepSeek-V4-Pro)
fi
if [[ $ACTION == check-hardware || $ACTION == serve || $ACTION == smoke || $ACTION == fill-replay ]]; then
  DOCKER_ARGS+=(--network host --ipc host --pid host --privileged
    --device /dev/kfd --device /dev/dri
    --security-opt seccomp=unconfined --security-opt label=disable --ulimit core=0:0)
else
  DOCKER_ARGS+=(--network none)
fi
if [[ $ACTION == serve || $ACTION == smoke || $ACTION == fill-replay ]]; then
  : "${MODEL_HOST_PATH:?Set MODEL_HOST_PATH to the DeepSeek V4 Pro directory}"
  : "${RUN_HOST_PATH:?Set RUN_HOST_PATH to an absolute output parent directory}"
  [[ $RUN_HOST_PATH == /* ]] || { printf 'RUN_HOST_PATH must be absolute\n' >&2; exit 2; }
  RUN_NAME=${2:-$(date -u +%Y%m%dT%H%M%SZ)-$ACTION}
  [[ $RUN_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
    printf 'RUN_NAME must be a single directory name\n' >&2; exit 2;
  }
  [[ ! -e $RUN_HOST_PATH/$RUN_NAME ]] || { printf 'Run directory already exists\n' >&2; exit 2; }
  mkdir -p -- "$RUN_HOST_PATH"
  DOCKER_ARGS+=(--mount "type=bind,src=$RUN_HOST_PATH,dst=/runs"
    -e "PORT=${PORT:-18000}" -e "NCCL_PORT=${NCCL_PORT:-19500}")
  ENTRY_ARGS+=(--out "/runs/$RUN_NAME" --idle-seconds "${IDLE_SECONDS:-300}"
    --timeout "${RUN_TIMEOUT:-3600}")
fi
exec docker "${DOCKER_ARGS[@]}" "$TEST_IMAGE" "${ENTRY_ARGS[@]}"
