#!/bin/bash

# node2(114)本地执行
GITHUB_WORKSPACE=/home/qizzhang/workspace/mori   # 按需改;确保这里已有 mori 源码

docker rm -f mori_ci 2>/dev/null || true
mkdir -p "$GITHUB_WORKSPACE"
cd "$GITHUB_WORKSPACE"

docker build --network=host \
  --build-arg BASE_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.8.0 \
  -t rocm/mori:ci -f docker/Dockerfile.dev .

CONTAINER_RUNTIME=docker ./docker/ci_run.sh --name mori_ci \
  -e MORI_RDMA_DEVICES=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8 \
  -e MORI_RDMA_SL=3 \
  -e MORI_RDMA_TC=104 \
  -e MORI_SOCKET_IFNAME=enp49s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp49s0f1np1 \
  -e NCCL_SOCKET_IFNAME=enp49s0f1np1 \
  -v "$GITHUB_WORKSPACE":"$GITHUB_WORKSPACE" \
  -w "$GITHUB_WORKSPACE" \
  rocm/mori:ci sleep infinity
