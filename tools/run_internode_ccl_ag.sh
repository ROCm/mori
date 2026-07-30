#!/bin/bash
# Run one rank of the cross-node hierarchical AllGather bit-exact smoke test via
# torchrun (2 nodes x 8 GPU = world 16). Wraps tests/python/ccl/bench_ag_perf_w16.py,
# which asserts byte-for-byte equality vs torch.distributed.all_gather_into_tensor
# for the selected mori handle (device = ibgda_sdma, hostproxy = hp_sdma).
#
# Usage:
#   run_internode_ccl_ag.sh --rank <0|1> --master-addr <ip> --ifname <nic> \
#                           --handle <device|hostproxy> [--master-port <port>] \
#                           [--sizes-mb 8,64] [--reps N] [--warmup N]
#
# --sizes-mb is COMMA-separated (no spaces): the caller ships this argv through a
# second round of word-splitting (ssh "${NODE2_CMD[*]}"), so a space-separated
# list would be torn apart. Commas survive; we convert them back below.
#
# GLOO/NCCL/MORI socket ifname are derived from --ifname. RDMA env
# (MORI_RDMA_DEVICES/SL/TC, NCCL_IB_*) is passed by the caller via docker exec -e.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
HANDLE="device"
SIZES_MB="8,64"
REPS=3
WARMUP=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --rank)         RANK="$2";        shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
    --ifname)       IFNAME="$2";      shift 2 ;;
    --handle)       HANDLE="$2";      shift 2 ;;
    --sizes-mb)     SIZES_MB="$2";    shift 2 ;;
    --reps)         REPS="$2";        shift 2 ;;
    --warmup)       WARMUP="$2";      shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

export GLOO_SOCKET_IFNAME="$IFNAME"
export NCCL_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"
export MORI_ENABLE_SDMA=1
# Fused hierarchical path; graph-replay off (the launch-collapse HIP-graph
# fallback is not bit-exact-safe on all archs yet -- keep the eager path here).
export MORI_HIER_FUSE_LOCAL=1
export MORI_HIER_FUSE_REMOTE=1
export MORI_HIER_LOCAL_PUSHONLY=1
export MORI_HIER_CUDA_GRAPH=0

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Comma-separated -> space-separated argv for bench_ag_perf_w16.py's nargs='+'.
read -r -a SIZES_ARR <<< "${SIZES_MB//,/ }"

exec timeout "${MORI_INTERNODE_TIMEOUT:-300}" torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  tests/python/ccl/bench_ag_perf_w16.py \
  --handle "$HANDLE" \
  --sizes-mb "${SIZES_ARR[@]}" \
  --reps "$REPS" \
  --warmup "$WARMUP"
