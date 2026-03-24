#!/bin/bash
# Run one rank of an internode dispatch/combine test via torchrun.
#
# Usage:
#   run_internode_test.sh --rank <0|1> --master-addr <ip> --ifname <nic> \
#                         --cmd <bench|stress> --max-tokens <N> \
#                         [--master-port <port>] [--kernel-type <v1>] [--num-qp <N>]
#
# Environment variables GLOO_SOCKET_IFNAME and MORI_SOCKET_IFNAME are set
# automatically from --ifname.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
CMD=""
KERNEL_TYPE="v1"
NUM_QP=2
MAX_TOKENS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --rank)         RANK="$2";         shift 2 ;;
    --master-addr)  MASTER_ADDR="$2";  shift 2 ;;
    --master-port)  MASTER_PORT="$2";  shift 2 ;;
    --ifname)       IFNAME="$2";       shift 2 ;;
    --cmd)          CMD="$2";          shift 2 ;;
    --kernel-type)  KERNEL_TYPE="$2";  shift 2 ;;
    --num-qp)       NUM_QP="$2";       shift 2 ;;
    --max-tokens)   MAX_TOKENS="$2";   shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME CMD MAX_TOKENS; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

export GLOO_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

exec timeout 120 torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --cmd "$CMD" \
  --kernel-type "$KERNEL_TYPE" \
  --num-qp "$NUM_QP" \
  --max-tokens "$MAX_TOKENS"
