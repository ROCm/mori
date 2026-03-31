#!/bin/bash
# Run one node of the two-node MORI-IO RDMA benchmark via torchrun.
#
# Usage:
#   run_internode_io_benchmark.sh \
#     --rank <0|1> \
#     --master-addr <ip-or-hostname> \
#     --ifname <nic> \
#     [--master-port <port>] \
#     [--host <io-engine-host>] \
#     -- [benchmark.py args...]
#
# The benchmark args after `--` are forwarded to tests/python/io/benchmark.py.
# The script always runs the RDMA backend in 2-node mode with nproc_per_node=1.
# Timeout can be overridden via MORI_IO_BENCH_TIMEOUT_SEC.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
HOST=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)         RANK="$2";        shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
    --ifname)       IFNAME="$2";      shift 2 ;;
    --host)         HOST="$2";        shift 2 ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

if [[ -z "$HOST" ]]; then
  HOST="$(ip -o -4 addr show dev "$IFNAME" | awk '{print $4}' | cut -d/ -f1 | head -n1 || true)"
fi

if [[ -z "$HOST" ]]; then
  HOST="$MASTER_ADDR"
fi

export GLOO_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

BENCH_TIMEOUT_SEC="${MORI_IO_BENCH_TIMEOUT_SEC:-600}"

exec timeout "$BENCH_TIMEOUT_SEC" torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m tests.python.io.benchmark \
  --backend rdma \
  --host "$HOST" \
  "${EXTRA_ARGS[@]}"
