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
NUMA_NODE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)         RANK="$2";        shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
    --ifname)       IFNAME="$2";      shift 2 ;;
    --host)         HOST="$2";        shift 2 ;;
    --numa)         NUMA_NODE="$2";   shift 2 ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

if [[ -z "$HOST" ]]; then
  HOST="$(
    python3 - "$IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys

ifname = sys.argv[1].encode("utf-8")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    packed = struct.pack("256s", ifname[:15])
    addr = fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24]
    print(socket.inet_ntoa(addr))
except OSError:
    pass
PY
  )"
fi

if [[ -z "$HOST" ]]; then
  echo "Failed to determine local host address for interface '$IFNAME'; pass --host explicitly" >&2
  exit 1
fi

export GLOO_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

BENCH_TIMEOUT_SEC="${MORI_IO_BENCH_TIMEOUT_SEC:-600}"

# Optional NUMA pinning. For host-memory multi-NIC runs this is REQUIRED to stay
# rail-safe: the multi-NIC pairing matches each side's rank-j NUMA-local NIC, so
# both nodes must select the SAME NIC subset. Pinning both nodes to the same NUMA
# node makes MatchCpuNics() return an identical, rail-aligned NIC ordering on both
# ends; without it the two nodes can land on different NUMA nodes and pair NICs
# across rails (fails on fabrics where rails are not interconnected).
NUMACTL=()
if [[ -n "$NUMA_NODE" ]]; then
  if command -v numactl >/dev/null 2>&1; then
    NUMACTL=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
    echo "[run_internode_io_benchmark] NUMA pinned to node $NUMA_NODE: ${NUMACTL[*]}"
  else
    echo "[run_internode_io_benchmark] ERROR: --numa $NUMA_NODE requested but numactl not found;" \
         "refusing to run multi-NIC host benchmark unpinned (cross-rail risk)." >&2
    exit 1
  fi
fi

exec "${NUMACTL[@]}" timeout "$BENCH_TIMEOUT_SEC" torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m tests.python.io.benchmark \
  --backend rdma \
  --host "$HOST" \
  "${EXTRA_ARGS[@]}"
