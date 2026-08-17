#!/bin/bash
# Run one node of the two-node MORI-IO RDMA benchmark.
#
# Usage:
#   run_internode_io_benchmark.sh \
#     --rank <0|1> \
#     --master-addr <ip-or-hostname> \
#     --ifname <nic> \
#     [--engine <cpp|python>] \
#     [--master-port <port>] \
#     [--host <io-engine-host>] \
#     -- [benchmark args...]
#
# Two benchmark engines are selectable with --engine (default: cpp):
#   cpp     the native tests/cpp/io/bench_engine binary (default). No Python
#           interpreter in the measurement loop, so numbers are cleaner and it
#           is the nixlbench-matching path. Requires the binary to be built
#           (see MORI_IO_BENCH_ENGINE_BIN below); brought up over MORI's own
#           socket bootstrap rather than torchrun.
#   python  the tests/python/io/benchmark.py torchrun path (kept for parity).
#
# The benchmark args after `--` are forwarded to the selected engine. Most flag
# names are shared; the C++ engine additionally accepts the Python spellings as
# aliases (see docs/MORI-IO-BENCHMARK.md "Differences from the Python flags").
# The script always runs the RDMA backend in 2-node mode.
# Timeout can be overridden via MORI_IO_BENCH_TIMEOUT_SEC.
# The C++ binary path can be overridden via MORI_IO_BENCH_ENGINE_BIN.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
HOST=""
NUMA_NODE=""
ENGINE="cpp"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)         RANK="$2";        shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
    --ifname)       IFNAME="$2";      shift 2 ;;
    --host)         HOST="$2";        shift 2 ;;
    --numa)         NUMA_NODE="$2";   shift 2 ;;
    --engine)       ENGINE="$2";      shift 2 ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

case "$ENGINE" in
  cpp|python) ;;
  *) echo "Invalid --engine '$ENGINE' (expected cpp or python)"; exit 1 ;;
esac

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

if [[ "$ENGINE" == "python" ]]; then
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
fi

# --- C++ engine (default) ---------------------------------------------------
# The C++ bench brings the two ranks together over MORI's own socket bootstrap
# (rank 0 is the rendezvous root) instead of torchrun, so the rendezvous args
# translate: --master-addr -> --master-ip (rank 0's IP, identical on both ranks)
# and this node's own reachable IP -> --self-ip (advertised in EngineDesc). The
# base --port is the bootstrap; the engine control planes take --port+1+rank.
BENCH_BIN="${MORI_IO_BENCH_ENGINE_BIN:-$REPO_ROOT/build/tests/cpp/bench_engine}"
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "[run_internode_io_benchmark] ERROR: C++ bench_engine not found at $BENCH_BIN" >&2
  echo "  build it with: cmake -B build -DBUILD_IO=ON -DBUILD_TESTS=ON &&" \
       "cmake --build build --target bench_engine -j" >&2
  echo "  or set MORI_IO_BENCH_ENGINE_BIN, or pass --engine python" >&2
  exit 1
fi

# The engine's shared libraries must be discoverable at runtime.
export LD_LIBRARY_PATH="$REPO_ROOT/build/src/io:$REPO_ROOT/build/src/application${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${NUMACTL[@]}" timeout "$BENCH_TIMEOUT_SEC" "$BENCH_BIN" \
  --rank "$RANK" \
  --master-ip "$MASTER_ADDR" \
  --self-ip "$HOST" \
  --port "$MASTER_PORT" \
  --backend rdma \
  "${EXTRA_ARGS[@]}"
