#!/usr/bin/env bash
# Multi-PROCESS control-plane pressure harness for the UMBP master metadata
# backend (Redis or in-memory).
#
# One standalone `umbp_master` is shared by PROCS OS processes x CLIENTS clients,
# all driving BatchRouteGet / BatchRoutePut / Heartbeat through the full
# Router/gRPC path (requires bench_kvevent_master_pressure built with
# --external-master, see that file). This reproduces the real
# "N-processes-per-machine + multi-machine" shape that the store microbench
# (1 process, N threads) and the in-process kvevent bench (1 process, N clients)
# cannot: each process has its own connection pool / gRPC channel / heartbeat.
#
# It is the primary *repeatable* benchmark for judging Redis-backend
# optimizations: scale PROCS/CLIENTS until the single-slot ceiling shows, then
# compare master per-RPC p50/p95/p99 + redis evalsha usec/call before vs after.
#
# Env knobs (all optional):
#   BACKEND     redis | inmemory                     (default redis)
#   REDIS_URI   tcp://host:6379                       (default tcp://127.0.0.1:6379)
#   PROCS       OS processes (== workers/machine)     (default 8)
#   CLIENTS     clients per process                   (default 2)
#   ROUNDS WARMUP BATCH GAP GETMODE KEYSPACE          (defaults 300 20 32 0 both 4096)
#               GETMODE: exists=BatchLookup(BatchExistsBlock, no RDMA);
#                        fetch/both=BatchRouteGet(route_get_batch)+RDMA fetch.
#   MORI_BUILD_DIR  path to mori build dir            (default <repo>/build)
#   REDIS_CLI   redis-cli path                        (default: PATH, else /tmp/umbp_redis_bench/redis-cli)
#   OUT         output dir                            (default ./mp_redis_out)
#   PORT METRICS standalone master ports             (default 15560 9092)
#
# For a multi-machine run: start ONE master (this script on the master host with
# PROCS/CLIENTS as desired), then run extra client-only processes on other hosts
# pointing bench_kvevent_master_pressure --external-master <master_ip>:<PORT>
# --node-address <that_host_ip>.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BUILD_DIR="${MORI_BUILD_DIR:-${REPO_ROOT}/build}"
BIN="${BUILD_DIR}/tests/cpp/umbp/distributed/bench_umbp_kvevent_master_pressure"
MASTER_BIN="${BUILD_DIR}/src/umbp/umbp_master"
PARSE="${SCRIPT_DIR}/parse_master_hist.py"

BACKEND="${BACKEND:-redis}"
REDIS_URI="${REDIS_URI:-tcp://127.0.0.1:6379}"
# Redis deployment: REDIS_CLUSTER=1 => cluster (REDIS_URI is the comma seed list);
# SHARD_URIS set => multi-endpoint (one instance per block shard); else single.
REDIS_CLUSTER="${REDIS_CLUSTER:-0}"
SHARD_URIS="${SHARD_URIS:-}"
PROCS="${PROCS:-8}"; CLIENTS="${CLIENTS:-2}"
ROUNDS="${ROUNDS:-300}"; WARMUP="${WARMUP:-20}"; BATCH="${BATCH:-32}"
GAP="${GAP:-0}"; GETMODE="${GETMODE:-both}"; KEYSPACE="${KEYSPACE:-4096}"
PORT="${PORT:-15560}"; METRICS="${METRICS:-9092}"
OUT="${OUT:-./mp_redis_out}"

REDIS_CLI="${REDIS_CLI:-$(command -v redis-cli || echo /tmp/umbp_redis_bench/redis-cli)}"
RHOST="${REDIS_URI#tcp://}"; RHOST="${RHOST%%:*}"
RPORT="${REDIS_URI##*:}"

if [[ ! -x "$BIN" ]]; then echo "ERROR: bench not built: $BIN (build with USE_REDIS_BACKEND=ON BUILD_TESTS=ON)"; exit 2; fi
mkdir -p "$OUT"; ulimit -n 1048576 2>/dev/null || true
export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-1024}"
HOSTIP="$(hostname -i 2>/dev/null | awk '{print $1}')"; HOSTIP="${HOSTIP:-127.0.0.1}"
LABEL="${BACKEND}_p${PROCS}c${CLIENTS}_gap${GAP}_${GETMODE}"
echo "=== ${LABEL}: total_clients=$((PROCS*CLIENTS)) backend=${BACKEND} redis=${REDIS_URI} ==="

# ---- start standalone master ----
pkill -f "umbp_master 0.0.0.0:${PORT}" 2>/dev/null; sleep 1
ENV="UMBP_METADATA_BACKEND=${BACKEND}"
if [[ "$BACKEND" == "redis" ]]; then
  REDIS_EXTRA="UMBP_REDIS_NAMESPACE=mp_${LABEL}_$(date +%s) UMBP_REDIS_POOL_SIZE=${UMBP_REDIS_POOL_SIZE:-32} UMBP_REDIS_CONNECT_TIMEOUT_MS=1000 UMBP_REDIS_SOCKET_TIMEOUT_MS=1000"
  if [[ "$REDIS_CLUSTER" == "1" ]]; then
    # Cluster: unique namespace isolates runs, so no cluster-wide FLUSHALL needed.
    "$REDIS_CLI" -h "$RHOST" -p "$RPORT" ping >/dev/null 2>&1 || { echo "ERROR: cannot reach cluster seed $RHOST:$RPORT"; exit 2; }
    ENV="${ENV} UMBP_REDIS_CLUSTER=1 UMBP_REDIS_URI=${REDIS_URI} ${REDIS_EXTRA}"
  elif [[ -n "$SHARD_URIS" ]]; then
    ENV="${ENV} UMBP_REDIS_SHARD_URIS=${SHARD_URIS} ${REDIS_EXTRA}"
  else
    "$REDIS_CLI" -h "$RHOST" -p "$RPORT" FLUSHALL   >/dev/null 2>&1 || { echo "ERROR: cannot reach redis $REDIS_URI"; exit 2; }
    "$REDIS_CLI" -h "$RHOST" -p "$RPORT" CONFIG RESETSTAT >/dev/null 2>&1
    ENV="${ENV} UMBP_REDIS_URI=${REDIS_URI} ${REDIS_EXTRA}"
  fi
fi
env $ENV UMBP_ROUTE_PUT_NODE_AFFINITY=local "$MASTER_BIN" "0.0.0.0:${PORT}" "$METRICS" > "${OUT}/master_${LABEL}.log" 2>&1 &
MPID=$!
for i in $(seq 1 30); do curl -sf "http://127.0.0.1:${METRICS}/metrics" >/dev/null 2>&1 && break; sleep 1; done

# ---- launch PROCS client processes (each CLIENTS clients) ----
pids=()
for p in $(seq 0 $((PROCS-1))); do
  "$BIN" --external-master "${HOSTIP}:${PORT}" \
         --node-id-prefix "p${p}-" --node-address "$HOSTIP" \
         --clients "$CLIENTS" --rounds "$ROUNDS" --warmup-rounds "$WARMUP" --batch "$BATCH" \
         --key-space "$KEYSPACE" --read-lag-rounds 1 --pattern rotate --get-mode "$GETMODE" \
         --gap-ms "$GAP" --mode baseline --put-affinity local --metrics-port 0 \
         > "${OUT}/${LABEL}_p${p}.csv" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
sleep 2  # let clients' final ReportMetrics flush land at the master

# ---- results: master per-RPC hist + redis cmdstats + aggregate qps ----
SUM="${OUT}/${LABEL}_summary.txt"
{
  echo "# ${LABEL}  procs=${PROCS} clients/proc=${CLIENTS} batch=${BATCH} gap=${GAP}ms get=${GETMODE}"
  echo "## master per-RPC latency (ms)"
  python3 "$PARSE" "http://127.0.0.1:${METRICS}/metrics" 2>/dev/null
  if [[ "$BACKEND" == "redis" ]]; then
    echo "## redis commandstats"
    "$REDIS_CLI" -h "$RHOST" -p "$RPORT" INFO commandstats 2>/dev/null | grep -E "evalsha|cmdstat_eval:"
    echo "## redis cpu"; "$REDIS_CLI" -h "$RHOST" -p "$RPORT" INFO cpu 2>/dev/null | grep used_cpu
  fi
  echo "## aggregate put/get qps (sum over processes)"
  awk -F, 'FNR>1 && $1!="mode"{p+=$14; g+=$15} END{printf "put_qps_total=%.0f get_qps_total=%.0f\n", p, g}' \
    "${OUT}/${LABEL}"_p*.csv 2>/dev/null
} | tee "$SUM"

kill "$MPID" 2>/dev/null; wait "$MPID" 2>/dev/null || true
echo "DONE -> $SUM"
