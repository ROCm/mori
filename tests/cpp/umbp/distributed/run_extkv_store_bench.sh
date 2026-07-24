#!/usr/bin/env bash
# External-KV store-level microbench sweep for the UMBP master metadata backend.
#
# Runs bench_umbp_master_metadata_store_extkv (isolated IMasterMetadataStore
# calls, no gRPC / RDMA) across the external-KV hot-path workloads and every
# backend topology, and prints a side-by-side comparison. This is the cleanest
# apples-to-apples signal for "which backend serves the external-KV hot path
# best" — the analogue of the guide's §3 store microbench, for external KV.
#
# It exists to answer, per interface:
#   - MatchExternalKv (count_as_hit=true)  -- hot read (+ hit-count write)
#   - MatchExternalKv (count_as_hit=false) -- pure read (isolates the hit cost)
#   - RegisterExternalKvIfAlive            -- hot write (BlockStored)
#   - UnregisterExternalKv                 -- write (BlockRemoved)
#   - GetExternalKvHitCounts               -- eviction/admin read
# and to expose the design fact that external-KV/hit state lives on ONE control
# hash tag ({umbp:<ns>}) — a single Redis slot/instance — so SHARDED and CLUSTER
# are NOT expected to scale this path, while DRAGONFLY (multi-threaded single
# instance) might. The sweep makes that visible.
#
# Env knobs (all optional):
#   BACKENDS   space list from: inmemory single sharded cluster dragonfly
#              (default "inmemory single sharded cluster dragonfly")
#   WORKLOADS  space list from: match match_nohit report revoke hitcounts mixed
#              (default "match match_nohit report revoke mixed")
#   THREADS SECONDS WARMUP KEYS BATCH NODES HIT_RATIO
#              (defaults 8 5 1 50000 32 8 1.0)
#   SINGLE_URI   tcp://host:port              (default tcp://127.0.0.1:6379)
#   DRAGONFLY_URI                             (default tcp://127.0.0.1:6380)
#   DRAGONFLY_BLOCK_SHARDS                    (default 8; matches Dragonfly threads)
#   SHARD_URIS   comma list                   (default 6390..6393 on localhost)
#   CLUSTER_URI  seed                         (default tcp://127.0.0.1:7000)
#   MORI_BUILD_DIR  build dir                 (default <repo>/build)
#   REDIS_CLI    redis-cli path               (default PATH, else /tmp/umbp_redis_bench/redis-cli)
#   OUT          output dir                   (default ./extkv_store_out)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${MORI_BUILD_DIR:-${REPO_ROOT}/build}"
BIN="${BUILD_DIR}/tests/cpp/umbp/distributed/bench_umbp_master_metadata_store_extkv"

BACKENDS="${BACKENDS:-inmemory single sharded cluster dragonfly}"
WORKLOADS="${WORKLOADS:-match match_nohit report revoke mixed}"
THREADS="${THREADS:-8}"; SECONDS_="${SECONDS:-5}"; WARMUP="${WARMUP:-1}"
KEYS="${KEYS:-50000}"; BATCH="${BATCH:-32}"; NODES="${NODES:-8}"; HIT_RATIO="${HIT_RATIO:-1.0}"

SINGLE_URI="${SINGLE_URI:-tcp://127.0.0.1:6379}"
DRAGONFLY_URI="${DRAGONFLY_URI:-tcp://127.0.0.1:6380}"
DRAGONFLY_BLOCK_SHARDS="${DRAGONFLY_BLOCK_SHARDS:-8}"
SHARD_URIS="${SHARD_URIS:-tcp://127.0.0.1:6390,tcp://127.0.0.1:6391,tcp://127.0.0.1:6392,tcp://127.0.0.1:6393}"
CLUSTER_URI="${CLUSTER_URI:-tcp://127.0.0.1:7000}"

REDIS_CLI="${REDIS_CLI:-$(command -v redis-cli || echo /tmp/umbp_redis_bench/redis-cli)}"
OUT="${OUT:-./extkv_store_out}"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: bench not built: $BIN"
  echo "  build: USE_REDIS_BACKEND=ON BUILD_UMBP=ON BUILD_TESTS=ON pip3 install -e . --no-build-isolation -v"
  echo "  or:    cmake --build $BUILD_DIR --target bench_umbp_master_metadata_store_extkv -j"
  exit 2
fi
mkdir -p "$OUT"
CSV="${OUT}/extkv_store_results.csv"
echo "topology,backend,workload,threads,nodes,batch,keys,hit_ratio,wall_s,ops,ops_per_s,keys_per_s,lat_us_p50,lat_us_p95,lat_us_p99,lat_us_max" > "$CSV"

host_port() { local u="${1#tcp://}"; echo "${u%%:*} ${u##*:}"; }
flush() {  # flush every port that this backend uses, so hit-counts start clean
  local ports="$*"
  for hp in $ports; do
    read -r h p <<<"$(host_port "$hp")"
    "$REDIS_CLI" -h "$h" -p "$p" FLUSHALL >/dev/null 2>&1
    "$REDIS_CLI" -h "$h" -p "$p" CONFIG RESETSTAT >/dev/null 2>&1
  done
}

# Emit the env prefix + a FLUSH target list for a given backend name.
setup_backend() {  # $1 backend -> prints "ENV|||FLUSH_PORTS" or "SKIP"
  local be="$1" ns="ek_$(date +%s)_$$_${RANDOM}"
  case "$be" in
    inmemory) echo "UMBP_METADATA_BACKEND=inmemory|||" ;;
    single)
      read -r h p <<<"$(host_port "$SINGLE_URI")"
      "$REDIS_CLI" -h "$h" -p "$p" ping >/dev/null 2>&1 || { echo SKIP; return; }
      echo "UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=${SINGLE_URI} UMBP_REDIS_NAMESPACE=${ns}|||${SINGLE_URI}" ;;
    dragonfly)
      read -r h p <<<"$(host_port "$DRAGONFLY_URI")"
      "$REDIS_CLI" -h "$h" -p "$p" ping >/dev/null 2>&1 || { echo SKIP; return; }
      echo "UMBP_METADATA_BACKEND=redis UMBP_REDIS_URI=${DRAGONFLY_URI} UMBP_REDIS_BLOCK_SHARDS=${DRAGONFLY_BLOCK_SHARDS} UMBP_REDIS_NAMESPACE=${ns}|||${DRAGONFLY_URI}" ;;
    sharded)
      local first="${SHARD_URIS%%,*}"; read -r h p <<<"$(host_port "$first")"
      "$REDIS_CLI" -h "$h" -p "$p" ping >/dev/null 2>&1 || { echo SKIP; return; }
      echo "UMBP_METADATA_BACKEND=redis UMBP_REDIS_SHARD_URIS=${SHARD_URIS} UMBP_REDIS_NAMESPACE=${ns}|||${SHARD_URIS//,/ }" ;;
    cluster)
      read -r h p <<<"$(host_port "$CLUSTER_URI")"
      "$REDIS_CLI" -h "$h" -p "$p" ping >/dev/null 2>&1 || { echo SKIP; return; }
      # Cluster: unique namespace isolates runs; no cluster-wide FLUSHALL.
      echo "UMBP_METADATA_BACKEND=redis UMBP_REDIS_CLUSTER=1 UMBP_REDIS_URI=${CLUSTER_URI} UMBP_REDIS_NAMESPACE=${ns}|||" ;;
    *) echo SKIP ;;
  esac
}

echo "=== extkv store sweep: threads=${THREADS} nodes=${NODES} batch=${BATCH} keys=${KEYS} hit_ratio=${HIT_RATIO} secs=${SECONDS_} ==="
for be in $BACKENDS; do
  spec="$(setup_backend "$be")"
  if [[ "$spec" == "SKIP" ]]; then echo "-- $be: unreachable, skipped"; continue; fi
  ENV="${spec%%|||*}"; FLUSH_PORTS="${spec##*|||}"
  for wl in $WORKLOADS; do
    [[ -n "$FLUSH_PORTS" ]] && flush $FLUSH_PORTS
    line="$(env $ENV "$BIN" --workload "$wl" --threads "$THREADS" --nodes "$NODES" \
              --seconds "$SECONDS_" --warmup-seconds "$WARMUP" --keys "$KEYS" \
              --batch "$BATCH" --hit-ratio "$HIT_RATIO" 2>/dev/null | tail -n +2)"
    echo "${be},${line}" >> "$CSV"
    echo "  ${be}/${wl}: ${line}"
  done
done

echo
echo "=== comparison (ops/s  p50us  p95us  p99us) ==="
awk -F, '
  { be=$1; wl=$3; ops=$11; p50=$13; p95=$14; p99=$15;
    val[be","wl]=sprintf("%8.0f %8.1f %8.1f %8.1f", ops, p50, p95, p99);
    bes[be]=1; wls[wl]=1 }
  END{
    printf "%-14s %-12s %8s %8s %8s %8s\n","backend","workload","ops/s","p50us","p95us","p99us";
    n=split("inmemory single sharded cluster dragonfly", order, " ");
    for(i=1;i<=n;i++){ be=order[i]; if(!(be in bes)) continue;
      for(wl in wls){ if((be","wl) in val) printf "%-14s %-12s %s\n", be, wl, val[be","wl]; } }
  }' "$CSV" | sort -k2,2 -k1,1
echo
echo "results CSV -> $CSV"
