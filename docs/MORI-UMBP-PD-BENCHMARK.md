# MORI UMBP PD Disaggregation Benchmark

Runs a prefill-decode disaggregated serving benchmark using
[SGLang](https://github.com/sgl-project/sglang) with mori's UMBP KV-cache transfer backend.

This guide is **xP1D-generic**: N prefill nodes (N >= 1) plus one decode node.
The 1P1D case is just N=1; the 2P1D case is N=2. Set `PREFILL_NODES` /
`PREFILL_IPS` to drive the topology — every step below loops over them.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Topology](#topology)
- [Configuration](#configuration)
- [Step 1 — Start Docker on all nodes](#step-1--start-docker-on-all-nodes)
- [Step 2 — Start Grafana and Prometheus](#step-2--start-grafana-and-prometheus)
- [Step 3 — Build mori on all nodes](#step-3--build-mori-on-all-nodes)
- [Step 4 — Create launcher scripts](#step-4--create-launcher-scripts)
- [Step 5 — Kill stale processes and launch prefill](#step-5--kill-stale-processes-and-launch-prefill)
- [Step 6 — Wait for ALL prefills ready](#step-6--wait-for-all-prefills-ready)
- [Step 7 — Launch decode and benchmark](#step-7--launch-decode-and-benchmark)
- [Step 8 — Monitor decode and benchmark](#step-8--monitor-decode-and-benchmark)
- [Step 9 — Show results](#step-9--show-results)
- [Teardown](#teardown)
- [Environment Variable Reference](#environment-variable-reference)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- N+1 nodes with ROCm-capable GPUs (8 GPUs per node, dp8ep8 topology):
  N prefill nodes + 1 decode node (N >= 1)
- NFS mount shared between all nodes (for logs, results, and mori/sglang source)
- SSH key access from the local machine to all nodes
- Docker with ROCm image available on all nodes
- mori built with `UMBP=ON` (Step 3 below)
- SGLang checked out at `$NFS_BASE/sglang` with `benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh`
  supporting both `PREFILL_URLS` (space-separated N prefill URLs) and the
  `wait_for_router_workers` gate (polls `/get_loads` after router is ready)

## Topology

```
┌──────────────────────────────────┐
│       PREFILL NODE [0] (PRIMARY) │
│                                  │
│  SGLang prefill server :30000    │
│  UMBP master           :15558  ◄────────────┐
│  KV events publisher   :5557     │          │
└──────────────────────────────────┘          │
                                              │
┌──────────────────────────────────┐          │   ┌──────────────────────────────────┐
│       PREFILL NODE [i] (i>=1)    │          │   │          DECODE NODE             │
│  (only present when N >= 2)      │          │   │                                  │
│                                  │          │   │  SGLang decode server  :30001    │
│  SGLang prefill server :30000    │          ├──►│  sglang_router         :8000     │
│  UMBP_MASTER_AUTO_START=false  ──┼──────────┤   │  benchmark client            ────┼─► localhost:8000
│  KV events publisher   :5557     │          │   │  UMBP_MASTER_AUTO_START=false ───┘
└──────────────────────────────────┘          │   │  KV events publisher   :5557     │
                                              │   │                                  │
                                              │   │  Grafana       :3000             │
                                              │   │  Prometheus    :9090             │
                                              │   └──────────────────────────────────┘
                                              │
                                  (UMBP master discovery / dp-rank registry)
```

All nodes share a **single UMBP master** that runs on `PREFILL_NODES[0]`
(the primary prefill). The primary's bench script auto-starts the master;
all other prefills (i>=1) and the decode node set
`UMBP_MASTER_AUTO_START=false` and connect to it. With N prefills + 1 decode
this means `(N+1) * 8 = 8(N+1)` dp ranks all register with one master.

The decode node hosts `sglang_router` (started by the bench script). The
router fans out to all N prefills via repeated `--prefill` flags driven by
the `PREFILL_URLS` env var.

## Configuration

Define these variables once before running any step. All subsequent code blocks
refer to them. **`PREFILL_NODES` and `PREFILL_IPS` are bash arrays of equal
length** — set one entry for 1P1D, two for 2P1D, N for xP1D.

```bash
# === Edit for your environment ===
USER_HOME="/home/youruser"             # home directory (same path on all nodes via NFS)
NFS_BASE="/nfs/users/youruser"         # NFS root containing sglang/ and mori/
SSH_KEY="$USER_HOME/.ssh/id_ed25519"   # SSH private key for node access

# Prefill nodes (xP1D). Index 0 is the PRIMARY: it hosts the UMBP master and
# must be started first. Indices 1..N-1 are SECONDARY: they connect to the
# primary's master. Both arrays MUST have the same length.
#
#   1P1D example:
#     PREFILL_NODES=("node-prefill-1")
#     PREFILL_IPS=("10.x.x.1")
#
#   2P1D example:
#     PREFILL_NODES=("node-prefill-1" "node-prefill-2")
#     PREFILL_IPS=("10.x.x.1" "10.x.x.2")
PREFILL_NODES=("node-prefill-1")
PREFILL_IPS=("10.x.x.1")

NODE_DECODE="node-hostname-decode"     # decode node hostname
IP_DECODE="10.x.x.y"                   # decode node IP

# Set to the image matching your AMD GPU platform, e.g.:
#   MI300X / MI325X (gfx942): rocm/sgl-dev:vX.Y.Z-rocm7XX-mi30x-YYYYMMDD
#   MI350X        (gfx950):   rocm/sgl-dev:vX.Y.Z-rocm7XX-mi35x-YYYYMMDD
DOCKER_IMAGE="rocm/sgl-dev:v0.5.9-rocm700-mi30x-20260316"

# Network interfaces — find with: ibstat | grep CA; ip link show
NCCL_IB_HCA="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
NET_IFNAME="ens14np0"          # TCP socket interface for GLOO/NCCL/MORI
MORI_RDMA_DEVICES="^mlx5_8"   # use ^ prefix to exclude; leave empty to use all devices

# Benchmark parameters
OUTPUT_LENGTH=100              # decode output tokens per request

# Extra Docker volume mounts (site-specific; clear if not needed)
EXTRA_MOUNTS=""
# Example: EXTRA_MOUNTS="-v /data/models:/models -v /apps:/apps"
# =================================

# === Derived (do not edit) ===
# Primary prefill: hosts the UMBP master.
NODE_PREFILL_PRIMARY="${PREFILL_NODES[0]}"
IP_PREFILL_PRIMARY="${PREFILL_IPS[0]}"
N_PREFILLS=${#PREFILL_NODES[@]}

# Sanity check arrays are same length and non-empty.
if (( N_PREFILLS == 0 )) || (( N_PREFILLS != ${#PREFILL_IPS[@]} )); then
    echo "ERROR: PREFILL_NODES and PREFILL_IPS must be non-empty and same length" >&2
    return 1 2>/dev/null || exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -i $SSH_KEY"
SCP="scp -o StrictHostKeyChecking=no -i $SSH_KEY"
CONTAINER="umbp-pd-bench"
RESULTS_BASE="$NFS_BASE/sglang/benchmark/hicache/results"

# All nodes (used by Steps 1, 3, Teardown).
ALL_NODES=( "${PREFILL_NODES[@]}" "$NODE_DECODE" )
```

## Step 1 — Start Docker on all nodes

Loops over every prefill node plus the decode node.

```bash
for NODE in "${ALL_NODES[@]}"; do
  $SSH $NODE "
    docker rm -f $CONTAINER 2>/dev/null || true
    docker run -d --name $CONTAINER \
      --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
      --device /dev/dri --device /dev/kfd \
      --network host --ipc host --group-add video \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
      -w $NFS_BASE \
      -v /nfs:/nfs \
      -v $USER_HOME:$USER_HOME \
      $EXTRA_MOUNTS \
      --shm-size 32G \
      $DOCKER_IMAGE sleep infinity
    docker ps --filter name=$CONTAINER --format 'table {{.Names}}\t{{.Status}}'" &
done
wait
echo "Docker started on ${#ALL_NODES[@]} node(s)"
```

## Step 2 — Start Grafana and Prometheus

Grafana and Prometheus run on the decode node. Dashboards are served from the
mori and SGLang source trees on NFS.

The Prometheus config is assembled locally so all N prefill targets can be
folded into one scrape job, then piped over SSH to the decode node.

```bash
# Build the multi-prefill targets list as YAML (nested under `- targets:`).
PREFILL_TARGETS=""
for ip in "${PREFILL_IPS[@]}"; do
    PREFILL_TARGETS+="          - '${ip}:30000'"$'\n'
done

cat > /tmp/pd_bench_prometheus.yml <<EOF
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: sglang_prefill
    static_configs:
      - targets:
${PREFILL_TARGETS%$'\n'}
        labels:
          role: prefill
  - job_name: sglang_decode
    static_configs:
      - targets: ['${IP_DECODE}:30001']
        labels:
          role: decode
  - job_name: umbp_master
    static_configs:
      - targets: ['${IP_PREFILL_PRIMARY}:9091']
EOF

# Push to the decode node.
$SCP /tmp/pd_bench_prometheus.yml $NODE_DECODE:/tmp/pd_bench_prometheus.yml

$SSH $NODE_DECODE "
  docker rm -f prometheus-pd 2>/dev/null || true
  docker run -d --name prometheus-pd \
    --network host \
    -v /tmp/pd_bench_prometheus.yml:/etc/prometheus/prometheus.yml:ro \
    prom/prometheus:latest \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus

  docker rm -f grafana-pd 2>/dev/null || true
  docker run -d --name grafana-pd \
    --network host \
    -v ${NFS_BASE}/sglang/examples/monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro \
    -v ${NFS_BASE}/sglang/examples/monitoring/grafana/dashboards/config:/etc/grafana/provisioning/dashboards:ro \
    -v ${NFS_BASE}/sglang/examples/monitoring/grafana/dashboards/json:/var/lib/grafana/dashboards:ro \
    -v ${NFS_BASE}/mori/examples/monitoring/grafana/dashboards:/var/lib/grafana/mori_dashboards:ro \
    -e GF_AUTH_ANONYMOUS_ENABLED=true \
    -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    -e GF_AUTH_BASIC_ENABLED=false \
    -e GF_USERS_ALLOW_SIGN_UP=false \
    -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/sglang-dashboard.json \
    grafana/grafana:latest

  sleep 4
  curl -sf http://localhost:9090/-/ready && echo 'Prometheus: READY' || echo 'Prometheus: NOT READY'
  curl -sf http://localhost:3000/api/health | python3 -c \"import sys,json; d=json.load(sys.stdin); print('Grafana:', 'HEALTHY' if d.get('database')=='ok' else 'NOT READY')\" 2>/dev/null || echo 'Grafana: NOT READY'
"
echo "=== Grafana:    http://${IP_DECODE}:3000 ==="
echo "=== Prometheus: http://${IP_DECODE}:9090 ==="
```

## Step 3 — Build mori on all nodes

Clears the per-node build directory so cmake starts clean, then builds with UMBP enabled.
Builds run in parallel across every prefill node and the decode node.

```bash
for NODE in "${ALL_NODES[@]}"; do
  $SSH $NODE "docker exec $CONTAINER bash -c '
    rm -rf ${NFS_BASE}/mori/build_\$(hostname) &&
    cd ${NFS_BASE}/mori && UMBP=ON bash build.sh'" &
done
wait
echo "mori builds done on ${#ALL_NODES[@]} node(s)"
```

## Step 4 — Create launcher scripts

All launchers point at a single UMBP master on the primary prefill node
(`${IP_PREFILL_PRIMARY}:15558`). Per-node behavior differs only on three
variables:

| Variable | Primary prefill (i=0) | Secondary prefill (i>=1) | Decode |
|---|---|---|---|
| `UMBP_MASTER_AUTO_START` | (default true) | `false` | `false` |
| `UMBP_NODE_ADDRESS` | `${PREFILL_IPS[0]}` | `${PREFILL_IPS[i]}` | `${IP_DECODE}` |
| `--role` | `prefill` | `prefill` | `decode` |

`UMBP_IO_ENGINE_PORT` and `UMBP_PEER_SERVICE_PORT` are required whenever
`UMBP_MASTER_ADDRESS` is set. `UMBP_NODE_ADDRESS` must be unique per node so
all `8 * (N+1)` dp ranks register distinct identities in the master.

> **Note on `USE_DUMMY_WEIGHTS`:** set to `true` below to skip loading real model
> weights, which speeds up startup and is useful for benchmarking transfer throughput.
> Set to `false` (or remove the variable) to run with actual weights.

```bash
# Helper that emits a prefill launcher for a given index.
# Index 0 is primary (auto-starts the UMBP master); indices >=1 connect to it.
emit_prefill_launcher() {
    local idx="$1"
    local node_ip="${PREFILL_IPS[$idx]}"
    local out="/tmp/launch_pd_prefill_${idx}.sh"
    local extra_master_line=""
    if (( idx > 0 )); then
        extra_master_line="export UMBP_MASTER_AUTO_START=false"
    fi
    cat > "$out" << LAUNCHEOF
#!/bin/bash
export PYTHONPATH=${NFS_BASE}/mori/python:/sgl-workspace/aiter
export MC_IB_TC=96
export MORI_ENABLE_SDMA=0
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_MORI_FP4_DISP=false
export SGLANG_MORI_FP8_DISP=true
export SGLANG_MORI_FP8_COMB=true
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048
export NCCL_IB_HCA=${NCCL_IB_HCA}
export GLOO_SOCKET_IFNAME=${NET_IFNAME}
export NCCL_SOCKET_IFNAME=${NET_IFNAME}
export MORI_SOCKET_IFNAME=${NET_IFNAME}
export MORI_RDMA_DEVICES=${MORI_RDMA_DEVICES}
export SGLANG_USE_AITER=1
export KV_CACHE_DTYPE=fp8_e4m3
export UMBP_MASTER_ADDRESS=${IP_PREFILL_PRIMARY}:15558
${extra_master_line}
export UMBP_NODE_ADDRESS=${node_ip}
export UMBP_MASTER_BIN=${NFS_BASE}/mori/build_\$(hostname)/src/umbp/umbp_master
export UMBP_IO_ENGINE_HOST=127.0.0.1
export UMBP_IO_ENGINE_PORT=16000
export UMBP_PEER_SERVICE_PORT=16001
export UMBP_CACHE_REMOTE_FETCHES=false
export ENABLE_KV_EVENTS=true
export KV_EVENTS_PUBLISHER=zmq
export KV_EVENTS_ENDPOINT=tcp://*:5557
export KV_EVENTS_TOPIC=
export USE_DUMMY_WEIGHTS=true
export MEM_FRACTION_STATIC=0.7
export MORI_GLOBAL_LOG_LEVEL=info
export MORI_LOG_FILE=${USER_HOME}/mori_prefill_${idx}.log
exec bash ${NFS_BASE}/sglang/benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh --role prefill
LAUNCHEOF
    chmod +x "$out"
}

# Build space-separated PREFILL_URLS list for the decode launcher.
# The bench script's xP1D contract: pass each prefill URL once, separated by
# a space. Bootstrap port is shared (DISAGG_BOOTSTRAP_PORT, default 8998).
PREFILL_URLS_LIST=""
for ip in "${PREFILL_IPS[@]}"; do
    PREFILL_URLS_LIST+="http://${ip}:30000 "
done
PREFILL_URLS_LIST="${PREFILL_URLS_LIST% }"   # trim trailing space

cat > /tmp/launch_pd_decode.sh << LAUNCHEOF
#!/bin/bash
export PYTHONPATH=${NFS_BASE}/mori/python:/sgl-workspace/aiter
export MC_IB_TC=96
export MORI_ENABLE_SDMA=0
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_MORI_FP4_DISP=false
export SGLANG_MORI_FP8_DISP=true
export SGLANG_MORI_FP8_COMB=true
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048
export NCCL_IB_HCA=${NCCL_IB_HCA}
export GLOO_SOCKET_IFNAME=${NET_IFNAME}
export NCCL_SOCKET_IFNAME=${NET_IFNAME}
export MORI_SOCKET_IFNAME=${NET_IFNAME}
export MORI_RDMA_DEVICES=${MORI_RDMA_DEVICES}
export SGLANG_USE_AITER=1
export KV_CACHE_DTYPE=fp8_e4m3
export UMBP_MASTER_ADDRESS=${IP_PREFILL_PRIMARY}:15558
export UMBP_MASTER_AUTO_START=false
export UMBP_NODE_ADDRESS=${IP_DECODE}
export UMBP_MASTER_BIN=${NFS_BASE}/mori/build_\$(hostname)/src/umbp/umbp_master
export UMBP_IO_ENGINE_HOST=127.0.0.1
export UMBP_IO_ENGINE_PORT=16000
export UMBP_PEER_SERVICE_PORT=16001
export UMBP_CACHE_REMOTE_FETCHES=false
export ENABLE_KV_EVENTS=true
export KV_EVENTS_PUBLISHER=zmq
export KV_EVENTS_ENDPOINT=tcp://*:5557
export KV_EVENTS_TOPIC=
export USE_DUMMY_WEIGHTS=true
export MEM_FRACTION_STATIC=0.7
export OUTPUT_LENGTH=${OUTPUT_LENGTH}
export PREFILL_URLS="${PREFILL_URLS_LIST}"
export MORI_GLOBAL_LOG_LEVEL=info
export MORI_LOG_FILE=${USER_HOME}/mori_decode.log
exec bash ${NFS_BASE}/sglang/benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh --role decode
LAUNCHEOF
chmod +x /tmp/launch_pd_decode.sh

# Emit one prefill launcher per node and ship them.
for i in "${!PREFILL_NODES[@]}"; do
    emit_prefill_launcher "$i"
    $SCP "/tmp/launch_pd_prefill_${i}.sh" \
        "${PREFILL_NODES[$i]}:${USER_HOME}/launch_pd_prefill.sh"
done
$SCP /tmp/launch_pd_decode.sh $NODE_DECODE:$USER_HOME/launch_pd_decode.sh
```

## Step 5 — Kill stale processes and launch prefill

Clears any leftover SGLang or UMBP master processes from a previous run on
every node, then launches prefill nodes. Startup order matters:

1. **Primary first** (`PREFILL_NODES[0]`) — its bench script auto-starts the
   shared UMBP master.
2. **Secondaries** (`PREFILL_NODES[1..N-1]`) — only after the master is up,
   since they connect to it with `UMBP_MASTER_AUTO_START=false`.

For 1P1D, the secondaries loop is a no-op.

```bash
# Stale-process cleanup on all nodes (parallel).
for NODE in "${ALL_NODES[@]}"; do
  $SSH $NODE 'docker exec umbp-pd-bench bash -c "
    pkill -9 -f sglang 2>/dev/null
    pkill -9 -f umbp_master 2>/dev/null
    sleep 2
    ss -tlnp | grep -E \":30000|:30001\" || echo prefill_decode_ports_free"' &
done
wait

# Marker file used by Step 6/8 to filter out previous-run log dirs on NFS.
RUN_START_MARKER="/tmp/pd_bench_run_start"
touch "$RUN_START_MARKER"

# 5a. Launch the PRIMARY prefill (auto-starts the UMBP master).
$SSH "$NODE_PREFILL_PRIMARY" "docker exec -d $CONTAINER bash -c \
  'bash $USER_HOME/launch_pd_prefill.sh > $USER_HOME/pd_bench_prefill.log 2>&1'"
echo "Primary prefill launched on $NODE_PREFILL_PRIMARY"

# 5b. Wait for the UMBP master to be reachable, then launch secondaries.
# Secondaries fail fast if the master is not yet listening on :15558.
if (( N_PREFILLS > 1 )); then
    echo "Waiting for UMBP master at ${IP_PREFILL_PRIMARY}:15558..."
    for i in $(seq 1 60); do
        if $SSH "$NODE_PREFILL_PRIMARY" "ss -tlnp 2>/dev/null | grep -q ':15558'"; then
            echo "  UMBP master is listening (took ${i}s window)."
            break
        fi
        sleep 5
        if (( i == 60 )); then
            echo "ERROR: UMBP master did not start within 5min — abort." >&2
            exit 1
        fi
    done

    for i in $(seq 1 $((N_PREFILLS - 1))); do
        node="${PREFILL_NODES[$i]}"
        $SSH "$node" "docker exec -d $CONTAINER bash -c \
          'bash $USER_HOME/launch_pd_prefill.sh > $USER_HOME/pd_bench_prefill.log 2>&1'"
        echo "Secondary prefill launched on $node (idx $i)"
    done
fi
```

## Step 6 — Wait for ALL prefills ready

Logs are on NFS, so they can be read directly without SSH or docker exec.
For xP1D we wait until every prefill node's `server_prefill.log` reports
`fired up`. Logs from previous runs are excluded via the `RUN_START_MARKER`
touched in Step 5.

The loop polls every 5 seconds and prints incremental output every 30 seconds.

```bash
sleep 10
declare -A LAST_LINES
for i in $(seq 1 180); do
  # Discover all server_prefill.log files newer than RUN_START_MARKER.
  mapfile -t LOGS < <(find "${RESULTS_BASE}/pd_disagg_prefill" \
                        -name server_prefill.log -newer "$RUN_START_MARKER" \
                        2>/dev/null | sort)

  # Count READY logs.
  READY_COUNT=0
  for LOG in "${LOGS[@]}"; do
    if grep -q "fired up" "$LOG" 2>/dev/null; then
      READY_COUNT=$(( READY_COUNT + 1 ))
    fi
  done

  if (( READY_COUNT >= N_PREFILLS )); then
    echo "=== All ${N_PREFILLS} prefill(s) READY (iter $i) ==="
    for LOG in "${LOGS[@]}"; do
      echo "--- $LOG (last 3) ---"
      tail -3 "$LOG"
    done
    break
  fi

  # Surface fatal errors from any prefill log.
  ERR_HIT=false
  for LOG in "${LOGS[@]}"; do
    if grep -qE "^Traceback \(most recent|^[[:space:]]*(RuntimeError|AssertionError|ImportError):|CUDA error:|out of memory|Segmentation fault|core dumped|^Killed" "$LOG" 2>/dev/null; then
      echo "=== ERROR in $LOG (iter $i) ==="
      tail -60 "$LOG"
      ERR_HIT=true
      break
    fi
  done
  $ERR_HIT && break

  # Periodic incremental tail per log.
  if (( i % 6 == 0 )); then
    echo "--- iter $i: ${READY_COUNT}/${N_PREFILLS} prefill(s) ready ---"
    for LOG in "${LOGS[@]}"; do
      TOTAL=$(wc -l < "$LOG" 2>/dev/null || echo 0)
      LAST="${LAST_LINES[$LOG]:-0}"
      if (( TOTAL > LAST )); then
        echo "  >>> $LOG (lines $LAST -> $TOTAL)"
        tail -n +"$((LAST + 1))" "$LOG" | head -10
        LAST_LINES[$LOG]=$TOTAL
      fi
    done
  fi

  # Wrapper logs on every prefill node — surface FATAL/FAILED early.
  WRAP_HIT=false
  for NODE in "${PREFILL_NODES[@]}"; do
    WLOG=$($SSH "$NODE" "cat $USER_HOME/pd_bench_prefill.log 2>/dev/null" || true)
    if echo "$WLOG" | grep -qiE "^\[.*\] FATAL|^\[.*\] FAILED"; then
      echo "=== FATAL in wrapper log on $NODE ==="
      echo "$WLOG" | tail -20
      WRAP_HIT=true
      break
    fi
  done
  $WRAP_HIT && break

  sleep 5
done
```

## Step 7 — Launch decode and benchmark

Decode connects to the already-running UMBP master on the primary prefill
node, and the bench script's `sglang_router` fans out to all N prefills via
`PREFILL_URLS`. The wrapper waits for all `N+1` workers to register on the
router before starting the bench (timeout: `ROUTER_WORKER_READY_TIMEOUT_SECS`,
default 180s).

```bash
$SSH $NODE_DECODE "docker exec -d $CONTAINER bash -c \
  'bash $USER_HOME/launch_pd_decode.sh > $USER_HOME/pd_bench_decode.log 2>&1'"
echo "Decode + benchmark launched (router will fan out to ${N_PREFILLS} prefill(s))"
```

## Step 8 — Monitor decode and benchmark

Polls every 15 seconds. Prints incremental server log every 60 seconds.
Exits when the benchmark reports completion or a fatal error is detected.

```bash
sleep 10
DECODE_READY=false
LAST=0
for i in $(seq 1 720); do
  # Filter out previous-run decode logs via RUN_START_MARKER from Step 5.
  LOG=$(find ${RESULTS_BASE}/pd_disagg_decode -name server_decode.log \
          -newer "$RUN_START_MARKER" 2>/dev/null | sort -r | head -1)
  if [[ -n "$LOG" ]]; then
    if ! $DECODE_READY && grep -q "fired up" "$LOG"; then
      echo "--- Decode READY (iter $i), benchmark starting ---"
      DECODE_READY=true
    fi
    if grep -qE "^Traceback \(most recent|^[[:space:]]*(RuntimeError|AssertionError|ImportError):|CUDA error:|out of memory|Segmentation fault|core dumped|^Killed" "$LOG"; then
      echo "=== ERROR in decode log (iter $i) ===" && tail -60 "$LOG" && break
    fi
    TOTAL=$(wc -l < "$LOG")
    if (( i % 4 == 0 && TOTAL > LAST )); then
      echo "--- decode log update (iter $i, lines $LAST→$TOTAL) ---"
      tail -n +"$((LAST + 1))" "$LOG" | head -15
      LAST=$TOTAL
    fi
  fi
  WLOG=$($SSH $NODE_DECODE "cat $USER_HOME/pd_bench_decode.log 2>/dev/null" || true)
  if echo "$WLOG" | grep -q "Benchmark finished in"; then
    echo "=== Benchmark COMPLETE (iter $i) ===" && echo "$WLOG" | tail -20 && break
  fi
  if echo "$WLOG" | grep -qiE "^\[.*\] FATAL|SERVER_CRASH|^\[.*\] FAILED"; then
    echo "=== FATAL in decode wrapper (iter $i) ===" && echo "$WLOG" | tail -30
    [[ -n "$LOG" ]] && tail -30 "$LOG"
    break
  fi
  sleep 15
done
```

## Step 9 — Show results

For xP1D, all N prefill logs are listed (one per node). The
`RUN_START_MARKER` filter ensures we only see this run's logs.

```bash
echo "=== Prefill server logs (this run, last 20 each) ==="
mapfile -t PLOGS < <(find ${RESULTS_BASE}/pd_disagg_prefill \
                       -name server_prefill.log -newer "$RUN_START_MARKER" \
                       2>/dev/null | sort)
for LOG in "${PLOGS[@]}"; do
  echo "--- $LOG ---"
  tail -20 "$LOG"
done

echo "=== Decode server log (last 20) ==="
find ${RESULTS_BASE}/pd_disagg_decode -name server_decode.log \
    -newer "$RUN_START_MARKER" 2>/dev/null | sort -r | head -1 | xargs tail -20

echo "=== Decode wrapper log (last 20) ==="
$SSH $NODE_DECODE "tail -20 $USER_HOME/pd_bench_decode.log"

echo "=== Summary ==="
find ${RESULTS_BASE}/pd_disagg_decode -name summary.txt \
    -newer "$RUN_START_MARKER" 2>/dev/null | sort -r | head -1 | xargs cat 2>/dev/null

echo "=== Metrics ==="
find ${RESULTS_BASE}/pd_disagg_decode -name performance_metrics.jsonl \
    -newer "$RUN_START_MARKER" 2>/dev/null | sort -r | head -1 | xargs tail -5 2>/dev/null
```

## Teardown

```bash
for NODE in "${ALL_NODES[@]}"; do
  $SSH $NODE "
    docker exec $CONTAINER bash -c 'pkill -9 -f sglang; pkill -9 -f umbp_master' 2>/dev/null || true
    docker rm -f $CONTAINER" &
done
$SSH $NODE_DECODE "docker rm -f prometheus-pd grafana-pd 2>/dev/null || true" &
wait
```

## Environment Variable Reference

### Topology (this guide's wrapper variables)

| Variable | Description |
|---|---|
| `PREFILL_NODES` | Bash array of prefill hostnames; index 0 is the **primary** (hosts the UMBP master). Length determines N in xP1D. |
| `PREFILL_IPS` | Bash array of prefill IPs; same length and order as `PREFILL_NODES`. |
| `NODE_PREFILL_PRIMARY` | Derived: `${PREFILL_NODES[0]}`. |
| `IP_PREFILL_PRIMARY` | Derived: `${PREFILL_IPS[0]}`; UMBP master listens on `${IP_PREFILL_PRIMARY}:15558`. |
| `N_PREFILLS` | Derived: `${#PREFILL_NODES[@]}`. |
| `NODE_DECODE` / `IP_DECODE` | The single decode node. |
| `ALL_NODES` | Derived: `("${PREFILL_NODES[@]}" "$NODE_DECODE")`. |

### UMBP / mori (consumed by `run_pd_disagg_bench_dp8ep8.sh`)

| Variable | Description |
|---|---|
| `UMBP_MASTER_ADDRESS` | `host:port` of the shared UMBP master (always `${IP_PREFILL_PRIMARY}:15558`) |
| `UMBP_MASTER_AUTO_START` | `true` on the primary prefill (default); `false` on every secondary prefill and on decode |
| `UMBP_NODE_ADDRESS` | This node's IP — must be unique per node so all `8 * (N+1)` dp ranks register distinct identities in the master |
| `UMBP_MASTER_BIN` | Path to the `umbp_master` binary (per-node build directory) |
| `UMBP_IO_ENGINE_HOST` | Host for the local IO engine listener (always `127.0.0.1`) |
| `UMBP_IO_ENGINE_PORT` | Port for the local IO engine; required when `UMBP_MASTER_ADDRESS` is set |
| `UMBP_PEER_SERVICE_PORT` | Port for peer-to-peer service; required when `UMBP_MASTER_ADDRESS` is set |
| `UMBP_CACHE_REMOTE_FETCHES` | Set `false` to disable remote fetch caching during benchmarking |
| `SGLANG_MORI_FP8_DISP` | Enable FP8 dispatch (reduces transfer size) |
| `SGLANG_MORI_FP8_COMB` | Enable FP8 combine |
| `MORI_RDMA_DEVICES` | RDMA devices to use; prefix `^` to exclude (e.g., `^mlx5_8`) |
| `MORI_SOCKET_IFNAME` | Network interface for mori's TCP socket |
| `KV_CACHE_DTYPE` | KV cache dtype; `fp8_e4m3` recommended for transfer efficiency |
| `KV_EVENTS_PUBLISHER` | KV event publisher backend (`zmq`) |
| `KV_EVENTS_ENDPOINT` | ZMQ endpoint for KV events (e.g., `tcp://*:5557`) |
| `ENABLE_KV_EVENTS` | Enable KV event publishing |
| `USE_DUMMY_WEIGHTS` | Skip loading real model weights (for throughput benchmarking) |
| `MEM_FRACTION_STATIC` | Fraction of GPU memory reserved for static KV cache |
| `OUTPUT_LENGTH` | Number of decode output tokens per request (decode node only) |
| `PREFILL_URLS` | Space-separated prefill URLs for the decode node's `sglang_router` (e.g. `"http://10.0.0.1:30000 http://10.0.0.2:30000"`); the bench script fans out one `--prefill` per entry |
| `PREFILL_URL` | [DEPRECATED] Single-prefill shortcut (1P1D back-compat); folded into `PREFILL_URLS` if set |
| `ROUTER_WORKER_READY_TIMEOUT_SECS` | Max seconds the wrapper waits for all `N+1` workers to report `load >= 0` on `/get_loads` after the router is ready. Default `180`. |

## Troubleshooting

**SSH access denied on prefill node from local machine**

Use the decode node as a jumphost to authorize your key on every prefill node.

```bash
PUBKEY=$(cat $USER_HOME/.ssh/id_ed25519.pub)
for NODE in "${PREFILL_NODES[@]}"; do
  $SSH $NODE_DECODE "ssh -o StrictHostKeyChecking=no $NODE \
    \"mkdir -p ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\""
done
```

**Server fails to start / ImportError**

Check `PYTHONPATH` includes both the mori Python bindings and aiter:

```bash
export PYTHONPATH=${NFS_BASE}/mori/python:/sgl-workspace/aiter
```

The NFS copy of mori (`${NFS_BASE}/mori/python`) is authoritative. The
docker-internal `/sgl-workspace/mori` may be an older version and should not
be used.

**Errors appear in the wrong log**

Always check the actual SGLang server log, not the wrapper log:

```bash
# Prefill
find ${RESULTS_BASE}/pd_disagg_prefill -name server_prefill.log | sort -r | head -1 | xargs tail -50

# Decode
find ${RESULTS_BASE}/pd_disagg_decode -name server_decode.log | sort -r | head -1 | xargs tail -50
```

**Port already in use**

Kill stale processes from a previous run (see Step 5) and verify the port is free.
For xP1D, only the primary prefill is expected to be listening on `:15558` (UMBP master);
secondary prefills should not be (they connect to the primary).

```bash
for NODE in "${PREFILL_NODES[@]}"; do
  echo "--- $NODE ---"
  $SSH $NODE "ss -tlnp | grep -E ':30000|:15558|:16000|:16001' || echo all_free"
done
echo "--- $NODE_DECODE ---"
$SSH $NODE_DECODE "ss -tlnp | grep -E ':30001|:16000|:16001|:8000' || echo all_free"
```

**Bench progress freezes partway**

If `bench_multiturn` sits at e.g. `4/8` for minutes with no new requests,
check the decode wrapper log for `WARNING: router workers not all healthy`.
That means `ROUTER_WORKER_READY_TIMEOUT_SECS` (default 180s) elapsed before
all `N+1` workers registered, and the bench then raced a `503 "No decode
workers available"`. Increase the timeout and re-run:

```bash
$SSH $NODE_DECODE "docker exec $CONTAINER bash -c \
  'ROUTER_WORKER_READY_TIMEOUT_SECS=300 bash $USER_HOME/launch_pd_decode.sh > $USER_HOME/pd_bench_decode.log 2>&1'"
```
