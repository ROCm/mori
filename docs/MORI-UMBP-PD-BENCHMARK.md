# MORI UMBP PD Disaggregation Benchmark

Runs a prefill-decode disaggregated serving benchmark across two nodes using
[SGLang](https://github.com/sgl-project/sglang) with mori's UMBP KV-cache transfer backend.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Topology](#topology)
- [Configuration](#configuration)
- [Step 1 — Start Docker on both nodes](#step-1--start-docker-on-both-nodes)
- [Step 2 — Start Grafana and Prometheus](#step-2--start-grafana-and-prometheus)
- [Step 3 — Build mori on both nodes](#step-3--build-mori-on-both-nodes)
- [Step 4 — Create launcher scripts](#step-4--create-launcher-scripts)
- [Step 5 — Kill stale processes and launch prefill](#step-5--kill-stale-processes-and-launch-prefill)
- [Step 6 — Wait for prefill ready](#step-6--wait-for-prefill-ready)
- [Step 7 — Launch decode and benchmark](#step-7--launch-decode-and-benchmark)
- [Step 8 — Monitor decode and benchmark](#step-8--monitor-decode-and-benchmark)
- [Step 9 — Show results](#step-9--show-results)
- [Teardown](#teardown)
- [Environment Variable Reference](#environment-variable-reference)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Two nodes with ROCm-capable GPUs (8 GPUs per node, dp8ep8 topology)
- NFS mount shared between both nodes (for logs, results, and mori/sglang source)
- SSH key access from the local machine to both nodes
- Docker with ROCm image available on both nodes
- mori built with `UMBP=ON` (Step 3 below)
- SGLang checked out at `$NFS_BASE/sglang` with `benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh`

## Topology

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│          PREFILL NODE            │      │          DECODE NODE             │
│                                  │      │                                  │
│  SGLang prefill server :30000    │      │  SGLang decode server  :30001    │
│  UMBP master           :15558    │◄────►│  benchmark client      :8000     │
│  KV events publisher   :5557     │      │  KV events publisher   :5557     │
│                                  │      │                                  │
│  Grafana/Prometheus              │      │                                  │
│    accessible from decode node   │      │  Grafana       :3000             │
│                                  │      │  Prometheus    :9090             │
└──────────────────────────────────┘      └──────────────────────────────────┘
```

Both nodes share a **single UMBP master** running on the prefill node. The prefill
bench script auto-starts it; decode sets `UMBP_MASTER_AUTO_START=false` and connects.

## Configuration

Define these variables once before running any step. All subsequent code blocks
refer to them.

```bash
# === Edit for your environment ===
USER_HOME="/home/youruser"             # home directory (same path on all nodes via NFS)
NFS_BASE="/nfs/users/youruser"         # NFS root containing sglang/ and mori/
SSH_KEY="$USER_HOME/.ssh/id_ed25519"   # SSH private key for node access

NODE_PREFILL="node-hostname-prefill"   # prefill node hostname
NODE_DECODE="node-hostname-decode"     # decode node hostname
IP_PREFILL="10.x.x.x"                 # prefill node IP (reachable from both nodes)
IP_DECODE="10.x.x.y"                  # decode node IP

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

SSH="ssh -o StrictHostKeyChecking=no -i $SSH_KEY"
SCP="scp -o StrictHostKeyChecking=no -i $SSH_KEY"
CONTAINER="umbp-pd-bench"
RESULTS_BASE="$NFS_BASE/sglang/benchmark/hicache/results"
```

## Step 1 — Start Docker on both nodes

```bash
for NODE in $NODE_PREFILL $NODE_DECODE; do
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
echo "Docker started on both nodes"
```

## Step 2 — Start Grafana and Prometheus

Grafana and Prometheus run on the decode node. Dashboards are served from the
mori and SGLang source trees on NFS.

```bash
$SSH $NODE_DECODE "cat > /tmp/pd_bench_prometheus.yml << 'EOF'
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: sglang_prefill
    static_configs:
      - targets: ['${IP_PREFILL}:30000']
        labels:
          role: prefill
  - job_name: sglang_decode
    static_configs:
      - targets: ['${IP_DECODE}:30001']
        labels:
          role: decode
  - job_name: umbp_master
    static_configs:
      - targets: ['${IP_PREFILL}:9091']
EOF"

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

## Step 3 — Build mori on both nodes

Clears the per-node build directory so cmake starts clean, then builds with UMBP enabled.

```bash
$SSH $NODE_PREFILL "docker exec $CONTAINER bash -c '
  rm -rf ${NFS_BASE}/mori/build_\$(hostname) &&
  cd ${NFS_BASE}/mori && UMBP=ON bash build.sh'" &
$SSH $NODE_DECODE  "docker exec $CONTAINER bash -c '
  rm -rf ${NFS_BASE}/mori/build_\$(hostname) &&
  cd ${NFS_BASE}/mori && UMBP=ON bash build.sh'" &
wait
echo "mori builds done"
```

## Step 4 — Create launcher scripts

Both launchers share a single UMBP master on the prefill node (`${IP_PREFILL}:15558`).
`UMBP_IO_ENGINE_PORT` and `UMBP_PEER_SERVICE_PORT` are required whenever
`UMBP_MASTER_ADDRESS` is set. `UMBP_NODE_ADDRESS` is set to each node's actual IP
so all 16 dp ranks (8 prefill + 8 decode) have unique identities in the master.

> **Note on `USE_DUMMY_WEIGHTS`:** set to `true` below to skip loading real model
> weights, which speeds up startup and is useful for benchmarking transfer throughput.
> Set to `false` (or remove the variable) to run with actual weights.

```bash
cat > /tmp/launch_pd_prefill.sh << LAUNCHEOF
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
export UMBP_MASTER_ADDRESS=${IP_PREFILL}:15558
export UMBP_NODE_ADDRESS=${IP_PREFILL}
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
export MORI_LOG_FILE=${USER_HOME}/mori_prefill.log
exec bash ${NFS_BASE}/sglang/benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh --role prefill
LAUNCHEOF

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
export UMBP_MASTER_ADDRESS=${IP_PREFILL}:15558
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
export PREFILL_URL=http://${IP_PREFILL}:30000
export MORI_GLOBAL_LOG_LEVEL=info
export MORI_LOG_FILE=${USER_HOME}/mori_decode.log
exec bash ${NFS_BASE}/sglang/benchmark/hicache/run_pd_disagg_bench_dp8ep8.sh --role decode
LAUNCHEOF

chmod +x /tmp/launch_pd_prefill.sh /tmp/launch_pd_decode.sh
$SCP /tmp/launch_pd_prefill.sh $NODE_PREFILL:$USER_HOME/launch_pd_prefill.sh
$SCP /tmp/launch_pd_decode.sh  $NODE_DECODE:$USER_HOME/launch_pd_decode.sh
```

## Step 5 — Kill stale processes and launch prefill

Clears any leftover SGLang or UMBP master processes from a previous run, then
starts the prefill server. The bench script auto-starts the shared UMBP master.

```bash
$SSH $NODE_PREFILL 'docker exec umbp-pd-bench bash -c "
  pkill -9 -f sglang 2>/dev/null
  pkill -9 -f umbp_master 2>/dev/null
  sleep 2
  ss -tlnp | grep :30000 || echo port_30000_free"' &
$SSH $NODE_DECODE  'docker exec umbp-pd-bench bash -c "
  pkill -9 -f sglang 2>/dev/null
  pkill -9 -f umbp_master 2>/dev/null
  sleep 2
  ss -tlnp | grep :30001 || echo port_30001_free"' &
wait

$SSH $NODE_PREFILL "docker exec -d $CONTAINER bash -c \
  'bash $USER_HOME/launch_pd_prefill.sh > $USER_HOME/pd_bench_prefill.log 2>&1'"
echo "Prefill launched"
```

## Step 6 — Wait for prefill ready

Logs are on NFS, so they can be read directly without SSH or docker exec.
The loop polls every 5 seconds and prints incremental log output every 30 seconds
until it detects `fired up` (server ready) or a fatal error.

```bash
sleep 10
LAST=0
for i in $(seq 1 180); do
  LOG=$(find ${RESULTS_BASE}/pd_disagg_prefill -name server_prefill.log 2>/dev/null | sort -r | head -1)
  if [[ -n "$LOG" ]]; then
    if grep -q "fired up" "$LOG"; then
      echo "=== Prefill READY (iter $i) ===" && tail -5 "$LOG" && break
    fi
    if grep -qE "^Traceback \(most recent|^[[:space:]]*(RuntimeError|AssertionError|ImportError):|CUDA error:|out of memory|Segmentation fault|core dumped|^Killed" "$LOG"; then
      echo "=== ERROR (iter $i) ===" && tail -60 "$LOG" && break
    fi
    TOTAL=$(wc -l < "$LOG")
    if (( i % 6 == 0 && TOTAL > LAST )); then
      echo "--- prefill log update (iter $i, lines $LAST→$TOTAL) ---"
      tail -n +"$((LAST + 1))" "$LOG" | head -15
      LAST=$TOTAL
    fi
  fi
  WLOG=$($SSH $NODE_PREFILL "cat $USER_HOME/pd_bench_prefill.log 2>/dev/null" || true)
  if echo "$WLOG" | grep -qiE "^\[.*\] FATAL|^\[.*\] FAILED"; then
    echo "=== FATAL in wrapper log ===" && echo "$WLOG" | tail -20 && break
  fi
  sleep 5
done
```

## Step 7 — Launch decode and benchmark

Decode connects to the already-running UMBP master on the prefill node.

```bash
$SSH $NODE_DECODE "docker exec -d $CONTAINER bash -c \
  'bash $USER_HOME/launch_pd_decode.sh > $USER_HOME/pd_bench_decode.log 2>&1'"
echo "Decode + benchmark launched"
```

## Step 8 — Monitor decode and benchmark

Polls every 15 seconds. Prints incremental server log every 60 seconds.
Exits when the benchmark reports completion or a fatal error is detected.

```bash
sleep 10
DECODE_READY=false
LAST=0
for i in $(seq 1 720); do
  LOG=$(find ${RESULTS_BASE}/pd_disagg_decode -name server_decode.log 2>/dev/null | sort -r | head -1)
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

```bash
echo "=== Prefill server log (last 20) ==="
find ${RESULTS_BASE}/pd_disagg_prefill -name server_prefill.log | sort -r | head -1 | xargs tail -20

echo "=== Decode server log (last 20) ==="
find ${RESULTS_BASE}/pd_disagg_decode -name server_decode.log | sort -r | head -1 | xargs tail -20

echo "=== Decode wrapper log (last 20) ==="
$SSH $NODE_DECODE "tail -20 $USER_HOME/pd_bench_decode.log"

echo "=== Summary ==="
find ${RESULTS_BASE}/pd_disagg_decode -name summary.txt | sort -r | head -1 | xargs cat 2>/dev/null

echo "=== Metrics ==="
find ${RESULTS_BASE}/pd_disagg_decode -name performance_metrics.jsonl | sort -r | head -1 | xargs tail -5 2>/dev/null
```

## Teardown

```bash
for NODE in $NODE_PREFILL $NODE_DECODE; do
  $SSH $NODE "
    docker exec $CONTAINER bash -c 'pkill -9 -f sglang; pkill -9 -f umbp_master' 2>/dev/null || true
    docker rm -f $CONTAINER" &
done
$SSH $NODE_DECODE "docker rm -f prometheus-pd grafana-pd 2>/dev/null || true" &
wait
```

## Environment Variable Reference

| Variable | Description |
|---|---|
| `UMBP_MASTER_ADDRESS` | `host:port` of the shared UMBP master (prefill node) |
| `UMBP_MASTER_AUTO_START` | Set `false` on decode to connect to prefill's master instead of starting a new one |
| `UMBP_NODE_ADDRESS` | This node's IP — must be unique per node so all dp ranks register distinct identities in the master |
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
| `PREFILL_URL` | URL of the prefill server for the decode benchmark client |

## Troubleshooting

**SSH access denied on prefill node from local machine**

Use the decode node as a jumphost to authorize your key:

```bash
PUBKEY=$(cat $USER_HOME/.ssh/id_ed25519.pub)
$SSH $NODE_DECODE "ssh -o StrictHostKeyChecking=no $NODE_PREFILL \
  \"mkdir -p ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\""
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

Kill stale processes from a previous run (see Step 5) and verify the port is free:

```bash
$SSH $NODE_PREFILL "ss -tlnp | grep ':30000\|:15558\|:16000\|:16001'"
$SSH $NODE_DECODE  "ss -tlnp | grep ':30001\|:16000\|:16001'"
```
