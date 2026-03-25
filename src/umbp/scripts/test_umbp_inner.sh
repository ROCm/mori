#!/usr/bin/env bash
set -e

# Inner script that runs inside the docker container.
# Called by test_umbp_integration.sh — not meant to be run directly on the host.
#
# Usage: test_umbp_inner.sh [branch] [storage_mode] [sglang_branch] [parallelism_mode]

MORI_BRANCH="${1:-main}"
STORAGE_MODE="${2:-local}"
SGLANG_BRANCH="${3:-main}"
PARALLELISM_MODE="${4:-dp_ep}"

# ===========================================================================
# Distributed UMBP settings
# ===========================================================================
UMBP_MASTER_LISTEN="${UMBP_MASTER_LISTEN:-0.0.0.0:50051}"
UMBP_MASTER_ADDRESS="${UMBP_MASTER_ADDRESS:-localhost:50051}"
UMBP_NODE_ADDRESS="${UMBP_NODE_ADDRESS:-localhost}"
UMBP_IO_ENGINE_PORTS="${UMBP_IO_ENGINE_PORTS:-50100,50101,50102,50103,50104,50105,50106,50107}"

# ===========================================================================
# Locate UMBP master binary
# ===========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORI_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UMBP_MASTER_BIN=""
HOSTNAME_SHORT="$(hostname)"
for d in "${MORI_DIR}/build_${HOSTNAME_SHORT}/src/umbp" "${MORI_DIR}/build/src/umbp"; do
    if [[ -x "${d}/umbp_master" ]]; then
        UMBP_MASTER_BIN="${d}/umbp_master"
        break
    fi
done

# ===========================================================================
# Helper functions
# ===========================================================================

start_umbp_master() {
    if [ "$STORAGE_MODE" != "distributed" ]; then
        return
    fi
    if [ -z "$UMBP_MASTER_BIN" ]; then
        echo "ERROR: umbp_master binary not found. Build mori with BUILD_UMBP=ON first."
        exit 1
    fi
    MASTER_LOG="umbp_master_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting UMBP Master on $UMBP_MASTER_LISTEN (connect via $UMBP_MASTER_ADDRESS)..."
    MORI_UMBP_LOG_LEVEL="${MORI_UMBP_LOG_LEVEL:-INFO}" "$UMBP_MASTER_BIN" "$UMBP_MASTER_LISTEN" > "$MASTER_LOG" 2>&1 &
    MASTER_PID=$!
    echo "UMBP Master PID=$MASTER_PID (log: $MASTER_LOG)"
    sleep 2
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "ERROR: UMBP Master died. Check $MASTER_LOG"
        cat "$MASTER_LOG"
        exit 1
    fi
    echo "  master=$UMBP_MASTER_ADDRESS  node_addr=$UMBP_NODE_ADDRESS"
}

run_bench_hicache() {
    local MODE="${1:-tp}"
    local STOR_MODE="${2:-local}"
    local SERVER_LOG="server_hicache_$(date +%Y%m%d_%H%M%S).log"
    local SERVER_PORT=30000
    local BENCH_DIR="/sgl-workspace/sglang/benchmark/gsm8k"

    echo "Server logs will be saved to: $SERVER_LOG"
    echo "Parallelism mode: $MODE | Storage mode: $STOR_MODE"

    # --- Start server ---
    export MORI_GLOBAL_LOG_LEVEL=INFO
    export MORI_UMBP_LOG_LEVEL=INFO
    export MORI_IO_LOG_LEVEL=ERROR
    export MORI_RDMA_SL=3
    export MORI_RDMA_TC=96

    SGLANG_MORI_FP8_DISP=false \
    MORI_SHMEM_MODE=ISOLATION \
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384 \
    NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
    GLOO_SOCKET_IFNAME=enp81s0f1 \
    NCCL_SOCKET_IFNAME=enp81s0f1 \
    MORI_SOCKET_IFNAME=enp81s0f1 \
    SGLANG_USE_AITER=1 \
    python -m sglang.launch_server \
      --enable-cache-report \
      --enable-metrics \
      --model-path /models/DSR1 \
      --tp-size 8 \
      $([ "$MODE" = "dp_ep" ] && echo "\
      --dp-size 8 \
      --ep-size 8 \
      --moe-a2a-backend mori \
      --deepep-mode normal \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --moe-dense-tp-size 1 \
      ") \
      --decode-log-interval 1 \
      --trust-remote-code \
      --watchdog-timeout 1000000 \
      --chunked-prefill-size 131072 \
      --attention-backend aiter \
      --kv-cache-dtype fp8_e4m3 \
      --max-total-tokens 1024 \
      --mem-fraction-static 0.5 \
      --enable-hierarchical-cache \
      --hicache-write-policy write_through \
      --hicache-mem-layout page_first \
      --hicache-ratio 5.0 \
      --hicache-storage-backend umbp \
      --hicache-storage-backend-extra-config "$(
        if [ "$STOR_MODE" = "distributed" ]; then
          cat <<EOFJSON
{
  "dram_capacity_bytes": 5368709120,
  "ssd_enabled": true,
  "ssd_storage_dir": "/tmp/umbp_ssd",
  "ssd_capacity_bytes": 5368709120,
  "auto_promote_on_read": true,
  "prefetch_threshold": 0,
  "master_address": "$UMBP_MASTER_ADDRESS",
  "node_address": "$UMBP_NODE_ADDRESS",
  "io_engine_host": "127.0.0.1",
  "io_engine_port": [$(echo "$UMBP_IO_ENGINE_PORTS" | tr ',' ', ')]
}
EOFJSON
        else
          cat <<EOFJSON
{
  "dram_capacity_bytes": 5368709120,
  "ssd_enabled": true,
  "ssd_storage_dir": "/tmp/umbp_ssd",
  "ssd_capacity_bytes": 5368709120,
  "auto_promote_on_read": true,
  "prefetch_threshold": 0
}
EOFJSON
        fi
      )" \
      > "$SERVER_LOG" 2>&1 &

    local SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"

    # --- Wait for server to be ready ---
    echo "Waiting for server to start on port $SERVER_PORT..."
    local MAX_WAIT=600
    local ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            echo "Server is ready after ${ELAPSED}s."
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process died. Check $SERVER_LOG for details."
            cat "$SERVER_LOG"
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become ready within ${MAX_WAIT}s."
        cat "$SERVER_LOG"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

    # --- Run benchmarks (cleanup server on exit) ---
    cleanup() {
        echo "Shutting down server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        if [ -n "$MASTER_PID" ]; then
            echo "Shutting down UMBP Master (PID: $MASTER_PID)..."
            kill "$MASTER_PID" 2>/dev/null || true
            wait "$MASTER_PID" 2>/dev/null || true
        fi
        echo "All processes stopped. Logs saved to: $SERVER_LOG"
    }
    trap cleanup EXIT

    echo "=== Benchmark run 1/2 ==="
    cd "$BENCH_DIR"
    python3 bench_sglang.py --num-questions 200

    echo "=== Benchmark run 2/2 ==="
    python3 bench_sglang.py --num-questions 200

    echo "=== Both benchmark runs complete ==="
}

# ===========================================================================
# Main
# ===========================================================================

echo "=== Step 1/4: Updating sglang (branch: $SGLANG_BRANCH) ==="
cd /sgl-workspace/sglang/ && git fetch hicache "$SGLANG_BRANCH"
if ! git checkout "$SGLANG_BRANCH"; then
    echo "ERROR: sglang branch '$SGLANG_BRANCH' not found."
    exit 1
fi
git pull --rebase

echo "=== Step 2/4: Building mori with UMBP ==="
cd /sgl-workspace/mori && git fetch origin "$MORI_BRANCH"
if ! git checkout "$MORI_BRANCH"; then
    echo "ERROR: mori branch '$MORI_BRANCH' not found."
    exit 1
fi
git pull --rebase
BUILD_UMBP=ON BUILD_TESTS=ON pip3 install -e /sgl-workspace/mori --no-build-isolation -v

echo "=== Step 3/4: Starting UMBP Master (if distributed) ==="
MASTER_PID=""
start_umbp_master

echo "=== Step 4/4: Running hicache benchmark ==="
run_bench_hicache "$PARALLELISM_MODE" "$STORAGE_MODE"

echo "=== UMBP integration test complete ==="
