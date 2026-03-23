#!/usr/bin/env bash
set -e

# Inner script that runs inside the docker container.
# Called by test_umbp_integration.sh — not meant to be run directly on the host.
#
# Usage: test_umbp_inner.sh [branch]

MORI_BRANCH="${1:-main}"

# ===========================================================================
# Helper functions
# ===========================================================================

run_bench_hicache() {
    local MODE="${1:-tp}"
    local SERVER_LOG="server_hicache_$(date +%Y%m%d_%H%M%S).log"
    local SERVER_PORT=30000
    local BENCH_DIR="/apps/ditian12/sglang/benchmark/gsm8k"

    echo "Server logs will be saved to: $SERVER_LOG"

    # --- Start server ---
    export MORI_APP_LOG_LEVEL=INFO
    export MORI_RDMA_SL=3
    export MORI_RDMA_TC=96

    PYTHONPATH=$PYTHONPATH:/apps/ditian12/sglang/python \
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
      --hicache-storage-backend-extra-config '{
        "dram_capacity_bytes": 5368709120,
        "ssd_enabled": true,
        "ssd_storage_dir": "/tmp/umbp_ssd",
        "ssd_capacity_bytes": 5368709120,
        "auto_promote_on_read": true,
        "prefetch_threshold": 0
      }' \
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
        echo "Server stopped. Logs saved to: $SERVER_LOG"
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

echo "=== Step 1/3: Updating sglang ==="
cd /sgl-workspace/sglang/ && git pull

echo "=== Step 2/3: Building mori with UMBP ==="
cd /sgl-workspace/mori && git checkout "$MORI_BRANCH" && git pull --rebase
BUILD_UMBP=ON BUILD_TESTS=ON pip3 install -e /apps/ditian12/mori --no-build-isolation -v

echo "=== Step 3/3: Running hicache benchmark ==="
run_bench_hicache

echo "=== UMBP integration test complete ==="
