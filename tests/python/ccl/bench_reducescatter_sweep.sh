#!/bin/bash
# ReduceScatter Latency Sweep: SDMA (MORI) vs RCCL
# Sweeps buffer sizes from 10MB to 128MB per PE (output)

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/mori:$PYTHONPATH
export HSA_NO_SCRATCH_RECLAIM=1

MORI_DIR="$HOME/mori"
RESULTS_FILE="$MORI_DIR/reducescatter_sweep_results.csv"

# uint32 = 4 bytes per element
# --elems = output elements per PE
declare -A SIZE_TO_ELEMS
SIZE_TO_ELEMS[10]=2621440
SIZE_TO_ELEMS[20]=5242880
SIZE_TO_ELEMS[40]=10485760
SIZE_TO_ELEMS[80]=20971520
SIZE_TO_ELEMS[128]=33554432

WORLD_SIZE=8
ITERS=10
WARMUP=10

echo "OutputSize_MB,SDMA_AvgTime_ms,SDMA_BW_GBs,RCCL_AvgTime_ms,RCCL_BW_GBs" > "$RESULTS_FILE"

for SIZE_MB in 10 20 40 80 128; do
    ELEMS=${SIZE_TO_ELEMS[$SIZE_MB]}
    INPUT_MB=$((SIZE_MB * WORLD_SIZE))
    echo ""
    echo "================================================================"
    echo "  Output: ${SIZE_MB}MB/PE, Input: ${INPUT_MB}MB/PE (${ELEMS} elements)"
    echo "================================================================"

    # --- SDMA ON (MORI ReduceScatterSdma) ---
    echo "[${SIZE_MB}MB] Running ReduceScatterSdma (SDMA ON) ..."
    SDMA_OUT=$(cd "$MORI_DIR" && python3 ./tests/python/ccl/test_reducescatter_overlap.py \
        --elems "$ELEMS" --world-size "$WORLD_SIZE" \
        --iterations "$ITERS" --warmup "$WARMUP" --enable-sdma 1 2>&1)

    SDMA_AVG=$(echo "$SDMA_OUT" | grep "Avg:" | tail -1 | awk '{print $4}' | sed 's/s,//')
    SDMA_BW=$(echo "$SDMA_OUT" | grep "Bandwidth:" | tail -1 | awk '{print $2}')

    echo "  SDMA Avg time: ${SDMA_AVG}s, BW: ${SDMA_BW} GB/s"

    # --- SDMA OFF (RCCL dist.reduce_scatter) ---
    echo "[${SIZE_MB}MB] Running RCCL ReduceScatter (SDMA OFF) ..."
    RCCL_OUT=$(cd "$MORI_DIR" && python3 ./tests/python/ccl/test_rccl_reducescatter.py \
        --elems "$ELEMS" --world-size "$WORLD_SIZE" \
        --iterations "$ITERS" --warmup "$WARMUP" 2>&1)

    RCCL_AVG=$(echo "$RCCL_OUT" | grep "Avg:" | tail -1 | awk '{print $4}' | sed 's/s,//')
    RCCL_BW=$(echo "$RCCL_OUT" | grep "Bandwidth:" | tail -1 | awk '{print $2}')

    echo "  RCCL Avg time: ${RCCL_AVG}s, BW: ${RCCL_BW} GB/s"

    SDMA_MS=$(echo "$SDMA_AVG" | awk '{printf "%.3f", $1 * 1000}')
    RCCL_MS=$(echo "$RCCL_AVG" | awk '{printf "%.3f", $1 * 1000}')

    echo "${SIZE_MB},${SDMA_MS},${SDMA_BW},${RCCL_MS},${RCCL_BW}" >> "$RESULTS_FILE"
done

echo ""
echo "================================================================"
echo "  REDUCESCATTER SWEEP COMPLETE — Results: $RESULTS_FILE"
echo "================================================================"
echo ""
cat "$RESULTS_FILE" | column -t -s','
