#!/bin/bash
# ReduceScatter + GEMM Overlap Sweep: SDMA vs RCCL across GEMM sizes
# Fixed ReduceScatter at 128MB/PE output, sweeps GEMM M=N=K from 4096 to 16384

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/mori:$PYTHONPATH
export HSA_NO_SCRATCH_RECLAIM=1

MORI_DIR="$HOME/mori"
RESULTS_FILE="$MORI_DIR/rs_overlap_sweep_results.csv"

RS_ELEMS=33554432   # 128 MB/PE output
WORLD_SIZE=8
ITERS=10
WARMUP=10

GEMM_SIZES="4096 8192 16384"

echo "GEMM_MNK,SDMA_RS_ms,SDMA_GEMM_ms,SDMA_Overlap_ms,SDMA_SeqTotal_ms,SDMA_Speedup,RCCL_RS_ms,RCCL_GEMM_ms,RCCL_Overlap_ms,RCCL_SeqTotal_ms,RCCL_Speedup" > "$RESULTS_FILE"

for GEMM_SIZE in $GEMM_SIZES; do
    echo ""
    echo "================================================================"
    echo "  GEMM M=N=K=${GEMM_SIZE}, ReduceScatter ${RS_ELEMS} output elems (128MB/PE)"
    echo "================================================================"

    # --- SDMA ON ---
    echo "[GEMM=${GEMM_SIZE}] Running SDMA + GEMM overlap ..."
    SDMA_OUT=$(cd "$MORI_DIR" && python3 ./tests/python/ccl/test_reducescatter_overlap.py \
        --elems "$RS_ELEMS" --world-size "$WORLD_SIZE" \
        --iterations "$ITERS" --warmup "$WARMUP" --enable-sdma 1 \
        --use-custom-stream --test-gemm-overlap \
        --gemm-m "$GEMM_SIZE" --gemm-n "$GEMM_SIZE" --gemm-k "$GEMM_SIZE" 2>&1)

    SDMA_RS=$(echo "$SDMA_OUT" | grep "ReduceScatter avg:" | tail -1 | awk '{print $3}' | sed 's/s//')
    SDMA_GEMM=$(echo "$SDMA_OUT" | grep "GEMM avg:" | tail -1 | awk '{print $3}' | sed 's/s//')
    SDMA_OVERLAP=$(echo "$SDMA_OUT" | grep "Overlap time (measured):" | tail -1 | awk '{print $4}' | sed 's/s//')
    SDMA_SEQ=$(echo "$SDMA_OUT" | grep "Sequential baseline:" | tail -1 | awk '{print $3}' | sed 's/s//')
    SDMA_SPEEDUP=$(echo "$SDMA_OUT" | grep "Speedup:" | tail -1 | awk '{print $2}')

    echo "  SDMA: RS=${SDMA_RS}s GEMM=${SDMA_GEMM}s Overlap=${SDMA_OVERLAP}s Speedup=${SDMA_SPEEDUP}"

    # --- RCCL ---
    echo "[GEMM=${GEMM_SIZE}] Running RCCL + GEMM overlap ..."
    RCCL_OUT=$(cd "$MORI_DIR" && python3 ./tests/python/ccl/test_rccl_reducescatter.py \
        --elems "$RS_ELEMS" --world-size "$WORLD_SIZE" \
        --iterations "$ITERS" --warmup "$WARMUP" \
        --use-custom-stream --test-gemm-overlap \
        --gemm-m "$GEMM_SIZE" --gemm-n "$GEMM_SIZE" --gemm-k "$GEMM_SIZE" 2>&1)

    RCCL_RS=$(echo "$RCCL_OUT" | grep "ReduceScatter avg:" | tail -1 | awk '{print $3}' | sed 's/s//')
    RCCL_GEMM=$(echo "$RCCL_OUT" | grep "GEMM avg:" | tail -1 | awk '{print $3}' | sed 's/s//')
    RCCL_OVERLAP=$(echo "$RCCL_OUT" | grep "Overlap time (measured):" | tail -1 | awk '{print $4}' | sed 's/s//')
    RCCL_SEQ=$(echo "$RCCL_OUT" | grep "Sequential baseline:" | tail -1 | awk '{print $3}' | sed 's/s//')
    RCCL_SPEEDUP=$(echo "$RCCL_OUT" | grep "Speedup:" | tail -1 | awk '{print $2}')

    echo "  RCCL: RS=${RCCL_RS}s GEMM=${RCCL_GEMM}s Overlap=${RCCL_OVERLAP}s Speedup=${RCCL_SPEEDUP}"

    SDMA_RS_MS=$(echo "$SDMA_RS" | awk '{printf "%.3f", $1 * 1000}')
    SDMA_GEMM_MS=$(echo "$SDMA_GEMM" | awk '{printf "%.3f", $1 * 1000}')
    SDMA_OVERLAP_MS=$(echo "$SDMA_OVERLAP" | awk '{printf "%.3f", $1 * 1000}')
    SDMA_SEQ_MS=$(echo "$SDMA_SEQ" | awk '{printf "%.3f", $1 * 1000}')
    RCCL_RS_MS=$(echo "$RCCL_RS" | awk '{printf "%.3f", $1 * 1000}')
    RCCL_GEMM_MS=$(echo "$RCCL_GEMM" | awk '{printf "%.3f", $1 * 1000}')
    RCCL_OVERLAP_MS=$(echo "$RCCL_OVERLAP" | awk '{printf "%.3f", $1 * 1000}')
    RCCL_SEQ_MS=$(echo "$RCCL_SEQ" | awk '{printf "%.3f", $1 * 1000}')

    echo "${GEMM_SIZE},${SDMA_RS_MS},${SDMA_GEMM_MS},${SDMA_OVERLAP_MS},${SDMA_SEQ_MS},${SDMA_SPEEDUP},${RCCL_RS_MS},${RCCL_GEMM_MS},${RCCL_OVERLAP_MS},${RCCL_SEQ_MS},${RCCL_SPEEDUP}" >> "$RESULTS_FILE"
done

echo ""
echo "================================================================"
echo "  RS OVERLAP SWEEP COMPLETE — Results: $RESULTS_FILE"
echo "================================================================"
echo ""
cat "$RESULTS_FILE" | column -t -s','
