#!/usr/bin/env bash
# ==========================================================================
# EP IntraNode Dispatch/Combine Performance Benchmark
#
# Sweeps over (token_count x dtype) combinations using tuning (quick mode)
# to find the best block/warp config for each, then records raw output and
# a best-performance summary.
#
# This script is for *performance evaluation and reporting*, not for
# generating runtime tuning configs (use batch_intranode_tuning.sh for that).
#
# Usage:
#   bash tools/bench_ep_performance.sh                          # full sweep (EP8)
#   bash tools/bench_ep_performance.sh --tokens "1,64,512"      # specific tokens
#   bash tools/bench_ep_performance.sh --world-size 4            # EP4
#   bash tools/bench_ep_performance.sh --dtypes "bf16"           # bf16 only
#   bash tools/bench_ep_performance.sh --output-dir /tmp/bench   # custom output
#
# Output directory layout:
#   <output-dir>/
#     raw/                          -- full tuning output per combo
#       dispatch_fp8_e4m3_64token.txt
#       dispatch_bf16_64token.txt
#       ...
#     summary.txt                   -- best performance summary
#     bench.log                     -- run log
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/tests/python/ops/bench_dispatch_combine.py"

# ---- Defaults ----
WORLD_SIZE=8
SMALL_TOKENS="1,2,4,8,16,32,64,128,256,512,768"
LARGE_TOKENS="4096,8192,16384,32768,65536,131072,262144,524288"
ALL_TOKENS="$SMALL_TOKENS,$LARGE_TOKENS"
TOKENS="$ALL_TOKENS"
DTYPES="fp8_e4m3,bf16"
OUTPUT_DIR=""
TIMEOUT=7200

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)       TOKENS="$2";        shift 2 ;;
        --dtypes)       DTYPES="$2";        shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --world-size)   WORLD_SIZE="$2";    shift 2 ;;
        --timeout)      TIMEOUT="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Output directory ----
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/bench_results/ep${WORLD_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR/raw"

LOG="$OUTPUT_DIR/bench.log"
SUMMARY="$OUTPUT_DIR/summary.txt"

# ---- Environment ----
cd "$REPO_ROOT"
export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$(pwd)/python:$(pwd):${PYTHONPATH:-}"
export MORI_TUNING_SCOPE=quick

# ---- SHMEM size mapping ----
get_shmem_size() {
    local t=$1
    if   (( t >= 524288 )); then echo "192G"
    elif (( t >= 262144 )); then echo "96G"
    elif (( t >= 65536  )); then echo "48G"
    elif (( t >= 16384  )); then echo "24G"
    else echo "6G"
    fi
}

# ---- GPU info ----
GPU_INFO=$(python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'{p.name} (CU={p.multi_processor_count})')
" 2>/dev/null)

{
    echo "============================================================"
    echo "EP${WORLD_SIZE} IntraNode Benchmark"
    echo "============================================================"
    echo "  GPU:           $GPU_INFO"
    echo "  world_size:    $WORLD_SIZE"
    echo "  tokens:        $TOKENS"
    echo "  dtypes:        $DTYPES"
    echo "  tuning_scope:  quick"
    echo "  output_dir:    $OUTPUT_DIR"
    echo "  started:       $(date)"
    echo "============================================================"
    echo ""
} | tee "$LOG"

# ---- Summary header ----
{
    echo "# EP${WORLD_SIZE} Benchmark Summary"
    echo "# GPU: $GPU_INFO | EP$WORLD_SIZE | $(date)"
    echo "#"
    echo "# columns: tokens dtype phase bw(GB/s) lat(us) block_num warp_per_block"
    echo "# bw = avg(per_rank_bw), lat = avg(per_rank_duration)"
    echo "#"
} > "$SUMMARY"

# ---- Build arrays ----
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS"
IFS=',' read -ra DTYPE_ARRAY <<< "$DTYPES"

TOTAL=$(( ${#TOKEN_ARRAY[@]} * ${#DTYPE_ARRAY[@]} ))
IDX=0

# ---- Main loop ----
for NTOKENS in "${TOKEN_ARRAY[@]}"; do
    for DTYPE in "${DTYPE_ARRAY[@]}"; do
        IDX=$((IDX + 1))
        SHMEM=$(get_shmem_size "$NTOKENS")
        LABEL="${DTYPE}_${NTOKENS}token"
        RAW_FILE="$OUTPUT_DIR/raw/dispatch_${LABEL}.txt"

        echo "[$IDX/$TOTAL] tokens=$NTOKENS dtype=$DTYPE shmem=$SHMEM  $(date)" | tee -a "$LOG"

        set +e
        MORI_SHMEM_HEAP_SIZE="$SHMEM" \
        timeout "$TIMEOUT" python "$BENCH_SCRIPT" \
            --cmd tuning \
            --world-size "$WORLD_SIZE" \
            --max-tokens "$NTOKENS" \
            --dtype "$DTYPE" \
            2>&1 | tee "$RAW_FILE" | grep -E "Performance Summary|Best " | tee -a "$LOG"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [[ $EXIT_CODE -eq 124 ]]; then
            echo "  !!! TIMEOUT !!!" | tee -a "$LOG"
        elif [[ $EXIT_CODE -ne 0 ]]; then
            echo "  !!! FAILED (exit $EXIT_CODE) !!!" | tee -a "$LOG"
        fi

        # Parse Best lines into summary
        python3 -c "
import re, sys
for line in open(sys.argv[1]):
    m = re.search(r'Best (Dispatch|Combine)\s+\(\S+\):\s+([\d.]+)\s+GB/s\s+latency=([\d.einfEINF]+)\s+us\s+at block_num=(\d+),\s*warp_per_block=(\d+)', line)
    if m:
        phase = m.group(1).lower()
        bw, lat = m.group(2), m.group(3)
        try: lat = f'{float(lat):.1f}'
        except: pass
        print(f'$NTOKENS $DTYPE {phase} {bw} {lat} {m.group(4)} {m.group(5)}')
" "$RAW_FILE" >> "$SUMMARY"
    done
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "All $TOTAL benchmarks complete.  $(date)" | tee -a "$LOG"
echo "  Summary: $SUMMARY" | tee -a "$LOG"
echo "  Raw:     $OUTPUT_DIR/raw/" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ---- Print summary ----
echo ""
echo "=== Best Performance Summary ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
