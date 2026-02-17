#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Intranode EP Dispatch+Combine Tuning Script
#
# Usage:
#   bash tools/intranode_tuning.sh                             # defaults
#   bash tools/intranode_tuning.sh --world-size 8              # EP8
#   bash tools/intranode_tuning.sh --max-tokens 4096           # high-BW mode
#   bash tools/intranode_tuning.sh --dtype bf16                # BF16 only
#   bash tools/intranode_tuning.sh --dtype fp4 --combine-dtype bf16 \
#        --combine-quant-type fp8_direct_cast                  # mixed FP4+FP8
#   bash tools/intranode_tuning.sh --zero-copy 0               # P2P write mode
#
# All arguments are forwarded to bench_dispatch_combine.py.
# A timestamped log file is auto-generated under logs/.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/tests/python/ops/bench_dispatch_combine.py"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

# ---- Defaults (override via CLI args) ----
WORLD_SIZE=4
MAX_TOKENS=128
ZERO_COPY=0
CMD=tuning
DTYPE=fp4
COMBINE_DTYPE=bf16
COMBINE_QUANT_TYPE=fp8_direct_cast
GPUS=""
SHMEM_MODE=""

# ---- Parse args to extract values for log naming ----
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --world-size)       WORLD_SIZE="$2";        shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2";        shift 2 ;;
        --zero-copy)        ZERO_COPY="$2";         shift 2 ;;
        --cmd)              CMD="$2";               shift 2 ;;
        --dtype)            DTYPE="$2";             shift 2 ;;
        --combine-dtype)    COMBINE_DTYPE="$2";     shift 2 ;;
        --combine-quant-type) COMBINE_QUANT_TYPE="$2"; shift 2 ;;
        --gpus)             GPUS="$2";              shift 2 ;;
        --shmem-mode)       SHMEM_MODE="$2";        shift 2 ;;
        *)                  EXTRA_ARGS+=("$1");     shift ;;
    esac
done

# ---- GPU visibility ----
if [[ -n "$GPUS" ]]; then
    export HIP_VISIBLE_DEVICES="$GPUS"
fi

# ---- Shared memory mode ----
if [[ -n "$SHMEM_MODE" ]]; then
    export MORI_SHMEM_MODE="$SHMEM_MODE"
fi

# ---- Build log filename ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ZC_TAG="zc${ZERO_COPY}"

if [[ -n "$COMBINE_DTYPE" && "$COMBINE_DTYPE" != "$DTYPE" ]]; then
    DTYPE_TAG="${DTYPE}disp_${COMBINE_DTYPE}comb"
else
    DTYPE_TAG="${DTYPE}"
fi

if [[ -n "$COMBINE_QUANT_TYPE" && "$COMBINE_QUANT_TYPE" != "none" ]]; then
    DTYPE_TAG="${DTYPE_TAG}_${COMBINE_QUANT_TYPE}"
fi

SHMEM_TAG=""
if [[ -n "$SHMEM_MODE" ]]; then
    SHMEM_TAG="_${SHMEM_MODE}"
fi

LOG_FILE="${LOG_DIR}/ep${WORLD_SIZE}_${DTYPE_TAG}_${MAX_TOKENS}tok_${ZC_TAG}${SHMEM_TAG}_${CMD}_${TIMESTAMP}.log"

# ---- Build python command ----
PY_ARGS=(
    --world-size "$WORLD_SIZE"
    --max-tokens "$MAX_TOKENS"
    --zero-copy  "$ZERO_COPY"
    --cmd        "$CMD"
    --dtype      "$DTYPE"
)

if [[ -n "$COMBINE_DTYPE" ]]; then
    PY_ARGS+=(--combine-dtype "$COMBINE_DTYPE")
fi
if [[ -n "$COMBINE_QUANT_TYPE" ]]; then
    PY_ARGS+=(--combine-quant-type "$COMBINE_QUANT_TYPE")
fi

PY_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---- Print summary and run ----
echo "============================================================"
echo "Intranode Tuning"
echo "============================================================"
echo "  world_size:          $WORLD_SIZE"
echo "  max_tokens:          $MAX_TOKENS"
echo "  zero_copy:           $ZERO_COPY"
echo "  cmd:                 $CMD"
echo "  dtype:               $DTYPE"
echo "  combine_dtype:       ${COMBINE_DTYPE:-same as dtype}"
echo "  combine_quant_type:  ${COMBINE_QUANT_TYPE:-none}"
echo "  gpus:                ${GPUS:-all}"
echo "  shmem_mode:          ${SHMEM_MODE:-default}"
echo "  log:                 $LOG_FILE"
echo "  extra args:          ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo "============================================================"
echo ""

python "$BENCH_SCRIPT" "${PY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
