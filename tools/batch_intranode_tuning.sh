#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Batch Intranode EP Dispatch+Combine Tuning Script
#
# Sweeps over multiple (hidden_dim, max_tokens) combinations, running tuning
# for each and accumulating results into a single JSON config file.
#
# Usage:
#   bash tools/batch_intranode_tuning.sh                           # defaults
#   bash tools/batch_intranode_tuning.sh --world-size 8 --dtype bf16
#   bash tools/batch_intranode_tuning.sh --tokens-list "128,512,4096"
#   bash tools/batch_intranode_tuning.sh --hidden-dims "2048,7168"
#   bash tools/batch_intranode_tuning.sh --dtype fp4 --combine-dtype bf16 \
#        --quant-type fp8_direct_cast
#   bash tools/batch_intranode_tuning.sh --config-output /path/to/out.json
#
# Results are saved to a JSON config file (auto-named or --config-output).
# A timestamped log is auto-generated under logs/.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/tests/python/ops/bench_dispatch_combine.py"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

# ---- Defaults ----
WORLD_SIZE=4
TOKENS_LIST="64,128,256,512,1024,2048,4096"
HIDDEN_DIMS="2048,4096,7168"
DTYPE=bf16
COMBINE_DTYPE=""
ZERO_COPY=1
QUANT_TYPE=none
CONFIG_OUTPUT=auto
GPUS=""
SHMEM_MODE=""

# ---- Parse args ----
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --world-size)       WORLD_SIZE="$2";        shift 2 ;;
        --tokens-list)      TOKENS_LIST="$2";       shift 2 ;;
        --hidden-dims)      HIDDEN_DIMS="$2";       shift 2 ;;
        --dtype)            DTYPE="$2";             shift 2 ;;
        --combine-dtype)    COMBINE_DTYPE="$2";     shift 2 ;;
        --zero-copy)        ZERO_COPY="$2";         shift 2 ;;
        --quant-type)       QUANT_TYPE="$2";        shift 2 ;;
        --config-output)    CONFIG_OUTPUT="$2";     shift 2 ;;
        --gpus)             GPUS="$2";              shift 2 ;;
        --shmem-mode)       SHMEM_MODE="$2";        shift 2 ;;
        *)                  EXTRA_ARGS+=("$1");     shift ;;
    esac
done

# ---- GPU visibility ----
if [[ -n "$GPUS" ]]; then
    export HIP_VISIBLE_DEVICES="$GPUS"
fi

if [[ -n "$SHMEM_MODE" ]]; then
    export MORI_SHMEM_MODE="$SHMEM_MODE"
fi

# ---- Convert comma-separated lists to arrays ----
IFS=',' read -ra TOKEN_ARRAY <<< "$TOKENS_LIST"
IFS=',' read -ra HIDDEN_DIM_ARRAY <<< "$HIDDEN_DIMS"

# ---- Build log filename ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COMB_TAG="${COMBINE_DTYPE:-$DTYPE}"
QUANT_TAG=""
if [[ "$QUANT_TYPE" != "none" ]]; then
    QUANT_TAG="_${QUANT_TYPE}"
fi
LOG_FILE="${LOG_DIR}/batch_ep${WORLD_SIZE}_${DTYPE}_${COMB_TAG}${QUANT_TAG}_zc${ZERO_COPY}_${TIMESTAMP}.log"

TOTAL_COMBOS=$(( ${#HIDDEN_DIM_ARRAY[@]} * ${#TOKEN_ARRAY[@]} ))
COMBO_IDX=0

# ---- Print summary ----
echo "============================================================"
echo "Batch Intranode Tuning"
echo "============================================================"
echo "  world_size:          $WORLD_SIZE"
echo "  tokens_list:         ${TOKEN_ARRAY[*]}"
echo "  hidden_dims:         ${HIDDEN_DIM_ARRAY[*]}"
echo "  dtype:               $DTYPE"
echo "  combine_dtype:       ${COMBINE_DTYPE:-same as dtype}"
echo "  zero_copy:           $ZERO_COPY"
echo "  quant_type:          $QUANT_TYPE"
echo "  config_output:       $CONFIG_OUTPUT"
echo "  gpus:                ${GPUS:-all}"
echo "  total combos:        $TOTAL_COMBOS"
echo "  log:                 $LOG_FILE"
echo "============================================================"
echo ""

# ---- Build common python args ----
PY_COMMON_ARGS=(
    --world-size "$WORLD_SIZE"
    --cmd tuning
    --dtype "$DTYPE"
    --zero-copy "$ZERO_COPY"
    --quant-type "$QUANT_TYPE"
    --save-tuning-config "$CONFIG_OUTPUT"
)

if [[ -n "$COMBINE_DTYPE" ]]; then
    PY_COMMON_ARGS+=(--combine-dtype "$COMBINE_DTYPE")
fi

PY_COMMON_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---- Run sweep ----
for HIDDEN_DIM in "${HIDDEN_DIM_ARRAY[@]}"; do
    for TOKENS in "${TOKEN_ARRAY[@]}"; do
        COMBO_IDX=$((COMBO_IDX + 1))
        echo ""
        echo "############################################################"
        echo "[$(date)] [$COMBO_IDX/$TOTAL_COMBOS] hidden_dim=$HIDDEN_DIM, max_tokens=$TOKENS"
        echo "############################################################"
        echo ""

        python "$BENCH_SCRIPT" \
            "${PY_COMMON_ARGS[@]}" \
            --max-tokens "$TOKENS" \
            --hidden-dim "$HIDDEN_DIM" \
            2>&1 | tee -a "$LOG_FILE"

        echo ""
        echo "[$(date)] Completed [$COMBO_IDX/$TOTAL_COMBOS]"
    done
done

echo ""
echo "============================================================"
echo "Batch tuning complete: $TOTAL_COMBOS combinations"
echo "Config: $CONFIG_OUTPUT"
echo "Log:    $LOG_FILE"
echo "============================================================"
