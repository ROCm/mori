#!/usr/bin/env bash
set -uo pipefail

# Master script: run the full EP16 internode tuning matrix (quick mode)
# Runs 6 kernel/dtype combos × 7 token sizes = 42 total tuning jobs
#
# Usage:
#   bash tools/run_all_internode_tuning.sh \
#       --master-addr <HOST0> --peer-host <USER>@<HOST1> --ifname <IFNAME> \
#       [--docker <CONTAINER>] [--ssh-key <KEY>] [--num-qp 2] [--tuning-scope quick]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BATCH_SCRIPT="$REPO_ROOT/tools/batch_internode_tuning.sh"
RESULT_LOG="$REPO_ROOT/logs/internode_tuning_results.md"

# ---- Defaults (override via CLI) ----
MASTER_ADDR=""
PEER_HOST=""
IFNAME=""
DOCKER=""
SSH_KEY=""
NUM_QP=2
TUNING_SCOPE="quick"
HIDDEN_DIMS="7168"
TOKENS_LIST="64,128,256,512,1024,2048,4096"
TIMEOUT_LARGE=7200

# ---- Parse args (pass-through to batch_internode_tuning.sh) ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master-addr)   MASTER_ADDR="$2";   shift 2 ;;
        --peer-host)     PEER_HOST="$2";     shift 2 ;;
        --ifname)        IFNAME="$2";        shift 2 ;;
        --docker)        DOCKER="$2";        shift 2 ;;
        --ssh-key)       SSH_KEY="$2";       shift 2 ;;
        --num-qp)        NUM_QP="$2";        shift 2 ;;
        --tuning-scope)  TUNING_SCOPE="$2";  shift 2 ;;
        --hidden-dims)   HIDDEN_DIMS="$2";   shift 2 ;;
        --tokens-list)   TOKENS_LIST="$2";   shift 2 ;;
        --timeout)       TIMEOUT_LARGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

for var in MASTER_ADDR PEER_HOST IFNAME; do
    [[ -z "${!var}" ]] && { echo "Error: --${var,,} is required"; exit 1; }
done

COMMON_ARGS=(
    --master-addr "$MASTER_ADDR"
    --peer-host "$PEER_HOST"
    --ifname "$IFNAME"
    --num-qp "$NUM_QP"
    --tuning-scope "$TUNING_SCOPE"
    --hidden-dims "$HIDDEN_DIMS"
)
[[ -n "$DOCKER" ]]  && COMMON_ARGS+=(--docker "$DOCKER")
[[ -n "$SSH_KEY" ]] && COMMON_ARGS+=(--ssh-key "$SSH_KEY")

# Auto-detect FP8 dtype: OCP (fp8_e4m3) vs FNUZ (fp8_e4m3_fnuz)
FP8_DTYPE=$(python3 -c "
import torch
if hasattr(torch, 'float8_e4m3fn'):
    print('fp8_e4m3')
elif hasattr(torch, 'float8_e4m3fnuz'):
    print('fp8_e4m3_fnuz')
else:
    print('bf16')
" 2>/dev/null || echo "fp8_e4m3")
echo "Auto-detected FP8 dtype: $FP8_DTYPE"

declare -A TUNING_MATRIX
TUNING_MATRIX["v1_fp4"]="--kernel-type v1 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast"
TUNING_MATRIX["v1_fp8"]="--kernel-type v1 --dtype $FP8_DTYPE --combine-dtype bf16 --quant-type none"
TUNING_MATRIX["v1_ll_fp4"]="--kernel-type v1_ll --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast"
TUNING_MATRIX["v1_ll_fp8"]="--kernel-type v1_ll --dtype $FP8_DTYPE --combine-dtype bf16 --quant-type none"
TUNING_MATRIX["async_ll_fp4"]="--kernel-type async_ll --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast"
TUNING_MATRIX["async_ll_fp8"]="--kernel-type async_ll --dtype $FP8_DTYPE --combine-dtype bf16 --quant-type none"

RUN_ORDER=(v1_ll_fp4 v1_ll_fp8 async_ll_fp4 async_ll_fp8 v1_fp4 v1_fp8)

mkdir -p "$REPO_ROOT/logs"

cat > "$RESULT_LOG" <<HEADER
# Internode EP16 Tuning Results

## Run Info
- Mode: $TUNING_SCOPE
- Tokens: $TOKENS_LIST
- Hidden dim: $HIDDEN_DIMS
- Master: $MASTER_ADDR → $PEER_HOST
- Date: $(date '+%Y-%m-%d %H:%M')

## Hang / Timeout / Failure Log

| Combo | Tokens | Status | Duration | Notes |
|-------|--------|--------|----------|-------|
HEADER

TOTAL_COMBOS=${#RUN_ORDER[@]}
COMBO_NUM=0
TOTAL_FAILED=0

for KEY in "${RUN_ORDER[@]}"; do
    COMBO_NUM=$((COMBO_NUM + 1))
    EXTRA_ARGS="${TUNING_MATRIX[$KEY]}"

    echo ""
    echo "################################################################"
    echo "# [$COMBO_NUM/$TOTAL_COMBOS] $KEY"
    echo "# Args: $EXTRA_ARGS"
    echo "# $(date)"
    echo "################################################################"
    echo ""

    START_TS=$(date +%s)

    set +e
    bash "$BATCH_SCRIPT" \
        "${COMMON_ARGS[@]}" \
        $EXTRA_ARGS \
        --tokens-list "$TOKENS_LIST" \
        --timeout "$TIMEOUT_LARGE"
    EXIT_CODE=$?
    set -e

    END_TS=$(date +%s)
    DURATION=$(( (END_TS - START_TS) / 60 ))

    if [[ $EXIT_CODE -ne 0 ]]; then
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        echo "| $KEY | ALL | FAILED (exit=$EXIT_CODE) | ${DURATION}min | batch script error |" >> "$RESULT_LOG"
        echo ""
        echo "!!! $KEY FAILED with exit code $EXIT_CODE after ${DURATION} min !!!"
        echo ""
    else
        echo "| $KEY | ALL | OK | ${DURATION}min | |" >> "$RESULT_LOG"
        echo ""
        echo "=== $KEY completed in ${DURATION} min ==="
        echo ""
    fi
done

echo "" >> "$RESULT_LOG"
echo "## Summary" >> "$RESULT_LOG"
echo "" >> "$RESULT_LOG"
echo "- Total combos: $TOTAL_COMBOS" >> "$RESULT_LOG"
echo "- Failed: $TOTAL_FAILED" >> "$RESULT_LOG"
echo "- Completed: $(date '+%Y-%m-%d %H:%M')" >> "$RESULT_LOG"

echo ""
echo "################################################################"
echo "# ALL DONE"
echo "# Total: $TOTAL_COMBOS combos"
echo "# Failed: $TOTAL_FAILED"
echo "# Results: $RESULT_LOG"
echo "################################################################"
