#!/usr/bin/env bash
set -euo pipefail

# Master script: run the full intra-node EP tuning matrix
# Sweeps EP2/4/8 × fp4/fp8 × zero-copy/non-zero-copy = 12 groups
#
# Usage:
#   bash tools/run_all_intranode_tuning.sh [--tuning-scope quick] [--gpus 0,1,2,3,4,5,6,7]
#
# Options are forwarded to batch_intranode_tuning.sh. Common overrides:
#   --tuning-scope quick|full    (default: quick)
#   --tokens-list "128,4096"     (default: 64,128,256,512,1024,2048,4096)
#   --gpus "0,1,2,3"            (default: all visible)
#   --timeout 3600               (default: 3600)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=("$@")

LOGFILE="${REPO_ROOT}/logs/all_intranode_tuning_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${REPO_ROOT}/logs"
echo "=== Full IntraNode Tuning Run ===" | tee "$LOGFILE"
echo "Started at: $(date)" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

run_group() {
    local desc="$1"
    shift
    echo "" | tee -a "$LOGFILE"
    echo "################################################################" | tee -a "$LOGFILE"
    echo "# $desc" | tee -a "$LOGFILE"
    echo "# Started at: $(date)" | tee -a "$LOGFILE"
    echo "################################################################" | tee -a "$LOGFILE"
    bash "${REPO_ROOT}/tools/batch_intranode_tuning.sh" "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
    echo "# $desc completed at: $(date)" | tee -a "$LOGFILE"
}

for EP in 2 4 8; do
    run_group "EP${EP}: fp4 + fp8_direct_cast + zero-copy" \
        --world-size "$EP" --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast

    run_group "EP${EP}: fp4 + fp8_direct_cast + non-zero-copy" \
        --world-size "$EP" --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --zero-copy 0

    run_group "EP${EP}: fp8 + none + zero-copy" \
        --world-size "$EP" --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none

    run_group "EP${EP}: fp8 + none + non-zero-copy" \
        --world-size "$EP" --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none --zero-copy 0
done

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "=== ALL 12 GROUPS COMPLETE ===" | tee -a "$LOGFILE"
echo "Finished at: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
