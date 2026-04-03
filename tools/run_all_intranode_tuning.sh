#!/usr/bin/env bash
set -euo pipefail

cd /apps/yutongwu/store/mori
export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH=$(pwd)/python:$(pwd):$PYTHONPATH

LOGFILE="/apps/yutongwu/store/mori/logs/all_intranode_tuning_$(date +%Y%m%d_%H%M%S).log"
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
    bash tools/batch_intranode_tuning.sh "$@" 2>&1 | tee -a "$LOGFILE"
    echo "# $desc completed at: $(date)" | tee -a "$LOGFILE"
}

# EP4 (4 groups)
run_group "EP4: fp4 + fp8_direct_cast + zero-copy" \
    --world-size 4 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --tuning-scope quick

run_group "EP4: fp4 + fp8_direct_cast + non-zero-copy" \
    --world-size 4 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --zero-copy 0 --tuning-scope quick

run_group "EP4: fp8_e4m3 + none + zero-copy" \
    --world-size 4 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --tuning-scope quick

run_group "EP4: fp8_e4m3 + none + non-zero-copy" \
    --world-size 4 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --zero-copy 0 --tuning-scope quick

# EP8 (4 groups)
run_group "EP8: fp4 + fp8_direct_cast + zero-copy" \
    --world-size 8 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --tuning-scope quick

run_group "EP8: fp4 + fp8_direct_cast + non-zero-copy" \
    --world-size 8 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --zero-copy 0 --tuning-scope quick

run_group "EP8: fp8_e4m3 + none + zero-copy" \
    --world-size 8 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --tuning-scope quick

run_group "EP8: fp8_e4m3 + none + non-zero-copy" \
    --world-size 8 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --zero-copy 0 --tuning-scope quick

# EP2 (4 groups)
run_group "EP2: fp4 + fp8_direct_cast + zero-copy" \
    --world-size 2 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --tuning-scope quick

run_group "EP2: fp4 + fp8_direct_cast + non-zero-copy" \
    --world-size 2 --dtype fp4 --combine-dtype bf16 --quant-type fp8_direct_cast --zero-copy 0 --tuning-scope quick

run_group "EP2: fp8_e4m3 + none + zero-copy" \
    --world-size 2 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --tuning-scope quick

run_group "EP2: fp8_e4m3 + none + non-zero-copy" \
    --world-size 2 --dtype fp8_e4m3 --combine-dtype bf16 --quant-type none --zero-copy 0 --tuning-scope quick

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "=== ALL 12 GROUPS COMPLETE ===" | tee -a "$LOGFILE"
echo "Finished at: $(date)" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
