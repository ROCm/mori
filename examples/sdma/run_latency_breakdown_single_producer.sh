#!/bin/bash

# Ranging copy size from 1KB (2^10) to 1GB (2^30)
MIN_COPY_SIZE=$((1<<8))
MAX_COPY_SIZE=$((1<<20))

TIMESTAMP=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
OUTPUT_DIR="p2p_xgmi_latency$TIMESTAMP"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

mkdir $OUTPUT_DIR
touch $SUMMARY_FILE

echo "==== Running shader_latency from $MIN_COPY_SIZE to $MAX_COPY_SIZE ===="
rm log.txt
for (( NUM_COPY_CMDS=0; NUM_COPY_CMDS<=1; NUM_COPY_CMDS++ ))
do
    RESULT_CSV="p2p_xgmi_latency_${NUM_COPY_CMDS}copies.csv"
    ./build/bench/sdma_latency --minCopySize $MIN_COPY_SIZE --maxCopySize $MAX_COPY_SIZE --numCopyCommands $NUM_COPY_CMDS --fineGrainedLatency -o $OUTPUT_DIR/$RESULT_CSV  >> log.txt
    if [ $NUM_COPY_CMDS -eq 0 ]; then
        cat $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    else
        tail -n +2 $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    fi
done
