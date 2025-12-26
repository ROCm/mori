#!/bin/bash

# Ranging copy size from 1KB (2^10) to 1GB (2^30)
MIN_COPY_SIZE=$((1<<10))
MAX_COPY_SIZE=$((1<<30))

TIMESTAMP=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
OUTPUT_DIR="p2p_xgmi_bandwidth"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

mkdir -p $OUTPUT_DIR
touch $SUMMARY_FILE

echo "==== Running shader_sdma from $MIN_COPY_SIZE to $MAX_COPY_SIZE ===="
rm -rf log.txt
for (( NUM_DST=1; NUM_DST<=7; NUM_DST++ ))
do
    echo "==== The GPU nums of destination is $NUM_DST ===="	
    RESULT_CSV="p2p_xgmi_bandwidth_${NUM_DST}dst.csv"
    #./build/bench/sdma_bw --minCopySize $MIN_COPY_SIZE --maxCopySize $MAX_COPY_SIZE --numCopyCommands 1 --numDestinations $NUM_DST -o $OUTPUT_DIR/$RESULT_CSV  >> log.txt
    ./build/examples/sdma_bw --minCopySize $MIN_COPY_SIZE --maxCopySize $MAX_COPY_SIZE --numCopyCommands 1 --numDestinations $NUM_DST
    #if [ $NUM_DST -eq 1 ]; then
        #cat $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    #else
        #tail -n +2 $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    #fi
done
