#!/bin/bash

# Ranging copy size from 1KB (2^10) to 1GB (2^30)
MIN_COPY_SIZE=$((1<<10))
MAX_COPY_SIZE=$((1<<20))
NUM_COPY_COMMMANDS=100

MIN_QUEUES_PER_DST=1
MAX_QUEUES_PER_DST=8

NUM_DST=7

MIN_WG_PER_QUEUE=2 
MAX_WG_PER_QUEUE=4

TIMESTAMP=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
OUTPUT_DIR="p2p_xgmi_multiproducer_bandwidth_$TIMESTAMP"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

mkdir $OUTPUT_DIR
touch $SUMMARY_FILE

echo "==== Running shader_sdma from $MIN_COPY_SIZE to $MAX_COPY_SIZE ===="
rm log.txt
for(( QUEUES_PER_DST=MIN_QUEUES_PER_DST; QUEUES_PER_DST<=MAX_QUEUES_PER_DST; QUEUES_PER_DST*=2))
do
    for(( WGS_PER_QUEUE=MIN_WG_PER_QUEUE; WGS_PER_QUEUE<=MAX_WG_PER_QUEUE; WGS_PER_QUEUE*=2))
    do
        for (( NUM_WAVES=1; NUM_WAVES<=1; NUM_WAVES*=2 ))
        do
            RESULT_CSV="p2p_xgmi_banwdith_${NUM_DST}dst_${QUEUES_PER_DST}queuesPerDst_${WGS_PER_QUEUE}wgsPerQ_${NUM_WAVES}waves_${NUM_COPY_COMMMANDS}copies.csv"
            ./build/bench/sdma_bw --minCopySize $MIN_COPY_SIZE --maxCopySize $MAX_COPY_SIZE --numCopyCommands $NUM_COPY_COMMMANDS --numOfQueuesPerDestination $QUEUES_PER_DST --numDestinations $NUM_DST --wgsPerQueue $WGS_PER_QUEUE --warpsPerWG $NUM_WAVES -o $OUTPUT_DIR/$RESULT_CSV  >> log.txt
            if [ $QUEUES_PER_DST -eq $MIN_QUEUES_PER_DST ] && [ $WGS_PER_QUEUE -eq $MIN_WG_PER_QUEUE ] && [ $NUM_WAVES -eq 1 ]; then
                cat $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
            else
                tail -n +2 $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
            fi
        done
    done
done
