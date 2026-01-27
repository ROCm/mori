/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */
 
#!/bin/bash

NUM_COPY_CMDS=10000

TIMESTAMP=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
OUTPUT_DIR="p2p_xgmi_packet_rate"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

mkdir -p $OUTPUT_DIR
touch $SUMMARY_FILE

echo "==== Running shader_sdma_rate ===="
rm -rf log.txt
for (( NUM_QUEUES=1; NUM_QUEUES<=8; NUM_QUEUES++ ))
do
    RESULT_CSV="p2p_xgmi_latency_${NUM_QUEUES}queues.csv"
    #./build/bench/sdma_packet_rate --minCopySize 64 --maxCopySize 64 --numCopyCommands $NUM_COPY_CMDS -w 1 -n 10 --numOfQueues $NUM_QUEUES  -o $OUTPUT_DIR/$RESULT_CSV  >> log.txt
    ./build/examples/sdma_rate --minCopySize 64 --maxCopySize 64 --numCopyCommands $NUM_COPY_CMDS -w 1 -n 10 --numOfQueues $NUM_QUEUES
    #if [ $NUM_QUEUES -eq 1 ]; then
    #    cat $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    #else
    #    tail -n +2 $OUTPUT_DIR/$RESULT_CSV >> $SUMMARY_FILE 
    #fi
done
