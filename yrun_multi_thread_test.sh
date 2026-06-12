#!/bin/bash
# set -x

GDB=
ROCPROF=

trap sigint_handler INT
sigint_handler() {
  echo "SIGINT caught!"
  #chmod 777 $DUMP_DIR/*
}

debug=${debug:-0}
profile=${profile:-0}

if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

if [[ ${profile} -eq 1 ]]; then
    #ROCPROF="rocprofv3 -i rocprof_counters.json -d $DUMP_DIR -o out --"
    #ROCPROF="rocprofv3 --stats --kernel-trace -d $DUMP_DIR -o out --"
    # ROCPROF="rocprofv3 --stats --hip-runtime-trace --memory-copy-trace -d $DUMP_DIR -o $DUMP_DIR/output.csv --"
    #--scratch-memory-trace ??
    #ROCPROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    #ROCPROF="rocprofv3 --stats --truncate-kernels --kernel-trace --output-format pftrace -d $DUMP_DIR --"
   # ROCPROF="rocprofv3 --kernel-trace --output-format pftrace -d $DUMP_DIR --"
    # --hip-runtime-trace
  ROCPROF="rocprofv2 --plugin perfetto --kernel-trace -d $DUMP_DIR "
    # ROCPROF="rocsys --session vv1 launch rocprofv2 --kernel-trace -d $DUMP_DIR"
fi


export HSA_NO_SCRATCH_RECLAIM=1
export HIP_FORCE_DEV_KERNARG=1

HIPLIB= #/tf/rocm-systems/projects/clr/build/hipamd/lib/libamdhip64.so
RCCL= #/tf/rccl/backup/librccl_7.1.1.so.1.0

export LD_PRELOAD=$HIPLIB:$RCCL

#| `MORI_SHMEM_MODE` | Heap mode: `"static"`, `"vmm"`, or `"isolation"` | `"static"` |
export MORI_SHMEM_MODE=STATIC_HEAP #ISOLATION #VMM_HEAP #  STATIC_HEAP
# export MORI_SHMEM_HEAP_TYPE=normal
export MORI_SHMEM_HEAP_SIZE=5G
export MORI_KERNEL_DIR=/tf/mori/build/lib/gfx942_mlx5
# export MORI_APP_LOG_LEVEL=DEBUG
# export MORI_SHMEM_LOG_LEVEL=DEBUG
# export MORI_CORE_LOG_LEVEL=DEBUG
# export MORI_OPS_LOG_LEVEL=DEBUG

# export ROCR_VISIBLE_DEVICES=4,5,6,7
# export HIP_VISIBLE_DEVICES=0,1,2,3

export MORI_DISABLE_P2P=0
export MORI_ENABLE_SDMA=1
# it looks like 1 channel gives the best performance
export MORI_SDMA_NUM_CHANNELS=1
export RS_MIN_SIGNAL_SLOTS_PER_DEV=2

# CCO socket rendezvous (single host, one process, one thread per GPU).
export MORI_SOCKET_IFNAME=lo

# this is go get rid of 'request to allocate mask for invalid number: Invalid argument'
export LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1 #:/lib/x86_64-linux-gnu/libibverbs.so.1

# Unified collectives_benchmark knobs (were RS_MODE / RS_LOG_PUSH_SLICES env vars,
# now runtime CLI flags). Override any of these from the environment.
#   COLL: reduce_scatter|all_reduce|all_gather|all_to_all|collective_permute
#         (aliases rs|ar|ag|a2a|cp)
COLL=${COLL:-all_reduce}
MODE=${MODE:-push}      # push|pull (reduce_scatter)
LOGS=${LOGS:-2}         # logS: S = 1<<LOGS slices (push path)
DTYPE=${DTYPE:-f32}     # f32|bf16|f16|s32|s64 (reduction collectives)
OP=${OP:-sum}           # sum|prod|min|max (reduction collectives)

TEST_NAME=collectives_benchmark
pkill -9 -c -f $TEST_NAME
#rm -f allgather_test_uid.bin zz*.log

NUM_GPUS=${NUM_GPUS:-4}
MIN_SIZE=${MIN_SIZE:-1024*1024}
MAX_SIZE=${MAX_SIZE:-1024*1024*128}

# MIN_SIZE=77777*4
# MAX_SIZE=77777991*4

TEST=./build/examples/$TEST_NAME
rm -f zzout_*.log

# /data/mori/perf record --call-graph fp -F 2999 -m 128M \
#   -- $TEST --coll $COLL --npes $NUM_GPUS --size 1024*1024*128 "$@"

for ((size = MIN_SIZE; size <= MAX_SIZE; size = size * 2)); do
  $GDB $ROCPROF $TEST --coll $COLL --npes $NUM_GPUS --size $size \
    --dtype $DTYPE --op $OP --mode $MODE --logS $LOGS $@ 2>&1 | tee -a zzout_0.log
done
exit 0

for ((pid = 0; pid < $NUM_PROCS; pid++ )); do
  gpus=$(seq -s, $((pid*NUM_GPUS_PER_PROCESS)) $((pid*NUM_GPUS_PER_PROCESS+NUM_GPUS_PER_PROCESS-1)))
  #HIP_VISIBLE_DEVICES=$gpus 
  $TEST $pid $NUM_PROCS $NUM_GPUS_PER_PROCESS $@ 2>&1 | tee zzout_$pid.log &
done

# mpirun --allow-run-as-root -np 2 $TEST 0 2 2>&1 | tee zzzrun.log
