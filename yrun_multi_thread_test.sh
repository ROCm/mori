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
export MORI_APP_LOG_LEVEL=DEBUG
export MORI_SHMEM_LOG_LEVEL=DEBUG
export MORI_CORE_LOG_LEVEL=DEBUG
export MORI_OPS_LOG_LEVEL=DEBUG

export MORI_DISABLE_P2P=0
export MORI_ENABLE_SDMA=0

# export HIP_VISIBLE_DEVICES=0,1
# export AMD_LOG_LEVEL=4

TORCH_LIBS=/usr/local/lib/python3.12/dist-packages/torch/lib
# TORCH=$TORCH_LIBS/libtorch.so

# this is go get rid of 'request to allocate mask for invalid number: Invalid argument'
export LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1 #:/lib/x86_64-linux-gnu/libibverbs.so.1
# export LD_LIBRARY_PATH=$TORCH_LIBS:$LD_LIBRARY_PATH

#mpirun --allow-run-as-root -np 2 ./build/examples/allgather_test $@
NUM_PROCS=1
NUM_GPUS_PER_PROCESS=2

pkill -9 -c -f allgather_test
rm -f allgather_test_uid.bin zz*.log

TEST=./build/examples/allgather_test
# TEST=./build/examples/sdma_put_test
for ((pid = 0; pid < $NUM_PROCS; pid++ )); do
  gpus=$(seq -s, $((pid*NUM_GPUS_PER_PROCESS)) $((pid*NUM_GPUS_PER_PROCESS+NUM_GPUS_PER_PROCESS-1)))
  #HIP_VISIBLE_DEVICES=$gpus 
  $TEST $pid $NUM_PROCS $NUM_GPUS_PER_PROCESS $@ 2>&1 | tee zzout_$pid.log &
done

# mpirun --allow-run-as-root -np 2 $TEST 0 2 2>&1 | tee zzzrun.log
