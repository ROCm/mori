#!/bin/bash
# set -x

XLA_DIR=.
DUMP_DIR=$XLA_DIR/uu_dump
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
export MORI_SHMEM_LOG_LEVEL=TRACE
export MORI_CORE_LOG_LEVEL=DEBUG
export MORI_OPS_LOG_LEVEL=DEBUG

# export HIP_VISIBLE_DEVICES=0,1
# export AMD_LOG_LEVEL=4

$GDB $ROCPROF ./build/examples/multithread_multi_gpu 2>&1 | tee zzzrun.log
