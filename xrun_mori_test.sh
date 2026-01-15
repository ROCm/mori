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

rm -rf $DUMP_DIR
mkdir -p $DUMP_DIR
#rm -f /dev/shm/nccl-*
#rm -rf swdev545325 swdev550718
rm -rf $XLA_DIR/profile/2025_*
rm -f graph_*_dot*
 
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_LOG_LEVEL=6
export MIOPEN_GEMM_ENFORCE_BACKEND=5
# export TF_ROCM_KEEP_XLA_TEMPFILES=1 

#export NCCL_PROTO=LL
#export NCCL_MIN_NCHANNELS=112 # does it help ??    
#export NCCL_MAX_NCHANNELS=8

# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,INIT,COLL 
#RCCL_KERNEL_COLL_TRACE_ENABLE=1
# RCCL debug tips https://uccl-project.github.io/posts/debug-nccl/

export TF_CPP_MIN_LOG_LEVEL=1 TF_CPP_MIN_VLOG_LEVEL=0 TF_CPP_MAX_LOG_LEVEL=5

#export TF_FORCE_UNIFIED_MEMORY=true
# export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_CLIENT_MEM_FRACTION=0.35
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

export HSA_NO_SCRATCH_RECLAIM=1
# export XLA_COMMAND_BUFFERS_USE_CACHED_ALLOCS=1
# export XLA_COMMAND_BUFFERS_MODULE_RE=jit_train_step.*
# export XLA_COMMAND_BUFFERS_USE_RENDEZVOUS=false

#export JAX_COMPILATION_CACHE_DIR=$XLA_DIR/jax_hsaco_cache
#export TF_XLA_HSACO_CACHE_DIR=/tf/hsaco_cache
export TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD=11111111111111
export HIP_FORCE_DEV_KERNARG=1
export PJRT_NPROC=8 
export OMP_NUM_THREADS=8 TORCH_NUM_THREADS=8 TORCH_NUM_INTEROP_THREADS=8
#export AMD_LOG_LEVEL=4

# No spaces in between!!
export TF_CPP_VMODULE=latency_hiding_scheduler=0,\
reduce_scatter_combiner=0,\
command_buffer_cmd=0,gemm_algorithm_picker=0,\
command_buffer_thunk=0,\
gpu_compiler=0,command_buffer_scheduling=0,\
gpu_command_buffer=0,rocm_command_buffer=0,\
gpu_executable=0,nccl_communicator=0,collective_pipeliner=0

#PYEXEC=$(pyenv which python)
PYEXEC=$(which python)

#export LD_PRELOAD=/opt/rocm/lib/libMIOpen.so.1:$HIPLIB:/tf/rccl/build/librccl.so
export XLA_FLAGS="--xla_gpu_enable_cublaslt=true \
           --xla_gpu_enable_latency_hiding_scheduler=false \
           --xla_gpu_autotune_level=0 \
            --xla_gpu_enable_nccl_comm_splitting=false \
            --xla_gpu_collectives_use_persistent_cliques=true \
           --xla_gpu_enable_triton_softmax_fusion=false \
           --xla_dump_to=$DUMP_DIR \
           --xla_gpu_enable_all_gather_combine_by_dim=false \
           --xla_gpu_enable_reduce_scatter_combine_by_dim=false \
            --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 \
            --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
            --xla_gpu_reduce_scatter_combine_threshold_bytes=0 \
            --xla_gpu_experimental_parallel_collective_overlap_limit=1 \
           --xla_gpu_graph_min_graph_size=5 \
           --xla_gpu_enable_command_buffer= \
           --xla_gpu_enable_triton_gemm=false \
           --xla_gpu_graph_enable_concurrent_region=true \
           --xla_gpu_force_compilation_parallelism=4 \
           --xla_gpu_enable_highest_priority_async_stream=true \
           --xla_gpu_memory_limit_slop_factor=95 \
           --xla_gpu_autotune_gemm_rtol=0.01"

NumProcs=8
TotalGpus=8

#HIPLIB=/tf/clr/build/hipamd/lib/libamdhip64.so
#export LD_LIBRARY_PATH=/usr/rocm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib

# this is go get rid of 'request to allocate mask for invalid number: Invalid argument'
#export LD_PRELOAD=/lib/x86_64-linux-gnu/libnuma.so.1:$HIPLIB:/opt/rocm/lib/libroctracer64.so
# export MORI_SHMEM_MODE=ISOLATION
export MORI_SHMEM_HEAP_SIZE=16G

# install debug packages for more insights
# apt install hip-runtime-amd-dbgsym hsa-rocr-dbgsym rocm-core-dbgsym rocprofiler-dbgsym rocprofiler-sdk-dbgsym

TEST=mori_playground.py
# TEST=jax_playground.py
#TEST=torch_test_dc.py
#TEST=examples/ops/dispatch_combine/test_dispatch_combine.py
# TEST=tests/python/ops/test_dispatch_combine.py

XBASE=$(basename $TEST)
pkill -9 -c -f python #$XBASE
rm -f $XLA_DIR/zzout_*.log

# PYEXEC=$(which pytest)
# PYTHONPATH=/tf/mori $PYEXEC -s $TEST
# exit 0


for ((pid = 0; pid < $NumProcs; pid++ )); do
  last_id=$(($NumProcs - 1))
  div=$(($TotalGpus/$NumProcs))
  gpus=$(seq -s, $((pid*div)) $((pid*div+div-1)))

  if [[ pid -eq last_id ]] && [[ ${debug} -eq 1 ]]; then
    NODE_RANK=$pid \
    HIP_VISIBLE_DEVICES=$gpus \
    $GDB $ROCPROF $PYEXEC $TEST --world_size=$NumProcs --rank=$pid 2>&1 | tee $XLA_DIR/zzout_$pid.log
  else
    NODE_RANK=$pid \
    HIP_VISIBLE_DEVICES=$gpus \
    $PYEXEC $TEST --world_size=$NumProcs --rank=$pid 2>&1 | tee $XLA_DIR/zzout_$pid.log &
  fi
done

chmod -R 777 $DUMP_DIR

# popd
