// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// JIT-compiled shmem device kernels: provides the globalGpuStates device
// variable and shmem runtime kernels (barrier, etc.).
// Loaded at runtime via hipModuleLoad during ShmemInit.

#include <hip/hip_runtime.h>

// JIT TU: same weak globalGpuStates path as mori_collective; not EP (no MORI_DEFINE_GPU_STATES).
// Policy: include/mori/shmem/shmem.hpp
#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

// globalGpuStates is provided by shmem.hpp for HIP/CUDA device builds.
// Keep only one definition in this TU to avoid redefinition with JIT build.

extern "C" __global__ void mori_shmem_barrier_all_block() { mori::shmem::ShmemBarrierAllBlock(); }
