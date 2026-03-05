// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// JIT-compiled shmem device kernels: provides the globalGpuStates device
// variable and shmem runtime kernels (barrier, etc.).
// Loaded at runtime via hipModuleLoad during ShmemInit.

#include <hip/hip_runtime.h>

#define MORI_SHMEM_NO_STATIC_INIT
#include "mori/shmem/shmem.hpp"

// Each .hsaco needs its own globalGpuStates, initialized via ShmemModuleInit.
namespace mori {
namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}
}  // namespace mori

extern "C" __global__ void mori_shmem_barrier_all_block() { mori::shmem::ShmemBarrierAllBlock(); }
