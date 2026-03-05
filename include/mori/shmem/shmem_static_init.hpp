// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Include this header in ONE hipcc-compiled source file to automatically
// define globalGpuStates and register the barrier kernel for ShmemBarrierOnStream.
// ShmemInit will then initialize everything without any manual step.
//
// Usage:
//   #include "mori/shmem/shmem_static_init.hpp"
//   int main() {
//     ShmemMpiInit(MPI_COMM_WORLD);
//     // globalGpuStates ready, ShmemBarrierOnStream works
//   }

#pragma once

#include <hip/hip_runtime.h>
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace shmem {

__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;

namespace {

__global__ void _shmem_barrier_all_block_kernel() { ShmemBarrierAllBlock(); }

void _staticBarrierLauncher(hipStream_t stream) {
  _shmem_barrier_all_block_kernel<<<1, 1, 0, stream>>>();
}

void* _getStaticGpuStatesAddr() {
  void* addr = nullptr;
  hipGetSymbolAddress(&addr, HIP_SYMBOL(mori::shmem::globalGpuStates));
  return addr;
}

struct _StaticShmemRegistrar {
  _StaticShmemRegistrar() {
    RegisterGpuStatesAddrProvider(_getStaticGpuStatesAddr);
    RegisterBarrierLauncher(_staticBarrierLauncher);
  }
};
static _StaticShmemRegistrar _s_registrar;

}  // namespace
}  // namespace shmem
}  // namespace mori
