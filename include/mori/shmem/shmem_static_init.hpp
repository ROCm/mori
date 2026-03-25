// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
