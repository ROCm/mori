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
#pragma once

#include "mori/shmem/shmem_api.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "mori/shmem/shmem_device_api.hpp"
#include "mori/shmem/shmem_device_kernels.hpp"
#include "mori/shmem/shmem_ibgda_kernels.hpp"
#include "mori/shmem/shmem_p2p_kernels.hpp"
#include "mori/shmem/shmem_sdma_kernels.hpp"
#endif

// When compiled with hipcc (C++ users with device code), define a weak
// globalGpuStates symbol per TU. This keeps non-RDC HIP builds linkable.
#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__)
namespace mori {
namespace shmem {

__device__ __attribute__((visibility("hidden"), weak)) GpuStates globalGpuStates;

namespace _static_init {

__attribute__((visibility("hidden"), weak)) void* _getGpuStatesAddr() {
  void* addr = nullptr;
  (void)hipGetSymbolAddress(&addr, HIP_SYMBOL(mori::shmem::globalGpuStates));
  return addr;
}

struct _GpuStatesRegistrar {
  _GpuStatesRegistrar() { RegisterGpuStatesAddrProvider(_getGpuStatesAddr); }
};
__attribute__((visibility("hidden"), weak)) _GpuStatesRegistrar _s_gpuStatesRegistrar;

#if !defined(MORI_SHMEM_NO_STATIC_INIT)
__global__ void _barrier_kernel() { ShmemBarrierAllBlock(); }

__attribute__((weak)) void _barrierLauncher(hipStream_t stream) {
  _barrier_kernel<<<1, 1, 0, stream>>>();
}

struct _BarrierRegistrar {
  _BarrierRegistrar() { RegisterBarrierLauncher(_barrierLauncher); }
};
__attribute__((weak)) _BarrierRegistrar _s_barrierRegistrar;
#endif  // !MORI_SHMEM_NO_STATIC_INIT

}  // namespace _static_init
}  // namespace shmem
}  // namespace mori
#endif  // __HIPCC__ || __HIP__ || __CUDACC__