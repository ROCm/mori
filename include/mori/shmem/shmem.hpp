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

// -----------------------------------------------------------------------------
// Policy: EP vs SDMA CCL vs JIT shmem_kernels (minimize shmem surface area)
// -----------------------------------------------------------------------------
// SDMA CCL "business" changes stay in shmem_api / memory.cpp (symmetric register)
// and shmem_sdma_kernels.hpp (SDMA put + expectedSignals). This header only gates
// the weak __device__ globalGpuStates + RegisterGpuStatesAddrProvider block.
//
// EP JIT (ep_common.hip -> ep_*.hip): set MORI_SHMEM_NO_STATIC_INIT only. Do NOT
// set MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES. shmem.hpp then does not emit a
// second globalGpuStates; each .hsaco defines it via MORI_DEFINE_GPU_STATES
// (default visibility; must match internal.hpp). Host uses ShmemModuleInit.
//
// mori_collective (static HIP in libmori_collective): same NO_STATIC_INIT, plus
// MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES from CMake so weak globalGpuStates
// (default visibility; matches internal.hpp / EP) and the addr provider emit here.
//
// shmem_kernels.hip (JIT): NO_STATIC_INIT + MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
// before including shmem.hpp — same weak path as collective, no MORI_DEFINE_GPU_STATES.
//
// RegisterGpuStatesAddrProvider callbacks are stored in a std::vector in runtime.cpp
// so CopyGpuStatesToDevice can update every registered static symbol (multiple HIP
// TUs and/or future modules). Do not revert to a single provider without a new design.
// -----------------------------------------------------------------------------

#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__)
namespace mori {
namespace shmem {

#if !defined(MORI_SHMEM_NO_STATIC_INIT) || defined(MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES)
__device__ __attribute__((visibility("default"), weak)) GpuStates globalGpuStates;

namespace _static_init {

__attribute__((visibility("default"), weak)) void* _getGpuStatesAddr() {
  void* addr = nullptr;
  (void)hipGetSymbolAddress(&addr, HIP_SYMBOL(mori::shmem::globalGpuStates));
  return addr;
}

struct _GpuStatesRegistrar {
  _GpuStatesRegistrar() { RegisterGpuStatesAddrProvider(_getGpuStatesAddr); }
};
__attribute__((visibility("default"), weak)) _GpuStatesRegistrar _s_gpuStatesRegistrar;

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
#endif  // !MORI_SHMEM_NO_STATIC_INIT || MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES

}  // namespace shmem
}  // namespace mori
#endif  // __HIPCC__ || __HIP__ || __CUDACC__
