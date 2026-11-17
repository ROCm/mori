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
//
// The signal-pad protocol and kernels below are adapted from PyTorch v2.15.0,
// torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh, under this license:
//
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin,
//                                                   Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio,
//                                                   Johnny Mariethoz)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// No torch headers: this TU only needs HIP, and hipcc stays off torch's include tree.
// Launches use hipLaunchKernelGGL, not <<<>>>, so torch's hipify leaves the file unchanged.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>

#include "symm_signal.hpp"

namespace mori {
namespace allocator {
namespace {

// The flat-address CAS torch uses on ROCm, with the scope spelled out.
template <int Order>
__device__ __forceinline__ uint32_t CasSystem(uint32_t* addr, uint32_t expected, uint32_t desired) {
  __hip_atomic_compare_exchange_strong(addr, &expected, desired, Order, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return expected;
}

// Spin until *addr flips from -> to; false once timeout_ticks of wall clock pass (0: never).
template <int Order>
__device__ __forceinline__ bool TryFlip(uint32_t* addr, uint32_t from, uint32_t to,
                                        uint64_t timeout_ticks) {
  const uint64_t start = wall_clock64();
  while (CasSystem<Order>(addr, from, to) != from) {
    if (timeout_ticks != 0 && wall_clock64() - start > timeout_ticks) return false;
  }
  return true;
}

__device__ __forceinline__ bool TryPut(uint32_t* addr, uint64_t timeout_ticks) {
  return TryFlip<__ATOMIC_RELEASE>(addr, 0, 1, timeout_ticks);
}

__device__ __forceinline__ bool TryWait(uint32_t* addr, uint64_t timeout_ticks) {
  return TryFlip<__ATOMIC_ACQUIRE>(addr, 1, 0, timeout_ticks);
}

__device__ __forceinline__ uint32_t* Slot(uint32_t** pads, int owner, int channel, int sender,
                                          int world_size) {
  return pads[owner] + static_cast<size_t>(world_size) * channel + sender;
}

// Kernels carry torch's names, so profiles and tests that look for torch's find these.
// One thread per peer: signal it, then wait for its signal.
__global__ void barrier_kernel(uint32_t** pads, int channel, int rank, int world_size,
                               uint64_t timeout_ticks, unsigned long timeout_ms) {
  const int peer = static_cast<int>(threadIdx.x);
  if (peer >= world_size || peer == rank) return;
  if (!TryPut(Slot(pads, peer, channel, rank, world_size), timeout_ticks)) {
    printf(
        "[FATAL] SymmetricMemory::barrier: rank %d failed to send signal to rank %d on channel "
        "%d after %lu milliseconds\n",
        rank, peer, channel, timeout_ms);
    __builtin_trap();
  }
  if (!TryWait(Slot(pads, rank, channel, peer, world_size), timeout_ticks)) {
    printf(
        "[FATAL] SymmetricMemory::barrier: rank %d failed to receive signal from rank %d on "
        "channel %d after %lu milliseconds\n",
        rank, peer, channel, timeout_ms);
    __builtin_trap();
  }
}

__global__ void put_signal_kernel(uint32_t** pads, int dst_rank, int channel, int rank,
                                  int world_size, uint64_t timeout_ticks,
                                  unsigned long timeout_ms) {
  if (!TryPut(Slot(pads, dst_rank, channel, rank, world_size), timeout_ticks)) {
    printf(
        "[FATAL] SymmetricMemory::put_signal: rank %d failed to send signal to rank %d on channel "
        "%d after %lu milliseconds\n",
        rank, dst_rank, channel, timeout_ms);
    __builtin_trap();
  }
}

__global__ void wait_signal_kernel(uint32_t** pads, int src_rank, int channel, int rank,
                                   int world_size, uint64_t timeout_ticks,
                                   unsigned long timeout_ms) {
  if (!TryWait(Slot(pads, rank, channel, src_rank, world_size), timeout_ticks)) {
    printf(
        "[FATAL] SymmetricMemory::wait_signal: rank %d failed to receive signal from rank %d on "
        "channel %d after %lu milliseconds\n",
        rank, src_rank, channel, timeout_ms);
    __builtin_trap();
  }
  __threadfence_system();
}

// wall_clock64() runs at the device's fixed wall-clock rate, reported in kHz = ticks per ms.
hipError_t TimeoutTicks(size_t timeout_ms, uint64_t* ticks) {
  *ticks = 0;
  if (timeout_ms == 0) return hipSuccess;
  int dev = 0, khz = 0;
  hipError_t err = hipGetDevice(&dev);
  if (err != hipSuccess) return err;
  err = hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev);
  if (err != hipSuccess) return err;
  if (khz <= 0) khz = 100000;  // unreported: assume the 100 MHz every gfx9+ part runs at
  const uint64_t rate = static_cast<uint64_t>(khz);
  *ticks = timeout_ms > std::numeric_limits<uint64_t>::max() / rate
               ? std::numeric_limits<uint64_t>::max()
               : static_cast<uint64_t>(timeout_ms) * rate;
  return hipSuccess;
}

constexpr int kMaxBlockThreads = 1024;

}  // namespace

hipError_t LaunchSignalPadBarrier(uint32_t** pads_dev, int channel, int rank, int world_size,
                                  size_t timeout_ms, hipStream_t stream) {
  if (world_size < 1 || world_size > kMaxBlockThreads) return hipErrorInvalidValue;
  uint64_t ticks = 0;
  hipError_t err = TimeoutTicks(timeout_ms, &ticks);
  if (err != hipSuccess) return err;
  hipLaunchKernelGGL(barrier_kernel, dim3(1), dim3(world_size), 0, stream, pads_dev, channel, rank,
                     world_size, ticks, static_cast<unsigned long>(timeout_ms));
  return hipGetLastError();
}

hipError_t LaunchSignalPadPut(uint32_t** pads_dev, int dst_rank, int channel, int rank,
                              int world_size, size_t timeout_ms, hipStream_t stream) {
  uint64_t ticks = 0;
  hipError_t err = TimeoutTicks(timeout_ms, &ticks);
  if (err != hipSuccess) return err;
  hipLaunchKernelGGL(put_signal_kernel, dim3(1), dim3(1), 0, stream, pads_dev, dst_rank, channel,
                     rank, world_size, ticks, static_cast<unsigned long>(timeout_ms));
  return hipGetLastError();
}

hipError_t LaunchSignalPadWait(uint32_t** pads_dev, int src_rank, int channel, int rank,
                               int world_size, size_t timeout_ms, hipStream_t stream) {
  uint64_t ticks = 0;
  hipError_t err = TimeoutTicks(timeout_ms, &ticks);
  if (err != hipSuccess) return err;
  hipLaunchKernelGGL(wait_signal_kernel, dim3(1), dim3(1), 0, stream, pads_dev, src_rank, channel,
                     rank, world_size, ticks, static_cast<unsigned long>(timeout_ms));
  return hipGetLastError();
}

}  // namespace allocator
}  // namespace mori
