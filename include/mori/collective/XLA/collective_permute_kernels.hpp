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

// ===========================================================================
// collective_permute_kernels.hpp
//
// SDMA-based "push" collective-permute for a single destination. PE myPe copies
// sendBuf into peer dstPe's recvBuf (self-copy when dstPe == myPe) and waits for
// srcPe's copy to land in our recvBuf (srcPe < 0 means no incoming).
//
// Compute-free: one block, logS=0, reusing StartSdmaScatter from
// collectives_common.hpp. sendBuf and recvBuf must be symmetric-heap pointers
// so the address-based SDMA put can translate local->peer (offset from
// heapBaseAddr).
// ===========================================================================
#pragma once

#include <cstdint>

#include "mori/collective/XLA/collectives_common.hpp"
#include "mori/shmem/internal.hpp"  // GetGlobalGpuStatesPtr

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::collective;

// Single-block, compute-free collective-permute (one destination).
//   sendBuf  : raw symmetric-heap pointer, numBytes to send
//   recvBuf  : raw symmetric-heap pointer, numBytes receive slot
//   dstPe    : peer that receives our sendBuf (may be myPe)
//   srcPe    : peer that sends into our recvBuf, or -1 if nobody does
//
// Phase 1: SDMA pushes sendBuf into dstPe's recvBuf (self included when
// dstPe == myPe), trailed by an ADD32 of 1 into the receiver's signalPtrs[0].
// Phase 2: if srcPe >= 0, thread 0 waits until the counter reaches 1, then an
// acquire fence makes the peer SDMA write visible. Finally thread 0 resets the
// counter for the next launch.
//
// Launch <<<1, 256>>> with npes <= 8 (SDMA fast path).
__global__ void CollectivePermutePushKernel(int npes, int dstPe, int srcPe,
                                            const void* __restrict__ sendBuf,
                                            void* __restrict__ recvBuf, size_t numBytes) {
  const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
  const size_t dstOff = reinterpret_cast<uintptr_t>(recvBuf) - heapBase;
  StartSdmaScatter(
      npes, /*logS=*/0, numBytes,
      [=](int peer) { return peer == dstPe; },
      [=](int) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(sendBuf); },
      [=](int) -> size_t { return dstOff; });

  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
  const uint32_t want = (srcPe >= 0) ? 1u : 0u;

  if (threadIdx.x == 0 && want) {
    auto* addr = Tglobal(reinterpret_cast<uint32_t*>(&heapObj->signalPtrs[0]));  // S=1
    while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
    }
  }
  __syncthreads();
  if (want) {
    // System-scope ACQUIRE: recvBuf was written by srcPe's SDMA over XGMI.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }

  if (threadIdx.x == 0) {
    heapObj->signalPtrs[0] = 0;
    __threadfence_system();  // keep it for next launch's peer SDMA
  }
}
