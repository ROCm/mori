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
// all_gather_kernels.hpp
//
// SDMA-based "push" all-gather. All-gather is the mirror of reduce-scatter's
// scatter step with NO reduction: each PE pushes its single local shard into
// every peer's output[myPe] slot -- including itself (SDMA self-copy). Since
// there is no GPU compute to overlap, this uses a single block and logS=0
// (S=1, one slice, one completion flag), reusing the generalized fused SDMA
// scatter (StartSdmaScatter, byte-based) from collectives_common.hpp.
// ===========================================================================
#pragma once

#include <cstdint>

#include "mori/collective/XLA/collectives_common.hpp"
#include "mori/shmem/internal.hpp"  // GetGlobalGpuStatesPtr

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::collective;

// Single-block, compute-free all-gather.
//   input  : raw symmetric-heap pointer, my shard of chunkBytes bytes
//   output : raw symmetric-heap pointer, npes*chunkBytes bytes (all shards)
//
// Phase 1: SDMA scatters my shard into every peer's output[myPe] slot (self
// included), each copy trailed by an ADD64 of 1 into the receiver's
// signalPtrs[0] counter. Phase 2: thread 0 waits until the counter reaches npes
// (self-copy included), then an acquire fence makes the peer SDMA writes to
// output visible. No Phase 3 -- output is already fully written by SDMA. Finally
// thread 0 resets the counter for the next launch.
//
// blockDim.x must be >= npes for block 0's fast-path packet build; launch with
// npes <= 8 (SDMA fast path) and a comfortable blockDim (256).
__global__ void AllGatherPushKernel(int myPe, int npes, const void* __restrict__ input,
                                    void* __restrict__ output, size_t chunkBytes) {
  // Single block: push my shard to every peer's output[myPe] slot (+ self). The
  // destination slot is myPe for every peer, so the byte offset is constant; the
  // source is our single shard (same for all peers).
  const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
  const size_t dstOff = reinterpret_cast<uintptr_t>(output) - heapBase +
                        myPe * chunkBytes;
  StartSdmaScatter(
      myPe, npes, /*logS=*/0, chunkBytes,
      [](int) { return true; },
      [=](int) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(input); },
      [=](int) -> size_t { return dstOff; });

  auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;

  // === Phase 2: wait until the counter reaches all npes senders. ===============
  if (threadIdx.x == 0) {
    const uint32_t want = static_cast<uint32_t>(npes);  // self-copy included
    // 32-bit ADD into the low dword -> poll 32 bits of the single counter.
    auto* addr = Tglobal(reinterpret_cast<uint32_t*>(&heapObj->signalPtrs[0]));  // S=1
    while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
    }
  }
  __syncthreads();
  // System-scope ACQUIRE: output was written by peers' SDMA over XGMI, so
  // invalidate against another device's writes before any reader trusts output.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  // No Phase 3: output already fully written by SDMA. Reset the counter for reuse.
  if (threadIdx.x == 0) {
    heapObj->signalPtrs[0] = 0;
    __threadfence_system();  // keep it for next launch's peer SDMA
  }
}
