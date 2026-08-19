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
// all_to_all_kernels.hpp
//
// SDMA-based "push" all-to-all. Mirrors RCCL's per-rank send/recv loop: PE myPe
// sends srcPtrs[p] to peer p, and peer p stores it in its recv slot for sender
// myPe (dstPtrs[myPe]). Self (p==myPe) is included via an SDMA self-copy, like
// all-gather. Compute-free: SDMA writes every peer's recv slot directly and the
// kernel only waits for completion.
//
// Reuses the shared fused SDMA routine StartSdmaScatter (collectives_common.hpp)
// via call-site lambdas: active=true (self included), per-peer source is an
// arbitrary pointer srcPtrs[peer], and the destination offset is constant
// (dstPtrs[myPe] - heapBase; every receiver stores our chunk at its own
// dstPtrs[myPe]-equivalent symmetric-heap offset).
// ===========================================================================
#pragma once

#include <cstdint>

#include "mori/collective/XLA/collectives_common.hpp"
#include "mori/shmem/internal.hpp"  // GetGlobalGpuStatesPtr

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::collective;

// Single-block, compute-free all-to-all.
//   srcPtrs : npes local symmetric-heap pointers; srcPtrs[p] sent to peer p.
//   dstPtrs : npes local symmetric-heap pointers; dstPtrs[p] receives peer p's chunk.
//   chunkBytes : size in bytes of each per-peer chunk.
//
// Phase 1: SDMA scatters each chunk into every peer's recv slot (self included),
// each copy trailed by an ADD64 of 1 into the receiver's signalPtrs[0] counter.
// Phase 2: thread 0 waits until the counter reaches npes (self-copy included),
// then an acquire fence makes the peer SDMA writes visible. No Phase 3. Finally
// thread 0 resets the counter for the next launch.
//
// blockDim.x must be >= npes for the fast-path packet build; launch <<<1, 256>>>
// with npes <= 8 (SDMA fast path).
__global__ void AllToAllPushKernel(int myPe, int npes,
                                   const AddressPair* __restrict__ pairs, size_t chunkBytes) {
  // Phase 1: push each send slot p to peer p's recv slot for sender myPe (self
  // included). The destination offset is constant (my slot on every peer, by
  // symmetric-heap layout); the source is an arbitrary per-peer pointer.
  const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
  const size_t off = reinterpret_cast<uintptr_t>(pairs[myPe].dest) - heapBase;
  StartSdmaScatter(
      npes, /*logS=*/0, chunkBytes,
      [](int) { return true; },
      [=](int peer) -> const uint8_t* {
        return reinterpret_cast<const uint8_t*>(pairs[peer].source);
      },
      [=](int) -> size_t { return off; });

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
  // System-scope ACQUIRE: recv slots were written by peers' SDMA over XGMI.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  // No Phase 3: recv buffers already fully written by SDMA. Reset counter for reuse.
  if (threadIdx.x == 0) {
    heapObj->signalPtrs[0] = 0;
    __threadfence_system();  // keep it for next launch's peer SDMA
  }
}
