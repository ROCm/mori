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

namespace mori {
namespace collective {

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
__global__ void AllToAllPushKernel(int myPe, int npes, const AddressPair* __restrict__ pairs,
                                   size_t chunkBytes, mori::cco::ccoDevComm devComm,
                                   mori::cco::ccoWindow_t heapWin) {
  // Phase 1: push each send slot p to peer p's recv slot for sender myPe (self
  // included). My recv slot has the SAME byte offset within the heap window on
  // every peer (all ranks Allocate identically), so compute it once from the
  // local window base and reuse it for every peer. The source is an arbitrary
  // per-peer local pointer (SDMA reads local VA directly).
  const size_t heapBase = reinterpret_cast<uintptr_t>(mori::cco::ccoGetLocalPtr(heapWin));
  const size_t off = reinterpret_cast<uintptr_t>(pairs[myPe].dest) - heapBase;
  StartSdmaScatter(
      devComm.sdma, npes, /*logS=*/0, chunkBytes,
      [](int) { return true; },
      [=](int peer) -> const uint8_t* {
        return reinterpret_cast<const uint8_t*>(pairs[peer].source);
      },
      [=](int peer) -> uint8_t* {
        return reinterpret_cast<uint8_t*>(mori::cco::ccoGetLsaPeerPtr(heapWin, peer, off));
      });

  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;

  // === Phase 2: wait until the counter reaches all npes senders. ===============
  if (threadIdx.x == 0) {
    const uint32_t want = static_cast<uint32_t>(npes);  // self-copy included
    // 32-bit ADD into the low dword -> poll 32 bits of the single counter.
    auto* addr = Iglobal(reinterpret_cast<uint32_t*>(&signalBuf[0]));  // S=1
    while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
    }
  }
  __syncthreads();
  // System-scope ACQUIRE: recv slots were written by peers' SDMA over XGMI.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  // No Phase 3: recv buffers already fully written by SDMA. Reset counter for reuse.
  // SYSTEM scope (sc0 sc1: write-through past L2) because peers ADD into this slot
  // by SDMA straight into HBM -- a plain store would leave a dirty line that could
  // evict on top of a later ADD and swallow it. That is also why no release fence
  // follows: there is nothing left in L2 to write back.
  //
  // NOTE: this clears at kernel END, so it still relies on the caller barriering
  // between launches (the benchmark does hipStreamSynchronize + ccoBarrierAll per
  // iteration); without one, a peer's NEXT launch's ADD can arrive before this
  // store and be overwritten by it, no matter the scope.
  if (threadIdx.x == 0) {
    StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[0], 0);
  }
}
} // namespace collective
} // namespace mori
