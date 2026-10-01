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

namespace mori {
namespace collective {

// Single-block, compute-free collective-permute (one destination).
//   sendBuf  : raw symmetric-heap pointer, numBytes to send
//   recvBuf  : raw symmetric-heap pointer, numBytes receive slot
//   dstPe    : peer that receives our sendBuf (may be myPe)
//   srcPe    : peer that sends into our recvBuf, or -1 if nobody does
//
//   ready    : symmetric-heap uint64[>= npes] ready tokens, zeroed once. Slot p of
//              my copy is written only by PE p and cleared only by me.
//
// Phase 0: pairwise handshake, no global barrier -- only the PEs of a send/recv
// pair synchronize, so PEs that do not launch this kernel are never waited on.
// The receiver posts a token into its source's ready[myPe]; the sender waits for
// its destination's token and consumes it before writing anything into it.
// Phase 1: SDMA pushes sendBuf into dstPe's recvBuf (self included when
// dstPe == myPe), trailed by an ADD64 of 1 into the receiver's signalPtrs[0].
// Phase 2: if srcPe >= 0, thread 0 waits until the counter reaches 1, then an
// acquire fence makes the peer SDMA write visible. Finally thread 0 resets the
// counter for the next launch.
//
// Launch <<<1, 256>>> with npes <= 8 (SDMA fast path).
__global__ void CollectivePermutePushKernel(int npes, int dstPe, int srcPe,
                                            const void* __restrict__ sendBuf,
                                            void* __restrict__ recvBuf, size_t numBytes,
                                            mori::cco::ccoDevComm devComm,
                                            mori::cco::ccoWindow_t heapWin,
                                            uint64_t* __restrict__ ready) {
  const int myLsaRank = heapWin->lsaRank;
  const uint32_t stride4G = heapWin->stride4G;
  // dstPe's token proves it has entered this launch, so by its stream order its
  // previous kernel (of any collective) is done with recvBuf and has cleared
  // signalBuf[0]. Without it, my copy could overwrite recvBuf before dstPe
  // consumed it, or my ADD could land before its end-of-launch reset and be lost.
  // Every PE posts its token before waiting, so a ring cannot deadlock. A
  // self-send needs no handshake: stream order covers it.
  if (threadIdx.x == 0) {
    if (srcPe >= 0 && srcPe != myLsaRank) {
      int32_t diff = (srcPe - myLsaRank) * static_cast<int32_t>(stride4G);
      auto* tok = cco::impl::global(reinterpret_cast<uint64_t*>(
          reinterpret_cast<uint8_t*>(&ready[myLsaRank]) + (static_cast<uint64_t>(diff) << 32)));
      __hip_atomic_store(tok, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (dstPe != myLsaRank) {
      auto* mine = cco::impl::global(&ready[dstPe]);
      while (__hip_atomic_load(mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
      // dstPe re-arms only after my copy below lands, so this clear cannot
      // swallow its next token -- provided it is visible before the doorbell.
      StreamStore<ESystemScope, sizeof(uint64_t)>(&ready[dstPe], 0);
      __threadfence_system();
    }
  }
  __syncthreads();

  // recvBuf's byte offset within the heap window is identical on every rank
  // (symmetric layout), so my local recvBuf pointer maps to peer p's copy by the
  // flat-VA rank delta (pe - lsaRank)*stride4G<<32 -- the same inline LSA
  // addressing the reduce-scatter/all-reduce push kernels use. winBase cancels
  // out, so this needs no ccoGetLocalPtr/ccoGetLsaPeerPtr round trip.
  uint8_t* const localDst = reinterpret_cast<uint8_t*>(recvBuf);
  StartSdmaScatter(
      devComm.sdma, npes, /*logS=*/0, numBytes,
      [=](int peer) { return peer == dstPe; },
      [=](int) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(sendBuf); },
      [=](int peer) -> uint8_t* {
        int32_t diff = (peer - myLsaRank) * static_cast<int32_t>(stride4G);
        return localDst + (static_cast<uint64_t>(diff) << 32);
      });

  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;
  const uint32_t want = (srcPe >= 0) ? 1u : 0u;

  if (threadIdx.x == 0 && want) {
    auto* addr = cco::impl::global(reinterpret_cast<uint32_t*>(&signalBuf[0]));  // S=1
    while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  if (want) {
    // System-scope ACQUIRE: recvBuf was written by srcPe's SDMA over XGMI.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }

  // Reset the counter for reuse. SYSTEM scope (sc0 sc1: write-through past L2)
  // because srcPe ADDs into this slot by SDMA straight into HBM -- a plain store
  // would leave a dirty line that could evict on top of a later ADD and swallow
  // it. That is also why no release fence follows: there is nothing left in L2 to
  // write back.
  //
  // Clearing at kernel END is safe: srcPe's next ADD only follows my next
  // token, which I post only once my next launch has started.
  if (threadIdx.x == 0) {
    StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[0], 0);
  }
}
} // namespace collective
} // namespace mori
