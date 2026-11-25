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

// One-shot all-to-all whose bytes move through the TDM engine. gfx125x only: like
// allreduce/twoshot_tdm_kernel.hpp, it is reached from inside the arch guard in ccl_kernels.hip.
//
// Same buffers and result as OneShotAll2allSdmaKernel: chunk p of `input` (elementCount elements)
// lands at offset myPe * elementCount of rank p's output transit buffer, this rank's own chunk
// included, so the host class is shared.
//
// The completion protocol differs. Every value below only grows, so nothing is cleared after a call
// and a slow rank cannot miss one.
//   flags[s]  in the host class's npes u64 flags: a token from rank s, 2g - 1 once s has started
//             call g, which means s is done with its output transit buffer from call g - 1 and may
//             be written; 2g once every byte s sends here in call g has landed.
//   sync      a per-handle array in ordinary device memory, allocated zeroed by
//             python/mori/ccl/collective.py: one slot per block, set to g when that block's copies
//             are done; a go word, set to g by block 0 once every peer has started call g; and the
//             number of the last finished call, from which every block reads g.
// Only block 0 talks to the peers; the other blocks wait for the go word. Measured on 4 gfx1250
// GPUs with graphs of 100 calls, 64 KiB to 32 MiB per GPU: every block exchanging the start tokens
// itself made each call 33-45 us longer, and with block 0 alone at the start, a slot per block was
// up to 5 us faster than every block adding to one counter in the flags.

#include <hip/amd_detail/amd_gfx1250_TDM.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "mori/collective/allreduce/twoshot_sdma_kernel.hpp"
#include "mori/collective/allreduce/twoshot_tdm_kernel.hpp"
#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// Tile geometry, in 4-byte elements: 1 KiB rows like the AllReduce tiles, 16 rows, 16 KiB a warp.
// MUST AGREE WITH _TDM_A2A_TILE_BYTES in python/mori/ccl/collective.py, which reserves this much LDS
// per warp; reserving less is an LDS overrun.
#define MORI_CCL_TDM_A2A_TILE_W 256
#define MORI_CCL_TDM_A2A_TILE_H 16
#define MORI_CCL_TDM_A2A_TILE_ELEMS (MORI_CCL_TDM_A2A_TILE_W * MORI_CCL_TDM_A2A_TILE_H)
#define MORI_CCL_TDM_A2A_TILE_BYTES (MORI_CCL_TDM_A2A_TILE_ELEMS * 4)

// Layout of sync, in u64 words. MUST AGREE WITH _TDM_A2A_SYNC_WORDS and _TDM_A2A_BLOCKS in
// python/mori/ccl/collective.py: the grid may not exceed the block slots.
#define MORI_CCL_TDM_A2A_MAX_BLOCKS 64
#define MORI_CCL_TDM_A2A_SYNC_GO MORI_CCL_TDM_A2A_MAX_BLOCKS
#define MORI_CCL_TDM_A2A_SYNC_CALL (MORI_CCL_TDM_A2A_MAX_BLOCKS + 1)
#define MORI_CCL_TDM_A2A_SYNC_WORDS (MORI_CCL_TDM_A2A_MAX_BLOCKS + 2)

// The SDMA kernel's arguments plus sync, which the host class does not own. MUST AGREE WITH
// _launch_tdm in python/mori/ccl/collective.py, which copies the 72 bytes prepare_sync fills and
// writes the sync pointer after them.
struct CclTdmAll2allArgs {
  CclAll2allArgs<uint32_t> base;
  uint64_t* sync;
};
static_assert(offsetof(CclTdmAll2allArgs, sync) == 72, "collective.py writes sync at byte 72");
static_assert(sizeof(CclTdmAll2allArgs) == 80, "collective.py packs 80 bytes");

namespace detail {

// One warp copies n <= MORI_CCL_TDM_A2A_TILE_ELEMS four-byte elements from local memory to dst
// through its LDS tile. TDM takes whole rows, at least two of them (the EP planner in
// src/ops/dispatch_combine_v2/tdm_align.hpp never issues a one-row shape either), and only when both
// addresses sit on a 128 B boundary. The lanes copy the rest.
__device__ __forceinline__ void CclTdmA2aWarpCopy(uint32_t* dst, const uint32_t* src, size_t n,
                                                  uint32_t* ldsTile, int laneId, int waveSize) {
  size_t done = 0;
  const uintptr_t addrBits = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  const int rows = static_cast<int>(n / MORI_CCL_TDM_A2A_TILE_W);
  if ((addrBits & 127) == 0 && rows >= 2) {
    const gfx1250_TDM_GROUP1 g1 = CclTdmShape2D(MORI_CCL_TDM_A2A_TILE_W, rows);
    CclTdmIssueLoad(ldsTile, src, g1);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    CclTdmIssueStore(dst, ldsTile, g1);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    done = static_cast<size_t>(rows) * MORI_CCL_TDM_A2A_TILE_W;
  }
  for (size_t i = done + laneId; i < n; i += waveSize) dst[i] = src[i];
}

__device__ __forceinline__ void CclTdmA2aWaitPeer(const uint64_t* flag, uint64_t token, int myPe,
                                                  int peer) {
  int spin = 0;
  bool warned = false;
  while (LoadControlFlag(flag) < token) {
    if (++spin > 100000000 && !warned) {
      printf("PE %d: TdmAll2all timeout waiting for peer %d (token %llu)\n", myPe, peer,
             static_cast<unsigned long long>(token));
      warned = true;
    }
  }
}

// Relaxed polls and one acquire after the last: an acquire on every poll measured 0.1-0.4 us a call
// slower.
__device__ __forceinline__ void CclTdmA2aWaitLocal(const uint64_t* word, uint64_t value, int myPe,
                                                   const char* what) {
  int spin = 0;
  bool warned = false;
  while (__scoped_atomic_load_n(word, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < value) {
    if (++spin > 100000000 && !warned) {
      printf("PE %d: TdmAll2all timeout waiting for %s (call %llu)\n", myPe, what,
             static_cast<unsigned long long>(value));
      warned = true;
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

}  // namespace detail

template <typename T>
__device__ void TdmAll2allKernel_body(int myPe, int npes, T* input,
                                      const application::SymmMemObjPtr inputTransitMemObj,
                                      const application::SymmMemObjPtr outputTransitMemObj,
                                      const application::SymmMemObjPtr flagsMemObj,
                                      size_t elementCount, uint64_t* sync) {
  static_assert(sizeof(T) % 4 == 0, "TdmAll2allKernel moves 4-byte words");
  (void)inputTransitMemObj;
  if (elementCount == 0 || npes <= 0) return;

  const size_t chunk4 = elementCount * (sizeof(T) / 4);
  const int waveSize = warpSize;
  const int warpId = static_cast<int>(threadIdx.x) / waveSize;
  const int laneId = static_cast<int>(threadIdx.x) % waveSize;
  const int warpsPerBlock = (static_cast<int>(blockDim.x) + waveSize - 1) / waveSize;

  extern __shared__ char _cclTdmSmem[];
  uint32_t* ldsTile = reinterpret_cast<uint32_t*>(_cclTdmSmem) +
                      static_cast<size_t>(warpId) * MORI_CCL_TDM_A2A_TILE_ELEMS;

  uint64_t* flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  uint64_t* go = sync + MORI_CCL_TDM_A2A_SYNC_GO;
  uint64_t* lastCall = sync + MORI_CCL_TDM_A2A_SYNC_CALL;
  const uint64_t grid = gridDim.x;
  // One lane per peer, all in warp 0. Lanes of one warp run the publish together and then spin
  // together, so a lane that is still waiting never holds back another lane's publish.
  const bool peerLane = warpId == 0 && laneId < npes && laneId != myPe;

  __shared__ uint64_t s_call;
  if (threadIdx.x == 0) {
    s_call = __scoped_atomic_load_n(lastCall, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1ULL;
  }
  __syncthreads();
  const uint64_t call = s_call;
  const uint64_t startToken = 2ULL * call - 1ULL;
  const uint64_t doneToken = 2ULL * call;

  if (blockIdx.x == 0) {
    if (peerLane) {
      PublishControlFlag(flagsMemObj, myPe, startToken, laneId);
      detail::CclTdmA2aWaitPeer(flags + laneId, startToken, myPe, laneId);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(go, call, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    }
  } else {
    if (threadIdx.x == 0) detail::CclTdmA2aWaitLocal(go, call, myPe, "block 0 to see every peer");
    __syncthreads();
  }

  // Item i is tile i / npes of one destination's chunk. Warp ids are warp-major (warp * grid +
  // block), so one trip of the loop is spread over every block. When grid is a multiple of npes the
  // destination is also rotated by i / grid, so a block's warps write to different peers rather
  // than all to the same one; (tile, destination) stays one-to-one because i / grid is constant
  // over the npes items of a tile.
  {
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(input);
    uint8_t* localOut = reinterpret_cast<uint8_t*>(outputTransitMemObj->localPtr);
    const size_t tilesPerChunk =
        (chunk4 + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) / MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t items = tilesPerChunk * static_cast<size_t>(npes);
    const size_t totalWarps = grid * static_cast<size_t>(warpsPerBlock);
    const bool rotate = (grid % static_cast<uint64_t>(npes)) == 0;

    for (size_t i = static_cast<size_t>(warpId) * grid + blockIdx.x; i < items; i += totalWarps) {
      const size_t tile = i / npes;
      const int destPe = static_cast<int>(rotate ? (i + i / grid) % npes : i % npes);
      const size_t off = tile * MORI_CCL_TDM_A2A_TILE_ELEMS;
      const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                           ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                           : chunk4 - off;
      // localPtr for the self chunk, so this does not depend on p2pPeerPtrs[myPe] being populated.
      uint8_t* dstRoot =
          destPe == myPe ? localOut : outputTransitMemObj->GetAs<uint8_t*>(destPe);
      uint32_t* dst =
          reinterpret_cast<uint32_t*>(dstRoot) + static_cast<size_t>(myPe) * chunk4 + off;
      detail::CclTdmA2aWarpCopy(dst, src4 + static_cast<size_t>(destPe) * chunk4 + off, n, ldsTile,
                                laneId, waveSize);
    }
  }

  // s_wait_tensorcnt(0) retired the engine; it does not order memory. The system-scope fence makes
  // this block's stores, tiles and lane remainders alike, visible before its slot goes up, and
  // __syncthreads() makes thread 0's fence cover every warp of the block.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    __scoped_atomic_store_n(sync + blockIdx.x, call, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
  }
  if (blockIdx.x != 0) return;

  if (threadIdx.x < grid) detail::CclTdmA2aWaitLocal(sync + threadIdx.x, call, myPe, "its blocks");
  __syncthreads();
  __threadfence_system();
  if (peerLane) {
    PublishControlFlag(flagsMemObj, myPe, doneToken, laneId);
    detail::CclTdmA2aWaitPeer(flags + laneId, doneToken, myPe, laneId);
  }
  __syncthreads();
  // Peers fenced their bytes before raising the token; match it before anything reads them.
  __threadfence_system();
  if (threadIdx.x == 0) {
    __scoped_atomic_store_n(lastCall, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

}  // namespace collective
}  // namespace mori
