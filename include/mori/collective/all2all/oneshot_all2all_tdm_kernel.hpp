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

// One-shot all-to-all whose bytes move through the TDM engine; it never touches an SDMA queue.
// gfx125x only: like allreduce/twoshot_tdm_kernel.hpp, it is reached from inside the arch guard in
// ccl_kernels.hip.
//
// Chunk p of `input` (elementCount elements) goes to rank p, at offset myPe * elementCount of that
// rank's result. Where the result ends up is fixed per handle by python/mori/ccl/collective.py:
//   output set   in the caller's output. Every rank writes its chunks into half (call % 2) of each
//                peer's output transit buffer and its own chunk straight into output; once a
//                peer's tiles have landed, the blocks copy them from the half into output.
//                Consecutive calls write different halves, so nobody waits for a peer before
//                writing.
//   output null  at the start of the output transit buffer, which the caller reads after the
//                kernel. Block 0 exchanges a start token with every peer and then lets the other
//                blocks go, so a rank writes into a peer only once the peer has started the call,
//                which means the peer is done with the previous result.
//
// Every rank deals the tiles to its blocks the same way, and a tile's destination does not depend
// on the sender, so the tiles that block b of rank s writes into rank r are the ones block b of rank
// r has to see land. Completion therefore goes block to block, and nothing gathers a rank's blocks:
// a block that wrote anything fences its writes and raises its word on every peer, and a block with
// tiles landing here waits for the same block's word from every peer. With output set, block 0 also
// waits for block 0 of every peer, which is what lets call g + 1 write without waiting: a peer that
// has raised its word for call g has started call g, so it has finished call g - 1, the last call to
// read the half that call g + 1 writes.
//
// Every value below only grows, so nothing is cleared after a call and a slow rank cannot miss one.
//   words     npes x gridDim.x u64 that collective.py reserves past the caller's room in each
//             rank's output transit buffer and zeroes: word s * gridDim.x + b is the last call in
//             which block b of rank s finished writing here.
//   flags[s]  in the host class's npes u64 flags, output null only: the last call rank s started.
//   sync      a per-handle array in ordinary device memory, allocated zeroed by collective.py: the
//             last call of each block, from which the block reads its call number, and a go word
//             that block 0 sets to g once every peer has started call g (output null).
// Measured on 4 gfx1250 GPUs (256 CUs each) with graphs of 100 calls: exchanging one word with every
// peer took about 3 us a call from block 0 alone and about 9 us from all 64 blocks, a fence each, so
// only the blocks that have tiles to send or to receive take part.

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
// python/mori/ccl/collective.py: the grid may exceed neither the block slots nor the words that
// collective.py reserves.
#define MORI_CCL_TDM_A2A_MAX_BLOCKS 64
#define MORI_CCL_TDM_A2A_SYNC_GO MORI_CCL_TDM_A2A_MAX_BLOCKS
#define MORI_CCL_TDM_A2A_SYNC_WORDS (MORI_CCL_TDM_A2A_MAX_BLOCKS + 1)

// The SDMA kernel's arguments plus what the host class does not own. MUST AGREE WITH _launch_tdm
// in python/mori/ccl/collective.py, which copies the 72 bytes prepare_sync fills and writes the
// rest after them.
struct CclTdmAll2allArgs {
  CclAll2allArgs<uint32_t> base;
  uint64_t* sync;
  // The caller's output, or nullptr to leave the result in the output transit buffer.
  void* output;
  // 4-byte elements from the start of one half of the output transit buffer to the next.
  uint64_t halfElems;
  // 4-byte elements from the start of the output transit buffer to the words, an even number.
  uint64_t wordsElems;
};
static_assert(offsetof(CclTdmAll2allArgs, sync) == 72, "collective.py writes sync at byte 72");
static_assert(offsetof(CclTdmAll2allArgs, output) == 80, "collective.py writes output at byte 80");
static_assert(offsetof(CclTdmAll2allArgs, halfElems) == 88,
              "collective.py writes halfElems at byte 88");
static_assert(offsetof(CclTdmAll2allArgs, wordsElems) == 96,
              "collective.py writes wordsElems at byte 96");
static_assert(sizeof(CclTdmAll2allArgs) == 104, "collective.py packs 104 bytes");

namespace detail {

// One warp copies n <= MORI_CCL_TDM_A2A_TILE_ELEMS four-byte elements from src to dst through its
// LDS tile. TDM takes whole rows, at least two of them (the EP planner in
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
                                      size_t elementCount, uint64_t* sync, T* output,
                                      size_t halfElems, size_t wordsElems) {
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

  uint32_t* const out4 = reinterpret_cast<uint32_t*>(output);
  uint32_t* const localTransit = reinterpret_cast<uint32_t*>(outputTransitMemObj->localPtr);
  const uint64_t grid = gridDim.x;
  const size_t blk = blockIdx.x;
  const size_t totalWarps = grid * static_cast<size_t>(warpsPerBlock);
  const size_t tilesPerChunk =
      (chunk4 + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) / MORI_CCL_TDM_A2A_TILE_ELEMS;
  const size_t items = tilesPerChunk * static_cast<size_t>(npes);
  const bool rotate = (grid % static_cast<uint64_t>(npes)) == 0;
  // One lane per peer, all in warp 0. Lanes of one warp run the publish together and then spin
  // together, so a lane that is still waiting never holds back another lane's publish.
  const bool peerLane = warpId == 0 && laneId < npes && laneId != myPe;
  // This block's word on peer laneId, and the same block's word from peer laneId here.
  uint64_t* peerWord = nullptr;
  const uint64_t* myWord = nullptr;
  if (peerLane) {
    peerWord = reinterpret_cast<uint64_t*>(
                   reinterpret_cast<uint32_t*>(outputTransitMemObj->GetAs<uint8_t*>(laneId)) +
                   wordsElems) +
               static_cast<size_t>(myPe) * grid + blk;
    myWord = reinterpret_cast<const uint64_t*>(localTransit + wordsElems) +
             static_cast<size_t>(laneId) * grid + blk;
  }

  __shared__ uint64_t s_call;
  if (threadIdx.x == 0) {
    s_call = __scoped_atomic_load_n(sync + blk, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1ULL;
  }
  __syncthreads();
  const uint64_t call = s_call;
  const size_t half = out4 != nullptr ? static_cast<size_t>(call & 1ULL) * halfElems : 0;

  if (out4 == nullptr) {
    uint64_t* go = sync + MORI_CCL_TDM_A2A_SYNC_GO;
    if (blk == 0) {
      if (peerLane) {
        PublishControlFlag(flagsMemObj, myPe, call, laneId);
        detail::CclTdmA2aWaitPeer(reinterpret_cast<const uint64_t*>(flagsMemObj->localPtr) + laneId,
                                  call, myPe, laneId);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __scoped_atomic_store_n(go, call, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
      }
    } else {
      if (threadIdx.x == 0) detail::CclTdmA2aWaitLocal(go, call, myPe, "block 0 to see every peer");
      __syncthreads();
    }
  }

  // Item i is tile i / npes of one destination's chunk. Warp ids are warp-major (warp * grid +
  // block), so one trip of the loop is spread over every block. When grid is a multiple of npes the
  // destination is also rotated by i / grid, so a block's warps write to different peers rather
  // than all to the same one; (tile, destination) stays one-to-one because i / grid is constant
  // over the npes items of a tile.
  {
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(input);
    // localPtr for the self chunk, so this does not depend on p2pPeerPtrs[myPe] being populated.
    uint32_t* const self = out4 != nullptr ? out4 : localTransit;
    for (size_t i = static_cast<size_t>(warpId) * grid + blk; i < items; i += totalWarps) {
      const size_t tile = i / npes;
      const int destPe = static_cast<int>(rotate ? (i + i / grid) % npes : i % npes);
      const size_t off = tile * MORI_CCL_TDM_A2A_TILE_ELEMS;
      const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                           ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                           : chunk4 - off;
      uint32_t* dstRoot =
          destPe == myPe
              ? self
              : reinterpret_cast<uint32_t*>(outputTransitMemObj->GetAs<uint8_t*>(destPe)) + half;
      detail::CclTdmA2aWarpCopy(dstRoot + static_cast<size_t>(myPe) * chunk4 + off,
                                src4 + static_cast<size_t>(destPe) * chunk4 + off, n, ldsTile,
                                laneId, waveSize);
    }
  }

  // This block owns items blk, blk + grid, ... on every rank; those bound for this rank are the
  // tiles it waits for.
  bool landsHere = false;
  for (size_t i = blk; i < items && !landsHere; i += grid) {
    landsHere = static_cast<int>(rotate ? (i + i / grid) % npes : i % npes) == myPe;
  }

  if (blk < items) {
    // s_wait_tensorcnt(0) retired the engine; it does not order memory. The system-scope fence
    // makes this block's stores, tiles and lane remainders alike, visible before its word goes up,
    // and __syncthreads() makes warp 0's fence cover every warp of the block.
    __syncthreads();
    if (warpId == 0) {
      __threadfence_system();
      if (peerLane) {
        __scoped_atomic_store_n(peerWord, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      }
    }
  }
  if (landsHere || (out4 != nullptr && blk == 0)) {
    if (peerLane) detail::CclTdmA2aWaitPeer(myWord, call, myPe, laneId);
    __syncthreads();
    // Peers fenced their bytes before raising the word; match it before anything reads them. Other
    // GPUs wrote them, which an agent-scope acquire does not cover.
    if (threadIdx.x == 0) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __syncthreads();
  }

  if (out4 != nullptr && landsHere) {
    // This block's tiles from every peer, one (tile, peer) pair a warp in turn.
    const uint32_t* from = localTransit + half;
    const size_t peers = static_cast<size_t>(npes - 1);
    size_t q = 0;
    for (size_t i = blk; i < items; i += grid) {
      if (static_cast<int>(rotate ? (i + i / grid) % npes : i % npes) != myPe) continue;
      const size_t off = (i / npes) * MORI_CCL_TDM_A2A_TILE_ELEMS;
      const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                           ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                           : chunk4 - off;
      for (size_t k = 0; k < peers; ++k, ++q) {
        if (q % static_cast<size_t>(warpsPerBlock) != static_cast<size_t>(warpId)) continue;
        const size_t srcPe = k < static_cast<size_t>(myPe) ? k : k + 1;
        const size_t at = srcPe * chunk4 + off;
        detail::CclTdmA2aWarpCopy(out4 + at, from + at, n, ldsTile, laneId, waveSize);
      }
    }
  }
  if (threadIdx.x == 0) {
    __scoped_atomic_store_n(sync + blk, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

}  // namespace collective
}  // namespace mori
