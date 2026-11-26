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
//                peer's output transit buffer and its own chunk straight into output. A warp
//                raises a tile's word on the receiver as soon as the tile is out, and a warp whose
//                own tiles are out copies tiles whose word is up from the half into output, so
//                the copy overlaps the push (detail::CclTdmA2aCopyOverlapped). Consecutive calls
//                write different halves, so nobody waits for a peer before writing. Calls of at
//                least kCclTdmA2aCopyAfterPushMinBytes a rank push every tile first, raise one
//                word per (peer, block) and copy afterwards (detail::CclTdmA2aCopyAfterPush).
//                If every rank registered its output (outputMemObj), those calls land in the
//                peers' outputs directly, after the start handshake of the next entry, and
//                nothing is copied.
//   output null  at the start of the output transit buffer, which the caller reads after the
//                kernel. Block 0 exchanges a start token with every peer and then lets the other
//                blocks go, so a rank writes into a peer only once the peer has started the call,
//                which means the peer is done with the previous result.
// Calls for which collective.py sets llElems skip all of the below and move as LL128 lines
// (detail::CclTdmA2aLl128): every rank stores its chunks into the LL128 area of each peer's output
// transit buffer, and every rank polls its own area and writes what lands into the result, in
// either mode. Such a call raises no word and exchanges no token.
//
// Every rank deals the tiles to its blocks the same way, and a tile's destination does not depend
// on the sender, so the tiles that block b of rank s writes into rank r are the ones block b of
// rank r has to see land. Completion therefore never gathers a rank's blocks. With output null it
// goes block to block: a block that wrote anything fences its writes and raises its word on every
// peer, and a block with tiles landing here waits for the same block's word from every peer. With
// output set it goes tile by tile, and every rank waits for at least one tile from every peer,
// which is what lets call g + 1 write without waiting: a peer that has raised a word for call g
// has started call g, so it is done with call g - 1, the last call to read the half that call
// g + 1 writes.
//
// Every value below only grows, so nothing is cleared after a call and a slow rank cannot miss one.
//   words     u64 that collective.py reserves past the caller's room in each rank's output transit
//             buffer and zeroes. Output set: npes x tileStride of them, kCclTdmA2aWordGap apart,
//             where tileStride is the number of tiles in halfElems / npes elements; word
//             s * tileStride + t is the last call in which rank s finished writing tile t of its
//             chunk here. Output null: npes x gridDim.x; word s * gridDim.x + b is the last call in
//             which block b of rank s finished writing here.
//   flags[s]  in the host class's npes u64 flags, output null only: the last call rank s started.
//   sync      a per-handle array in ordinary device memory, allocated zeroed by collective.py: the
//             last call of each block, from which the block reads its call number, and a go word
//             that block 0 sets to g once every peer has started call g (output null).
// Measured on 4 gfx1250 GPUs (256 CUs each) with graphs of 100 calls: exchanging one word with
// every peer took about 3 us a call from block 0 alone and about 9 us from all 64 blocks, a fence
// each, so with output null only the blocks that have tiles to send or to receive take part.

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
// MUST AGREE WITH _TDM_A2A_TILE_BYTES in python/mori/ccl/collective.py, which reserves this much
// LDS per warp; reserving less is an LDS overrun.
#define MORI_CCL_TDM_A2A_TILE_W 256
#define MORI_CCL_TDM_A2A_TILE_H 16
#define MORI_CCL_TDM_A2A_TILE_ELEMS (MORI_CCL_TDM_A2A_TILE_W * MORI_CCL_TDM_A2A_TILE_H)
#define MORI_CCL_TDM_A2A_TILE_BYTES (MORI_CCL_TDM_A2A_TILE_ELEMS * 4)

// Layout of sync, in u64 words. MUST AGREE WITH _TDM_A2A_SYNC_WORDS and _TDM_A2A_MAX_BLOCKS in
// python/mori/ccl/collective.py: the grid may exceed neither the block slots nor the words that
// collective.py reserves.
#define MORI_CCL_TDM_A2A_MAX_BLOCKS 128
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
  // 4-byte elements from the start of the output transit buffer to the LL128 area, a multiple of
  // 128, or 0 to move the call through TDM.
  uint64_t llElems;
  // 4-byte elements from the start of one half of the LL128 area to the next, a multiple of 128.
  uint64_t llHalfElems;
  // `output` as a symmetric object when every rank registered its output
  // (All2allSdma::register_output_buffer), a null object otherwise. Large calls then land in the
  // peers' outputs directly.
  application::SymmMemObjPtr outputMemObj;
};
static_assert(offsetof(CclTdmAll2allArgs, sync) == 72, "collective.py writes sync at byte 72");
static_assert(offsetof(CclTdmAll2allArgs, output) == 80, "collective.py writes output at byte 80");
static_assert(offsetof(CclTdmAll2allArgs, halfElems) == 88,
              "collective.py writes halfElems at byte 88");
static_assert(offsetof(CclTdmAll2allArgs, wordsElems) == 96,
              "collective.py writes wordsElems at byte 96");
static_assert(offsetof(CclTdmAll2allArgs, llElems) == 104,
              "collective.py writes llElems at byte 104");
static_assert(offsetof(CclTdmAll2allArgs, llHalfElems) == 112,
              "collective.py writes llHalfElems at byte 112");
static_assert(offsetof(CclTdmAll2allArgs, outputMemObj) == 120,
              "collective.py writes outputMemObj at byte 120");
static_assert(sizeof(CclTdmAll2allArgs) == 136, "collective.py packs 136 bytes");

namespace detail {

// One warp copies n <= MORI_CCL_TDM_A2A_TILE_ELEMS four-byte elements from src to dst through its
// LDS tile. TDM takes whole rows, at least two of them (the EP planner in
// src/ops/dispatch_combine_v2/tdm_align.hpp never issues a one-row shape either), and only when
// both addresses sit on a 128 B boundary. The lanes copy the rest.
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

// CclTdmA2aWarpCopy with system scope on every store; on the TDM store that is bits 3-4 of the
// builtin's last operand. Returns whether the lanes stored any of it.
__device__ __forceinline__ bool CclTdmA2aWarpCopySys(uint32_t* dst, const uint32_t* src, size_t n,
                                                     uint32_t* ldsTile, int laneId, int waveSize) {
  typedef int _v4i __attribute__((ext_vector_type(4)));
  typedef int _v8i __attribute__((ext_vector_type(8)));
  constexpr int kCpolScopeSys = 0x18;
  size_t done = 0;
  const uintptr_t addrBits = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  const int rows = static_cast<int>(n / MORI_CCL_TDM_A2A_TILE_W);
  if ((addrBits & 127) == 0 && rows >= 2) {
    const gfx1250_TDM_GROUP1 g1 = CclTdmShape2D(MORI_CCL_TDM_A2A_TILE_W, rows);
    CclTdmIssueLoad(ldsTile, src, g1);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    gfx1250_TDM_GROUP0 g0;
    g0.ldsAddr(reinterpret_cast<uintptr_t>(ldsTile));
    g0.globalAddr(reinterpret_cast<uintptr_t>(dst));
    _v4i z4{0, 0, 0, 0};
    _v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_store_from_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, kCpolScopeSys);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    done = static_cast<size_t>(rows) * MORI_CCL_TDM_A2A_TILE_W;
  }
  for (size_t i = done + laneId; i < n; i += waveSize) {
    __scoped_atomic_store_n(dst + i, src[i], __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  return done < n;
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

// LL128: a 128-byte line holds 120 payload bytes and, in its last 8, the number of the call it
// belongs to. A receiver that reads its own call number in every line of a unit knows the unit's
// payload is this call's, so the path needs no fence and no word. One wave32 store of 16 bytes a
// lane fills a 512-byte unit of 4 lines and 480 payload bytes: lane l writes slot l % 8 of line
// l / 8, and slot 7 carries 8 payload bytes and the call number. This relies on a peer seeing a
// line that one store wrote either whole or not at all.
constexpr size_t kCclLlUnitBytes = 512;
constexpr size_t kCclLlLinePayload = 120;
constexpr size_t kCclLlUnitPayload = 480;
typedef int CclLlV4i __attribute__((ext_vector_type(4)));

// Eight bytes at byte h (a multiple of 8) of a chunk of `limit` bytes (a multiple of 4), zeros past
// its end. al8: the chunk starts on an 8-byte boundary.
__device__ __forceinline__ uint2 CclLlLoad8(const char* src, size_t h, size_t limit, bool al8) {
  if (h + 8 <= limit) {
    if (al8) return *reinterpret_cast<const uint2*>(src + h);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(src + h);
    return make_uint2(p[0], p[1]);
  }
  if (h < limit) return make_uint2(*reinterpret_cast<const uint32_t*>(src + h), 0u);
  return make_uint2(0u, 0u);
}

__device__ __forceinline__ void CclLlStore8(char* dst, size_t h, uint2 v, size_t limit, bool al8) {
  if (h + 8 <= limit) {
    if (al8) {
      *reinterpret_cast<uint2*>(dst + h) = v;
    } else {
      uint32_t* p = reinterpret_cast<uint32_t*>(dst + h);
      p[0] = v.x;
      p[1] = v.y;
    }
  } else if (h < limit) {
    *reinterpret_cast<uint32_t*>(dst + h) = v.x;
  }
}

// The whole call as LL128 lines, wave32 only. Rank s's lines for call c sit in half c & 1 of every
// receiver's LL128 area (llOffBytes is this call's half), unit u of its chunk at byte
// (s * units + u) * 512 of the half. A rank stores call c + 2 into a half only after finishing
// call c + 1, which needed every peer to have started call c + 1 and so to be done reading call c
// out of that half; a TDM call in between orders it the same way, as it too waits for every peer.
// The area holds nothing but lines and is zeroed before the first call, so a line left there by an
// earlier call always shows an older call number. Warp `warp` takes every totalWarps-th pair from
// itself on, sending and then receiving: pair (k, u) is unit u of the chunk for rank myPe + 1 + k
// and of the chunk from rank myPe - 1 - k.
__device__ __forceinline__ void CclTdmA2aLl128(
    int myPe, int npes, const void* input, void* result,
    const application::SymmMemObjPtr& outputTransitMemObj, size_t chunkBytes, uint64_t call,
    size_t llOffBytes, size_t warp, size_t totalWarps, int laneId, int waveSize) {
  const size_t units = (chunkBytes + kCclLlUnitPayload - 1) / kCclLlUnitPayload;
  const size_t np = static_cast<size_t>(npes);
  const size_t me = static_cast<size_t>(myPe);
  const size_t pairs = (np - 1) * units;
  const int lp = laneId & 7;
  const size_t laneOff =
      static_cast<size_t>(laneId >> 3) * kCclLlLinePayload + static_cast<size_t>(lp) * 16;
  const char* const in8 = static_cast<const char*>(input);
  char* const out8 = static_cast<char*>(result);
  const bool inAl8 = ((reinterpret_cast<uintptr_t>(input) | chunkBytes) & 7) == 0;
  const bool outAl8 = ((reinterpret_cast<uintptr_t>(result) | chunkBytes) & 7) == 0;

  for (size_t i = warp; i < pairs; i += totalWarps) {
    const size_t k = i / units;
    const size_t u = i - k * units;
    const size_t dest = (me + 1 + k) % np;
    const char* const src = in8 + dest * chunkBytes;
    const size_t h = u * kCclLlUnitPayload + laneOff;
    const uint2 a = CclLlLoad8(src, h, chunkBytes, inAl8);
    const uint2 b = lp == 7
                        ? make_uint2(static_cast<uint32_t>(call), static_cast<uint32_t>(call >> 32))
                        : CclLlLoad8(src, h + 8, chunkBytes, inAl8);
    char* const dst =
        reinterpret_cast<char*>(outputTransitMemObj->GetAs<uint8_t*>(static_cast<int>(dest))) +
        llOffBytes + (me * units + u) * kCclLlUnitBytes + static_cast<size_t>(laneId) * 16;
    *reinterpret_cast<uint4*>(dst) = make_uint4(a.x, a.y, b.x, b.y);
  }

  {
    const size_t chunk4 = chunkBytes / 4;
    const uint32_t* const s4 = reinterpret_cast<const uint32_t*>(in8 + me * chunkBytes);
    uint32_t* const d4 = reinterpret_cast<uint32_t*>(out8 + me * chunkBytes);
    const bool al16 =
        ((reinterpret_cast<uintptr_t>(s4) | reinterpret_cast<uintptr_t>(d4)) & 15) == 0;
    const size_t per = static_cast<size_t>(waveSize) * 4;
    for (size_t j = warp * per; j < chunk4; j += totalWarps * per) {
      const size_t e = j + static_cast<size_t>(laneId) * 4;
      if (al16 && e + 4 <= chunk4) {
        *reinterpret_cast<uint4*>(d4 + e) = *reinterpret_cast<const uint4*>(s4 + e);
      } else {
        for (size_t q = e; q < e + 4 && q < chunk4; ++q) d4[q] = s4[q];
      }
    }
  }

  const char* const land =
      reinterpret_cast<const char*>(outputTransitMemObj->localPtr) + llOffBytes;
  for (size_t i = warp; i < pairs; i += totalWarps) {
    const size_t k = i / units;
    const size_t u = i - k * units;
    const size_t srcPe = (me + np - 1 - k) % np;
    const CclLlV4i* const p = reinterpret_cast<const CclLlV4i*>(
        land + (srcPe * units + u) * kCclLlUnitBytes + static_cast<size_t>(laneId) * 16);
    CclLlV4i x;
    int spin = 0;
    bool warned = false;
    while (true) {
      asm volatile("" ::: "memory");
      x = __builtin_nontemporal_load(p);
      const uint64_t seen =
          (static_cast<uint64_t>(static_cast<uint32_t>(x.w)) << 32) | static_cast<uint32_t>(x.z);
      if (__ballot(lp == 7 && seen != call) == 0ull) break;
      if (++spin > 100000000 && !warned) {
        if (laneId == 0) {
          printf("PE %d: TdmAll2all timeout waiting for LL128 lines from peer %d (call %llu)\n",
                 myPe, static_cast<int>(srcPe), static_cast<unsigned long long>(call));
        }
        warned = true;
      }
    }
    char* const dst = out8 + srcPe * chunkBytes;
    const size_t h = u * kCclLlUnitPayload + laneOff;
    CclLlStore8(dst, h, make_uint2(static_cast<uint32_t>(x.x), static_cast<uint32_t>(x.y)),
                chunkBytes, outAl8);
    if (lp != 7) {
      CclLlStore8(dst, h + 8, make_uint2(static_cast<uint32_t>(x.z), static_cast<uint32_t>(x.w)),
                  chunkBytes, outAl8);
    }
  }
}

// With output set, consecutive words of a peer sit this many u64 apart, so no two share a 128-byte
// line. Measured on 4 gfx1250 GPUs: words alone, exchanged by 64 blocks, took 9.7 us a call 8 bytes
// apart and 6.0 us 256 bytes apart. MUST AGREE WITH _TDM_A2A_WORD_GAP_BYTES in
// python/mori/ccl/collective.py, which reserves the words.
constexpr size_t kCclTdmA2aWordGap = 32;

// The whole TDM call with output set. Items are dealt as in the zero-copy path of
// TdmAll2allKernel_body. After each tile for a peer, the warp raises that tile's word on the peer.
// tileStride comes from the half, not the call, so a peer that is already in the next call, with
// another count, never raises a word this call waits on. A warp whose own tiles are out takes
// (landing tile, source) pairs from the LDS counter `next`, in the order the peers push them,
// waits for the tile's word and copies the tile from the half into output: the tiles that landed
// first are copied while later ones are still being pushed. Measured on 4 gfx1250 GPUs with graphs
// of 100 calls at 1 GiB a rank: 814.0 us a call, against 883.5 when every block waited for all of
// its tiles before copying any; leaving the result in the transit buffer took 613.8.
__device__ __forceinline__ void CclTdmA2aCopyOverlapped(
    int myPe, int npes, const uint32_t* src4, uint32_t* out4, uint32_t* localTransit,
    const application::SymmMemObjPtr& outputTransitMemObj, size_t chunk4, uint64_t call,
    size_t halfElems, size_t wordsElems, size_t grid, size_t blk, int warpId, size_t totalWarps,
    unsigned* next, uint32_t* ldsTile, int laneId, int waveSize) {
  const size_t np = static_cast<size_t>(npes);
  const size_t me = static_cast<size_t>(myPe);
  const size_t items =
      (chunk4 + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) / MORI_CCL_TDM_A2A_TILE_ELEMS * np;
  const bool rotate = grid % np == 0;
  const size_t half = static_cast<size_t>(call & 1ULL) * halfElems;
  const size_t tileStride =
      (halfElems / np + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) / MORI_CCL_TDM_A2A_TILE_ELEMS;

  for (size_t i = static_cast<size_t>(warpId) * grid + blk; i < items; i += totalWarps) {
    const size_t tile = i / np;
    const size_t destPe = rotate ? (i + i / grid) % np : i % np;
    const size_t off = tile * MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                         ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                         : chunk4 - off;
    const uint32_t* const src = src4 + destPe * chunk4 + off;
    if (destPe == me) {
      CclTdmA2aWarpCopy(out4 + me * chunk4 + off, src, n, ldsTile, laneId, waveSize);
      continue;
    }
    uint32_t* const peer =
        reinterpret_cast<uint32_t*>(outputTransitMemObj->GetAs<uint8_t*>(static_cast<int>(destPe)));
    // With system scope on the TDM store the tile is not left in this GPU's cache, so the word
    // follows s_wait_tensorcnt(0) without the system fence of the zero-copy path, whose global_wb
    // writes this GPU's dirty lines back: a fence per tile measured 6.4 us a call slower at 32 MiB
    // a rank. Lanes' stores are not counted by tensorcnt, so a tile that has any takes the fence.
    if (CclTdmA2aWarpCopySys(peer + half + me * chunk4 + off, src, n, ldsTile, laneId, waveSize)) {
      __threadfence_system();
    }
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    if (laneId == 0) {
      uint64_t* const word = reinterpret_cast<uint64_t*>(peer + wordsElems) +
                             (me * tileStride + tile) * kCclTdmA2aWordGap;
      __scoped_atomic_store_n(word, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
  }

  // This block pushes items blk + grid * j on every rank, and the ones bound here, which are the
  // ones it receives, are those with j = j0, j0 + period, ...
  size_t j0 = np;
  for (size_t j = 0; j < np && j0 == np; ++j) {
    const size_t i = blk + grid * j;
    if ((rotate ? (i + i / grid) % np : i % np) == me) j0 = j;
  }
  if (j0 == np) return;
  size_t period = np;
  for (size_t j = j0 + 1; j < j0 + np && period == np; ++j) {
    const size_t i = blk + grid * j;
    if ((rotate ? (i + i / grid) % np : i % np) == me) period = j - j0;
  }
  const size_t first = blk + grid * j0;
  const size_t stride = grid * period;
  const size_t landing = first < items ? (items - 1 - first) / stride + 1 : 0;
  const size_t peers = np - 1;
  const size_t pairs = landing * peers;
  const uint32_t* const from = localTransit + half;
  const uint64_t* const words = reinterpret_cast<const uint64_t*>(localTransit + wordsElems);
  while (true) {
    unsigned q32 = 0;
    if (laneId == 0) q32 = atomicAdd(next, 1u);
    const size_t q = __shfl(q32, 0);
    if (q >= pairs) break;
    const size_t tile = (first + stride * (q / peers)) / np;
    const size_t k = q % peers;
    const size_t srcPe = k < me ? k : k + 1;
    const size_t off = tile * MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                         ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                         : chunk4 - off;
    if (laneId == 0) {
      CclTdmA2aWaitPeer(words + (srcPe * tileStride + tile) * kCclTdmA2aWordGap, call, myPe,
                        static_cast<int>(srcPe));
    }
    // Another GPU wrote the tile, which an agent-scope acquire does not cover.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    const size_t at = srcPe * chunk4 + off;
    CclTdmA2aWarpCopy(out4 + at, from + at, n, ldsTile, laneId, waveSize);
  }
}

// Calls of at least this many bytes a rank (count * npes * 4) with output set go through
// CclTdmA2aCopyAfterPush instead of CclTdmA2aCopyOverlapped. MUST AGREE WITH
// _TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES in python/mori/ccl/collective.py, which only documents and
// tests the boundary.
constexpr size_t kCclTdmA2aCopyAfterPushMinBytes = 64ULL << 20;

// The whole TDM call with output set, for large calls: every warp pushes its tiles, then the block
// raises one word on every peer, and then the same warps copy what landed into output.
//
// Measured on 4 gfx1250 GPUs with graphs of 100 calls at 1 GiB a rank. Inside a compute unit a copy
// does not overlap a push, wherever its warps sit: 4 warps pushing took 570.9 us, 4 other warps
// copying took 236.4 us, and both together 857.4 us. A word per tile cost the pushing warp 2.5 us
// a tile (812.6 us against 570.9). So the words CclTdmA2aCopyOverlapped pays for buy no overlap
// once the push is long, and one word a block does the same job: 700.3 us a call against 813.6,
// 185.5 against 211.9 at 256 MiB a rank and 57.5 against 61.3 at 64 MiB. At 32 and 16 MiB a rank
// the exchange between all blocks costs more than it saves (37.4 us against 36.6, 27.8 against
// 22.8), which is where the bound above comes from.
//
// Item i = t * (npes - 1) + k is tile t of the chunk for rank (myPe + 1 + k) % npes, and warp
// g = warpId * grid + blk takes items g, g + totalWarps, ... The tile that lands here under index i
// comes from rank (myPe - 1 - k) mod npes and was pushed by warp g of that rank, so the words of
// block blk cover exactly the tiles block blk copies. The word of (rank s, block b) is u64
// (s * wordStride + b) * wordGap of the words: with output set the caller passes the per-tile
// words' stride and gap and guarantees that grid of them exist per rank; values only grow, so a
// call through either function never mistakes the other's word for its own.
// Every block raises and waits, so a rank ends call g only once every peer has pushed all of call
// g, which orders the halves as CclTdmA2aCopyOverlapped does. Needs npes >= 2.
//
// The tiles go to element peerOff of each peer's output transit buffer and the chunk this rank
// keeps to res4. With `from` set, what landed is then copied from `from` into res4; with nullptr
// the result stays where it landed (the zero-copy call, whose res4 is the transit buffer). There
// the same order of work measured 593.8 us a call at 1 GiB a rank against 614.3 for the loop that
// mixes this rank's own tiles into the push and lets only the blocks with tiles raise and wait,
// 162.6 against 171.1 at 256 MiB, 36.9 against 38.8 at 32 MiB, and 29.9 against 28.7 at 16 MiB.
//
// landObj is where the tiles land on a peer: the output transit buffer, or the outputs the ranks
// registered, in which case peerOff is 0, res4 is this rank's registered output and nothing is
// copied either. The words always live in the output transit buffer.
constexpr size_t kCclTdmA2aPushFirstMinBytes = 32ULL << 20;
__device__ __forceinline__ void CclTdmA2aCopyAfterPush(
    int myPe, int npes, const uint32_t* src4, uint32_t* res4, const uint32_t* from,
    uint32_t* localTransit, const application::SymmMemObjPtr& landObj,
    const application::SymmMemObjPtr& outputTransitMemObj, size_t chunk4, uint64_t call,
    size_t peerOff, size_t wordsElems, size_t wordStride, size_t wordGap, size_t grid, size_t blk,
    int warpId, size_t totalWarps, uint32_t* ldsTile, int laneId, int waveSize) {
  const size_t np = static_cast<size_t>(npes);
  const size_t me = static_cast<size_t>(myPe);
  const size_t peers = np - 1;
  const size_t tiles = (chunk4 + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) / MORI_CCL_TDM_A2A_TILE_ELEMS;
  const size_t items = tiles * peers;
  const size_t first = static_cast<size_t>(warpId) * grid + blk;

  for (size_t i = first; i < items; i += totalWarps) {
    const size_t destPe = (me + 1 + i % peers) % np;
    const size_t off = i / peers * MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                         ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                         : chunk4 - off;
    uint32_t* const peer =
        reinterpret_cast<uint32_t*>(landObj->GetAs<uint8_t*>(static_cast<int>(destPe)));
    CclTdmA2aWarpCopySys(peer + peerOff + me * chunk4 + off, src4 + destPe * chunk4 + off, n,
                         ldsTile, laneId, waveSize);
  }

  // One lane per peer, all in warp 0, as in the zero-copy path. s_wait_tensorcnt(0) retired the
  // engine and the tiles went out at system scope; the fence of warp 0 after the barrier covers
  // what the lanes of any warp stored of a tile's tail.
  const bool peerLane = warpId == 0 && laneId < npes && laneId != myPe;
  __syncthreads();
  if (warpId == 0) {
    __threadfence_system();
    if (peerLane) {
      uint64_t* const word =
          reinterpret_cast<uint64_t*>(
              reinterpret_cast<uint32_t*>(outputTransitMemObj->GetAs<uint8_t*>(laneId)) +
              wordsElems) +
          (me * wordStride + blk) * wordGap;
      __scoped_atomic_store_n(word, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
  }

  // The chunk this rank keeps, while the peers' words are on their way.
  for (size_t t = first; t < tiles; t += totalWarps) {
    const size_t off = t * MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                         ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                         : chunk4 - off;
    const size_t at = me * chunk4 + off;
    CclTdmA2aWarpCopy(res4 + at, src4 + at, n, ldsTile, laneId, waveSize);
  }

  if (peerLane) {
    CclTdmA2aWaitPeer(reinterpret_cast<const uint64_t*>(localTransit + wordsElems) +
                          (static_cast<size_t>(laneId) * wordStride + blk) * wordGap,
                      call, myPe, laneId);
  }
  __syncthreads();
  // Other GPUs wrote the tiles, which an agent-scope acquire does not cover.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  if (from == nullptr) return;
  for (size_t i = first; i < items; i += totalWarps) {
    const size_t srcPe = (me + np - 1 - i % peers) % np;
    const size_t off = i / peers * MORI_CCL_TDM_A2A_TILE_ELEMS;
    const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                         ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                         : chunk4 - off;
    const size_t at = srcPe * chunk4 + off;
    CclTdmA2aWarpCopy(res4 + at, from + at, n, ldsTile, laneId, waveSize);
  }
}

// A rank writes into memory a peer's caller reads between calls (the start of the output transit
// buffer, or a registered output) only once that peer has started the call. Block 0 exchanges a
// start token with every peer, one lane per peer so that a lane still waiting never holds back
// another lane's publish, and then lets the other blocks go through the go word.
__device__ __forceinline__ void CclTdmA2aStart(int myPe, int npes,
                                               const application::SymmMemObjPtr& flagsMemObj,
                                               uint64_t* go, uint64_t call, size_t blk, int warpId,
                                               int laneId) {
  if (blk == 0) {
    if (warpId == 0 && laneId < npes && laneId != myPe) {
      PublishControlFlag(flagsMemObj, myPe, call, laneId);
      CclTdmA2aWaitPeer(reinterpret_cast<const uint64_t*>(flagsMemObj->localPtr) + laneId, call,
                        myPe, laneId);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(go, call, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    }
  } else {
    if (threadIdx.x == 0) CclTdmA2aWaitLocal(go, call, myPe, "block 0 to see every peer");
    __syncthreads();
  }
}

}  // namespace detail

template <typename T>
__device__ __forceinline__ void TdmAll2allKernel_body(
    int myPe, int npes, T* input, const application::SymmMemObjPtr inputTransitMemObj,
    const application::SymmMemObjPtr outputTransitMemObj,
    const application::SymmMemObjPtr flagsMemObj, size_t elementCount, uint64_t* sync, T* output,
    size_t halfElems, size_t wordsElems, size_t llElems, size_t llHalfElems,
    const application::SymmMemObjPtr outputMemObj) {
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

  __shared__ uint64_t s_call;
  __shared__ unsigned s_nextPair;
  if (threadIdx.x == 0) {
    s_call = __scoped_atomic_load_n(sync + blk, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1ULL;
    s_nextPair = 0;
  }
  __syncthreads();
  const uint64_t call = s_call;

  if (llElems != 0) {
    detail::CclTdmA2aLl128(myPe, npes, input, out4 != nullptr ? out4 : localTransit,
                           outputTransitMemObj, chunk4 * 4, call,
                           (llElems + static_cast<size_t>(call & 1ULL) * llHalfElems) * 4,
                           static_cast<size_t>(warpId) * grid + blk, totalWarps, laneId, waveSize);
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(sync + blk, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    return;
  }

  if (out4 != nullptr) {
    // The same on every rank: it depends on the call's size and the handle's half only.
    const size_t tileStride =
        (halfElems / static_cast<size_t>(npes) + MORI_CCL_TDM_A2A_TILE_ELEMS - 1) /
        MORI_CCL_TDM_A2A_TILE_ELEMS;
    const bool afterPush =
        npes >= 2 &&
        chunk4 * 4 * static_cast<size_t>(npes) >= detail::kCclTdmA2aCopyAfterPushMinBytes &&
        tileStride >= grid;
    if (afterPush && outputMemObj.gpu != nullptr) {
      // Every rank registered its output: the peers' tiles land in it, after the start handshake.
      detail::CclTdmA2aStart(myPe, npes, flagsMemObj, sync + MORI_CCL_TDM_A2A_SYNC_GO, call, blk,
                             warpId, laneId);
      detail::CclTdmA2aCopyAfterPush(
          myPe, npes, reinterpret_cast<const uint32_t*>(input), out4, nullptr, localTransit,
          outputMemObj, outputTransitMemObj, chunk4, call, 0, wordsElems, tileStride,
          detail::kCclTdmA2aWordGap, grid, blk, warpId, totalWarps, ldsTile, laneId, waveSize);
    } else if (afterPush) {
      const size_t half = static_cast<size_t>(call & 1ULL) * halfElems;
      detail::CclTdmA2aCopyAfterPush(myPe, npes, reinterpret_cast<const uint32_t*>(input), out4,
                                     localTransit + half, localTransit, outputTransitMemObj,
                                     outputTransitMemObj, chunk4, call, half, wordsElems,
                                     tileStride, detail::kCclTdmA2aWordGap, grid, blk, warpId,
                                     totalWarps, ldsTile, laneId, waveSize);
    } else {
      detail::CclTdmA2aCopyOverlapped(myPe, npes, reinterpret_cast<const uint32_t*>(input), out4,
                                      localTransit, outputTransitMemObj, chunk4, call, halfElems,
                                      wordsElems, grid, blk, warpId, totalWarps, &s_nextPair,
                                      ldsTile, laneId, waveSize);
    }
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(sync + blk, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    return;
  }

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

  detail::CclTdmA2aStart(myPe, npes, flagsMemObj, sync + MORI_CCL_TDM_A2A_SYNC_GO, call, blk,
                         warpId, laneId);

  // Large calls: push, one word per (peer, block), and nothing to copy. The words are the same
  // ones the loop below raises, word (rank, block) at u64 rank * grid + block.
  if (npes >= 2 && chunk4 * 4 * static_cast<size_t>(npes) >= detail::kCclTdmA2aPushFirstMinBytes) {
    detail::CclTdmA2aCopyAfterPush(myPe, npes, reinterpret_cast<const uint32_t*>(input),
                                   localTransit, nullptr, localTransit, outputTransitMemObj,
                                   outputTransitMemObj, chunk4, call, 0, wordsElems, grid, 1, grid,
                                   blk, warpId, totalWarps, ldsTile, laneId, waveSize);
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(sync + blk, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    return;
  }

  // Item i is tile i / npes of one destination's chunk. Warp ids are warp-major (warp * grid +
  // block), so one trip of the loop is spread over every block. When grid is a multiple of npes the
  // destination is also rotated by i / grid, so a block's warps write to different peers rather
  // than all to the same one; (tile, destination) stays one-to-one because i / grid is constant
  // over the npes items of a tile.
  {
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(input);
    for (size_t i = static_cast<size_t>(warpId) * grid + blk; i < items; i += totalWarps) {
      const size_t tile = i / npes;
      const int destPe = static_cast<int>(rotate ? (i + i / grid) % npes : i % npes);
      const size_t off = tile * MORI_CCL_TDM_A2A_TILE_ELEMS;
      const size_t n = (off + MORI_CCL_TDM_A2A_TILE_ELEMS <= chunk4)
                           ? static_cast<size_t>(MORI_CCL_TDM_A2A_TILE_ELEMS)
                           : chunk4 - off;
      // localPtr for the self chunk, so this does not depend on p2pPeerPtrs[myPe] being populated.
      uint32_t* dstRoot =
          destPe == myPe
              ? localTransit
              : reinterpret_cast<uint32_t*>(outputTransitMemObj->GetAs<uint8_t*>(destPe));
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
  if (landsHere) {
    if (peerLane) detail::CclTdmA2aWaitPeer(myWord, call, myPe, laneId);
    __syncthreads();
    // Peers fenced their bytes before raising the word; match it before anything reads them. Other
    // GPUs wrote them, which an agent-scope acquire does not cover.
    if (threadIdx.x == 0) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    __scoped_atomic_store_n(sync + blk, call, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

}  // namespace collective
}  // namespace mori
