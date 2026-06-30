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
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// Core ring AllGather data movement, parameterized on an explicit per-chunk
// byte size (``peChunkSize``). Factored out as a ``__device__`` helper so it
// can be reused both by the existing ``__global__`` entry point below (used by
// the inter-node executor, which derives the chunk size from the buffer) and by
// the JIT-launched hierarchical entry point (which passes an explicit chunk
// size so a single fixed-size symmetric ring buffer can serve variable message
// sizes). The ring buffer holds ``npes`` chunks of ``peChunkSize`` bytes; on
// entry only PE ``myPe``'s own chunk is filled. After ``npes-1`` rounds every
// PE holds all chunks in PE order. Equal per-PE chunks => no last-chunk special
// case is needed.
// Sub-group ring AllGather over an arithmetic sub-group of global PEs:
// ``{peBase, peBase+peStride, ..., peBase+(ringSize-1)*peStride}``. This PE is
// at position ``ringPos`` within that sub-group. The ring buffer holds
// ``ringSize`` chunks; on entry only slot ``ringPos`` is filled. After
// ``ringSize-1`` rounds every member holds all ``ringSize`` chunks in ring
// order. The whole-world ring is the special case ``peBase=0, peStride=1``.
//
// This is the inter-node phase of the hierarchical AllGather: the ring runs
// over node-leaders (or same-local-index ranks across nodes), so neighbours are
// reached over RDMA while same-node members go P2P -- exactly the DESIGN intent.
// ``numBlocksOverride`` / ``bidOverride`` let this body run as
// a SUB-RANGE of a larger fused grid. When >=0 they replace the grid-derived
// ``gridDim.x`` / ``blockIdx.x`` so the ring can occupy blocks [0, ringBlocks)
// of a fused launch while OTHER blocks of the same grid run the intra-node SDMA
// local-block gather CONCURRENTLY (NIC || XGMI in ONE kernel, no host-side
// wait_stream merge -- the fused recv+reassemble lever this work proved reaches
// RCCL parity, ported here. Both default to -1, preserving the
// historical grid-derived geometry BYTE-FOR-BYTE -- inert until a fused launcher
// passes them.
inline __device__ void AllGatherRingSubGroupKernelBody(
    int ringPos, int ringSize, int peBase, int peStride,
    const application::SymmMemObjPtr memObj, const application::SymmMemObjPtr flagsObj,
    size_t peChunkSize, int numQp = 1, int numBlocksOverride = -1, int bidOverride = -1) {
  int nextPos = (ringPos + 1) % ringSize;
  int nextPeer = peBase + nextPos * peStride;
  int maxRounds = ringSize - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const int threadLinearId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int warpId = threadLinearId / warpSize;
  const int warpsPerBlock = threadsPerBlock / warpSize;

  // M4: MULTI-BLOCK ring ("channels", RCCL-style). The single-block
  // ring (numQp warps in ONE CTA, each on its own QP) saturates at ~63 GB/s vs
  // RCCL's ~150 (-18: numQp 4->8 gave 0 gain -> warps-in-one-CTA is
  // exhausted). RCCL instead drives many CTAs (channels) concurrently. Here each
  // block ``bid`` of ``gridDim.x`` handles a DISJOINT 16B-aligned sub-range of
  // EVERY chunk and uses its OWN flag region [bid*ringSize, (bid+1)*ringSize), so
  // blocks never alias data or flags. Block bid issues its put on qpId=bid so
  // each channel drives a distinct QP -- the union still tiles each chunk exactly
  // => byte-identical result. ONLY engaged for true RDMA neighbours (same-node
  // P2P/SDMA lowers to ONE anvil queue per (src,dst); multiple CTAs hammering it
  // overflow the retry budget and coredump, ). For a non-RDMA neighbour
  // (single-node simulation) only block 0 runs the proven single-block path and
  // the other blocks return, keeping single-node bit-exact + crash-free.
  const int numBlocks = (numBlocksOverride >= 0) ? numBlocksOverride : static_cast<int>(gridDim.x);
  const int bid = (bidOverride >= 0) ? bidOverride : static_cast<int>(blockIdx.x);
  application::TransportType nextXportMb =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerRdmaMb = (nextXportMb == application::TransportType::RDMA);
  bool multiBlock = (numBlocks > 1 && peerRdmaMb);
  if (numBlocks > 1 && !peerRdmaMb && bid != 0) {
    // Single-node simulation with a multi-block launch: only block 0 works.
    return;
  }
  // Per-block 16B-aligned sub-range of the chunk (full chunk when single-block).
  size_t blkOff = 0;
  size_t blkBytes = peChunkSize;
  int flagBase = 0;
  if (multiBlock) {
    const size_t kAlignB = 16;
    size_t nUnits = (peChunkSize + kAlignB - 1) / kAlignB;
    size_t unitsPerBlk = (nUnits + numBlocks - 1) / numBlocks;
    size_t startUnit = static_cast<size_t>(bid) * unitsPerBlk;
    size_t endUnit = startUnit + unitsPerBlk;
    if (endUnit > nUnits) endUnit = nUnits;
    blkOff = startUnit * kAlignB;
    size_t blkEnd = endUnit * kAlignB;
    if (blkEnd > peChunkSize) blkEnd = peChunkSize;
    blkBytes = (blkOff < blkEnd) ? (blkEnd - blkOff) : 0;
    flagBase = bid * ringSize;
  }

  // M4: multi-QP fan-out gate. The whole motivation (see the long
  // NOTE below) is that a single warp on a single QP under-fills the NIC vs
  // RCCL's many channels. We fan the per-round put across ``numQp`` QPs (warp w
  // -> qpId=w, disjoint 16B-aligned sub-range) -- but ONLY when the neighbour is
  // reached over RDMA. For a same-node neighbour ShmemPutMemNbiWarp lowers to a
  // single anvil SDMA queue per (src,dst); multiple warps hammering it overflow
  // the retry budget and coredump (validated-negative, ). So we read the
  // neighbour's transport at runtime: single-node simulation (P2P/SDMA) keeps
  // the proven single-warp path and stays bit-exact; only a true cross-node
  // (RDMA) neighbour fans out. Gated additionally on numQp>1 so the flat
  // whole-world ring (numQp defaults to 1) is byte-for-byte unchanged.
  application::TransportType nextXport =
      shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Multi-block and within-block multi-QP fan-out are
  // mutually exclusive: in multi-block mode each CTA already drives its own QP
  // (qpId=bid) on its own sub-range, so a single warp per block is correct.
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  for (int i = 0; i < maxRounds; i++) {
    // Chunk slots are indexed by ring position, not global PE.
    int sendDataRank = (ringPos - i + ringSize) % ringSize;
    int recvDataRank = (ringPos - i - 1 + ringSize) % ringSize;

    size_t chunkBaseOffset = static_cast<size_t>(sendDataRank) * peChunkSize;
    // NOTE (M4, ): tried splitting this put across ALL 8 warps in the
    // block (disjoint 16B-aligned sub-ranges) to parallelize the transfer.
    // VALIDATED NEGATIVE RESULT on device (single-node ring, GPUs 0-3): it
    // crashes with anvil "submitPacket: Retry limit exceeded" -> GPU coredump.
    // Cause: the same-node neighbour path of ShmemPutMemNbiWarp lowers to one
    // anvil SDMA queue per (src,dst) pair; all warps target the SAME nextPeer
    // chunk region, so 8 warps hammer ONE SDMA queue and overflow its retry
    // budget. Multi-warp puts only help when each warp drives a distinct queue/
    // QP (as the intra SDMA gather does, one warp per peer). For a single-peer
    // ring round there is one queue, so a single warp is correct. Kept as-is.
    // NOTE (M4, ): this put is the DOMINANT cost of the xnode hier
    // AllGather. True 2-node bench (n09-21+n09-29, fp32 64MiB/rank, world=8
    // N=2,G=4, >=3 reps) of the  overlap commit (4a2feeb9):
    //   mori min=13.59ms 39.5 GB/s  vs  rccl min=3.55ms 151.1 GB/s (~3.8x).
    // Within noise of the  baseline (40.3 GB/s) => the quiet/spin
    // overlap is a NO-OP at this size: the round is BANDWIDTH-bound, not
    // latency-bound. ROOT CAUSE of the under-fill: this single warp issues
    // the entire chunk on a SINGLE QP. ShmemPutMemNbiWarp<RDMA> issues from
    // lane 0 with qpId defaulting to 0, while the transport provisions
    // numQpPerPe (default 4, MORI_NUM_QP_PER_PE) QPs/peer -- we use 1 of 4.
    // NEXT LEVER: fan the chunk across all numQpPerPe QPs (warp w -> disjoint
    // 16B-aligned sub-range on qpId=w) so multiple QPs drive the NIC in
    // parallel. CORRECTNESS GATE (must hold before landing): the flag bump
    // below must follow a quiet that drains ALL used QPs -- ShmemQuietThread
    // (int pe) (RDMA) loops qpId 0..numQpPerPe-1, whereas the current
    // ShmemQuietThread(nextPeer, memObj) SDMA-typed call must be confirmed to
    // cover every QP, else the receiver's flag fires before tail QPs land.
    // Gate fan-out on numQp>1 only for the all-RDMA sub-group ring; the flat
    // single-node ring (P2P, one anvil queue) must stay single-warp.
    // IMPLEMENTED: the runtime-gated fan-out described above.
    if (multiBlock) {
      // RCCL-style channel: this CTA puts only its sub-range [blkOff, blkOff+
      // blkBytes) of the chunk, on qpId=bid (a distinct QP per channel). Warp 0
      // issues; the union of all CTAs' sub-ranges tiles the chunk exactly =>
      // byte-identical result.
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        shmem::ShmemPutMemNbiWarp(memObj, subOff, memObj, subOff, blkBytes, nextPeer, bid);
      }
    } else if (fanOut) {
      // Split the chunk into ``useWarps`` disjoint 16B-aligned sub-ranges; warp
      // w drives its sub-range on qpId=w. The union tiles the chunk exactly (the
      // last warp absorbs the unaligned tail), so the byte image is identical to
      // a single whole-chunk put -- only the QP fan-out differs.
      if (warpId < useWarps) {
        const size_t kAlign = 16;
        size_t nUnits = (peChunkSize + kAlign - 1) / kAlign;  // # of 16B units
        size_t unitsPerWarp = (nUnits + useWarps - 1) / useWarps;
        size_t startUnit = static_cast<size_t>(warpId) * unitsPerWarp;
        size_t endUnit = startUnit + unitsPerWarp;
        if (endUnit > nUnits) endUnit = nUnits;
        if (startUnit < endUnit) {
          size_t subStart = startUnit * kAlign;
          size_t subEnd = endUnit * kAlign;
          if (subEnd > peChunkSize) subEnd = peChunkSize;  // clamp tail
          size_t subOff = chunkBaseOffset + subStart;
          shmem::ShmemPutMemNbiWarp(memObj, subOff, memObj, subOff, subEnd - subStart, nextPeer,
                                    warpId);
        }
      }
      // All fan-out warps must finish ISSUING their puts before thread 0 drains
      // the QPs and bumps the flag, else the receiver's flag could fire before a
      // tail QP's data lands. (Only added on the fan-out path -- the single-warp
      // path keeps the  thread schedule unchanged.)
      __syncthreads();
    } else if (warpId == 0) {
      shmem::ShmemPutMemNbiWarp(memObj, chunkBaseOffset, memObj, chunkBaseOffset, peChunkSize,
                                nextPeer);
    }

    // M4: overlap the OUTBOUND drain (quiet + flag bump to nextPeer)
    // with the INBOUND recv-flag wait (from prevPeer). These two waits are on
    // independent network directions, but the previous code serialized them on
    // thread 0 (quiet+bump, syncthreads, then spin) so their latencies added.
    // Split them across two threads so they proceed concurrently; the single
    // trailing __syncthreads makes both finish before the round ends (this also
    // drops one __syncthreads/round vs the old two-phase form).
    // NOTE: true round-level pipelining (issuing the next round's put during
    // this recv-wait) is IMPOSSIBLE here -- round i+1 sends sendDataRank =
    // (ringPos-i-1), which is EXACTLY the recvDataRank received in round i, a
    // hard data dependency (you forward onward precisely what you just got).
    if (threadLinearId == 0) {
      // Drain the outbound put before bumping the receiver's flag. On the
      // fan-out path the put used numQp RDMA QPs, so we must quiet ALL of them:
      // ShmemQuietThread(pe) (RDMA) loops qpId 0..numQpPerPe-1. The single-warp
      // path keeps the original SDMA/P2P-typed quiet (memObj overload).
      if (multiBlock) {
        // This CTA put on exactly qpId=bid; drain ONLY that QP. Draining ALL QPs
        // (ShmemQuietThread(pe)) from every block would poll the same completion
        // queues concurrently across CTAs and race. Per-QP quiet keeps each
        // channel independent (RCCL-style), and this block bumps only its own
        // flag region, so the receiver's per-block flag fires only after THIS
        // block's data has landed.
        shmem::ShmemQuietThread(nextPeer, bid);
      } else if (fanOut) {
        // Fan-out issued from numQp QPs within ONE block; ShmemQuietThread(pe)
        // drains ALL QPs so the receiver's flag never fires before a tail QP's
        // data lands.
        shmem::ShmemQuietThread(nextPeer);
      } else {
        shmem::ShmemQuietThread(nextPeer, memObj);
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
          flagsObj, (flagBase + sendDataRank) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD,
          nextPeer);
    } else if (threadLinearId == warpSize) {
      // Each round the sender increments a DISTINCT flag slot (index
      // recvDataRank = sendDataRank on the receiver), so every slot is
      // incremented exactly once over the ringSize-1 rounds -- 0 -> 1.
      // Wait for THIS round's slot to become nonzero (not a cumulative count;
      // the previous "!= i+1" form only held for ringSize==2 / a single round).
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flagsArray + flagBase + recvDataRank) == 0) {
        spinCount++;
        if (spinCount > 10000000) {  // Increased timeout threshold
          printf("ringPos %d: Timeout waiting from ringPos %d (round %d, slot still 0)\n", ringPos,
                 recvDataRank, i);
          break;
        }
      }
    }
    __syncthreads();
  }

  // Each block resets ONLY its own flag region [flagBase, flagBase+ringSize) so
  // concurrent channels never race (single-block: flagBase=0, ringSize slots).
  for (int idx = threadLinearId; idx < ringSize; idx += threadsPerBlock) {
    flagsArray[flagBase + idx] = 0;
  }
  __syncthreads();
  if (threadLinearId == 0) {
    __threadfence_system();
  }
}

// Whole-world ring (flat): position == global PE, stride 1, base 0.
inline __device__ void AllGatherRingKernelBody(int myPe, int npes,
                                               const application::SymmMemObjPtr memObj,
                                               const application::SymmMemObjPtr flagsObj,
                                               size_t peChunkSize) {
  AllGatherRingSubGroupKernelBody(myPe, npes, /*peBase=*/0, /*peStride=*/1, memObj, flagsObj,
                                  peChunkSize);
}

template <typename T>
__global__ void AllGatherRingKernel(int myPe, int npes, const application::SymmMemObjPtr memObj,
                                    const application::SymmMemObjPtr flagsObj) {
  // Existing executor path: chunk size derived from the (exactly npes*chunk)
  // buffer. Equal-sized chunks, so size/npes is exact.
  AllGatherRingKernelBody(myPe, npes, memObj, flagsObj, memObj->size / npes);
}

}  // namespace collective
}  // namespace mori
