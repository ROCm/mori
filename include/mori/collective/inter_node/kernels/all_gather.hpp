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

// s_sleep backoff between inter-node landing-spin polls. Timing-only, memory order
// unchanged (post-loop __threadfence_system still gates visibility, bit-exact); 0 => OFF.
constexpr int kHierInterPollSleep = 0;

// Ring AllGather over an arithmetic PE sub-group {peBase + k*peStride : k in
// [0,ringSize)}; this PE is at ringPos. peChunkSize is explicit so one fixed-size
// symmetric ring buffer serves variable sizes. On entry only slot ringPos is
// filled; after ringSize-1 rounds every member holds all chunks in ring order.
// Equal per-PE chunks => no last-chunk special case. Whole-world ring is the
// special case peBase=0, peStride=1. Inter-node phase of the hierarchical
// AllGather: the ring runs over cross-node leaders (RDMA neighbours), same-node P2P.
//
// numBlocksOverride/bidOverride replace the grid-derived gridDim.x/blockIdx.x so
// the ring can occupy a sub-range of blocks in a fused grid while other blocks run
// the intra-node SDMA gather (NIC + XGMI in one kernel). Both -1 => grid-derived
// geometry, inert until a fused launcher passes them.
//
// chunkReadyFlags (default nullptr, inert): per-channel landing signal pipelining the
// inter-node ring with intra-node SDMA reassembly. Device array of >=numBlocks uint64_t in
// cached HBM, zeroed before launch. Block bid publishes chunkReadyFlags[bid] only after its
// inbound sub-range has landed AND been made visible via the receiver's system acquire +
// __threadfence_system. nullptr => byte-for-byte identical standalone ring.
inline __device__ void AllGatherRingSubGroupKernelBody(
    int ringPos, int ringSize, int peBase, int peStride, const application::SymmMemObjPtr memObj,
    const application::SymmMemObjPtr flagsObj, size_t peChunkSize, int numQp = 1,
    int numBlocksOverride = -1, int bidOverride = -1,
    uint64_t* chunkReadyFlags = nullptr,
    int deepPipe = 1,
    int deepPipeQuiet = 0, bool useWriteFence = false) {
  int nextPos = (ringPos + 1) % ringSize;
  int nextPeer = peBase + nextPos * peStride;
  int maxRounds = ringSize - 1;

  uint64_t* flagsArray = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const int threadLinearId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int warpId = threadLinearId / warpSize;
  const int warpsPerBlock = threadsPerBlock / warpSize;

  // Multi-block ring ("channels", RCCL-style): each block bid drives a disjoint
  // 16B-aligned sub-range on its own QP (qpId=bid) and flag region
  // [bid*ringSize,(bid+1)*ringSize) (no data/flag aliasing, union byte-identical).
  // Only for RDMA neighbours: same-node P2P/SDMA has one anvil queue per (src,dst) that
  // multiple CTAs would crash, so a non-RDMA neighbour runs only block 0 (bit-exact).
  const int numBlocks = (numBlocksOverride >= 0) ? numBlocksOverride : static_cast<int>(gridDim.x);
  const int bid = (bidOverride >= 0) ? bidOverride : static_cast<int>(blockIdx.x);
  application::TransportType nextXportMb = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerRdmaMb = (nextXportMb == application::TransportType::RDMA);
  bool multiBlock = (numBlocks > 1 && peerRdmaMb);
  if (numBlocks > 1 && !peerRdmaMb && bid != 0) {
    // Single-node simulation with a multi-block launch: only block 0 works.
    return;
  }
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

  // Multi-QP fan-out gate: a single warp/QP under-fills the NIC, so fan the per-round put
  // across numQp QPs (warp w -> qpId=w, disjoint 16B sub-range) -- only for an RDMA
  // neighbour (same-node anvil single-queue would crash) and only when numQp>1 (flat
  // whole-world ring stays byte-for-byte unchanged).
  application::TransportType nextXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Mutually exclusive with multi-block (there each CTA already drives qpId=bid).
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  int prevPos = (ringPos - 1 + ringSize) % ringSize;
  int prevPeer = peBase + prevPos * peStride;
  application::TransportType prevXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[prevPeer];
  bool prevIsRdma = (prevXport == application::TransportType::RDMA);

  // WRITE-PUSH (SEND-CQ) per-channel landing fence for the multiBlock AG: channel bid
  // pushes its sub-range as a fused put-with-signal (flag AMO after the WRITE, same QP,
  // RC in-order => flag can't beat data), draining only qpId=bid's CQE. Gated maxRounds==1.
  bool multiBlockWrite =
      (useWriteFence && peerIsRdma && prevIsRdma && maxRounds == 1 && multiBlock);

  __syncthreads();

  // DEEP-SQ TEMPORAL PIPELINE (MORI_HIER_DEEP_PIPE=P): split the chunk into P temporal
  // sub-chunks with per-sub-chunk put-with-signal; p's flag fires (RC in-order) before
  // p+1's, so a reassembly worker pushes p over XGMI while p+1.. still cross the NIC.
  // Single-round all-RDMA fan-out only; self-contained. INERT when deepPipe<=1 (byte-identical).
  bool deepPipeEngaged =
      (deepPipe > 1 && peerIsRdma && prevIsRdma && maxRounds == 1 && !multiBlock &&
       chunkReadyFlags != nullptr && useWarps >= 1);
  if (deepPipeEngaged) {
    // Clamp P so every sub-chunk carries >= kMinSubChunkB (1 MiB): tiny per-sub-chunk
    // WQEs starve the NIC and the extra flag round-trips can deadlock small transfers.
    // Bit-exact -- P only partitions the same contiguous bytes, issued+landed in order.
    const size_t kMinSubChunkB = static_cast<size_t>(1) << 20;
    int reqPmax = static_cast<int>(peChunkSize / kMinSubChunkB);
    if (reqPmax < 1) reqPmax = 1;
    const int P = (deepPipe < reqPmax) ? deepPipe : reqPmax;
    const int sendRank = ringPos;  // maxRounds==1: send our own chunk
    const size_t chunkBaseOffsetSend = static_cast<size_t>(sendRank) * peChunkSize;
    const size_t kAlignDP = 16;
    const size_t nUnits = (peChunkSize + kAlignDP - 1) / kAlignDP;
    const size_t unitsPerP = (nUnits + static_cast<size_t>(P) - 1) / static_cast<size_t>(P);
    const int sw = useWarps;  // sender fan-out warps per sub-chunk (== numQp fan-out)
    auto dpRange = [&](int p, size_t& sU, size_t& eU) {
      sU = static_cast<size_t>(p) * unitsPerP;
      eU = sU + unitsPerP;
      if (eU > nUnits) eU = nUnits;
      if (sU > eU) sU = eU;
    };
    // Per-sub-chunk active-warp count, computed identically to the sender tiling.
    auto activeOf = [&](int p) -> int {
      size_t sU, eU;
      dpRange(p, sU, eU);
      if (sU >= eU) return 0;
      size_t subUnits = eU - sU;
      size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
      if (upw == 0) upw = 1;
      int a = static_cast<int>((subUnits + upw - 1) / upw);
      if (a > sw) a = sw;
      if (a < 1) a = 1;
      return a;
    };
    // DEEP_PIPE_QUIET: scale-robust per-sub-chunk landing fence. The put-signal AMO can
    // beat its own data at large sizes and WRITE_WITH_IMM is HW-unavailable, so each sub-chunk
    // p rides its OWN QP (plain put) drained via SEND-CQ before its flag => flag never precedes
    // landing. Requires chunkReadyFlags.
    if (deepPipeQuiet) {
      // Disjoint QP GROUP per sub-chunk: group(p) = QPs [p*g, p*g+g), g = sw/P; each
      // fans across g QPs for BW. When P>sw groups wrap (g=1, p%sw) and share a QP.
      // Draining a sub-chunk's whole group before its AMO is the landing fence.
      const int g = (sw >= P) ? (sw / P) : 1;  // QPs per sub-chunk
      auto grpBase = [&](int p) -> int { return (sw >= P) ? (p * g) : (p % sw); };
      auto nonEmptyDP = [&](int p) -> bool {
        size_t sU, eU;
        dpRange(p, sU, eU);
        return sU < eU;
      };
      // SENDER: warp w -> sub-chunk p=w/g (sw>=P), lane wl=w%g on qpId=grpBase(p)+wl.
      if (warpId < sw) {
        int p, wl, gg;
        if (sw >= P) {
          p = warpId / g;
          wl = warpId % g;
          gg = g;
          if (p >= P) p = -1;  // extra warps idle
        } else {
          p = warpId;  // P>sw: each of first sw warps -> its own sub-chunk group start
          wl = 0;
          gg = 1;
        }
        if (p >= 0) {
          for (int pp = p; pp < P;
               pp += (sw >= P ? P : sw)) {  // sw<P: warp handles pp = warpId, warpId+sw, ...
            size_t sU, eU;
            dpRange(pp, sU, eU);
            if (sU >= eU) {
              if (sw >= P)
                break;
              else
                continue;
            }
            size_t subUnits = eU - sU;
            size_t upl = (subUnits + static_cast<size_t>(gg) - 1) / static_cast<size_t>(gg);
            if (upl == 0) upl = 1;
            size_t lS = sU + static_cast<size_t>(wl) * upl;
            size_t lE = lS + upl;
            if (lE > eU) lE = eU;
            if (lS >= lE) {
              if (sw >= P)
                continue;
              else
                continue;
            }
            size_t so = lS * kAlignDP;
            size_t eo = lE * kAlignDP;
            if (eo > peChunkSize) eo = peChunkSize;
            size_t off = chunkBaseOffsetSend + so;
            int qp = grpBase(pp) + wl;
            shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, eo - so, nextPeer, qp);
            if (sw >= P) break;  // sw>=P: one warp issues exactly one sub-chunk tile
          }
        }
      }
      __syncthreads();
      // PARALLEL SEND-DRAIN + RECV-PUBLISH: one warp-leader per sub-chunk drains its OWN
      // disjoint QP group (=> disjoint WQ/CQ, no shared-completion race, bit-exact), AMOs
      // the remote flag p, waits its inbound flag p, publishes chunkReadyFlags[p]. Safe
      // only when sw>=P AND P<=warpsPerBlock (a leader warp per p); else serial thread-0.
      if (sw >= P && P <= warpsPerBlock) {
        // Per-group lock-free join (shared arrival counter); groups fire independently.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        // P==1: all sw drain warps are one group, so __syncthreads is the natural join
        // (the P>1 atomic-counter spin would serialize independent groups).
        if (P == 1) {
          if (warpLead && myWarp < sw && nonEmptyDP(0)) {
            shmem::ShmemQuietThread(nextPeer, grpBase(0) + myWarp);  // g==sw, wl==myWarp
            __threadfence_system();                                  // this QP's bytes visible
          }
          __syncthreads();  // all g QP drains + fences complete (uniform block join)
          if (threadLinearId == 0 && nonEmptyDP(0)) {
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                           (flagBase + 0) * sizeof(uint64_t), 1,
                                                           core::atomicType::AMO_ADD, nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + 0) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
              if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + 0,
                                          static_cast<uint64_t>(1));
          }
          __syncthreads();  // keep the block uniform before the shared trailing logic
        } else {
          __shared__ unsigned int dpGrpDrained[64];  // arrivals per sub-chunk group
          for (int i = threadLinearId; i < P; i += threadsPerBlock) dpGrpDrained[i] = 0u;
          __syncthreads();
          // Drain warp d in [0,sw) -> group p=d/g, lane wl=d%g, drains QP grpBase(p)+wl.
          if (warpLead && myWarp < sw) {
            int d = myWarp;
            int p = d / g;
            int wl = d % g;
            if (p < P && nonEmptyDP(p)) {
              shmem::ShmemQuietThread(nextPeer, grpBase(p) + wl);
              __threadfence_system();           // this QP's landed bytes visible
              atomicAdd(&dpGrpDrained[p], 1u);  // signal group arrival
              if (wl == 0) {
                // Group leader: wait for all g QPs to land (atomicAdd(.,0) is an atomic
                // load), then AMO the remote flag + spin our own inbound flag + publish.
                long long gspin = 0;
                while (atomicAdd(&dpGrpDrained[p], 0u) < static_cast<unsigned int>(g)) {
                  if (++gspin > 10000000000LL) __builtin_trap();
                }
                shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                               (flagBase + p) * sizeof(uint64_t), 1,
                                                               core::atomicType::AMO_ADD, nextPeer);
                long long spin = 0;
                while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                       static_cast<uint64_t>(1)) {
                  // Landing fence MUST wait for the group to land; publishing on a
                  // timeout would allow a stale read. Abort loudly instead.
                  if (++spin > 10000000000LL) __builtin_trap();
                  if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
                }
                __threadfence_system();
                core::AtomicStoreSeqCstSystem(
                    chunkReadyFlags + p,
                    static_cast<uint64_t>(1));
              }
            }
          }
        }  // end P>1 per-group counter join
      } else if (sw <= warpsPerBlock) {
        // Wrap parallel drain (P>sw, sub-chunks share QPs via grpBase=p%sw): each of the sw
        // disjoint QP groups gets a merged leader walking p=w,w+sw,... in order (in-order
        // per QP == landing proof). Requires sw<=warpsPerBlock.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        if (warpLead && myWarp < sw) {
          const int w = myWarp;
          for (int p = w; p < P; p += sw) {
            if (!nonEmptyDP(p)) continue;
            shmem::ShmemQuietThread(nextPeer, grpBase(p));  // grpBase(p)==p%sw==w
            __threadfence_system();
            shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                           (flagBase + p) * sizeof(uint64_t), 1,
                                                           core::atomicType::AMO_ADD, nextPeer);
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
              if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
            __threadfence_system();
            core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                          static_cast<uint64_t>(1));
          }
        }
      } else if (threadLinearId == 0) {
        // WRAP fallback (P>sw AND sw>warpsPerBlock): serial drain, groups share QPs.
        for (int p = 0; p < P; ++p) {
          if (!nonEmptyDP(p)) continue;
          shmem::ShmemQuietThread(nextPeer, grpBase(p));
          __threadfence_system();
          shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(
              flagsObj, (flagBase + p) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, nextPeer);
        }
      } else if (threadLinearId == warpSize) {
        for (int p = 0; p < P; ++p) {
          if (nonEmptyDP(p)) {
            long long spin = 0;
            while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                   static_cast<uint64_t>(1)) {
              if (++spin > 10000000000LL) __builtin_trap();
              if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
            }
          }
          __threadfence_system();
          core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                        static_cast<uint64_t>(1));
        }
      }
      __syncthreads();
      // Parallel paths already drained every send QP; only the serial fallback needs the
      // trailing full-QP re-drain for buffer reuse.
      const bool dpParallelDrained =
          (sw >= P && P <= warpsPerBlock) || (sw <= warpsPerBlock);
      if (!dpParallelDrained) {
        if (threadLinearId == 0) {
          shmem::ShmemQuietThread(nextPeer);
        }
        __syncthreads();
      }
      for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
        flagsArray[flagBase + idx] = 0;
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      return;
    }
    // SENDER: warp w sends its 16B tile of each sub-chunk p on qpId=w; all P ride
    // qpId=w back-to-back (deep SQ) so p lands before p+1 -- temporal order preserved.
    if (warpId < sw) {
      for (int p = 0; p < P; ++p) {
        size_t sU, eU;
        dpRange(p, sU, eU);
        if (sU >= eU) break;
        size_t subUnits = eU - sU;
        size_t upw = (subUnits + static_cast<size_t>(sw) - 1) / static_cast<size_t>(sw);
        if (upw == 0) upw = 1;
        size_t wS = sU + static_cast<size_t>(warpId) * upw;
        size_t wE = wS + upw;
        if (wE > eU) wE = eU;
        if (wS >= wE) continue;
        size_t so = wS * kAlignDP;
        size_t eo = wE * kAlignDP;
        if (eo > peChunkSize) eo = peChunkSize;
        size_t off = chunkBaseOffsetSend + so;
        shmem::ShmemPutMemNbiSignalWarp(memObj, off, memObj, off, eo - so, flagsObj,
                                        (flagBase + p) * sizeof(uint64_t), 1,
                                        core::atomicType::AMO_ADD, nextPeer, warpId);
      }
    }
    // RECEIVER: publish chunkReadyFlags[p] in temporal order (reassembly can push p while
    // later sub-chunks still cross the NIC).
    if (threadLinearId == 0) {
      for (int p = 0; p < P; ++p) {
        int active = activeOf(p);
        if (active > 0) {
          long long spin = 0;
          while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + p) <
                 static_cast<uint64_t>(active)) {
            if (++spin > 10000000000LL) __builtin_trap();
            if (kHierInterPollSleep) __builtin_amdgcn_s_sleep(kHierInterPollSleep);
          }
        }
        __threadfence_system();
        core::AtomicStoreSeqCstSystem(chunkReadyFlags + p,
                                      static_cast<uint64_t>(1));
      }
    }
    __syncthreads();
    // Trailing barrier: the recv flag-slot reset (threads>0) is ordered before any peer's
    // next-op AMO by the entry barrier in prepare.
    for (int idx = threadLinearId; idx < P; idx += threadsPerBlock) {
      flagsArray[flagBase + idx] = 0;
    }
    if (threadLinearId == 0) {
      shmem::ShmemQuietThread(nextPeer);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    return;
  }

  for (int i = 0; i < maxRounds; i++) {
    // Chunk slots are indexed by ring position, not global PE.
    int sendDataRank = (ringPos - i + ringSize) % ringSize;
    int recvDataRank = (ringPos - i - 1 + ringSize) % ringSize;

    size_t chunkBaseOffset = static_cast<size_t>(sendDataRank) * peChunkSize;

    if (multiBlockWrite) {
      // Warp 0 pushes this channel's sub-range as ONE fused put-with-signal (see the
      // multiBlockWrite landing fence above).
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        shmem::ShmemPutMemNbiSignalWarp(memObj, subOff, memObj, subOff, blkBytes, flagsObj,
                                        (flagBase + sendDataRank) * sizeof(uint64_t), 1,
                                        core::atomicType::AMO_ADD, nextPeer, bid);
      }
      __syncthreads();
      // No explicit send-CQ quiet needed (fused signal's flag implies landing). Receiver
      // spins this channel's inbound flag; system acquire + fence make it visible.
      if (threadLinearId == warpSize && blkBytes > 0) {
        int spinCount = 0;
        while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) < 1) {
          if (++spinCount > 10000000) break;
        }
        __threadfence_system();
      }
      __syncthreads();
      continue;
    }
    // Same-node round stays single-warp (one anvil queue per (src,dst)); an RDMA
    // neighbour fans across QPs (warp w -> qpId=w), the flag bump gated on a quiet
    // draining ALL used QPs.
    auto putDeep = [&](size_t off, size_t bytes, int qp) {
      shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, bytes, nextPeer, qp);
    };
    if (multiBlock) {
      // RCCL-style channel: this CTA puts only its sub-range on qpId=bid.
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        putDeep(subOff, blkBytes, bid);
      }
    } else if (fanOut) {
      // Split the chunk into useWarps disjoint 16B-aligned sub-ranges (last warp absorbs
      // the tail), warp w on qpId=w.
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
          putDeep(subOff, subEnd - subStart, warpId);
        }
      }
      // All fan-out warps must finish issuing before thread 0 drains the QPs and bumps
      // the flag, else the receiver's flag could fire before a tail QP's data lands.
      __syncthreads();
    } else if (warpId == 0) {
      putDeep(chunkBaseOffset, peChunkSize, 0);
    }

    // Round-level pipelining is impossible: round i+1 forwards exactly what round i
    // received (hard data dependency).
    if (threadLinearId == 0) {
      // Drain the outbound put before bumping the receiver's flag.
      if (multiBlock) {
        // Drain ONLY qpId=bid: draining all QPs from every block would poll the same CQs
        // concurrently across CTAs and race (per-QP quiet keeps channels independent).
        shmem::ShmemQuietThread(nextPeer, bid);
      } else if (fanOut) {
        // Fan-out used numQp QPs; drain ALL so the flag never fires before a tail lands.
        shmem::ShmemQuietThread(nextPeer);
      } else if (peerIsRdma) {
        // CORRECTNESS (flag-beats-data): the SDMA-typed memObj quiet drains only the
        // P2P/SDMA path, so an RDMA neighbour's flag AMO could land before its data PUT
        // drains => stale bytes. Use the transport-aware quiet (RDMA -> all QPs).
        shmem::ShmemQuietThread(nextPeer);
      } else {
        shmem::ShmemQuietThread(nextPeer, memObj);
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                     (flagBase + sendDataRank) * sizeof(uint64_t),
                                                     1, core::atomicType::AMO_ADD, nextPeer);
    } else if (threadLinearId == warpSize) {
      int spinCount = 0;
      // SYSTEM-scope acquire: flag and chunk are both written by a REMOTE peer's RDMA; a
      // relaxed load establishes no happens-before, so the chunk would not be coherently
      // visible to this GPU's forward-put/copy-OUT (stale bytes under FSDP overlap).
      // Acquire + system fence make the peer's data visible without a host sync.
      while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) < 1) {
        spinCount++;
        if (spinCount > 10000000) {
          break;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  // Per-chunk landing publish: fenced system-scoped store so a concurrent reassembly block
  // can push this landed sub-range over XGMI without a global finish barrier. INERT when
  // chunkReadyFlags==nullptr (standalone ring byte-for-byte identical).
  if (chunkReadyFlags != nullptr && threadLinearId == 0) {
    __threadfence_system();
    core::AtomicStoreSeqCstSystem(chunkReadyFlags + bid,
                                  static_cast<uint64_t>(1));
  }
  // Each block resets ONLY its own flag region so concurrent channels never race.
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
  // Executor path: equal-sized chunks, so size/npes is exact.
  AllGatherRingKernelBody(myPe, npes, memObj, flagsObj, memObj->size / npes);
}

}  // namespace collective
}  // namespace mori
// Direct-land coherence note: the NIC writing straight into the output tensor leaves the
// SDMA reassembly read stale -- not a landing/ordering race (fusing the flag onto the data
// QP does not fix it) but SDMA-read coherence of the coarse-grained cached output tensor,
// whose line the NIC write does not invalidate for the copy engine. The default path reads
// the fine-grained RDMA ring buffer (SDMA-read-coherent) and is correct; a fix would land
// RDMA into fine-grained coherent staging.
