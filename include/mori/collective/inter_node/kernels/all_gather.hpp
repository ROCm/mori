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

// s_sleep cycles between polls on the long REMOTE-landing spins (timing-only;
// memory order unchanged). 0 => OFF, guarded call compiles out (byte-identical).
constexpr int kHierInterPollSleep = 0;

// Ring AllGather over the arithmetic sub-group {peBase + k*peStride : k<ringSize};
// this PE at ``ringPos``. Ring buffer holds ``ringSize`` chunks of ``peChunkSize``,
// slot indexed by ring position; after ringSize-1 rounds every member holds all
// chunks. Equal chunks => no last-chunk special case. Whole-world ring = peBase 0,
// peStride 1. Inter-node phase of the hierarchical AG (neighbours over RDMA).
//
// ``numBlocksOverride``/``bidOverride`` (default -1, inert): override grid geometry
// so the ring occupies a sub-range of a fused grid alongside intra-node SDMA blocks.
// ``chunkReadyFlags`` (default nullptr, inert): per-chunk landing signal in cached
// HBM (>=numBlocks uint64_t, caller-zeroed). Block ``bid`` publishes flag[bid] once
// its inbound sub-range has landed and been made visible (system acquire + fence);
// a reassembly block spins on it and SDMA-pushes that sub-range while later channels
// still cross the NIC. Dependency is this PE's own landing (local spin, no barrier).
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

  // Multi-block ring ("channels", RCCL-style): each block ``bid`` handles a
  // disjoint 16B-aligned sub-range of every chunk on its own flag region
  // [bid*ringSize, (bid+1)*ringSize) and its own qpId=bid; union tiles each chunk
  // exactly (byte-identical). RDMA neighbours only: same-node P2P/SDMA has one
  // anvil queue per (src,dst), so non-RDMA neighbours run only block 0.
  const int numBlocks = (numBlocksOverride >= 0) ? numBlocksOverride : static_cast<int>(gridDim.x);
  const int bid = (bidOverride >= 0) ? bidOverride : static_cast<int>(blockIdx.x);
  application::TransportType nextXportMb = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
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

  // Multi-QP fan-out gate: fan the put across ``numQp`` QPs (warp w -> qpId=w) only
  // for RDMA neighbours and numQp>1. Same-node P2P/SDMA has one anvil queue, so it
  // stays single-warp; flat whole-world ring (numQp=1) byte-for-byte unchanged.
  application::TransportType nextXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[nextPeer];
  bool peerIsRdma = (nextXport == application::TransportType::RDMA);
  // Multi-block and within-block fan-out are mutually exclusive (multi-block already
  // drives one QP per CTA), so single warp per block there.
  int useWarps = (!multiBlock && numQp > 1 && peerIsRdma) ? numQp : 1;
  if (useWarps > warpsPerBlock) useWarps = warpsPerBlock;
  bool fanOut = (useWarps > 1);

  int prevPos = (ringPos - 1 + ringSize) % ringSize;
  int prevPeer = peBase + prevPos * peStride;
  application::TransportType prevXport = shmem::GetGlobalGpuStatesPtr()->transportTypes[prevPeer];
  bool prevIsRdma = (prevXport == application::TransportType::RDMA);

  // Per-channel WRITE-PUSH landing fence (maxRounds==1 only): fused put-with-signal
  // on qpId=bid (flag AMO after data WRITE, RC in-order => flag can't beat data),
  // per-channel send-CQ drain, receiver spins its own inbound flag (system acquire).
  bool multiBlockWrite =
      (useWriteFence && peerIsRdma && prevIsRdma && maxRounds == 1 && multiBlock);

  __syncthreads();

  // ==========================================================================
  // DEEP-SQ TEMPORAL PIPELINE (deepPipe=P): split the chunk into P sub-chunks issued
  // back-to-back on the useWarps QP fan-out with per-sub-chunk put-with-signal, so
  // p's flag fires (RC in-order) before p+1's and a reassembly worker pushes p over
  // XGMI while later sub-chunks cross the NIC. Single-round (ringSize==2) all-RDMA
  // path only; publishes P chunkReadyFlags in temporal order and returns before the
  // round loop. INERT when deepPipe<=1 (byte-identical).
  bool deepPipeEngaged =
      (deepPipe > 1 && peerIsRdma && prevIsRdma && maxRounds == 1 && !multiBlock &&
       chunkReadyFlags != nullptr && useWarps >= 1);
  if (deepPipeEngaged) {
    // Clamp P so each sub-chunk carries >= kMinSubChunkB (1 MiB): tiny per-sub-chunk
    // WQEs starve the NIC and can deadlock. Bit-exact (P only partitions the same
    // contiguous bytes issued+landed in order).
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
    // Uniform temporal split: sub-chunk p = [p*unitsPerP, (p+1)*unitsPerP), clamped.
    auto dpRange = [&](int p, size_t& sU, size_t& eU) {
      sU = static_cast<size_t>(p) * unitsPerP;
      eU = sU + unitsPerP;
      if (eU > nUnits) eU = nUnits;
      if (sU > eU) sU = eU;
    };
    // Per-sub-chunk active-warp count, computed identically to the sender tiling.
    // active(p) = # of warps that actually get a non-empty tile of sub-chunk p.
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
    // DEEP_PIPE_QUIET: per-sub-chunk landing fence via SEND-CQ drain (the put-signal
    // AMO can beat its data >=64MB). Each sub-chunk p rides its own QP; plain put,
    // then drain qpId=(p%sw) and only then AMO flag slot p, so the flag never precedes
    // the landing while the temporal pipeline holds. Bit-exact; requires chunkReadyFlags.
    if (deepPipeQuiet) {
      // Disjoint QP group per sub-chunk (g = sw/P QPs each) so sub-chunks stay on
      // distinct QPs and each still fans for full BW. group(p) = QPs [p*g, p*g+g).
      // When P>sw groups wrap (g==1, p%sw): sub-chunks share a QP, drain covers the
      // later one (bit-exact, less overlap). Draining a group's g QPs before its AMO
      // is the landing fence.
      const int g = (sw >= P) ? (sw / P) : 1;  // QPs per sub-chunk
      auto grpBase = [&](int p) -> int { return (sw >= P) ? (p * g) : (p % sw); };
      auto nonEmptyDP = [&](int p) -> bool {
        size_t sU, eU;
        dpRange(p, sU, eU);
        return sU < eU;
      };
      // SENDER: warp w -> sub-chunk p=w/g (sw>=P), lane wl=w%g drives qpId
      // grpBase(p)+wl on its 16B tile.
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
            // Tile sub-chunk pp across gg QPs; lane wl takes its slice.
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
      // Parallel per-sub-chunk drain+publish: one warp-leader per sub-chunk p (thread
      // p*warpSize) drains its own QP group, AMOs remote flag p, waits its inbound flag
      // p, publishes chunkReadyFlags[p]. Disjoint QP groups => disjoint WQ/CQ, bit-exact.
      // Single leader per p (not split send/recv) so a leader warp exists for every p on
      // wave64 (P<=warpsPerBlock). Only when sw>=P AND P<=warpsPerBlock; else serial
      // thread-0 fallback.
      if (sw >= P && P <= warpsPerBlock) {
        // Each sub-chunk's g QPs get their own drain warp (concurrent completion
        // polls), then a lock-free per-group join (shared arrival counter) lets the
        // group leader AMO+publish only after all g QPs land. Groups fire independently
        // (no global barrier); bit-exact.
        const int myWarp = threadLinearId / warpSize;
        const bool warpLead = (threadLinearId % warpSize) == 0;
        // P==1: all sw drain warps are one group, so __syncthreads is the join. Each
        // quiets its QP + fences before the barrier, then thread 0 AMOs / spins the
        // inbound flag / publishes. (P>1 uses the per-group counter join below.)
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
          // Drain warp d -> group p=d/g, lane wl=d%g, drains QP grpBase(p)+wl. All g
          // lanes participate (empty tile drains a no-op CQ) so the count reaches g.
          if (warpLead && myWarp < sw) {
            int d = myWarp;
            int p = d / g;
            int wl = d % g;
            if (p < P && nonEmptyDP(p)) {
              shmem::ShmemQuietThread(nextPeer, grpBase(p) + wl);
              __threadfence_system();           // this QP's landed bytes visible
              atomicAdd(&dpGrpDrained[p], 1u);  // signal group arrival
              if (wl == 0) {
                // Group leader: wait all g QPs landed, then AMO remote flag + spin our
                // inbound flag + publish. atomicAdd(.,0) is an atomic load of the counter.
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
                  // Landing fence must wait; a timeout escape that published anyway
                  // would allow a stale read. Abort loudly instead of corrupting bytes.
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
        }  // end else (P>1 per-group counter join; P==1 uses the block-barrier fast join)
      } else if (sw <= warpsPerBlock) {
        // Wrap parallel drain (P>sw, groups share QPs but the sw QP groups are
        // disjoint): each group w gets one leader warp walking p=w,w+sw,... in order,
        // draining QP w (in-order == landing proof), AMO flag p, wait inbound flag p,
        // publish. sw leaders concurrent on disjoint QP groups. Requires sw<=warpsPerBlock.
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
      // Parallel paths already drained every send QP group per sub-chunk, so the
      // trailing full-QP re-drain is redundant there (buffer-reuse safety already met);
      // keep it only on the serial fallback.
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
    // SENDER: warp w sends its 16B tile of each sub-chunk on qpId=w; all P ride
    // qpId=w back-to-back (deep SQ) so temporal landing order holds.
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
    // RECEIVER: per sub-chunk spin the flag sum (== active senders) then publish
    // chunkReadyFlags[p] in temporal order.
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
    // Overlap the buffer-reuse send-QP drain (thread 0) with the flag-slot reset
    // (threads>0) under one trailing barrier: disjoint state (chunkReadyFlags already
    // published; prepare's entry barrier orders each PE's reset before any peer's AMO).
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
  // ==========================================================================

  for (int i = 0; i < maxRounds; i++) {
    // Chunk slots are indexed by ring position, not global PE.
    int sendDataRank = (ringPos - i + ringSize) % ringSize;
    int recvDataRank = (ringPos - i - 1 + ringSize) % ringSize;

    size_t chunkBaseOffset = static_cast<size_t>(sendDataRank) * peChunkSize;

    if (multiBlockWrite) {
      // Multi-block write-push: warp 0 pushes this channel's sub-range on qpId=bid as
      // one fused put-with-signal (data WRITE + flag AMO on the same RC QP), so the
      // flag can never beat the data (bit-exact).
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        shmem::ShmemPutMemNbiSignalWarp(memObj, subOff, memObj, subOff, blkBytes, flagsObj,
                                        (flagBase + sendDataRank) * sizeof(uint64_t), 1,
                                        core::atomicType::AMO_ADD, nextPeer, bid);
      }
      __syncthreads();
      // No explicit send-CQ quiet: the fused signal already lands after the data
      // (RC in-order). Receiver spins this channel's inbound flag (system acquire +
      // fence) to make the sub-range coherently visible.
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
    // Same-node ring round is single-warp: ShmemPutMemNbiWarp lowers to one anvil
    // SDMA queue per (src,dst), so splitting across warps overflows its retry budget.
    // For the all-RDMA sub-group ring (numQp>1) fan the chunk across QPs (warp w ->
    // 16B sub-range on qpId=w); the flag bump must follow a quiet draining ALL QPs.
    auto putDeep = [&](size_t off, size_t bytes, int qp) {
      shmem::ShmemPutMemNbiWarp(memObj, off, memObj, off, bytes, nextPeer, qp);
    };
    if (multiBlock) {
      // RCCL-style channel: this CTA puts only its sub-range on qpId=bid; warp 0
      // issues, union of CTAs tiles the chunk exactly (byte-identical).
      if (warpId == 0 && blkBytes > 0) {
        size_t subOff = chunkBaseOffset + blkOff;
        putDeep(subOff, blkBytes, bid);
      }
    } else if (fanOut) {
      // Split the chunk into useWarps disjoint 16B sub-ranges; warp w on qpId=w. Union
      // tiles exactly (last warp absorbs the tail) => byte-identical to a whole put.
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
      // All fan-out warps must finish issuing before thread 0 drains and bumps the
      // flag, else the flag could fire before a tail QP's data lands.
      __syncthreads();
    } else if (warpId == 0) {
      putDeep(chunkBaseOffset, peChunkSize, 0);
    }

    // Outbound drain+flag-bump (to nextPeer) and inbound recv-flag wait (from
    // prevPeer) are independent directions, run on two threads to overlap. Rounds
    // can't pipeline: round i+1 forwards exactly what round i received.
    if (threadLinearId == 0) {
      // Drain the outbound put before bumping the flag. Fan-out used numQp RDMA QPs,
      // so ShmemQuietThread(pe) drains all; single-warp keeps the SDMA/P2P quiet.
      if (multiBlock) {
        // Drain only qpId=bid; a per-block ShmemQuietThread(pe) would race the shared
        // CQs across CTAs. Per-QP quiet keeps channels independent (RCCL-style).
        shmem::ShmemQuietThread(nextPeer, bid);
      } else if (fanOut) {
        // Fan-out used numQp QPs in one block; drain ALL so the flag never fires
        // before a tail QP's data lands.
        shmem::ShmemQuietThread(nextPeer);
      } else if (peerIsRdma) {
        // RDMA neighbour: the SDMA-typed memObj quiet does NOT drain the RDMA send
        // queue, so the flag AMO could land before the data PUT drains (stale read).
        // Use the transport-aware quiet (drains all numQpPerPe QPs).
        shmem::ShmemQuietThread(nextPeer);
      } else {
        shmem::ShmemQuietThread(nextPeer, memObj);
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsObj,
                                                     (flagBase + sendDataRank) * sizeof(uint64_t),
                                                     1, core::atomicType::AMO_ADD, nextPeer);
    } else if (threadLinearId == warpSize) {
      // Each round bumps a distinct flag slot exactly once (0 -> 1); wait per-slot.
      int spinCount = 0;
      // System-scope acquire + threadfence: the flag and its chunk are written by a
      // remote peer's RDMA AMO/put, so a relaxed load would leave the chunk stale for
      // this GPU's forward-put/copy-OUT. Mirrors the intra SDMA gather receiver.
      while (core::AtomicLoadSeqCstSystem(flagsArray + flagBase + recvDataRank) < 1) {
        spinCount++;
        if (spinCount > 10000000) {  // Increased timeout threshold
          break;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  // Publish this channel's landing (fence + system-scoped store) so a reassembly
  // block can push sub-range bid over XGMI without a global finish barrier.
  // INERT when chunkReadyFlags==nullptr (standalone ring byte-identical).
  if (chunkReadyFlags != nullptr && threadLinearId == 0) {
    __threadfence_system();
    core::AtomicStoreSeqCstSystem(chunkReadyFlags + bid,
                                  static_cast<uint64_t>(1));
  }
  // Each block resets only its own flag region so concurrent channels never race.
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
  // Equal-sized chunks, so size/npes is exact.
  AllGatherRingKernelBody(myPe, npes, memObj, flagsObj, memObj->size / npes);
}

}  // namespace collective
}  // namespace mori
// Direct-land coherence note: direct-land (NIC writing straight into the output
// tensor) leaves the SDMA reassembly read stale -- an SDMA-read coherence issue on the
// coarse-grained cached output tensor, not a landing/ordering race. The default path
// reads the fine-grained RDMA-scratch ring buffer (SDMA-read-coherent) and is correct.
