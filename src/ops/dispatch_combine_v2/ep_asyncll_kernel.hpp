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
//
// EP AsyncLL on cco, ported from src/ops/dispatch_combine/low_latency_async.cpp.
//
// What AsyncLL is, and why it is a separate family from dispatch_ll: the work is
// cut into nine kernels so the SEND half and the RECV half are separate launches
// with a host-visible seam between them. The caller runs
//
//     dispatch_send()  ->  [its own compute]  ->  dispatch_recv()
//
// and the transfers drain under that compute. dispatch_ll does send and recv in
// one kernel with a grid-wide DispatchSync at the end (EpDispatchInterNodeV2LL_body),
// so there is no seam to put anything in. That seam is the whole feature; the
// routing differs too (AsyncLL deduplicates per destination RANK, v2_ll does not
// deduplicate at all and sends one entry per node per token).
//
// The transport is ccoSdma, the intra-node copy engine. v1 reached the same
// engines through shmem's ShmemPutMemNbiThread, which dispatches per peer on
// globalGpuStates->transportTypes[pe] and lands on SDMA for a same-node peer, so
// this is the same hardware path with a different front end. Two deliberate
// departures from v1, both measured as wins on MI355X (see the SendTransfer and
// RecvTransfer comments):
//
//   * the per-peer payload goes out as ONE copy, not sliced per QP;
//   * the token count rides a single slot per source rank rather than one per QP.
//
// Cross-node peers go over ccoGda instead, through the same EpInterNode* shim
// the v2 / v2_ll families use. The protocol is unchanged -- same slots, same
// count word, same epoch -- and the branch is on `destPe / gpuPerNode != myNode`
// in the four transfer passes. Two things differ by transport, both deliberate:
// a cross-node payload is STRIPED over numQpPerPe QPs where a same-node one is
// sent whole, and the remote count and epoch are atomic ADDs because that is the
// only remote update ccoGda offers at an arbitrary window offset.
//
// Note that AsyncLL talks to every peer directly. On a fabric whose rails are not
// connected to each other this cannot work at all, and the symptom is not an
// error: QP setup happens over TCP and succeeds, then no data moves. The
// v1/v1_ll families aggregate through a same-rail proxy and are immune, which is
// why MORI_ENABLE_RAIL_ONLY is only ever claimed for them. tools/ has a probe
// for the cco half of this; the fabric half is an ib_write_bw across two
// different rails.

#pragma once

// Brings DEF_COMMON_VARS, index_t, EpDispatchCombineArgs, the EpInterNode* ccoGda
// shim used by the internode hooks, and the MORI_EP_INTERNODE_CCO_ENTRY* macros
// the generated TU expands. Including it rather than restating any of that is
// what keeps the two families from drifting apart.
#include "src/ops/dispatch_combine_v2/ep_internode_kernel.hpp"

#ifndef MORI_ASYNCLL_EARLY_NOTIFY
#define MORI_ASYNCLL_EARLY_NOTIFY 0
#endif
#ifndef MORI_ASYNCLL_FUSED_RECV
#define MORI_ASYNCLL_FUSED_RECV 0
#endif
#if MORI_ASYNCLL_FUSED_RECV && !MORI_ASYNCLL_EARLY_NOTIFY
#error "MORI_ASYNCLL_FUSED_RECV requires MORI_ASYNCLL_EARLY_NOTIFY"
#endif

namespace mori {
namespace ops {
namespace v2 {
namespace asyncll {

// The host resolves this protocol per op and carries it in every Plan's cfg.
// LSA and multi-node configurations retain the original protocol.
template <EpInterNodeKernelCfg kConfig>
inline constexpr bool EarlyNotify =
    MORI_ASYNCLL_EARLY_NOTIFY && kConfig.worldSize == kConfig.gpuPerNode;

// ---------------------------------------------------------------------------
// Signal slots.
//
// v1 indexes the recv-count region by (srcPe, qpId) because its transfer is
// striped over numQpPerPe RDMA queue pairs and each stripe announces its own
// chunk. An SDMA queue is not a QP: queues exist so independent WARPS can issue
// in parallel, and splitting one peer's copy across them is measurably slower
// (one peer is one xGMI link, so the pieces only get smaller -- 600 KB in one
// op runs at ~46 GB/s on this part, the same bytes in two at ~35). This port
// therefore sends each peer's payload whole and needs exactly one count slot per
// source rank. The region is still allocated at v1's per-QP size, so these
// indices address a prefix of it and the unused tail stays zero.
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t RecvCountSlot(int srcPe) {
  return static_cast<size_t>(srcPe) * sizeof(uint64_t);
}

// The local staging for a value that is about to be SDMA'd out. ccoSdma moves
// window to window and has no immediate form, so a number has to exist at a real
// address before it can be sent; v1 put the same number straight in the WQE with
// ShmemPutUint64ImmNbiThread. Two words per destination: [0] the token count,
// [1] the combine barrier epoch. They are in flight at different times, but
// separate slots cost 8 bytes per peer and remove the question.
__device__ __forceinline__ size_t SendCountSlot(int destPe) {
  return static_cast<size_t>(destPe) * 2 * sizeof(uint64_t);
}
__device__ __forceinline__ size_t SendFlagSlot(int destPe) {
  return (static_cast<size_t>(destPe) * 2 + 1) * sizeof(uint64_t);
}

// One issuing warp per queue, which is the rule that matters for SDMA: several
// warps posting to one queue is correct but serialises on the in-order commit
// chain. The transfer kernels run one issuing warp per destination PE, so keying
// the queue on the PE gives each its own as long as the comm was created with
// enough; the modulo only folds peers together when it was not.
// The reverse map's encoding: recv slot -> which rank's which input token landed
// there. This is a PUBLIC output -- callers decode it as `tis / maxTokPerRank`,
// `tis % maxTokPerRank` -- so the stride is maxTokPerRank and NOT the wider
// FlatTokenIndex stride the forward slot map uses. EpSrcTokIndex in
// ep_intranode_kernel.hpp carries the same contract for the intranode family,
// with a comment about FlyDSL publishing the identical encoding; a handle has to
// decode the same way whichever kernel produced it.
__device__ __forceinline__ int SrcTokenIndex(const EpInterNodeDeviceCfg& config, int pe,
                                             int srcTokId) {
  return pe * config.MaxNumTokensToSendPerRank() + srcTokId;
}

__device__ __forceinline__ int QueueForPe(const ::mori::cco::ccoDevComm& comm, int destPe) {
  const uint32_t n = comm.sdma.sdmaNumQueue;
  return (n == 0) ? 0 : static_cast<int>(static_cast<uint32_t>(destPe) % n);
}

// The first active lane places the payload, the second its ready word, in one
// reservation and with one doorbell. A zero-byte payload still sends the word.
// Keep exceptionally large, multi-packet copies separate: the chunked put's
// per-lane loop must not interleave the ready word with an unfinished payload.
__device__ __forceinline__ void PostPayloadAndNotify(
    const ::mori::cco::ccoDevComm& comm, int peer, int queueId, ::mori::cco::ccoWindow_t dstWin,
    size_t dstOffset, ::mori::cco::ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
    ::mori::cco::ccoWindow_t notifyDstWin, size_t notifyDstOffset,
    ::mori::cco::ccoWindow_t notifySrcWin, size_t notifySrcOffset) {
  ::mori::cco::ccoSdma sdma{comm};
  const int lane = threadIdx.x & (warpSize - 1);
  if (bytes > ::mori::cco::CCO_SDMA_MAX_COPY_BYTES) {
    sdma.put<::mori::cco::ccoCoopWarp>(peer, dstWin, dstOffset, srcWin, srcOffset, bytes, queueId);
    if (lane == 0)
      sdma.put(peer, notifyDstWin, notifyDstOffset, notifySrcWin, notifySrcOffset, sizeof(uint64_t),
               queueId);
  } else if ((lane == 0 && bytes != 0) || lane == 1) {
    const bool notify = lane == 1;
    sdma.put<::mori::cco::ccoCoopThread, false, false, ::mori::cco::ccoSdmaOptFlagsDefault,
             ::mori::cco::ccoSdmaThreadSameQueue>(
        peer, notify ? notifyDstWin : dstWin, notify ? notifyDstOffset : dstOffset,
        notify ? notifySrcWin : srcWin, notify ? notifySrcOffset : srcOffset,
        notify ? sizeof(uint64_t) : bytes, queueId);
  }
}

}  // namespace asyncll

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLSlotAssign -- dispatch pass 1                         */
/* ---------------------------------------------------------------------------------------------- */
// Routing only: no payload moves here. Groups of numExpertPerToken lanes inside a
// warp cooperate on one token, each lane holding its own expert's destination, so
// the deduplication is a register shuffle rather than a reload.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSlotAssign_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  constexpr int numEpt = kConfig.numExpertPerToken;
  const int tokensPerWarp = warpSize / numEpt;
  const int expertIdx = laneId % numEpt;
  const int inWarpTokIdx = laneId / numEpt;
  const int baseLane = inWarpTokIdx * numEpt;

  for (int warpTokBase = globalWarpId * tokensPerWarp; warpTokBase < args.curRankNumToken;
       warpTokBase += globalWarpNum * tokensPerWarp) {
    const int tokenId = warpTokBase + inWarpTokIdx;
    // warpSize need not be a multiple of numEpt (64 % 6 == 4 at top-k 6): the
    // trailing lanes alias the next warp's first token, so one entry would get two
    // slots and the orphan -- never written by SendCopy -- is read as a token by
    // the receiver. The test at top-k 6/9/10/12 exists for this.
    //
    // Dropping a whole group keeps the shuffle below warp-coherent: inWarpTokIdx
    // is group-uniform, so a group is either entirely in or entirely out.
    if ((inWarpTokIdx >= tokensPerWarp) || (tokenId >= args.curRankNumToken)) continue;

    const int i = tokenId * numEpt + expertIdx;
    const index_t destExpert = args.tokenIndices[i];
    // A negative expert id is the caller's "drop this slot" sentinel.
    const bool dropped = (destExpert < 0);
    const index_t destPe = dropped ? 0 : (destExpert / config.numExpertPerRank);

    // Deduplicate by shuffling the earlier lanes' destinations out of their
    // registers. Every lane runs the same trip count so __shfl stays uniform --
    // an early exit here is what makes it read an inactive lane.
    bool isDuplicate = false;
    for (int j = 0; j < numEpt - 1; ++j) {
      const index_t prevPe = __shfl(destPe, baseLane + j);
      const bool prevDropped = __shfl(static_cast<int>(dropped), baseLane + j) != 0;
      if ((j < expertIdx) && !prevDropped) isDuplicate = isDuplicate || (prevPe == destPe);
    }

    // An out-of-range expert id (an EPLB physical id past worldSize*numExpertPerRank)
    // would index destPeTokenCounter and SendBufSlotOffset out of bounds. Folded
    // into the same null branch as the duplicate rather than skipped early, again
    // to keep the shuffle above coherent.
    const bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
    if (dropped || isDuplicate || peOutOfRange) {
      args.dispDestTokIdMap[i] = NullSendBufSlotOffset(config);
    } else {
      const index_t destTokId = atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = SendBufSlotOffset(config, destPe, destTokId);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpDispatchAsyncLLSendCopy -- dispatch pass 2                         */
/* ---------------------------------------------------------------------------------------------- */
// Payload only: the slots are already decided, so several warps can split one
// token's hidden dimension and the copy is pure bandwidth.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  if constexpr (asyncll::EarlyNotify<kConfig>) {
    // SlotAssign has completed. The following SendTransfer kernel may hand
    // these words directly to SDMA without an additional system fence.
    if (globalWarpId == 0 && laneId < npes) {
      uint8_t* counts = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
      *reinterpret_cast<uint64_t*>(counts + asyncll::SendCountSlot(laneId)) =
          static_cast<uint64_t>(core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId)) + 1;
    }
  }
  const index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  // v1 divides by totalEntries unguarded, which is an integer division by zero on
  // a rank that was handed no tokens -- reachable, and covered by the ragged
  // token-count test.
  if (totalEntries <= 0) return;

  const index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    const index_t entryId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;

    const index_t destTokOffset = args.dispDestTokIdMap[entryId];
    // Deduplicated / dropped entries carry the out-of-range sentinel.
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;

    const index_t srcTokId = entryId / config.numExpertPerToken;
    uint8_t* dst = stagingPtr + static_cast<size_t>(destTokOffset) * xferBytes;

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenOffset,
          static_cast<const uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    // The metadata is small and goes with the first sub-warp, so the others stay
    // on the payload.
    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes, reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes,
          indexBytes);
      if (args.weightsBuf)
        core::WarpCopy<uint8_t, 1>(
            dst + hiddenBytes + indexBytes,
            reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes, weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        core::WarpCopy<uint8_t, 1>(dst + hiddenBytes + indexBytes + weightBytes,
                                   args.scalesBuf + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0)
        reinterpret_cast<index_t*>(dst + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
            static_cast<index_t>(asyncll::SrcTokenIndex(config, myPe, srcTokId));
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*            EpDispatchAsyncLLSendCopyLsa -- dispatch passes 2 and 3, fused (LSA only)           */
/* ---------------------------------------------------------------------------------------------- */
// The same gather, landed in the peer instead of in local staging.
//
// The unfused pair moves the payload twice: SendCopy gathers scattered tokens
// into staging at SendBufSlotOffset(destPe, slot), then SendTransferLsa bulk
// copies that contiguous range into the peer. The second pass exists because
// ccoSdma needs a local, contiguous source to hand the engine. A peer store has
// no such requirement -- the grid is already doing the move -- so for this
// transport the gather can address the peer directly and the bulk copy
// disappears, taking one of AsyncLL's three passes over the payload with it.
//
// Self is the exception: RecvCopy reads this rank's own slice out of staging,
// not out of dispatchInp, so entries for myPe still land where they did.
//
// It also absorbs the count publication, which is a handful of words and the
// only other thing SendTransferLsa did. RecvTransfer is still what sends those
// counts, one kernel boundary later, so the payload is visible to the peer
// before its count is.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendCopyLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // Unconditional, not gated on having tokens: a rank handed none still has to
  // publish its zero or every peer waits on it forever.
  if (globalWarpId == 0 && laneId < npes) {
    uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    *reinterpret_cast<uint64_t*>(sendCount + asyncll::SendCountSlot(laneId)) =
        static_cast<uint64_t>(core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId)) + 1;
  }

  const index_t totalEntries = args.curRankNumToken * config.numExpertPerToken;
  if (totalEntries <= 0) return;

  const index_t warpsPerToken = (globalWarpNum + totalEntries - 1) / totalEntries;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < totalEntries * warpsPerToken; i += globalWarpNum) {
    const index_t entryId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;

    const index_t destTokOffset = args.dispDestTokIdMap[entryId];
    if (destTokOffset >= NullSendBufSlotOffset(config)) continue;

    const int destPe = PeFromSendBufSlotOffset(config, destTokOffset);
    const int slotId = SlotIdFromSendBufSlotOffset(config, destTokOffset);
    const index_t srcTokId = entryId / config.numExpertPerToken;

    // The two halves of SendBufSlotOffset's symmetry, as in the SDMA transfer:
    // locally a slot is indexed by its DESTINATION, in the peer by its SOURCE.
    uint8_t* dst;
    if (destPe == myPe) {
      dst = stagingPtr + static_cast<size_t>(destTokOffset) * xferBytes;
    } else {
      dst = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>(destPe) +
            static_cast<size_t>(SendBufSlotOffset(config, myPe, slotId)) * xferBytes;
    }

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      // Unroll 4, not 1: see the LSA section's note on store concurrency. This is
      // the peer write, and it is the single biggest lever on this transport.
      core::WarpCopy<uint8_t, 4>(
          dst + hiddenOffset,
          static_cast<const uint8_t*>(args.inpTokenBuf) + srcTokId * hiddenBytes + hiddenOffset,
          len);
    }

    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          dst + hiddenBytes, reinterpret_cast<uint8_t*>(args.tokenIndices) + srcTokId * indexBytes,
          indexBytes);
      if (args.weightsBuf)
        core::WarpCopy<uint8_t, 1>(
            dst + hiddenBytes + indexBytes,
            reinterpret_cast<uint8_t*>(args.weightsBuf) + srcTokId * weightBytes, weightBytes);
      if (args.scalesBuf && (scaleBytes > 0))
        core::WarpCopy<uint8_t, 1>(dst + hiddenBytes + indexBytes + weightBytes,
                                   args.scalesBuf + srcTokId * scaleBytes, scaleBytes);
      if (laneId == 0)
        reinterpret_cast<index_t*>(dst + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
            static_cast<index_t>(asyncll::SrcTokenIndex(config, myPe, srcTokId));
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLSendTransfer -- dispatch pass 3                       */
/* ---------------------------------------------------------------------------------------------- */
// Issue and return. Nothing is waited on here: that is what makes the seam between
// dispatch_send and dispatch_recv worth anything -- the copies drain under
// whatever the caller runs next.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendTransfer_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};

  // One issuing warp per destination: warp 0 of the block that owns the PE. More
  // warps on one peer would not make the copy faster (one peer is one link) and
  // would serialise on that queue's commit chain.
  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;

    const int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);

    if constexpr (asyncll::EarlyNotify<kConfig>) {
      if (destPe == myPe) {
        if (laneId == 0) {
          uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
          core::AtomicStoreReleaseSystem(
              reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)),
              static_cast<uint64_t>(tokenNum) + 1);
        }
      } else {
        const size_t remoteBase =
            static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes;
        const size_t localBase =
            static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;
        asyncll::PostPayloadAndNotify(
            comm, destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe),
            EpInterNodeWindow(args.reg(args.offDispatchInp)),
            EpInterNodeOffset(args.reg(args.offDispatchInp), remoteBase),
            EpInterNodeWindow(args.reg(args.offStaging)),
            EpInterNodeOffset(args.reg(args.offStaging), localBase),
            static_cast<size_t>(tokenNum) * xferBytes,
            EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
            EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe)),
            EpInterNodeWindow(args.reg(args.offSendCount)),
            EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendCountSlot(destPe)));
      }
      continue;
    }

    // Publish the count into the local staging word now, while the value is in
    // hand; RecvTransfer is what sends it, after it has drained the payload.
    if (laneId == 0) {
      uint64_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint64_t*>();
      *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sendCount) +
                                   asyncll::SendCountSlot(destPe)) =
          static_cast<uint64_t>(tokenNum) + 1;
    }

    // Self needs no transfer: RecvCopy reads this rank's own slice straight out
    // of staging, exactly as v1 does.
    if (destPe == myPe) continue;
    if (tokenNum <= 0) continue;

    const size_t remoteBase = static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes;
    const size_t localBase = static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;

    if (destPe / config.gpuPerNode != myNode) {
      // Cross-node: RDMA write, STRIPED over the QPs. This is the opposite of
      // what the same-node branch below does, and both are right for their
      // transport. Splitting an SDMA copy only makes each piece smaller on the
      // one link a peer has; splitting an RDMA transfer spreads it over QPs that
      // really do progress in parallel (the op README records that
      // num_qp_per_pe=1 starves this path by about 1.5x).
      //
      // No signal rides along: this phase only posts. RecvTransfer drains and
      // then announces, which is v1's shape and keeps this launch non-blocking.
      const int chunk = core::CeilDiv(tokenNum, config.numQpPerPe);
      for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) {
        const int first = qpId * chunk;
        const int count = min(chunk, tokenNum - first);
        if (count <= 0) break;
        EpInterNodePut(comm, args.reg(args.offDispatchInp),
                       remoteBase + static_cast<size_t>(first) * xferBytes,
                       args.reg(args.offStaging),
                       localBase + static_cast<size_t>(first) * xferBytes,
                       static_cast<size_t>(count) * xferBytes, destPe, qpId);
      }
      continue;
    }

    const int q = asyncll::QueueForPe(comm, destPe);
    // The destination addresses ITS copy of dispatchInp at this rank's slice, and
    // the source is this rank's staging at the destination's slice -- the two
    // halves of SendBufSlotOffset's symmetry, and the one place a transposed
    // index silently delivers another rank's tokens.
    //
    // Whole, not sliced per QP. This is the measured departure from v1: at 64
    // tokens a peer's payload is ~600 KB, which SDMA moves at ~46 GB/s in one op
    // and ~35 GB/s as two 300 KB ones. Warp scope so lane 0 posts for the warp
    // without an explicit if.
    sdma.put<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                       EpInterNodeWindow(args.reg(args.offDispatchInp)),
                                       EpInterNodeOffset(args.reg(args.offDispatchInp), remoteBase),
                                       EpInterNodeWindow(args.reg(args.offStaging)),
                                       EpInterNodeOffset(args.reg(args.offStaging), localBase),
                                       static_cast<size_t>(tokenNum) * xferBytes, q);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpDispatchAsyncLLRecvTransfer -- dispatch pass 4                       */
/* ---------------------------------------------------------------------------------------------- */
// Drain, announce, then wait for everyone else's announcement.
//
// The quiet lives here rather than in SendTransfer for the same reason v1 puts it
// here: a quiet in the send kernel would block the launch that is supposed to
// return immediately, and the seam would buy nothing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvTransfer_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;

    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    // Both drains are WARP-COLLECTIVE and must stay above the laneId guard:
    // ccoSdma::quietQueue syncs the group, and EpInterNodeQuiet takes ccoGda's
    // warp-level CQ poll lock. Called from one lane they deadlock rather than
    // misbehave, which is why the guard is below them and not above.
    if (destPe != myPe) {
      if (sameNode) {
        // Our payload to this peer is out of the queue before its count word
        // goes, so the peer seeing a non-zero count has the tokens too.
        sdma.quietQueue<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                                  asyncll::QueueForPe(comm, destPe));
      } else {
        // Same argument over RDMA, but every QP the send leg striped across has
        // to be drained -- EpInterNodeQuiet loops them, flushing each context.
        EpInterNodeQuiet(comm, destPe, config.numQpPerPe);
      }
    }

    // Early notifications were already placed after their payloads by SendTransfer.
    // Keep local completion: incoming ready words do not prove our sources are free.
    if constexpr (asyncll::EarlyNotify<kConfig>) continue;
    if (laneId != 0) continue;

    const uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
    const uint64_t countValue =
        *reinterpret_cast<const uint64_t*>(sendCount + asyncll::SendCountSlot(destPe));

    if (destPe == myPe) {
      // No engine for a local store; the prefix sum still has to see the slot.
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)), countValue);
    } else if (sameNode) {
      sdma.put(destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offRecvTokenNum)),
               EpInterNodeOffset(args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe)),
               EpInterNodeWindow(args.reg(args.offSendCount)),
               EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendCountSlot(destPe)),
               sizeof(uint64_t), asyncll::QueueForPe(comm, destPe));
    } else {
      // Cross-node. An ADD rather than a store, because that is the only remote
      // update ccoGda offers at an arbitrary window offset -- and it is exactly
      // equivalent here: combine clears this slot every round, so adding
      // (tokenNum + 1) to zero lands the same value v1 stores with
      // ShmemPutUint64ImmNbiThread.
      EpInterNodeAtomicAdd(comm, args.reg(args.offRecvTokenNum), asyncll::RecvCountSlot(myPe),
                           countValue, destPe);
    }
  }

  // Wait for every source to have announced. One lane per source rank; the count
  // is biased by one so that a rank sending zero tokens still publishes a value
  // distinguishable from "has not arrived".
  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();
  if (blockId == 0 && warpId == 0 && laneId < npes) {
    const uint64_t* slot =
        reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
    while (core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(slot)) == 0) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpDispatchAsyncLLRecvCopy -- dispatch pass 5                         */
/* ---------------------------------------------------------------------------------------------- */
// Unpack what arrived into the dispatch output, and build the two maps combine
// needs. Per-PE output offsets come from a warp prefix sum rather than an atomic,
// so every block derives the same layout independently.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  // The absolute bound only. The prefix sum below is one wavefront wide, so the
  // real limit is this arch's wave -- 32 on gfx1250, not 64 -- and waveSize is
  // not part of the rendered cfg, so it cannot be said here. The exact check is
  // host-side in EpAsyncLLRenderSource, which every compile goes through and
  // which does know it. This catches only a worldSize no wave could hold.
  static_assert(kConfig.worldSize <= 64,
                "the per-PE offset prefix sum is one wavefront wide; worldSize must fit it");

  const int blocksPerPe = blockNum / npes;
  if (blocksPerPe <= 0) return;
  const int destPe = blockId / blocksPerPe;
  if (destPe >= npes) return;

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();

  uint64_t myPeTokens = 0;
  if (laneId < npes)
    myPeTokens =
        *reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId)) - 1;

  uint64_t inclusive = myPeTokens;
  for (int offset = 1; offset < npes; offset <<= 1) {
    const uint64_t n = __shfl_up(inclusive, offset);
    if (laneId >= offset) inclusive += n;
  }
  uint64_t peExclusive = __shfl_up(inclusive, 1);
  if (laneId == 0) peExclusive = 0;

  const uint64_t peOffset = __shfl(peExclusive, destPe);
  const uint64_t recvTokenNum = __shfl(myPeTokens, destPe);
  const uint64_t totalTokens = __shfl(inclusive, npes - 1);

  const uint8_t* stagingPtr = (destPe != myPe)
                                  ? args.reg(args.offDispatchInp)->template GetAs<const uint8_t*>()
                                  : args.reg(args.offStaging)->template GetAs<const uint8_t*>();
  stagingPtr += static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;

  const int peWarps = blocksPerPe * warpNum;
  const int localWarpId = (blockId % blocksPerPe) * warpNum + warpId;
  const index_t warpsPerToken =
      (recvTokenNum > 0) ? static_cast<index_t>((peWarps + recvTokenNum - 1) / recvTokenNum) : 1;
  const size_t hiddenBytesPerWarp =
      ((hiddenBytes + warpsPerToken - 1) / warpsPerToken + 15) & ~static_cast<size_t>(15);

  for (int i = localWarpId; i < static_cast<int>(recvTokenNum) * warpsPerToken; i += peWarps) {
    const index_t tokenId = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;
    const index_t destTokId = static_cast<index_t>(peOffset) + tokenId;

    const size_t hiddenOffset = inTokenPartId * hiddenBytesPerWarp;
    if (hiddenOffset < hiddenBytes) {
      const size_t len = min(hiddenBytesPerWarp, hiddenBytes - hiddenOffset);
      core::WarpCopy<uint8_t, 1>(args.reg(args.offDispatchOut)->template GetAs<uint8_t*>() +
                                     destTokId * hiddenBytes + hiddenOffset,
                                 stagingPtr + tokenId * xferBytes + hiddenOffset, len);
    }

    if (inTokenPartId == 0) {
      core::WarpCopy<uint8_t, 1>(
          args.reg(args.offOutIndices)->template GetAs<uint8_t*>() + destTokId * indexBytes,
          stagingPtr + tokenId * xferBytes + hiddenBytes, indexBytes);
      if (args.weightsBuf)
        core::WarpCopy<uint8_t, 1>(
            args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>() +
                destTokId * weightBytes,
            stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
      if (scaleBytes > 0)
        core::WarpCopy<uint8_t, 1>(
            args.reg(args.offOutScales)->template GetAs<uint8_t*>() + destTokId * scaleBytes,
            stagingPtr + tokenId * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
      if (laneId == 0) {
        // Where this token came from, so combine's send leg can put the result
        // back in the slot it arrived in.
        args.dispReceiverIdxMap[destTokId] = SendBufSlotOffset(config, destPe, tokenId);
        args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>()[destTokId] =
            reinterpret_cast<const index_t*>(stagingPtr + tokenId * xferBytes + hiddenBytes +
                                             indexBytes + weightBytes + scaleBytes)[0];
      }
    }
  }

  // One warp in the grid owns the epilogue: publish the total, clear what the
  // next call reads, and advance the combine barrier epoch.
  if (globalWarpId == 0) {
    if (laneId == 0) args.totalRecvTokenNum[0] = static_cast<index_t>(totalTokens);
    // The recv counts are deliberately NOT cleared here. Combine's send leg needs
    // to know how many tokens each peer sent us -- that is exactly how many we owe
    // it back -- and this is the only place that number exists. v1 clears it in
    // EpCombineLowLatencyAsyncSendTransfer for the same reason, after reading it.
    if (laneId < npes) args.destPeTokenCounter[laneId] = 0;
    if (laneId == 0) {
      args.dispatchGridBarrier[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1ull);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                   LSA transfer variants -- same protocol, peer stores instead                  */
/* ---------------------------------------------------------------------------------------------- */
//
// Drop-in replacements for the two dispatch transfer passes, for the case where
// every peer is on this node. The protocol, the slot arithmetic and the count
// word are unchanged; only how the bytes cross differs.
//
// The copy engine tops out near 60 GB/s and needs ~6 us to start, while a peer
// store over the flat LSA VA runs at xGMI speed, so at these sizes (a per-peer
// payload of ~600 KB at 64 tokens x 7168 x bf16) the engine is latency-bound and
// a store should win.
//
// It does at 64 tokens and nowhere else. Measured on an IDLE, RESERVED MI355X
// node (crsuse2 v2-008) at EP8, hidden 7168, bf16, dispatch+combine as a CUDA
// graph; two full sweeps agreed to within 0.3 percentage points at every point:
//
//   tok/rank      64     128     256     512    1024    2048   (us)
//   sdma       102.0   136.8   194.3   292.1   523.1   982.0
//   lsa         97.7   138.9   201.5   320.3   581.4  1094.2
//   delta      -4.2%   +1.6%   +3.6%   +9.7%  +11.1%  +11.4%
//
// The reasoning above predicted the opposite sign at the top end -- the engine
// does saturate there, so peer stores "should" win -- and the measurement says
// lsa loses by a widening margin instead. Two things the prediction left out:
// the transfer is a kernel of its own, so nothing overlaps it either way and
// SDMA's separate hardware is pure gain; and a grid doing the copy contends with
// its own HBM traffic, where the engines do not. The ordinary v2 intranode
// dispatch, which writes straight into the peer and stages nothing, reaches
// 856.9 us on the same point, so the grid is not short of bandwidth -- it is
// short of bandwidth *while also* running the staging copies this protocol adds.
//
// That staging is why neither transport closes the gap to v2 native: AsyncLL
// moves each payload THREE times (input -> staging, staging -> peer, peer ->
// dispatch output) because that is what an RDMA leg needs, and changing the
// middle hop leaves the other two alone.
//
// So: sdma is the default on evidence, not just because it is the internode
// shape. lsa is worth keeping for the small-batch case and for a caller with an
// empty seam, since it also costs CUs that an SDMA transfer leaves free.
//
// STORE CONCURRENCY IS THE LEVER ON THIS TRANSPORT. The three peer-facing copies
// here (the fused gather and the two bulk transfers) carry Unroll=4 rather than
// WarpCopy's default of 1. At Unroll=1 a lane has exactly ONE 16 B store in
// flight and the warp walks a 14 KB token as 28 dependent round trips to the
// peer; raising it to 4 keeps four in flight and the latency stops being
// serial. Note that EpCombineAsyncLLSendCopy_body, whose copy is LOCAL, was
// already at 4 -- only the peer-facing sites had been left at 1, which is the
// wrong way round, since a peer store is the one with the long latency to hide.
//
// Measured on an idle gfx1250 node at EP4, hidden 7168, topk 8, bf16, graph;
// five interleaved rounds per arm, medians, arms differing only in
// MORI_SOURCE_ROOT. us for the dispatch+combine pair:
//
//   tok/rank        128      512     2048
//   Unroll=1      646.5    813.6   1364.1
//   Unroll=4      644.3    715.4    974.5
//   delta         -0.3%   -12.1%   -28.6%
//
// The dispatch half alone is -13.5% / -45.1% / -53.9%: 202 -> 438 GB/s at 2048
// in bench_ep.py's accounting. Unroll=8 was measured in the same matrix and is
// NOT better -- it ties at 512 and 2048 and loses 1.4% at 128, so 4 is the
// setting, not merely a direction.
//
// Two things tried in the same matrix and rejected, both documented in
// python/mori/ops/dispatch_combine_v2/README.md: moving the transfer onto the
// gfx1250 TDM engine, and padding the slot stride to a 128 B row.
//
// INTERMITTENT HANG observed on gfx125x UNDER CUDA GRAPH CAPTURE before the
// count-reset ordering fix below. The numbers above are MI355X (gfx950, wave64).
// On gfx1250 at EP4, hidden 7168, 64 tokens:
//
//   lsa  + eager          passes every run, output verifies
//   lsa  + graph capture  hung 6 runs out of 8
//   sdma + graph capture  passes every run
//
// These runs observed hangs only for LSA under graph capture. They do not
// establish that eager execution or other architectures are immune. An earlier
// note incorrectly called LSA broken on wave32 based on one capturing benchmark
// run compared with a single-node test that never captures.
//
// A protocol race exists independently of payload visibility: CombineRecvTransferLsa
// used to publish its epoch, wait for peers, and only THEN clear recv counts. A
// fast peer could pass the epoch handshake and publish its next dispatch count
// before our late clear erased it. SDMA clears in CombineSendTransfer, before
// the epoch announcement, so it does not have this window. LSA now clears each
// peer's count before its release epoch store, paired with an acquire wait.
//
// Fixing it removes the hang. Eight interleaved pairs on an idle gfx1250 at the
// shape above, base and fixed alternating, no contaminated samples:
//
//   baseline  1 pass / 7 hang
//   fixed     8 pass / 0 hang
//
// The two arms shared one host library and differed only in MORI_SOURCE_ROOT, so
// the JIT compiled a different kernel for each. That was checked rather than
// assumed -- an #error planted in the baseline tree fired when the root pointed
// there and did not fire when it pointed at the fixed tree. Without that check
// "both arms passed" cannot be told from "both arms ran the same code", which is
// how an earlier 8/8-vs-8/8 run reached no conclusion.
//
// Why eager never hit it, when the race does not need graphs: the window is the
// gap between a peer passing the epoch handshake and our clear landing, and
// eager's host round trip between kernels is wide enough that ranks rarely sit
// in it. Replay closes that gap and the ranks run shoulder to shoulder. So
// "eager passes" was never evidence against this race -- only evidence that the
// window was wide. Reading it as a contradiction cost a round of disbelief.
//
// A separate payload-visibility defect has not been established; adding a fence
// after WarpCopy alone would not fix a count that a later kernel overwrites with
// zero.
//
// Measure the RATE before believing any claim about this. Several conclusions
// here were drawn from single runs and were wrong; at a 7-in-8 hit rate a
// passing run means nothing, and on a shared box an interfering job can hide the
// failure entirely -- a 55-run matrix on a busy node saw 0 hangs where this
// 16-run matrix on an idle one saw 7.
//
// test_asyncll.py drives eager and the sdma transport, which is why the gfx1250
// enablement passed without touching this.

namespace asyncll {

// Splits one peer's byte range across the warps assigned to it. Chunks are
// 16 B-aligned so every warp's slice starts on a vector boundary; WarpCopy is
// what actually moves them.
struct LsaShare {
  int destPe;
  size_t offset;
  size_t bytes;
};

__device__ __forceinline__ LsaShare LsaShareFor(int globalWarpId, int globalWarpNum, int npes,
                                                size_t totalBytes) {
  const int warpsPerPe = globalWarpNum / npes;
  LsaShare share{globalWarpId / (warpsPerPe > 0 ? warpsPerPe : 1), 0, 0};
  if (warpsPerPe <= 0 || share.destPe >= npes) {
    share.destPe = -1;
    return share;
  }
  const int chunkId = globalWarpId % warpsPerPe;
  const size_t perWarp =
      ((totalBytes + warpsPerPe - 1) / warpsPerPe + 15) & ~static_cast<size_t>(15);
  share.offset = static_cast<size_t>(chunkId) * perWarp;
  share.bytes = (share.offset < totalBytes) ? min(perWarp, totalBytes - share.offset) : 0;
  return share;
}

}  // namespace asyncll

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLSendTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // Publish this rank's counts into the local staging word, as the SDMA variant
  // does. RecvTransfer is still what hands them to the peers.
  if (globalWarpId == 0 && laneId < npes) {
    uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    *reinterpret_cast<uint64_t*>(sendCount + asyncll::SendCountSlot(laneId)) =
        static_cast<uint64_t>(core::AtomicLoadRelaxed(args.destPeTokenCounter + laneId)) + 1;
  }

  // Every warp in the grid carries bytes, rather than one warp per peer: a peer
  // store has no engine to hand the work to, so the copy is only as wide as the
  // grid made it.
  const int warpsPerPe = globalWarpNum / npes;
  if (warpsPerPe <= 0) return;
  const int destPe = globalWarpId / warpsPerPe;
  if (destPe >= npes || destPe == myPe) return;
  // Single-node by construction: a peer store needs the peer inside this rank's
  // flat LSA window, which only same-node ranks are. The host gate refuses
  // async_ll_transport="lsa" on a multi-node config, so this is unreachable
  // rather than a silent drop -- kept as a guard, not as a fallback.
  if (destPe / config.gpuPerNode != myNode) return;

  const int tokenNum = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
  if (tokenNum <= 0) return;

  const asyncll::LsaShare share = asyncll::LsaShareFor(globalWarpId, globalWarpNum, npes,
                                                       static_cast<size_t>(tokenNum) * xferBytes);
  if (share.bytes == 0) return;

  uint8_t* dst = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>(destPe) +
                 static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes + share.offset;
  const uint8_t* src = args.reg(args.offStaging)->template GetAs<const uint8_t*>() +
                       static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes +
                       share.offset;
  core::WarpCopy<uint8_t, 4>(dst, src, share.bytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  // No drain to do: the payload stores were issued by the PREVIOUS kernel, and a
  // kernel boundary already made them visible system-wide. That is the same
  // argument the SDMA variant uses for its count word, applied to the payload --
  // and it is why this variant needs no quiet at all.
  if (globalWarpId == 0 && laneId < npes) {
    const int destPe = laneId;
    const uint8_t* sendCount = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
    const uint64_t countValue =
        *reinterpret_cast<const uint64_t*>(sendCount + asyncll::SendCountSlot(destPe));
    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    if (destPe == myPe || sameNode) {
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>(destPe);
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(myPe)), countValue);
    }
    // No cross-node arm, for the same reason the payload has none: this variant
    // is single-node and the host gate enforces it.
  }

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();
  if (blockId == 0 && warpId == 0 && laneId < npes) {
    const uint64_t* slot =
        reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
    while (core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(slot)) == 0) {
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;

  const uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<const uint8_t*>();

  // The counts are READ here and cleared in the next pass. They cannot be
  // cleared here: every warp in the grid sizes its copy from them, and a
  // grid-wide "all reads done" point does not exist inside one kernel --
  // __syncthreads is per block. The SDMA variant gets away with clearing in
  // place only because there the whole peer is one warp's business.
  const int warpsPerPe = globalWarpNum / npes;
  if (warpsPerPe <= 0) return;
  const int destPe = globalWarpId / warpsPerPe;
  if (destPe >= npes) return;

  // Biased by one, so 0 means "no dispatch announced this round". Test the raw
  // value rather than subtracting first: an unpaired combine finds 0 here and
  // `announced - 1` would be UINT64_MAX, which the old `tokenNum > 0` test waves
  // through and LsaShareFor then turns into a grid of peer stores running far
  // past the buffer. Same defect as the SDMA combine send; see its comment.
  const uint64_t announced = core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(
      reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe))));

  if (destPe != myPe && destPe / config.gpuPerNode == myNode && announced > 1) {
    const uint64_t tokenNum = announced - 1;
    const asyncll::LsaShare share =
        asyncll::LsaShareFor(globalWarpId, globalWarpNum, npes, tokenNum * hiddenBytes);
    if (share.bytes > 0) {
      uint8_t* dst = args.reg(args.offCombineInp)->template GetAs<uint8_t*>(destPe) +
                     static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes +
                     share.offset;
      const uint8_t* src = args.reg(args.offCombineStaging)->template GetAs<const uint8_t*>() +
                           static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes +
                           share.offset;
      core::WarpCopy<uint8_t, 4>(dst, src, share.bytes);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvTransferLsa_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  const uint64_t flag = args.crossDeviceBarrierFlag[0];

  // Same epoch handshake as the SDMA variant, with a peer store in place of the
  // engine. The payload stores landed in the previous kernel, so publishing the
  // epoch here is already ordered behind them.
  if (globalWarpId == 0 && laneId < npes) {
    const int destPe = laneId;
    if (destPe == myPe || destPe / config.gpuPerNode == myNode) {
      // All warps finished reading these counts in SendTransferLsa. Re-arm this
      // peer's slot BEFORE allowing it to start the next dispatch: if we clear
      // after the epoch wait, a faster peer can already have written its next
      // count and our late zero would erase it, leaving dispatch spinning forever.
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(
          reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe)), uint64_t{0});
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>(destPe);
      // The same lane clears the slot and notifies its writer. The peer's acquire
      // must observe this clear before its next dispatch can reuse the slot.
      core::AtomicStoreReleaseSystem(barrier + myPe, flag);
    }
  }

  if (blockId == 0 && warpId == 0 && laneId < npes) {
    uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    while (__hip_atomic_load(barrier + laneId, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) !=
           flag) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpCombineAsyncLLSendCopy -- combine pass 1                           */
/* ---------------------------------------------------------------------------------------------- */
// Put each post-expert token back into the staging slot it was dispatched from.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  if constexpr (asyncll::EarlyNotify<kConfig>) {
    if (globalWarpId == 0 && laneId < npes) {
      uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      uint64_t* count = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId));
      uint8_t* sendCounts = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
      // Dispatch recv drained its sends, so its old count source may be reused.
      *reinterpret_cast<uint64_t*>(sendCounts + asyncll::SendCountSlot(laneId)) =
          core::AtomicLoadRelaxedSystem(count) - 1;
      core::AtomicStoreRelaxedSystem(count, uint64_t{0});
      *reinterpret_cast<uint64_t*>(sendCounts + asyncll::SendFlagSlot(laneId)) =
          args.crossDeviceBarrierFlag[0];
      // This kernel must complete before SendTransfer publishes the epoch. In
      // particular the count clear must land before a peer starts its next round.
    }
  }
  const index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // Combine's own staging, not dispatch's: writing dispatch's here would race
  // its transfer still reading it. See asyncll_regions.
  uint8_t* stagingPtr = args.reg(args.offCombineStaging)->template GetAs<uint8_t*>();

  for (int tokenId = globalWarpId; tokenId < totalRecvTokenNum; tokenId += globalWarpNum) {
    index_t stagingTokId = 0;
    if (laneId == 0) stagingTokId = args.dispReceiverIdxMap[tokenId];
    stagingTokId = __shfl(stagingTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        stagingPtr + static_cast<size_t>(stagingTokId) * hiddenBytes,
        static_cast<const uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes, hiddenBytes);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpCombineAsyncLLSendTransfer -- combine pass 2                         */
/* ---------------------------------------------------------------------------------------------- */
// The dispatch transfer with the endpoints swapped: what arrived in dispatchInp
// goes back to combineInp, at the same slot arithmetic and the same stride. The
// payload is hidden only -- no indices, no source id -- so the stride is
// hiddenBytes rather than xferBytes.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendTransfer_body(EpDispatchCombineArgs args,
                                                  const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};

  uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;

    if constexpr (asyncll::EarlyNotify<kConfig>) {
      const uint8_t* sendCounts = args.reg(args.offSendCount)->template GetAs<const uint8_t*>();
      uint64_t tokenNum = 0;
      if (laneId == 0)
        tokenNum = *reinterpret_cast<const uint64_t*>(sendCounts + asyncll::SendCountSlot(destPe));
      tokenNum = __shfl(static_cast<unsigned long long>(tokenNum), 0);
      if (destPe == myPe) {
        if (laneId == 0) {
          uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
          const uint64_t flag =
              *reinterpret_cast<const uint64_t*>(sendCounts + asyncll::SendFlagSlot(destPe));
          core::AtomicStoreReleaseSystem(barrier + myPe, flag);
        }
      } else {
        const size_t remoteBase =
            static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes;
        const size_t localBase =
            static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes;
        asyncll::PostPayloadAndNotify(
            comm, destPe - args.lsaBase, asyncll::QueueForPe(comm, destPe),
            EpInterNodeWindow(args.reg(args.offCombineInp)),
            EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
            EpInterNodeWindow(args.reg(args.offCombineStaging)),
            EpInterNodeOffset(args.reg(args.offCombineStaging), localBase),
            static_cast<size_t>(tokenNum) * hiddenBytes,
            EpInterNodeWindow(args.reg(args.offCrossDeviceBarrier)),
            EpInterNodeOffset(args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t)),
            EpInterNodeWindow(args.reg(args.offSendCount)),
            EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)));
      }
      continue;
    }

    // How many tokens this peer sent us during dispatch is how many we owe it
    // back. Dispatch left the slot alone precisely so this read works; clear it
    // now, which is also what re-arms the "0 means not arrived" test the next
    // dispatch's RecvTransfer spins on.
    uint64_t announced = 0;
    if (laneId == 0) {
      uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe));
      announced = core::AtomicLoadRelaxedSystem(slot);
      core::AtomicStoreRelaxedSystem(slot, uint64_t{0});
    }
    announced = __shfl(static_cast<unsigned long long>(announced), 0);

    // The count is biased by one, so 0 is "no dispatch announced this round"
    // rather than "zero tokens" -- and subtracting from it wraps. A combine with
    // no dispatch behind it (combine() twice on one routing, combine() before any
    // dispatch, combine_recv() after a whole combine) lands here with 0, and the
    // `tokenNum == 0` test below cannot catch UINT64_MAX: sdma.put would then be
    // handed a ~2^64-byte copy. Reproduced on 8xMI308X as a PERMISSION_FAULT on
    // every GPU in the job. Nothing this kernel can do makes the call correct, so
    // this only keeps a caller's mistake from becoming a memory fault; the host
    // raises on it first (see the call-order check in dispatch_combine_op.py).
    if (announced == 0) continue;
    const uint64_t tokenNum = announced - 1;

    if (destPe == myPe) continue;
    if (tokenNum == 0) continue;

    const size_t remoteBase = static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes;
    const size_t localBase =
        static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes;

    if (destPe / config.gpuPerNode != myNode) {
      // Striped over the QPs, as the dispatch send leg is and for the same
      // reason. RecvTransfer drains them.
      const int chunk = core::CeilDiv(static_cast<int>(tokenNum), config.numQpPerPe);
      for (int qpId = 0; qpId < config.numQpPerPe; ++qpId) {
        const int first = qpId * chunk;
        const int count = min(chunk, static_cast<int>(tokenNum) - first);
        if (count <= 0) break;
        EpInterNodePut(comm, args.reg(args.offCombineInp),
                       remoteBase + static_cast<size_t>(first) * hiddenBytes,
                       args.reg(args.offCombineStaging),
                       localBase + static_cast<size_t>(first) * hiddenBytes,
                       static_cast<size_t>(count) * hiddenBytes, destPe, qpId);
      }
      continue;
    }

    const int q = asyncll::QueueForPe(comm, destPe);
    sdma.put<::mori::cco::ccoCoopWarp>(
        destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCombineInp)),
        EpInterNodeOffset(args.reg(args.offCombineInp), remoteBase),
        EpInterNodeWindow(args.reg(args.offCombineStaging)),
        EpInterNodeOffset(args.reg(args.offCombineStaging), localBase), tokenNum * hiddenBytes, q);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                         EpCombineAsyncLLRecvTransfer -- combine pass 3                         */
/* ---------------------------------------------------------------------------------------------- */
// Drain our sends, then an all-ranks handshake on a monotone epoch. The epoch
// comes from crossDeviceBarrierFlag, which dispatch's RecvCopy bumped, so it
// never repeats across launches and the slots need no clearing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvTransfer_body(EpDispatchCombineArgs args,
                                                  const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  ::mori::cco::ccoSdma sdma{comm};
  const uint64_t flag = args.crossDeviceBarrierFlag[0];

  for (int destPe = blockId; destPe < npes; destPe += blockNum) {
    if (warpId != 0) continue;
    const bool sameNode = (destPe / config.gpuPerNode == myNode);
    // Warp-collective, above the laneId guard. See the note in dispatch's
    // RecvTransfer: from a single lane these deadlock.
    if (destPe != myPe) {
      if (sameNode)
        sdma.quietQueue<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                                  asyncll::QueueForPe(comm, destPe));
      else
        EpInterNodeQuiet(comm, destPe, config.numQpPerPe);
    }

    if constexpr (asyncll::EarlyNotify<kConfig>) continue;
    if (laneId != 0) continue;

    uint8_t* sendFlags = args.reg(args.offSendCount)->template GetAs<uint8_t*>();
    core::AtomicStoreRelaxedSystem(
        reinterpret_cast<uint64_t*>(sendFlags + asyncll::SendFlagSlot(destPe)), flag);
    // The copy engine reads this word through its own path, not through the
    // wavefront's cache hierarchy, and the store above is in THIS kernel rather
    // than a previous launch -- so without the fence the engine can read the slot
    // before the value lands and ship a stale epoch. The peer then waits on a
    // barrier value that never arrives: a hang, not a wrong answer. Dispatch's
    // count word needs no equivalent because it is written in SendTransfer and
    // sent from RecvTransfer, with a kernel boundary in between.
    __threadfence_system();

    if (destPe == myPe) {
      uint8_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint8_t*>();
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint64_t*>(barrier) + myPe, flag);
    } else if (sameNode) {
      sdma.put(destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCrossDeviceBarrier)),
               EpInterNodeOffset(args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t)),
               EpInterNodeWindow(args.reg(args.offSendCount)),
               EpInterNodeOffset(args.reg(args.offSendCount), asyncll::SendFlagSlot(destPe)),
               sizeof(uint64_t), asyncll::QueueForPe(comm, destPe));
    } else {
      // Cross-node publishes the epoch as an ADD OF ONE, which is the same value
      // the two branches above store. crossDeviceBarrierFlag is bumped once per
      // round by dispatch's RecvCopy, and this rank writes a given peer's slot
      // exactly once per round, so after round N the slot has been incremented N
      // times and holds N -- precisely the epoch the wait below compares against.
      // ccoGda offers only an add at an arbitrary window offset, and here that is
      // enough rather than a compromise.
      EpInterNodeAtomicAdd(comm, args.reg(args.offCrossDeviceBarrier), myPe * sizeof(uint64_t),
                           1ull, destPe);
    }
  }

  if (blockId == 0 && warpId == 0 && laneId < npes) {
    uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    while (core::AtomicLoadRelaxedSystem(barrier + laneId) != flag) {
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpCombineAsyncLLRecvCopy -- combine pass 4                           */
/* ---------------------------------------------------------------------------------------------- */
// Reduce each local token's top-k copies back into one. The sources are found by
// decoding dispDestTokIdMap, the same map dispatch wrote, so a dropped or
// deduplicated slot decodes to pe == worldSize and contributes nothing.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  constexpr int kTopk = kConfig.numExpertPerToken;

  extern __shared__ char epAsyncLlShared[];
  T** srcPtrs = reinterpret_cast<T**>(epAsyncLlShared) + warpId * kTopk;

  if (args.curRankNumToken == 0) return;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  for (int i = globalWarpId; i < args.curRankNumToken * mwIter.warpsPerItem; i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    for (int j = laneId; j < kTopk; j += warpSize) {
      const index_t destTokId = args.dispDestTokIdMap[tokenId * kTopk + j];
      const index_t destPe = PeFromSendBufSlotOffset(config, destTokId);

      if (destPe < npes) {
        // Own tokens were never transferred -- they are still in combine's
        // staging where SendCopy put them.
        T* base = (destPe != myPe) ? args.reg(args.offCombineInp)->template GetAs<T*>()
                                   : args.reg(args.offCombineStaging)->template GetAs<T*>();
        srcPtrs[j] = base + static_cast<size_t>(destTokId) * hiddenDim + hiddenDimOffset;
      } else {
        srcPtrs[j] = nullptr;
      }
    }

    T* outPtr =
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset;
    core::WarpAccum<T, 4>(outPtr, srcPtrs, nullptr, kTopk, hiddenDimSize);
  }
}

#if MORI_ASYNCLL_FUSED_RECV
// The control wave checks one outgoing queue per lane. Thread-scope quiet has
// no collective barrier and these queues only wait on independently progressing
// SDMA engines. Kernel completion waits for this wave, so sources remain owned
// until all outgoing copies finish, without making other CTAs wait on this CTA.
template <EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void EpAsyncLLFusedQuiet(EpDispatchCombineArgs args,
                                                    const ::mori::cco::ccoDevComm& comm) {
  static_assert(asyncll::EarlyNotify<kConfig>,
                "fused AsyncLL receive requires early notifications on a single node");
  const int lane = threadIdx.x & (warpSize - 1);
  if (blockIdx.x == 0 && threadIdx.x < warpSize && lane < kConfig.worldSize && lane != args.rank) {
    ::mori::cco::ccoSdma sdma{comm};
    sdma.quietQueue<::mori::cco::ccoCoopThread>(lane - args.lsaBase,
                                                asyncll::QueueForPe(comm, lane));
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchAsyncLLRecvCopyFused_body(EpDispatchCombineArgs args,
                                                    const ::mori::cco::ccoDevComm& comm) {
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint8_t* counts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
      uint64_t* slot = reinterpret_cast<uint64_t*>(counts + asyncll::RecvCountSlot(lane));
      while (__hip_atomic_load(slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0)
        __builtin_amdgcn_s_sleep(1);
    }
    // Pair the successful system-scope observations with an acquire before
    // handing the ready data to every consumer wave in this CTA.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  // HIP's barrier includes global/local workgroup release and acquire fences:
  // propagate the first wave's system acquire to every consumer in this CTA.
  // All threads participate, including CTAs whose copy body has no work.
  __syncthreads();
  EpDispatchAsyncLLRecvCopy_body<kConfig, T>(args);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLRecvCopyFused_body(EpDispatchCombineArgs args,
                                                   const ::mori::cco::ccoDevComm& comm) {
  const uint64_t flag = args.crossDeviceBarrierFlag[0];
  EpAsyncLLFusedQuiet<kConfig>(args, comm);
  const int lane = threadIdx.x & (warpSize - 1);
  if (threadIdx.x < warpSize) {
    if (lane < kConfig.worldSize) {
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
      while (__hip_atomic_load(barrier + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != flag)
        __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  __syncthreads();
  // Even a zero-token rank must execute both completion paths above before the
  // original copy body returns early and the caller can start the next round.
  EpCombineAsyncLLRecvCopy_body<kConfig, T>(args);
}
#endif

}  // namespace v2
}  // namespace ops
}  // namespace mori

// The entry macros come from ep_internode_kernel.hpp; the generated TU picks the
// _LOCAL form for the passes that take no communicator.
