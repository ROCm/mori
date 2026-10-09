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
// Cross-node is not implemented here. The hooks are marked TODO(internode) and
// name the ccoGda helpers in ep_internode_kernel.hpp that already do the job for
// the v2 / v2_ll families.

#pragma once

// Brings DEF_COMMON_VARS, index_t, EpDispatchCombineArgs, the EpInterNode* ccoGda
// shim used by the internode hooks, and the MORI_EP_INTERNODE_CCO_ENTRY* macros
// the generated TU expands. Including it rather than restating any of that is
// what keeps the two families from drifting apart.
#include "src/ops/dispatch_combine_v2/ep_internode_kernel.hpp"

namespace mori {
namespace ops {
namespace v2 {
namespace asyncll {

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

    if (destPe / config.gpuPerNode != myNode) {
      // TODO(internode): EpInterNodePut(comm, args.reg(args.offDispatchInp), ...,
      // proxyPe, qpId) -- the ccoGda shim in ep_internode_kernel.hpp already does
      // exactly this for the v2/v2_ll families. Unreachable today: the host gate
      // refuses an AsyncLL config spanning nodes.
      continue;
    }

    const int q = asyncll::QueueForPe(comm, destPe);
    // The destination addresses ITS copy of dispatchInp at this rank's slice, and
    // the source is this rank's staging at the destination's slice -- the two
    // halves of SendBufSlotOffset's symmetry, and the one place a transposed
    // index silently delivers another rank's tokens.
    const size_t remoteOffset = static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * xferBytes;
    const size_t localOffset =
        static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * xferBytes;

    // Whole, not sliced per QP. This is the measured departure from v1: at 64
    // tokens a peer's payload is ~600 KB, which SDMA moves at ~46 GB/s in one op
    // and ~35 GB/s as two 300 KB ones. Warp scope so lane 0 posts for the warp
    // without an explicit if.
    sdma.put<::mori::cco::ccoCoopWarp>(
        destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offDispatchInp)),
        EpInterNodeOffset(args.reg(args.offDispatchInp), remoteOffset),
        EpInterNodeWindow(args.reg(args.offStaging)),
        EpInterNodeOffset(args.reg(args.offStaging), localOffset),
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
    if (destPe != myPe && sameNode) {
      // Our payload to this peer is out of the queue before its count word goes,
      // so the peer seeing a non-zero count has the tokens too.
      sdma.quietQueue<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                                asyncll::QueueForPe(comm, destPe));
    }

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
    }
    // TODO(internode): the cross-node leg is EpInterNodeQuiet + the chunk-flag
    // atomic, as DispatchInterNodeLLSend does.
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
// It does, but far less than that reasoning suggests. Measured on MI355X at EP8,
// hidden 7168, bf16, dispatch+combine as a CUDA graph:
//
//            64 tok/rank     128 tok/rank
//   sdma     103.0 us        133.8 us
//   lsa       96.4 us        135.0 us
//
// Six microseconds at 64 tokens and nothing at 128. The reason is that the
// transfer is not where the time goes: AsyncLL's protocol moves each payload
// THREE times (input -> staging, staging -> peer, peer -> dispatch output),
// because that is what an RDMA leg needs, and changing the middle hop leaves the
// other two alone. For comparison the ordinary v2 intranode dispatch, which
// writes straight into the peer and stages nothing, does the same shape in
// 59.5 / 89.6 us.
//
// So this variant is worth having for the latency, not for a transformation --
// and it costs CUs, because the copy occupies the grid instead of an engine and
// therefore cannot drain under the caller's work the way an SDMA transfer can.
// Which one is right depends on what the caller puts in the seam, which is why
// this is a config choice rather than a replacement.

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
  if (destPe / config.gpuPerNode != myNode) return;  // TODO(internode)

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
  core::WarpCopy<uint8_t, 1>(dst, src, share.bytes);
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
    // TODO(internode): cross-node ranks still need the GDA signal.
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

  const uint64_t tokenNum =
      core::AtomicLoadRelaxedSystem(const_cast<uint64_t*>(
          reinterpret_cast<const uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe)))) -
      1;

  if (destPe != myPe && destPe / config.gpuPerNode == myNode && tokenNum > 0) {
    const asyncll::LsaShare share =
        asyncll::LsaShareFor(globalWarpId, globalWarpNum, npes, tokenNum * hiddenBytes);
    if (share.bytes > 0) {
      uint8_t* dst = args.reg(args.offCombineInp)->template GetAs<uint8_t*>(destPe) +
                     static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes +
                     share.offset;
      const uint8_t* src = args.reg(args.offStaging)->template GetAs<const uint8_t*>() +
                           static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes +
                           share.offset;
      core::WarpCopy<uint8_t, 1>(dst, src, share.bytes);
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
      uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>(destPe);
      core::AtomicStoreRelaxedSystem(barrier + myPe, flag);
    }
  }

  if (blockId == 0 && warpId == 0 && laneId < npes) {
    uint64_t* barrier = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    while (core::AtomicLoadRelaxedSystem(barrier + laneId) != flag) {
    }
  }

  // Clear the dispatch recv counts, re-arming the "0 means not arrived" test the
  // next dispatch spins on. Here rather than in the pass that reads them: that
  // one sizes every warp's copy from these words and has no grid-wide point
  // after the last read, whereas this is a separate launch and therefore is one.
  if (globalWarpId == 0 && laneId < npes) {
    uint8_t* recvCounts = args.reg(args.offRecvTokenNum)->template GetAs<uint8_t*>();
    core::AtomicStoreRelaxedSystem(
        reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(laneId)), uint64_t{0});
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                           EpCombineAsyncLLSendCopy -- combine pass 1                           */
/* ---------------------------------------------------------------------------------------------- */
// Put each post-expert token back into the staging slot it was dispatched from.
template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAsyncLLSendCopy_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  const index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

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

    // How many tokens this peer sent us during dispatch is how many we owe it
    // back. Dispatch left the slot alone precisely so this read works; clear it
    // now, which is also what re-arms the "0 means not arrived" test the next
    // dispatch's RecvTransfer spins on.
    uint64_t tokenNum = 0;
    if (laneId == 0) {
      uint64_t* slot = reinterpret_cast<uint64_t*>(recvCounts + asyncll::RecvCountSlot(destPe));
      tokenNum = core::AtomicLoadRelaxedSystem(slot) - 1;
      core::AtomicStoreRelaxedSystem(slot, uint64_t{0});
    }
    tokenNum = __shfl(static_cast<unsigned long long>(tokenNum), 0);

    if (destPe == myPe) continue;
    if (destPe / config.gpuPerNode != myNode) continue;  // TODO(internode)
    if (tokenNum == 0) continue;

    const int q = asyncll::QueueForPe(comm, destPe);
    sdma.put<::mori::cco::ccoCoopWarp>(
        destPe - args.lsaBase, EpInterNodeWindow(args.reg(args.offCombineInp)),
        EpInterNodeOffset(args.reg(args.offCombineInp),
                          static_cast<size_t>(SendBufSlotOffset(config, myPe, 0)) * hiddenBytes),
        EpInterNodeWindow(args.reg(args.offStaging)),
        EpInterNodeOffset(args.reg(args.offStaging),
                          static_cast<size_t>(SendBufSlotOffset(config, destPe, 0)) * hiddenBytes),
        tokenNum * hiddenBytes, q);
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
    if (destPe != myPe && sameNode)
      sdma.quietQueue<::mori::cco::ccoCoopWarp>(destPe - args.lsaBase,
                                                asyncll::QueueForPe(comm, destPe));

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
        T* base = (destPe != myPe) ? args.reg(args.offCombineInp)->template GetAs<T*>()
                                   : args.reg(args.offStaging)->template GetAs<T*>();
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

}  // namespace v2
}  // namespace ops
}  // namespace mori

// The entry macros come from ep_internode_kernel.hpp; the generated TU picks the
// _LOCAL form for the passes that take no communicator.
