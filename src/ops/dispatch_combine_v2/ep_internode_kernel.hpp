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

// DEVICE ONLY. Included by the generated TU, never by a host source.
//
// Two parts, in this order:
//
//   1. the transport shim -- one EpInterNode* function per shmem call the port
//      replaced, on ccoGda. New code; nothing in the original corresponds to it.
//   2. EP internode v1 dispatch + combine, ported from
//      src/ops/dispatch_combine/internode_v1.cpp.
//
// Part 2 is mechanical, in the same sense as ep_intranode_kernel.hpp: the
// algorithm, the block specialisation, the chunk-flag protocol and the proxy-PE
// topology are unchanged. `diff` it against the original -- part 1 shows up as
// one leading block, skip it -- and everything else that comes back is the
// communication layer:
//
//   shmem::ShmemPutMemNbiSignalThread      -> EpInterNodePutSignal      (3 sites)
//   shmem::ShmemAtomicTypeNonFetchThread   -> EpInterNodeAtomicAdd      (4 sites)
//   shmem::ShmemPutTypeNbiWarp             -> EpInterNodePut            (2 sites)
//   shmem::ShmemInt32WaitUntilGreaterThan  -> EpInterNodeWaitGt         (1 site)
//   shmem::ShmemQuietThread                -> EpInterNodeQuiet          (1 site)
//
// plus the `const ccoDevComm& comm` those calls need, threaded down from the
// entry points, and the entry points themselves (extern "C", one per JIT
// module, taking the comm by value alongside the v1 arg block).
//
// mori/shmem is deliberately not included. It reaches its endpoints through a
// device global that the host fills on every hipModuleLoad, and a JIT module
// would have to be taught that handshake; cco takes the communicator as a
// kernel argument, which is the whole reason this port exists.

#pragma once

#include <type_traits>

#include "mori/application/application_device_types.hpp"
#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif
// The cfg half only. ep_internode_spec.hpp is host-only -- it pulls in the
// Compiler, which this TU is the OUTPUT of and must not depend on.
#include "mori/ops/dispatch_combine_v2/ep_internode_cfg.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"

// mori/core/utils/utils.hpp defines `warpSize` as an object-like macro, while
// cco.hpp declares mori::cco::impl::warpSize(). Whichever lands first wins, so
// including cco from EP code is otherwise include-order dependent. Hide the
// macro across the cco headers -- cco never wants it, it uses its own
// wavefront-size builtin -- and restore it for the kernel, which does.
#pragma push_macro("warpSize")
#undef warpSize
#include "mori/cco/cco.hpp"
#include "mori/cco/cco_scale_out.hpp"
#pragma pop_macro("warpSize")

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                    Transport shim (cco-GDA)                                    */
/* ---------------------------------------------------------------------------------------------- */
// The five cross-node operations the kernel below needs. Argument order is kept
// identical to the shmem calls they replaced, so the call sites read the same.
//
// Only cross-node traffic is here. Intra-node peer access is untouched: it was
// always a direct store through SymmMemObj::p2pPeerPtrs, and MallocSymm fills
// those from ccoGetPeerPtr on the cco backend, so GetAs<T*>(pe) keeps working.
//
// Scope, matching what the replaced shmem calls actually did:
//
//   EpInterNodePutSignal / EpInterNodePut / EpInterNodeAtomicAdd   thread scope: every active lane
//       posts its own data WQE, and the flag atomic comes once from the group
//       leader. Callers are divergent (they sit under `if (laneId == 0)` and
//       inside the dedup ballot), so these must not contain a group barrier.
//   EpInterNodeQuiet                                 warp-collective. Its one call site
//       is warp-uniform, and ccoGda::flush requires at least warp scope because
//       it takes a warp-level CQ poll lock.
//
// EpInterNodePut replaces ShmemPutTypeNbiWarp, whose RDMA implementation is
// `if (laneId == 0) <thread put>` -- the warp never split the transfer -- so it
// keeps that shape rather than promising a group barrier it could not honour
// under divergence.

// The NIC provider is fixed at compile time. The JIT toolchain passes the
// matching -DMORI_DEVICE_NIC_*; without it cco_scale_out.hpp falls through to
// its mlx5 #else and a bnxt host silently builds mlx5 WQEs.
inline constexpr ::mori::core::ProviderType kEpInterNodeProvider = CCO_GDA_BUILD_PROVIDER;

// A buffer's position in the arena window. Every EP buffer is a sub-region of
// one registered window (EpDispatchCombineHandle::MallocSymm), and RDMA names a
// remote buffer as (window, byte offset) with iova=0 -- there is no peer VA to
// hand the NIC, which is why SymmMemObj carries these two fields.
__device__ __forceinline__ ::mori::cco::ccoWindow_t EpInterNodeWin(
    const application::SymmMemObjPtr obj) {
  return reinterpret_cast<::mori::cco::ccoWindow_t>(obj->ccoWin);
}

__device__ __forceinline__ size_t EpInterNodeOff(const application::SymmMemObjPtr obj,
                                                 size_t byteOffset) {
  return obj->ccoWinOffset + byteOffset;
}

// ── the one primitive ccoGda does not expose ─────────────────────────────────
//
// v1's chunk-flag protocol needs a remote atomic add at an *arbitrary* window
// offset: one slot per (node, chunk) in EP's own arena, count scaling with the
// token capacity, polled and cleared by the receiver. ccoGda's remote actions
// (ccoGda_SignalInc / ccoGda_SignalAdd) address only the resource window's
// signal pool -- a fixed gdaSignalCount slots read through waitSignal's
// consume-on-read shadow -- so neither the address nor the semantics fit.
//
// Rather than grow the facade with a remote action that only this op wants, the
// two lookups the facade would do -- (window, offset) -> (rkey, raddr) and
// (pe, qpId) -> endpoint -- happen here and the WQE is posted through
// mori::cco::impl. That is the level ccoGdaBarrierSession composes at, and
// cco.hpp spells out the same lkey/rkey/raddr triple for kernels that do it.
// No warp protocol is duplicated: signalImpl posts one WQE for its caller, and
// putImpl runs the same warp aggregation the facade would have run.

__device__ __forceinline__ ::mori::core::RdmaEndpointDevice* EpInterNodeEndpoint(
    const ::mori::cco::ccoDevComm& comm, int pe, int qpId) {
  auto* ibgda = const_cast<::mori::cco::ccoIbgdaContext*>(&comm.ibgda);
  // endpoints and peerRkeys are world-indexed; v1 always passes a world rank,
  // which is what CCO_TEAM_WORLD means at the ccoGda calls below.
  return &ibgda->endpoints[pe * ibgda->numQpPerPe + (qpId % ibgda->numQpPerPe)];
}

__device__ __forceinline__ uint32_t EpInterNodeRkey(const application::SymmMemObjPtr obj, int pe) {
  return EpInterNodeWin(obj)->ibgdaWin.peerRkeys[pe];
}

__device__ __forceinline__ void EpInterNodeAtomicAdd(const ::mori::cco::ccoDevComm& comm,
                                                     const application::SymmMemObjPtr dst,
                                                     size_t dstOffset, uint64_t value, int pe,
                                                     int qpId = 0) {
  ::mori::core::RdmaEndpointDevice* ep = EpInterNodeEndpoint(comm, pe, qpId);
  ::mori::cco::impl::signalImpl<kEpInterNodeProvider>(ep, ep->qpn, EpInterNodeOff(dst, dstOffset),
                                                      EpInterNodeRkey(dst, pe),
                                                      ::mori::cco::ccoGdaSignalAdd, value);
}

// RDMA write with the flag atomic fused into the same reservation. The ordering
// the protocol wants -- a receiver that sees the counter move has the payload
// too -- comes from the QP: an RC responder executes requests in PSN order, and
// putImpl places the signal WQE at base + numActiveLanes, after every lane's
// write, with the slot index doubling as the PSN (BNXT reserves its signal PSN
// behind the data PSNs for the same reason).
//
// This is the signalled put shmem had. Going through ccoGda's facade cannot
// express it: ccoGda_SignalAdd resolves its target as
// signalId * sizeof(uint64_t) in the resource window, and v1's chunk flags live
// at an arbitrary offset in EP's own arena, so the facade would have to issue a
// bare put and leave us to post the atomic as a second reservation -- one extra
// doorbell and CQE per chunk. impl::putImpl takes the signal as a raw
// (raddr, rkey, op, arg), which is exactly the triple EpInterNodeAtomicAdd already
// hands to impl::signalImpl, so the flag buffer drops straight in.
//
// One atomic per warp group, posted by putImpl's leader lane. That is the shape
// shmem had and the dedup call site depends on it: its active lanes carry the
// same flag slot and the same value, so a single add per group is the protocol,
// not a saving. Every active lane must also agree on pe and qpId -- calling
// putImpl directly skips the facade's group-by-peer ballot, and all three call
// sites are warp-uniform in both (proxyPe follows the warp's node, and
// startTokenIdx is a multiple of warpSize so tokenId / warpSize is too).
__device__ __forceinline__ void EpInterNodePutSignal(
    const ::mori::cco::ccoDevComm& comm, const application::SymmMemObjPtr dst, size_t dstOffset,
    const application::SymmMemObjPtr src, size_t srcOffset, size_t bytes,
    const application::SymmMemObjPtr signal, size_t signalOffset, uint64_t signalValue, int pe,
    int qpId) {
  ::mori::core::RdmaEndpointDevice* ep = EpInterNodeEndpoint(comm, pe, qpId);
  ::mori::cco::impl::putImpl<kEpInterNodeProvider>(
      ep, ep->qpn, EpInterNodeOff(src, srcOffset), EpInterNodeWin(src)->ibgdaWin.lkey,
      EpInterNodeOff(dst, dstOffset), EpInterNodeRkey(dst, pe), bytes, /*hasSignal=*/true,
      EpInterNodeOff(signal, signalOffset), EpInterNodeRkey(signal, pe),
      ::mori::cco::ccoGdaSignalAdd, signalValue);
}

__device__ __forceinline__ void EpInterNodePut(const ::mori::cco::ccoDevComm& comm,
                                               const application::SymmMemObjPtr dst,
                                               size_t dstOffset,
                                               const application::SymmMemObjPtr src,
                                               size_t srcOffset, size_t bytes, int pe, int qpId) {
  if ((threadIdx.x & (warpSize - 1)) != 0) return;
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template put<::mori::cco::CCO_TEAM_WORLD, ::mori::cco::ccoGdaThreadIndependent>(
      pe, EpInterNodeWin(dst), EpInterNodeOff(dst, dstOffset), EpInterNodeWin(src),
      EpInterNodeOff(src, srcOffset), bytes, ::mori::cco::ccoGda_NoSignal{},
      ::mori::cco::ccoCoopThread{});
}

// Drains every stripe: flush(peer) only polls the QP belonging to its own
// context, so flushing one would leave the puts issued on the other qpIds
// outstanding. numQp is config.numQpPerPe, which is also what the devComm was
// created with (gdaContextCount).
__device__ __forceinline__ void EpInterNodeQuiet(const ::mori::cco::ccoDevComm& comm, int pe,
                                                 int numQp) {
  for (int q = 0; q < numQp; ++q) {
    ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, q};
    gda.template flush<::mori::cco::CCO_TEAM_WORLD>(pe, ::mori::cco::ccoCoopWarp{});
  }
}

// Spin on a local symmetric slot until a peer publishes a positive value. No
// transport of its own: the write arrives either by intra-node store or by the
// NIC, and ShmemInt32WaitUntilGreaterThan was the same system-scope relaxed load
// in a loop.
__device__ __forceinline__ int32_t EpInterNodeWaitGt(int32_t* addr, int32_t val) {
  int32_t observed;
  while (true) {
    observed = core::AtomicLoadRelaxedSystem(addr);
    if (observed > val) break;
  }
  return observed;
}

// ── compile-time configuration ───────────────────────────────────────────────
//
// v1 reads its whole configuration out of args.config, so
// `flat / config.MaxNumTokensToRecv()` and friends are a scalar load feeding a
// full integer division, repeated in every inner loop. EpInterNodeKernelCfg carries the
// fields that cannot change on a live handle as an NTTP -- the same
// `template <EpCfg kCfg, typename T>` shape ep_intranode_kernel.hpp uses.
//
// The last mile differs from intranode, because the code does. There the body is
// self-contained and `constexpr int kNpes = kCfg.worldSize;` is the whole story.
// Here the body is only the entry: the work is spread over forty-odd helpers
// that all read `config.xxx` (77 sites for numExpertPerToken alone), plus free
// functions like SendBufSlotOffset that take a `const EpDispatchCombineConfig&`.
// So the entry writes the constants over its BY-VALUE copy of args instead, and
// DEF_COMMON_VARS binds `config` to that. Every helper takes `args` by reference
// from the entry, so after inlining these stores are the only definitions their
// loads can see: constant propagation reaches all of them and not one signature
// changes.
//
// The fields left alone -- rank, blockNum, warpNumPerBlock -- keep the values
// the host launched with; see EpInterNodeKernelCfg for why they are not specialised.
template <::mori::ops::v2::EpInterNodeKernelCfg kConfig>
__device__ __forceinline__ void EpInterNodeBindConfig(EpDispatchCombineConfig& config) {
  config.worldSize = kConfig.worldSize;
  config.hiddenDim = kConfig.hiddenDim;
  config.scaleDim = kConfig.scaleDim;
  config.scaleTypeSize = kConfig.scaleTypeSize;
  config.maxTokenTypeSize = kConfig.maxTokenTypeSize;
  config.maxNumInpTokenPerRank = kConfig.maxNumInpTokenPerRank;
  config.numExpertPerRank = kConfig.numExpertPerRank;
  config.numExpertPerToken = kConfig.numExpertPerToken;
  config.maxTotalRecvTokens = kConfig.maxTotalRecvTokens;
  config.gpuPerNode = kConfig.gpuPerNode;
  config.numQpPerPe = kConfig.numQpPerPe;
  config.quantType = kConfig.quantType;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpDispatchInterNodeV1Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */
namespace v1 {
template <typename T>
inline __device__ void DispatchIntraNodeBlock(EpDispatchCombineArgs<T>& args, int tokenId,
                                              int expId, int destPe, int& localPeTokenCounter) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (!args.replayMode) {
    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.dispDestTokIdMap[tokenExpertId] = FlatTokenIndex(config, destPe, destTokId);

      core::AtomicStoreRelaxedSystem(
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destTokId,
          static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId)));
    }
    destTokId = __shfl(destTokId, 0);
  } else {
    // Replay routing: reuse the slot recorded by a prior cache-routing dispatch.
    index_t flat = args.dispDestTokIdMap[tokenExpertId];
    destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
  }
  // Skip per-PE counter in replay routing (caller's totalRecvTokenNum is already correct).
  if (!args.replayMode && laneId == (destPe % config.gpuPerNode)) localPeTokenCounter++;
  size_t srcTokOffset = tokenId * hiddenDim;
  size_t destTokOffset = destTokId * hiddenDim;

  T* remoteTokenPtr = args.interNodeV1TokBufs.dispatchOut->template GetAs<T*>(destPe);
  const T* localTokenPtr = args.inpTokenBuf;
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, hiddenDim);

  index_t* remoteIndexPtr = args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe);
  const index_t* localIndexPtr = args.tokenIndices;
  core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
                 localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  float* remoteWeightPtr = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe);
  const float* localWeightPtr = args.weightsBuf;
  core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
                 localWeightPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  if (args.scalesBuf && (scaleBytes > 0)) {
    core::WarpCopy(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
        args.scalesBuf + tokenId * scaleBytes, scaleBytes);
  }
}

template <typename T>
inline __device__ void DispatchIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchIntra);

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  int localPeTokenCounter = 0;

  for (int i = warpId; i < (endTokenIdx - startTokenIdx) * config.numExpertPerToken; i += warpNum) {
    index_t tokenId = i / config.numExpertPerToken + startTokenIdx;
    index_t expertOffset = startTokenIdx * config.numExpertPerToken + i;
    index_t destExpert = args.tokenIndices[expertOffset];
    if (destExpert < 0) {
      if (!args.replayMode && laneId == 0)
        args.dispDestTokIdMap[expertOffset] = NullFlatTokenIndex(config);
      continue;
    }
    index_t destPe = destExpert / config.numExpertPerRank;
    int destNode = destPe / config.gpuPerNode;

    int lanePe = -1, laneNode = -1;
    if (laneId < numExpertPerToken) {
      index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + laneId];
      // Sentinel lanes get a unique impossible destPe so dedup cannot false-match.
      lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                : (laneExpert / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    };

    // Deduplicate
    index_t inTokenExpertId = i % numExpertPerToken;
    if (destNode == myNode) {
      if (__any((laneId < inTokenExpertId) && (destPe == lanePe))) {
        if (!args.replayMode && laneId == 0)
          args.dispDestTokIdMap[expertOffset] = NullFlatTokenIndex(config);
        continue;
      }
      DispatchIntraNodeBlock(args, tokenId, inTokenExpertId, destPe, localPeTokenCounter);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <typename T, bool DEDUP>
inline __device__ void DispatchInterNodeSend(EpDispatchCombineArgs<T>& args,
                                             const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeSend);

  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, args.rdmaBlockNum);

  int startTokenIdx = blockChunkNum * blockId * warpSize;
  int endTokenIdx = std::min(startTokenIdx + blockChunkNum * warpSize, args.curRankNumToken);

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    if (DEDUP) {
      for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
        bool shouldSend = false;
        for (int e = 0; e < config.numExpertPerToken; e++) {
          index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + e];
          if (laneExpert < 0) continue;
          int destNode = laneExpert / config.numExpertPerRank / config.gpuPerNode;
          if (destNode == i) {
            shouldSend |= true;
            if (!args.replayMode)
              args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = NullFlatTokenIndex(config);
          }
        }
        uint64_t mask = __ballot(shouldSend) & __activemask();
        uint64_t num = __popcll(mask);

        if (num == 0) continue;

        // atomicAdd runs in both paths so blockFlagCounter stays in sync with cache routing.
        index_t flag = 0;
        index_t flagSlotId = 0;
        if (laneId == 0) {
          flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
          flag = num + 1;
        }
        flag = __shfl(flag, 0);
        flagSlotId = __shfl(flagSlotId, 0);

        if (args.replayMode) {
          // Recover the deterministic flag slot from the cached send map.
          int firstSender = __ffsll(static_cast<unsigned long long>(mask)) - 1;
          index_t myCached = shouldSend ? args.interNodeDispSendMap[nNodes * tokenId + i] : 0;
          flagSlotId = __shfl(myCached, firstSender) / warpSize;
        }

        index_t destTokIdOffset = flagSlotId * warpSize;

        uint64_t warpOffset = 0;
        if (laneId > 0) warpOffset = __popcll(mask << (warpSize - laneId));
        index_t destTokId = destTokIdOffset + warpOffset;

        if (shouldSend) {
          bool prev = (laneId > 0) ? ((mask >> (laneId - 1)) & 1ULL) : 0;
          int count = 0;
          if (!prev) {
            count = 1;
            for (int i = laneId + 1; i < warpSize; i++) {
              if ((mask >> i) & 1ULL) {
                count++;
              } else {
                break;
              }
            }
          }
          size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
          if (count > 0) {
            size_t stagingTokOffset = tokenId * xferBytes;
            int qpId = (tokenId / warpSize) % config.numQpPerPe;
            EpInterNodePutSignal(comm, args.interNodeV1TokBufs.dispatchInp, remoteIdx * xferBytes,
                                 args.interNodeV1TokBufs.dispatchStaging, stagingTokOffset,
                                 count * xferBytes, args.interNodeChunkFlagMemObj,
                                 (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), flag,
                                 proxyPe, qpId);
          }
          if (!args.replayMode) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
        }
      }
    } else {
      for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
        bool shouldSend = false;
        for (int e = 0; e < config.numExpertPerToken; e++) {
          index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + e];
          if (laneExpert < 0) continue;
          int destNode = laneExpert / config.numExpertPerRank / config.gpuPerNode;
          if (destNode == i) {
            shouldSend |= true;
            args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = NullFlatTokenIndex(config);
          }
        }

        index_t flagSlotId = 0;
        if (laneId == 0) {
          flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
        }
        flagSlotId = __shfl(flagSlotId, 0);

        index_t destTokIdOffset = flagSlotId * warpSize;
        index_t destTokId = destTokIdOffset + laneId;

        size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
        if (laneId == 0) {
          index_t tokenNum = std::min(tokenId + warpSize, endTokenIdx) - tokenId;
          size_t stagingTokOffset = tokenId * xferBytes;
          int qpId = (tokenId / warpSize) % config.numQpPerPe;
          EpInterNodePutSignal(comm, args.interNodeV1TokBufs.dispatchInp, remoteIdx * xferBytes,
                               args.interNodeV1TokBufs.dispatchStaging, stagingTokOffset,
                               tokenNum * xferBytes, args.interNodeChunkFlagMemObj,
                               (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), tokenNum + 1,
                               proxyPe, qpId);
        }
        if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    // laneId == myNode would signal slot myNode on a proxy that shares our node,
    // and DispatchInterNodeRecv only ever polls slots of remote nodes, so that
    // write is dead. It is also unroutable: RAIL connects cross-node same-rail
    // peers only, so a local peer has no QP and EpInterNodeAtomicAdd would fault.
    // shmem's atomic resolved a local peer to a plain store, hence no guard here
    // originally.
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      EpInterNodeAtomicAdd(comm, args.nodeRecvTokenNumMemObj, myNode * sizeof(uint64_t),
                           numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;
  }
}

template <typename T>
inline __device__ void DispatchInterNodeLLSend(EpDispatchCombineArgs<T>& args,
                                               const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeLLSend);

  // Then send to other nodes
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, args.rdmaBlockNum);
  int chunkStartTokenIdx = blockChunkNum * blockId * warpSize;
  int chunkEndTokenIdx =
      std::min(chunkStartTokenIdx + blockChunkNum * warpSize, args.curRankNumToken);
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);

    for (int tokenId = chunkStartTokenIdx + laneId; tokenId < chunkEndTokenIdx;
         tokenId += warpSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e] /
                       config.numExpertPerRank / config.gpuPerNode;
        if (destNode == i) {
          shouldSend |= true;
          args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = NullFlatTokenIndex(config);
        }
      }

      index_t flagSlotId = 0;
      if (laneId == 0) {
        flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
      }
      flagSlotId = __shfl(flagSlotId, 0);

      index_t destTokIdOffset = flagSlotId * warpSize;
      index_t destTokId = destTokIdOffset + laneId;

      size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
      if (laneId == 0) {
        index_t tokenNum = std::min(tokenId + warpSize, chunkEndTokenIdx) - tokenId;
        size_t stagingTokOffset = tokenId * xferBytes;
        int qpId = (tokenId / warpSize) % config.numQpPerPe;

        EpInterNodePutSignal(comm, args.interNodeV1TokBufs.dispatchInp, remoteIdx * xferBytes,
                             args.interNodeV1TokBufs.dispatchStaging, stagingTokOffset,
                             tokenNum * xferBytes, args.interNodeChunkFlagMemObj,
                             (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), tokenNum + 1,
                             proxyPe, qpId);
      }
      if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[1], 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    // Skips the local node for the same reason as DispatchInterNodeSend above.
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      EpInterNodeAtomicAdd(comm, args.nodeRecvTokenNumMemObj, myNode * sizeof(uint64_t),
                           numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[1] = 0;
  }
}

template <typename T>
inline __device__ void DispatchInterNodeRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeRecv);

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.interNodeV1TokBufs.dispatchInp->template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;
  int totalChunkNum = 0;

  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += args.rdmaBlockNum) {
    int k = bid / (numRecvBlock * (nNodes - 1));
    int i = (bid / numRecvBlock) % (nNodes - 1);

    int node = (myNode + 1 + i) % nNodes;
    int startTokenIdx = k * warpSize;

    uint64_t thisChunkTokenNum = 0;
    index_t nodeFlag = 0;
    if (laneId == 0) {
      while (1) {
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;

        nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
      }
    }
    thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;
    nodeFlag = __shfl(nodeFlag, 0) - 1;
    totalChunkNum += thisChunkTokenNum;

    int endTokenIdx = startTokenIdx + thisChunkTokenNum;

    for (int j = startTokenIdx + (blockId % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
         j += numRecvBlock * warpNum) {
      int tokIdx = SendBufSlotOffset(config, node, j);
      index_t* indices = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
      // Sentinel lanes (-1 expert) get a unique impossible destPe to avoid false dup-matches.
      int lanePe = -1;
      if (laneId < config.numExpertPerToken) {
        index_t laneExpert = indices[laneId];
        lanePe = (laneExpert < 0) ? (-1 - static_cast<int>(laneId))
                                  : (laneExpert / config.numExpertPerRank);
        assert((laneExpert < 0) || ((lanePe < config.worldSize) && (lanePe >= 0)));
      }
      index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes +
                                                    indexBytes + weightBytes + scaleBytes)[0];

      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destPe = __shfl(lanePe, e);
        bool isSentinelSlot = (destPe < 0);
        int destNode = isSentinelSlot ? -1 : destPe / config.gpuPerNode;

        // HSA-RCA Signature 1 guard: in Release builds NDEBUG strips the
        // assert at :387, so an out-of-range expert id (e.g. EPLB physical id
        // >= worldSize*numExpertPerRank, PR #254) yields destPe >= worldSize
        // and an OOB GetAs/WarpCopy/atomicAdd -> HSA page fault. Treat any
        // out-of-range destPe as a dropped token via the existing skip path.
        bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
        bool shouldSkip = peOutOfRange || isSentinelSlot || (destNode != myNode) ||
                          __any((laneId < e) && (destPe == lanePe));
        if (shouldSkip) {
          if (!args.replayMode && laneId == 0)
            args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] =
                NullFlatTokenIndex(config);
          continue;
        }
        int destTokId = 0;
        if (!args.replayMode) {
          if (laneId == 0) {
            destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] =
                FlatTokenIndex(config, destPe, destTokId);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
          }
          destTokId = __shfl(destTokId, 0);
        } else {
          // Replay: pull cached recv-side slot.
          index_t flat = args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e];
          destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
        }
        if (!args.replayMode && (destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
        core::WarpCopy<uint8_t, 4>(
            args.interNodeV1TokBufs.dispatchOut->template GetAs<uint8_t*>(destPe) +
                destTokId * hiddenBytes,
            stagingPtr + tokIdx * xferBytes, hiddenBytes);
        core::WarpCopy<uint8_t, 4>(
            args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
        core::WarpCopy<uint8_t, 4>(
            args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>(destPe) +
                destTokId * weightBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
        if ((scaleBytes > 0)) {
          core::WarpCopy<uint8_t, 4>(
              args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
              stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes + weightBytes, scaleBytes);
        }
      }
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <typename T>
inline __device__ void DispatchInterNodeLLRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeLLRecv);

  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.interNodeV1TokBufs.dispatchInp->template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;

  // expert -> token -> node
  for (int i = globalWarpId;
       i < config.MaxNumTokensToSendPerRank() * config.numExpertPerToken * (nNodes - 1);
       i += args.rdmaBlockNum * warpNum) {
    int expertId = i % config.numExpertPerToken;
    int tokenId = i / config.numExpertPerToken % config.MaxNumTokensToSendPerRank();
    int nodeId = i / config.numExpertPerToken / config.MaxNumTokensToSendPerRank();

    int node = (myNode + 1 + nodeId) % nNodes;
    int k = tokenId / warpSize;
    int startTokenIdx = k * warpSize;

    // Poll completion flags
    uint64_t thisChunkTokenNum = 0;
    index_t nodeFlag = 0;
    if (laneId == 0) {
      uint64_t barrierFlag = args.crossDeviceBarrierFlag[0];
      while (1) {
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;

        nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
      }
    }
    thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;
    int endTokenIdx = startTokenIdx + thisChunkTokenNum;
    if (tokenId >= endTokenIdx) continue;

    int globalTokenId = SendBufSlotOffset(config, node, tokenId);
    index_t* indices =
        reinterpret_cast<index_t*>(stagingPtr + globalTokenId * xferBytes + hiddenBytes);
    int lanePe = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = indices[laneId] / config.numExpertPerRank;
      assert((lanePe < config.worldSize) && (lanePe >= 0));
    }
    index_t srcTokId =
        reinterpret_cast<index_t*>(stagingPtr + globalTokenId * xferBytes + hiddenBytes +
                                   indexBytes + weightBytes + scaleBytes)[0];

    int destPe = __shfl(lanePe, expertId);
    int destNode = destPe / config.gpuPerNode;
    // HSA-RCA Signature 1 guard (mirror of the :396 site): out-of-range destPe
    // (assert at :493 stripped under NDEBUG) is dropped instead of writing OOB.
    bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
    bool shouldSkip =
        peOutOfRange || (destNode != myNode) || __any((laneId < expertId) && (destPe == lanePe));
    if (shouldSkip) {
      if (laneId == 0)
        args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
            NullFlatTokenIndex(config);
      continue;
    }

    int destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
          FlatTokenIndex(config, destPe, destTokId);
      args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
    }
    if ((destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
    destTokId = __shfl(destTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        args.interNodeV1TokBufs.dispatchOut->template GetAs<uint8_t*>(destPe) +
            destTokId * hiddenBytes,
        stagingPtr + globalTokenId * xferBytes, hiddenBytes);
    core::WarpCopy<uint8_t, 4>(
        args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>(destPe) +
            destTokId * weightBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
    if ((scaleBytes > 0)) {
      core::WarpCopy<uint8_t, 4>(
          args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
          stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes + weightBytes,
          scaleBytes);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <typename T>
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args,
                                    const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchSync);

  int nodePeOffset = myNode * config.gpuPerNode;
  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.dispatchGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == globalWarpNum) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      index_t numTokenSignal = core::AtomicLoadSeqCstSystem(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      core::AtomicStoreSeqCstSystem(signal, numTokenSignal);
    }
    if (laneId == 0)
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int destPe = nodePeOffset + laneId; destPe < (nodePeOffset + config.gpuPerNode);
         destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = EpInterNodeWaitGt(signal, 0) - 1;
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      __threadfence_system();
      // reset local counter
      core::AtomicStoreSeqCstSystem(signal, 0);
      core::AtomicStoreSeqCstSystem(args.destPeTokenCounter + destPe, 0);
    }

    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1);
      __hip_atomic_store(args.combineGridBarrier + 1, 0u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  for (int i = globalWarpId; i < nNodes; i += globalWarpNum) {
    // The send loops above target only remote nodes ((myNode + 1 + i) % nNodes),
    // so the local node never has GDA traffic to drain. Skipping it is not just
    // an optimisation: endpoints is world-indexed but RAIL only connects
    // cross-node same-rail peers, so a local proxyPe's slot is zero-filled --
    // and on a single node the mask collapses to NONE and endpoints is null.
    // flush() indexes it without a reachability check, so quieting the local
    // node faults. shmem's ShmemQuietThread tolerated the self peer, which is
    // why the original loop covered every node.
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    EpInterNodeQuiet(comm, proxyPe, config.numQpPerPe);
  }
}

}  // namespace v1

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchInterNodeV1Kernel_body(EpDispatchCombineArgs<T> args,
                                                 const ::mori::cco::ccoDevComm& comm) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  if (blockId < args.rdmaBlockNum) {
    v1::DispatchInterNodeSend<T, true>(args, comm);
    v1::DispatchInterNodeRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args, comm);
}

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchCopyToStaging_body(EpDispatchCombineArgs<T> args) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::EpDispatchCopyToStaging);
  if (args.curRankNumToken == 0) return;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  // First copy to staging buffer
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    uint8_t* stagingPtr = args.interNodeV1TokBufs.dispatchStaging->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenDimOffset * sizeof(T),
                               reinterpret_cast<uint8_t*>(args.inpTokenBuf) +
                                   tokenId * hiddenBytes + hiddenDimOffset * sizeof(T),
                               hiddenDimSize * sizeof(T));
    if (inTokenPartId != 0) continue;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenBytes,
                               reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes,
                               indexBytes);
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes,
                               reinterpret_cast<uint8_t*>(args.weightsBuf) + tokenId * weightBytes,
                               weightBytes);
    if (args.scalesBuf && (scaleBytes > 0))
      core::WarpCopy<uint8_t, 4>(
          stagingPtr + stagingTokOffset + hiddenBytes + indexBytes + weightBytes,
          reinterpret_cast<uint8_t*>(args.scalesBuf) + tokenId * scaleBytes, scaleBytes);
    if (laneId == 0)
      reinterpret_cast<index_t*>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes +
                                 weightBytes + scaleBytes)[0] =
          static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
  }
}

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T, bool EnableStdMoE>
__device__ void EpDispatchInterNodeV1KernelLowLatency_body(EpDispatchCombineArgs<T> args,
                                                           const ::mori::cco::ccoDevComm& comm) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  if (blockId < args.rdmaBlockNum) {
    v1::DispatchInterNodeLLSend<T>(args, comm);
    v1::DispatchInterNodeLLRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args, comm);

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpCombineInterNodeV1Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */
namespace v1 {

template <typename T>
inline __device__ void CombineSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineSync);

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, blockNum);
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
#ifndef ENABLE_STANDARD_MOE_ADAPT
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if (args.config.quantType == QuantType::Fp8DirectCast) {
      using Fp8T = core::CombineInternalFp8;
      Fp8T* dst = args.interNodeV1TokBufs.combineInp->template GetAs<Fp8T*>();
      const T* src = args.inpTokenBuf;
      const size_t base = tokenId * hiddenDim;
      core::WarpCastBf16ToCombineInternalFp8<T>(dst + base, src + base, hiddenDim, laneId);
    } else {
      core::WarpCopy(args.interNodeV1TokBufs.combineInp->template GetAs<T*>() + tokenId * hiddenDim,
                     args.inpTokenBuf + tokenId * hiddenDim, hiddenDim);
    }
  }
#endif
  if (args.weightsBuf) {
    for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
      core::WarpCopy(
          args.shmemInpWeightsMemObj->template GetAs<float*>() + tokenId * config.numExpertPerToken,
          args.weightsBuf + tokenId * config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

namespace combine_impl {

template <typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeTyped(EpDispatchCombineArgs<T>& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes) {
  DEF_COMMON_VARS;

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes + myNode, 0) * tokCombXferBytes;

  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
        srcPtrs[laneId] = args.interNodeV1TokBufs.combineInp->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    core::WarpAccum<TokT, 4>(reinterpret_cast<TokT*>(stagingPtr + tokenId * tokCombXferBytes),
                             srcPtrs, nullptr, config.numExpertPerToken, hiddenDim);
    if (args.weightsBuf) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * tokCombXferBytes + tokHiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeLLTyped(EpDispatchCombineArgs<T>& args,
                                                        size_t tokHiddenBytes,
                                                        size_t tokCombXferBytes) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int xgmiWarpNum = xgmiBlockNum * warpNum;

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes + myNode, 0) * tokCombXferBytes;

  MultiWarpIter mwIter(xgmiWarpNum, args.curRankNumToken, hiddenDim);

  for (int i = globalWarpId - blockOffset * warpNum;
       i < (args.curRankNumToken * mwIter.warpsPerItem); i += xgmiWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
        srcPtrs[laneId] = args.interNodeV1TokBufs.combineInp->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim + hiddenDimOffset;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    core::WarpAccum<TokT, 4>(
        reinterpret_cast<TokT*>(stagingPtr + tokenId * tokCombXferBytes) + hiddenDimOffset, srcPtrs,
        nullptr, config.numExpertPerToken, hiddenDimSize);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * tokCombXferBytes + tokHiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeTyped(EpDispatchCombineArgs<T>& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes,
                                                      const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  index_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>();

  int totalBids = 0;
  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += args.rdmaBlockNum) {
    totalBids++;
  }

  int processedCount = 0;
  int batchStart = 0;
  while (processedCount < totalBids) {
    uint32_t processedMask = 0;
    int currentBatchSize = std::min(totalBids - processedCount, 32);

    while (processedMask !=
           ((currentBatchSize == 32) ? 0xFFFFFFFF : ((1u << currentBatchSize) - 1))) {
      int bidIdx = 0;
      for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
           bid += args.rdmaBlockNum) {
        if (bidIdx < batchStart) {
          bidIdx++;
          continue;
        }
        if (bidIdx >= batchStart + currentBatchSize) break;

        int relativeIdx = bidIdx - batchStart;
        if (!((processedMask >> relativeIdx) & 1)) {
          int k = bid / (numRecvBlock * (nNodes - 1));
          int i = (bid / numRecvBlock) % (nNodes - 1);
          int node = (myNode + 1 + i) % nNodes;

          uint64_t thisChunkTokenNum = 0;
          int startTokenIdx = k * warpSize;

          if (laneId == 0) {
            thisChunkTokenNum = chunkFlag[node * maxChunkNum + k];
            if (thisChunkTokenNum == 0) {
              index_t nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
              if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
                thisChunkTokenNum = 1;
              }
            }
          }
          thisChunkTokenNum = __shfl(thisChunkTokenNum, 0);

          if (thisChunkTokenNum > 0) {
            thisChunkTokenNum -= 1;
            int endTokenIdx = startTokenIdx + thisChunkTokenNum;

            for (int j = startTokenIdx + (bid % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
                 j += numRecvBlock * warpNum) {
              int tokIdx = SendBufSlotOffset(config, node, j);

              if (laneId < config.numExpertPerToken) {
                srcPtrs[laneId] = nullptr;
                srcWeightsPtr[laneId] = nullptr;
                index_t destTokId =
                    args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId];
                index_t destPe = PeFromFlatTokenIndex(config, destTokId);
                index_t destNode = destPe / config.gpuPerNode;
                if (destNode == myNode) {
                  index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
                  srcPtrs[laneId] =
                      args.interNodeV1TokBufs.combineInp->template GetAs<TokT*>(destPe) +
                      destLocalTokId * hiddenDim;
                  srcWeightsPtr[laneId] =
                      args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                      destLocalTokId * config.numExpertPerToken;
                }
                // routing-handle callers own this tensor, hence no need to reset.
                if (args.dispTokIdToSrcTokIdLocal == nullptr) {
                  args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId] = 0;
                }
              }

              core::WarpAccum<TokT, 4>(
                  reinterpret_cast<TokT*>(stagingPtr + tokIdx * tokCombXferBytes), srcPtrs, nullptr,
                  config.numExpertPerToken, hiddenDim);

              if (args.weightsBuf) {
                core::WarpAccum<float, 4>(
                    reinterpret_cast<float*>(stagingPtr + tokIdx * tokCombXferBytes +
                                             tokHiddenBytes),
                    srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
              }
            }

            index_t finished = 0;
            if (laneId == 0)
              finished = atomicAdd(&args.interNodeChunkFlagCombine[node * maxChunkNum + k], 1);
            finished = __shfl(finished, 0);
            if ((finished + 1) >= (numRecvBlock * warpNum)) {
              if (laneId == 0) {
                core::AtomicStoreSeqCstSystem(
                    args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>() +
                        node * maxChunkNum + k,
                    uint64_t{0});
                core::AtomicStoreRelaxedSystem(
                    args.interNodeChunkFlagCombine + node * maxChunkNum + k, index_t{0});
              }
              int proxyPe = node * config.gpuPerNode + (config.rank % config.gpuPerNode);
              int qpId = k % config.numQpPerPe;
              EpInterNodePut(
                  comm, args.interNodeV1TokBufs.staging,
                  SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                  args.interNodeV1TokBufs.staging,
                  SendBufSlotOffset(config, node, startTokenIdx) * tokCombXferBytes,
                  thisChunkTokenNum * tokCombXferBytes, proxyPe, qpId);
            }
          }
          processedMask |= (1u << relativeIdx);
        }
        bidIdx++;
      }
    }
    processedCount += currentBatchSize;
    batchStart += currentBatchSize;
  }

  // Ensure all prior writes (in particular zeroing interNodeChunkFlagMemObj) are visible
  // to other nodes before participating in the cross-device barrier, so a remote node
  // never observes a non-zero flag that is subsequently overwritten with zero
  __threadfence_system();

  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);

  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      core::AtomicStoreSeqCstSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if ((laneId < nNodes) &&
        (laneId != myNode)) {  // avoid setting myNode, it will be set in intra node branch
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      for (int i = 0; i < config.numQpPerPe; i++) {
        EpInterNodeAtomicAdd(comm, args.crossDeviceBarrierMemObj,
                             args.config.rank * sizeof(uint64_t), 1, proxyPe, i);
      }
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) !=
             (barrierFlag * config.numQpPerPe)) {
      }
    }
  }
}

template <typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeLLTyped(EpDispatchCombineArgs<T>& args,
                                                        size_t tokHiddenBytes,
                                                        size_t tokCombXferBytes,
                                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>();

  int rdmaWarpNum = args.rdmaBlockNum * warpNum;
  for (int n = 0; n < (nNodes - 1); n++) {
    int node = (myNode + n + 1) % nNodes;
    uint64_t nodeCount = nodeRecvTokenNum[node];
    if (nodeCount > 0) nodeCount -= 1;
    if (nodeCount == 0) continue;

    // int warpsPerToken = (rdmaWarpNum + nodeCount - 1) / nodeCount;
    // NOTE: Using a fixed value of 4 for warpsPerToken instead of the dynamic formula above is
    // an intentional tuning choice.
    int warpsPerToken = 4;
    size_t hiddenDimPerWarp = (hiddenDim + warpsPerToken - 1) / warpsPerToken;

    for (int i = globalWarpId; i < (nodeCount * warpsPerToken); i += rdmaWarpNum) {
      int tokenId = i / warpsPerToken;
      int k = tokenId / warpSize;
      int startTokenIdx = k * warpSize;
      uint64_t thisChunkTokenNum = chunkFlag[node * maxChunkNum + k];
      thisChunkTokenNum -= (thisChunkTokenNum > 0) ? 1 : 0;
      if ((tokenId - startTokenIdx) < thisChunkTokenNum) {
        int inTokenPartId = i % warpsPerToken;
        size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
        size_t hiddenDimSize = (hiddenDimOffset < hiddenDim)
                                   ? std::min(hiddenDim - hiddenDimOffset, hiddenDimPerWarp)
                                   : size_t{0};

        int globalTokenId = SendBufSlotOffset(config, node, tokenId);
        if (laneId < config.numExpertPerToken) {
          srcPtrs[laneId] = nullptr;
          srcWeightsPtr[laneId] = nullptr;
          index_t destTokId =
              args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + laneId];
          index_t destPe = PeFromFlatTokenIndex(config, destTokId);
          index_t destNode = destPe / config.gpuPerNode;
          if (destNode == myNode) {
            index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
            srcPtrs[laneId] = args.interNodeV1TokBufs.combineInp->template GetAs<TokT*>(destPe) +
                              destLocalTokId * hiddenDim + hiddenDimOffset;
            srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                    destLocalTokId * config.numExpertPerToken;
          }
        }
        core::WarpAccum<TokT, 4>(
            reinterpret_cast<TokT*>(stagingPtr + globalTokenId * tokCombXferBytes) +
                hiddenDimOffset,
            srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
        if (args.weightsBuf && (inTokenPartId == 0)) {
          core::WarpAccum<float, 4>(
              reinterpret_cast<float*>(stagingPtr + globalTokenId * tokCombXferBytes +
                                       tokHiddenBytes),
              srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
        }
      }

      index_t finished = 0;
      if (laneId == 0)
        finished = atomicAdd(&args.interNodeChunkFlagCombine[node * maxChunkNum + k], 1);
      finished = __shfl(finished, 0);
      if ((finished + 1) >= (warpsPerToken * warpSize)) {
        if (laneId == 0) {
          core::AtomicStoreSeqCstSystem(
              args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>() + node * maxChunkNum + k,
              uint64_t{0});
          core::AtomicStoreRelaxedSystem(args.interNodeChunkFlagCombine + node * maxChunkNum + k,
                                         index_t{0});
        }
        int proxyPe = node * config.gpuPerNode + (config.rank % config.gpuPerNode);
        int qpId = k % config.numQpPerPe;
        EpInterNodePut(comm, args.interNodeV1TokBufs.staging,
                       SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                       args.interNodeV1TokBufs.staging,
                       SendBufSlotOffset(config, node, startTokenIdx) * tokCombXferBytes,
                       thisChunkTokenNum * tokCombXferBytes, proxyPe, qpId);
      }
    }
  }

  // Ensure all prior writes (in particular zeroing interNodeChunkFlagMemObj) are visible
  // to other nodes before participating in the cross-device barrier, so a remote node
  // never observes a non-zero flag that is subsequently overwritten with zero
  __threadfence_system();
  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[0], 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);

  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      core::AtomicStoreSeqCstSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if ((laneId < nNodes) &&
        (laneId != myNode)) {  // avoid setting myNode, it will be set in intra node branch
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      for (int i = 0; i < config.numQpPerPe; i++) {
        EpInterNodeAtomicAdd(comm, args.crossDeviceBarrierMemObj,
                             args.config.rank * sizeof(uint64_t), 1, proxyPe, i);
      }
      __threadfence_system();
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    // Wait other nodes
    uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) !=
             (barrierFlag * config.numQpPerPe)) {
      }
    }
  }
}

}  // namespace combine_impl

template <typename T>
inline __device__ void CombineIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNode);
  if (args.config.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineIntraNodeTyped<TokT>(args, tokHiddenBytes, tokCombXferBytes);
    return;
  }

  combine_impl::CombineIntraNodeTyped<T>(args, hiddenBytes, combXferBytes);
}

template <typename T>
inline __device__ void CombineIntraNodeLL(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNodeLL);

  if (args.curRankNumToken == 0) return;
  if (args.config.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineIntraNodeLLTyped<TokT>(args, tokHiddenBytes, tokCombXferBytes);
    return;
  }
  combine_impl::CombineIntraNodeLLTyped<T>(args, hiddenBytes, combXferBytes);
}

template <typename T>
inline __device__ void CombineInterNode(EpDispatchCombineArgs<T>& args,
                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNode);

  if (args.config.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineInterNodeTyped<TokT>(args, tokHiddenBytes, tokCombXferBytes, comm);
    return;
  }
  combine_impl::CombineInterNodeTyped<T>(args, hiddenBytes, combXferBytes, comm);
}

template <typename T>
inline __device__ void CombineInterNodeLL(EpDispatchCombineArgs<T>& args,
                                          const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL);
  if (args.config.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineInterNodeLLTyped<TokT>(args, tokHiddenBytes, tokCombXferBytes, comm);
    return;
  }
  combine_impl::CombineInterNodeLLTyped<T>(args, hiddenBytes, combXferBytes, comm);
}
}  // namespace v1

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineInterNodeV1Kernel_body(EpDispatchCombineArgs<T> args,
                                                const ::mori::cco::ccoDevComm& comm) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;

  if (blockId < args.rdmaBlockNum) {
    v1::CombineInterNode(args, comm);
  } else {
    v1::CombineIntraNode(args);
  }
}

namespace combine_all_impl {

template <typename T>
__forceinline__ __device__ void EpCombineAllInternalFp8(EpDispatchCombineArgs<T>& args,
                                                        size_t fp8HiddenBytes,
                                                        size_t fp8CombXferBytes) {
  DEF_COMMON_VARS;
  using Fp8T = core::CombineInternalFp8;

  extern __shared__ char sharedMem[];
  Fp8T** srcPtrs = reinterpret_cast<Fp8T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes, 0) * fp8CombXferBytes;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + laneId];
      if (laneExpert >= 0) {
        lanePe = laneExpert / config.numExpertPerRank;
        laneNode = lanePe / config.gpuPerNode;
      }
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base = stagingPtr + SendBufSlotOffset(config, n, mappedId) * fp8CombXferBytes;
        srcPtrs[n] = reinterpret_cast<Fp8T*>(base) + hiddenDimOffset;
        srcWeightsPtrs[n] = reinterpret_cast<float*>(base + fp8HiddenBytes);
      }
    }

    T* out = args.interNodeV1TokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
             hiddenDimOffset;
    core::WarpAccumCombineInternalFp8ToBf16(out, reinterpret_cast<const Fp8T* const*>(srcPtrs),
                                            nNodes, laneId, hiddenDimSize);

    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

template <typename T>
__forceinline__ __device__ void EpCombineAllGeneric(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes, 0) * combXferBytes;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      index_t laneExpert = args.tokenIndices[tokenId * numExpertPerToken + laneId];
      if (laneExpert >= 0) {
        lanePe = laneExpert / config.numExpertPerRank;
        laneNode = lanePe / config.gpuPerNode;
      }
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base = stagingPtr + SendBufSlotOffset(config, n, mappedId) * combXferBytes;
        srcPtrs[n] = reinterpret_cast<T*>(base) + hiddenDimOffset;
        srcWeightsPtrs[n] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }
    core::WarpAccum<T, 4>(args.interNodeV1TokBufs.combineOut->template GetAs<T*>() +
                              tokenId * hiddenDim + hiddenDimOffset,
                          srcPtrs, nullptr, nNodes, hiddenDimSize);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

}  // namespace combine_all_impl

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAll_body(EpDispatchCombineArgs<T> args) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::EpCombineAll);

  if (globalWarpId == 0) {
    // routing-handle callers own this tensor hence no need to reset.
    if (laneId == 0 && args.dispTokIdToSrcTokIdLocal == nullptr) args.totalRecvTokenNum[0] = 0;
    if (laneId < nNodes) args.blockFlagCounter[laneId] = 0;
  }
  if (args.curRankNumToken == 0) return;
  if (args.config.quantType == QuantType::Fp8DirectCast) {
    using Fp8T = core::CombineInternalFp8;
    const size_t fp8HiddenBytes = hiddenDim * sizeof(Fp8T);
    const size_t fp8CombXferBytes =
        (args.weightsBuf == nullptr) ? fp8HiddenBytes : fp8HiddenBytes + weightBytes;
    combine_all_impl::EpCombineAllInternalFp8(args, fp8HiddenBytes, fp8CombXferBytes);
    return;
  }
  combine_all_impl::EpCombineAllGeneric(args);
}

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T, bool EnableStdMoE>
__device__ void EpCombineInterNodeV1KernelLowLatency_body(EpDispatchCombineArgs<T> args,
                                                          const ::mori::cco::ccoDevComm& comm) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;

  // If EnableStdMoE, call ConvertCombineInputDevice first to convert standard MoE format
#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertCombineInput<T>(args, myPe);
  }
#endif

  if (blockId < args.rdmaBlockNum) {
    v1::CombineInterNodeLL(args, comm);
  } else {
    v1::CombineIntraNodeLL(args);
  }
}

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSync_body(EpDispatchCombineArgs<T> args) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  v1::CombineSync(args);
}

template <::mori::ops::v2::EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSyncBarrier_body(EpDispatchCombineArgs<T> args) {
  EpInterNodeBindConfig<kConfig>(args.config);
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::EpCombineSyncBarrier);
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  barrierFlag = __shfl(barrierFlag, 0);
  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(destPe) + args.config.rank,
        barrierFlag);
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + destPe) != barrierFlag) {
    }
  }
}

}  // namespace moe
}  // namespace mori

// ---------------------------------------------------------------------------
// JIT entry points.
//
// The AOT build wraps each body in a `template <typename T> __global__` and lets
// ep_common.hip stamp out one symbol per dtype. A JIT module compiles exactly
// one kernel, so the generated TU invokes one of these macros instead.
//
// The argument is the v1 POD block plus the communicator. Passing the comm by
// value is the whole point of the port: mori-shmem would have needed a device
// global filled by the host after every hipModuleLoad.
// ---------------------------------------------------------------------------

namespace mori {
namespace moe {

// EpDispatchCombineArgsRaw is static_asserted to share a layout with
// EpDispatchCombineArgs<T> (dispatch_combine.hpp), which is how the AOT launcher
// already passes it.
template <typename T>
__device__ __forceinline__ EpDispatchCombineArgs<T> EpInterNodeAsArgs(
    const ::mori::ops::v2::EpInterNodeCcoArgs& a) {
  EpDispatchCombineArgs<T> typed;
  __builtin_memcpy(&typed, &a.raw, sizeof(EpDispatchCombineArgsRaw));
  return typed;
}

}  // namespace moe
}  // namespace mori

// `kConfig` and `TokT` are not macro arguments. The generated TU defines both
// under those names just above the entry, exactly as ep_spec.cpp emits
// `constexpr EpCfg kCfg` / `using TokT` and then instantiates
// `EpDispatchBody<kCfg, TokT>`. A Cfg could not be a macro argument in any case:
// the commas inside its brace initialiser would be taken as argument separators.

// Kernels that reach the network.
#define MORI_EP_INTERNODE_CCO_ENTRY(entry, body)                                          \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs a) {               \
    ::mori::moe::body<kConfig, TokT>(::mori::moe::EpInterNodeAsArgs<TokT>(a), a.devComm); \
  }

// The LL pair, additionally specialised on the standard-MoE adapter.
#define MORI_EP_INTERNODE_CCO_ENTRY_STDMOE(entry, body, StdMoE)                                   \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs a) {                       \
    ::mori::moe::body<kConfig, TokT, StdMoE>(::mori::moe::EpInterNodeAsArgs<TokT>(a), a.devComm); \
  }

// Staging, sync and the final reduction: local or intra-node only, so they take
// no communicator. They still take the same argument struct, so the host has one
// launch path for the whole sequence.
#define MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(entry, body)                         \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs a) {    \
    ::mori::moe::body<kConfig, TokT>(::mori::moe::EpInterNodeAsArgs<TokT>(a)); \
  }
