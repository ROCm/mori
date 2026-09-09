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

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif
// The cfg half only. ep_internode_spec.hpp is host-only -- it pulls in the
// Compiler, which this TU is the OUTPUT of and must not depend on.
#include "mori/ops/dispatch_combine_v2/ep_internode_cfg.hpp"
// The argument surface, the config the bodies read, the flat-index helpers and
// MultiWarpIter -- everything this TU used to take from v1's dispatch_combine.hpp,
// common.hpp and application_device_types.hpp. Those three are gone: they cost 219
// transitive headers per JIT compile against 14 here, and dispatch_combine.hpp is
// where the ENABLE_PROFILER / ENABLE_STANDARD_MOE_ADAPT conditional struct tail
// lives that the JIT toolchain never defines and nothing was checking.
#include "mori/ops/dispatch_combine_v2/ep_internode_args.hpp"

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
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// v2 spellings, under the names the bodies below already use.
//
// The bodies live in this namespace and say `index_t`, `QuantType::...`,
// `EpDispatchCombineArgs` and the flat-index helpers unqualified. Introducing
// the v2 types under those names is what lets the argument surface change
// without touching the ~250 `args.` sites, the 39 index-helper calls or the 29
// signatures -- the surface moves, the algorithm does not, and the two want to
// be separately reviewable.
//
// Scaffolding with a definite end: moving these bodies out of mori::moe deletes
// this block and makes v1's namespace unreachable from the v2 tree. Nothing
// else belongs here.
// ---------------------------------------------------------------------------
using index_t = ep_index_t;
using QuantType = EpQuantType;

// v1's spelling, now a plain alias: the struct stopped depending on T when
// inpTokenBuf became void*, matching the intranode EpArgs.
using EpDispatchCombineArgs = EpInterNodeArgs;

// v1's common.hpp macro, with the config type swapped. Everything else is
// verbatim, including the assert: numExpertPerToken must fit in a ballot.
#define DEF_COMMON_VARS                                                    \
  constexpr EpInterNodeDeviceCfg config = EpInterNodeDeviceCfgOf(kConfig); \
  int thdId = threadIdx.x;                                                 \
  int thdNum = blockDim.x;                                                 \
  int laneId = threadIdx.x & (warpSize - 1);                               \
  int warpId = thdId / warpSize;                                           \
  int warpNum = blockDim.x / warpSize;                                     \
  int blockNum = gridDim.x;                                                \
  int blockId = blockIdx.x;                                                \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;                 \
  int globalThdNum = gridDim.x * blockDim.x;                               \
  int globalWarpId = blockIdx.x * warpNum + warpId;                        \
  int globalWarpNum = gridDim.x * warpNum;                                 \
  int nullTokenId = NullFlatTokenIndex(config);                            \
  int myPe = args.rank;                                                    \
  int npes = config.worldSize;                                             \
  int myNode = myPe / config.gpuPerNode;                                   \
  int nNodes = npes / config.gpuPerNode;                                   \
  int numExpertPerToken = config.numExpertPerToken;                        \
  static_assert(kConfig.numExpertPerToken < warpSize,                       \
                "numExpertPerToken must fit in a ballot");                   \
  size_t hiddenDim = config.HiddenDimSz();                                 \
  size_t hiddenBytes = config.HiddenBytes(sizeof(T));                      \
  size_t indexBytes = config.IndexBytes();                                 \
  size_t weightBytes = config.WeightBytes();                               \
  size_t srcTokenIdBytes = config.SrcTokenIdBytes();                       \
  size_t scaleBytes = config.ScaleBytes();                                 \
  size_t xferBytes = config.XferBytesPerToken(sizeof(T));                  \
  size_t combXferBytes = (args.weightsBuf == nullptr) ? hiddenBytes : hiddenBytes + weightBytes;

/* ---------------------------------------------------------------------------------------------- */
/*                                    Transport shim (cco-GDA)                                    */
/* ---------------------------------------------------------------------------------------------- */
// The five cross-node operations the kernel below needs. Argument order is kept
// identical to the shmem calls they replaced, so the call sites read the same.
//
// Only cross-node traffic is here. Intra-node peer access is untouched: it was
// always a direct store through the flat LSA VA, and the arena window covers
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
// hand the NIC, which is why a region is (window, offset).
__device__ __forceinline__ ::mori::cco::ccoWindow_t EpInterNodeWin(const EpInterNodeRegion obj) {
  return reinterpret_cast<::mori::cco::ccoWindow_t>(obj->win);
}

__device__ __forceinline__ size_t EpInterNodeOff(const EpInterNodeRegion obj, size_t byteOffset) {
  return obj->off + byteOffset;
}

// ── the remote actions ───────────────────────────────────────────────────────
//
// v1's chunk-flag protocol needs a remote atomic add at an *arbitrary* window
// offset: one slot per (node, chunk) in EP's own arena, count scaling with the
// token capacity, polled and cleared by the receiver. ccoGda_SignalInc /
// ccoGda_SignalAdd cannot name that target -- they resolve to
// `signalId * sizeof(uint64_t)` in the DevComm's own resourceWindow -- so EP
// passes ccoGda_WindowSignalAdd, which carries (window, offset, value) instead
// of a slot id. Same NIC atomic, same single reservation; only the raddr/rkey
// lookup differs.
//
// The tradeoff that comes with naming your own window: waitSignal / readSignal /
// resetSignal do not see these, because they are written against the resource
// window's signal pool and its consume-on-read shadow. EP polls and clears the
// flags itself, which is what its protocol did anyway.

__device__ __forceinline__ ::mori::cco::ccoGda_WindowSignalAdd EpInterNodeFlagAdd(
    const EpInterNodeRegion flag, size_t flagOffset, uint64_t value) {
  return ::mori::cco::ccoGda_WindowSignalAdd{EpInterNodeWin(flag), EpInterNodeOff(flag, flagOffset),
                                             value};
}

__device__ __forceinline__ void EpInterNodeAtomicAdd(const ::mori::cco::ccoDevComm& comm,
                                                     const EpInterNodeRegion dst, size_t dstOffset,
                                                     uint64_t value, int pe, int qpId = 0) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template signal<::mori::cco::CCO_TEAM_WORLD>(pe, EpInterNodeFlagAdd(dst, dstOffset, value),
                                                   ::mori::cco::ccoCoopThread{});
}

// RDMA write with the flag atomic fused into the same reservation. The ordering
// the protocol wants -- a receiver that sees the counter move has the payload
// too -- comes from the QP: an RC responder executes requests in PSN order, and
// putImpl places the signal WQE at base + numActiveLanes, after every lane's
// write, with the slot index doubling as the PSN (BNXT reserves its signal PSN
// behind the data PSNs for the same reason).
//
// This is the signalled put shmem had, and it stays ONE reservation: issuing a
// bare put and posting the atomic separately would cost an extra doorbell and
// CQE per chunk, on the critical path of a latency-bound kernel.
//
// ccoGdaThreadAggregate, not ThreadIndependent: the latter runs a group-by-peer
// ballot before posting, which every call site here would resolve in a single
// iteration anyway -- all three are warp-uniform in pe and qpId (proxyPe follows
// the warp's node, and startTokenIdx is a multiple of warpSize so
// tokenId / warpSize is too). Aggregate skips the ballot and posts directly,
// which is the shape shmem had. The warp aggregation itself is unaffected: it
// lives in putImpl, below the ThreadMode branch, so one atomic per warp group
// from the leader lane either way -- the shape the dedup call site depends on,
// since its active lanes carry the same flag slot and the same value.
__device__ __forceinline__ void EpInterNodePutSignal(const ::mori::cco::ccoDevComm& comm,
                                                     const EpInterNodeRegion dst, size_t dstOffset,
                                                     const EpInterNodeRegion src, size_t srcOffset,
                                                     size_t bytes, const EpInterNodeRegion signal,
                                                     size_t signalOffset, uint64_t signalValue,
                                                     int pe, int qpId) {
  ::mori::cco::ccoGda<kEpInterNodeProvider> gda{comm, qpId};
  gda.template put<::mori::cco::CCO_TEAM_WORLD, ::mori::cco::ccoGdaThreadAggregate>(
      pe, EpInterNodeWin(dst), EpInterNodeOff(dst, dstOffset), EpInterNodeWin(src),
      EpInterNodeOff(src, srcOffset), bytes, EpInterNodeFlagAdd(signal, signalOffset, signalValue),
      ::mori::cco::ccoCoopThread{});
}

__device__ __forceinline__ void EpInterNodePut(const ::mori::cco::ccoDevComm& comm,
                                               const EpInterNodeRegion dst, size_t dstOffset,
                                               const EpInterNodeRegion src, size_t srcOffset,
                                               size_t bytes, int pe, int qpId) {
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

// Device timestamps, opt-in. args.dbgTsBuf is null unless MORI_EP_DEV_TS is set
// on the host, so each site below is one wave-uniform scalar compare against a
// kernarg. EVERY site sits OUTSIDE a spin loop: the whole point is to time the
// loops, and an instruction added inside one would change what is measured.
//
// wall_clock64() and not mori::cco::clock64(): the latter is
// __builtin_readcyclecounter(), the SHADER clock, which moves with DVFS. This
// measurement compares a slow round against a fast one on a box where the
// shader clock has been observed to vary by several percent, so a shader-clock
// timer cannot tell "waited longer" from "clocked lower". wall_clock64 is the
// fixed reference counter; measured on this gfx942 rig at 100.008 MHz.
constexpr int kEpDbgTsSlots = 24;

__device__ __forceinline__ void EpDbgTs(const EpDispatchCombineArgs& args, int slot) {
  if (args.dbgTsBuf == nullptr) return;
  args.dbgTsBuf[args.dbgRound * kEpDbgTsSlots + slot] = static_cast<uint64_t>(wall_clock64());
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpDispatchInterNodeV1Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */
namespace internode {
template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchIntraNodeBlock(EpDispatchCombineArgs& args, int tokenId,
                                              int expId, int destPe, int& localPeTokenCounter) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (!args.replayMode) {
    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.dispDestTokIdMap[tokenExpertId] = FlatTokenIndex(config, destPe, destTokId);

      core::AtomicStoreRelaxedSystem(
          args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(destPe) + destTokId,
          static_cast<index_t>(FlatTokenIndex(config, myPe, tokenId)));
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

  T* remoteTokenPtr = args.reg(args.offDispatchOut)->template GetAs<T*>(destPe);
  const T* localTokenPtr = static_cast<const T*>(args.inpTokenBuf);
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, hiddenDim);

  index_t* remoteIndexPtr = args.reg(args.offOutIndices)->template GetAs<index_t*>(destPe);
  const index_t* localIndexPtr = args.tokenIndices;
  core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
                 localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  float* remoteWeightPtr = args.reg(args.offDispatchOutWeights)->template GetAs<float*>(destPe);
  const float* localWeightPtr = args.weightsBuf;
  core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
                 localWeightPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  if (args.scalesBuf && (scaleBytes > 0)) {
    core::WarpCopy(
        args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
        args.scalesBuf + tokenId * scaleBytes, scaleBytes);
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchIntraNode(EpDispatchCombineArgs& args) {
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
      DispatchIntraNodeBlock<kConfig, T>(args, tokenId, inTokenExpertId, destPe, localPeTokenCounter);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <EpInterNodeKernelCfg kConfig, typename T, bool DEDUP>
inline __device__ void DispatchInterNodeSend(EpDispatchCombineArgs& args,
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
    int proxyPe = i * config.gpuPerNode + (myPe % config.gpuPerNode);
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
            EpInterNodePutSignal(comm, args.reg(args.offDispatchInp), remoteIdx * xferBytes,
                                 args.reg(args.offDispatchStaging), stagingTokOffset,
                                 count * xferBytes, args.reg(args.offChunkFlag),
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
          EpInterNodePutSignal(comm, args.reg(args.offDispatchInp), remoteIdx * xferBytes,
                               args.reg(args.offDispatchStaging), stagingTokOffset,
                               tokenNum * xferBytes, args.reg(args.offChunkFlag),
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
    // v1 needed no guard because shmem dispatches per peer on
    // globalGpuStates->transportTypes[pe] (shmem_device_api.hpp): a same-node
    // peer is TransportType::P2P, so the atomic became a local XGMI atomic and
    // never touched a NIC. ccoGda has no such dispatch -- cco_scale_out.hpp
    // mentions neither P2P nor locality -- so scale-out is the only path it can
    // take, and it has nowhere to send this.
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      EpInterNodeAtomicAdd(comm, args.reg(args.offNodeRecvTokenNum), myNode * sizeof(uint64_t),
                           numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeLLSend(EpDispatchCombineArgs& args,
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
    int proxyPe = i * config.gpuPerNode + (myPe % config.gpuPerNode);

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

        // Slots 1/2 bracket the ONE thing that is purely local on the send
        // side: building the WQE and ringing the doorbell. At 4 tokens there is
        // exactly one post per pass so this is exact; above 64 tokens it is
        // last-write-wins and reports the final chunk.
        EpDbgTs(args, 1);
        EpInterNodePutSignal(comm, args.reg(args.offDispatchInp), remoteIdx * xferBytes,
                             args.reg(args.offDispatchStaging), stagingTokOffset,
                             tokenNum * xferBytes, args.reg(args.offChunkFlag),
                             (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), tokenNum + 1,
                             proxyPe, qpId);
        EpDbgTs(args, 2);
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
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      EpInterNodeAtomicAdd(comm, args.reg(args.offNodeRecvTokenNum), myNode * sizeof(uint64_t),
                           numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[1] = 0;
  }
  // End of the send half. This is a fan-in (atomicAdd), not a barrier -- nobody
  // blocks -- so slot 3 measures warp-arrival skew plus the tail atomic, and
  // must not be read as "barrier time".
  if ((globalWarpId == 0) && (laneId == 0)) EpDbgTs(args, 3);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeRecv(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeRecv);

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

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
            destTokId =
                atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe), 1);
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] =
                FlatTokenIndex(config, destPe, destTokId);
            args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(destPe)[destTokId] =
                srcTokId;
          }
          destTokId = __shfl(destTokId, 0);
        } else {
          // Replay: pull cached recv-side slot.
          index_t flat = args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e];
          destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
        }
        if (!args.replayMode && (destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
        core::WarpCopy<uint8_t, 4>(args.reg(args.offDispatchOut)->template GetAs<uint8_t*>(destPe) +
                                       destTokId * hiddenBytes,
                                   stagingPtr + tokIdx * xferBytes, hiddenBytes);
        core::WarpCopy<uint8_t, 4>(
            args.reg(args.offOutIndices)->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
        core::WarpCopy<uint8_t, 4>(
            args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>(destPe) +
                destTokId * weightBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
        if ((scaleBytes > 0)) {
          core::WarpCopy<uint8_t, 4>(
              args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) +
                  destTokId * scaleBytes,
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

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchInterNodeLLRecv(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeLLRecv);

  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.reg(args.offDispatchInp)->template GetAs<uint8_t*>();

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
      // Slots 4/5 bracket the wait for the peer's write to land. Guarded to one
      // warp so the 32 spinning lanes do not all write the same slot; at 4
      // tokens globalWarp 0 runs the outer loop exactly once, so it is
      // unambiguous which wait this is.
      if (globalWarpId == 0) EpDbgTs(args, 4);
      while (1) {
        thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkTokenNum > 0) break;

        nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
          thisChunkTokenNum = 1;
          break;
        }
      }
      if (globalWarpId == 0) EpDbgTs(args, 5);
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
      destTokId = atomicAdd(args.reg(args.offDispTokOffset)->template GetAs<index_t*>(destPe), 1);
      assert(destTokId < config.MaxNumTokensToRecv() &&
             "Total recv token overflow: increase maxTotalRecvTokens");
      args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
          FlatTokenIndex(config, destPe, destTokId);
      args.reg(args.offDispTokIdToSrcTokId)->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
    }
    if ((destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
    destTokId = __shfl(destTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offDispatchOut)->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
        stagingPtr + globalTokenId * xferBytes, hiddenBytes);
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offOutIndices)->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes, indexBytes);
    core::WarpCopy<uint8_t, 4>(
        args.reg(args.offDispatchOutWeights)->template GetAs<uint8_t*>(destPe) +
            destTokId * weightBytes,
        stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes, weightBytes);
    if ((scaleBytes > 0)) {
      core::WarpCopy<uint8_t, 4>(
          args.reg(args.offOutScales)->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
          stagingPtr + globalTokenId * xferBytes + hiddenBytes + indexBytes + weightBytes,
          scaleBytes);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
  // Slot 6 closes the recv half. 5->6 is everything AFTER the peer data has
  // landed: unpacking the staging slots and WarpCopy-ing each token into the
  // destination rank's buffer, which for a same-node destPe is an XGMI peer
  // write. The 4-token measurement put the whole regime delta in this span,
  // with the post and the spin both flat, so it needs its own bracket.
  if ((globalWarpId == 0) && (laneId == 0)) EpDbgTs(args, 6);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void DispatchSync(EpDispatchCombineArgs& args,
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
      index_t* signal = args.reg(args.offRecvTokenNum)->template GetAs<index_t*>(destPe) + myPe;
      core::AtomicStoreSeqCstSystem(signal, numTokenSignal);
    }
    if (laneId == 0)
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    index_t* recvTokenNums = args.reg(args.offRecvTokenNum)->template GetAs<index_t*>();
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
      args.reg(args.offDispTokOffset)->template GetAs<index_t*>()[0] = 0;
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
    int proxyPe = i * config.gpuPerNode + (myPe % config.gpuPerNode);
    EpInterNodeQuiet(comm, proxyPe, config.numQpPerPe);
  }
}

}  // namespace internode

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchInterNodeV1Kernel_body(EpDispatchCombineArgs args,
                                                 const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  if (blockId < args.rdmaBlockNum) {
    internode::DispatchInterNodeSend<kConfig, T, true>(args, comm);
    internode::DispatchInterNodeRecv<kConfig, T>(args);
  } else {
    internode::DispatchIntraNode<kConfig, T>(args);
  }
  internode::DispatchSync<kConfig, T>(args, comm);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchCopyToStaging_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::EpDispatchCopyToStaging);

  // Zero the receive counter here rather than from the host. DispatchSync
  // accumulates into it with atomicAdd, so it has to start at 0 every dispatch;
  // EpCombineAll clears it at the end of the pair, but that only covers a caller
  // that always combines. The host's `total_recv.zero_()` covered the rest at the
  // cost of a whole fill kernel enqueued AHEAD of the dispatch sequence on the
  // same stream -- it delayed the kernels it was protecting. This is the first
  // pass of that sequence and runs entirely before dispatch_ll, so stream order
  // is the ordering guarantee. Before the empty-input return: zero tokens still
  // needs the counter cleared.
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 12);
  if (globalThdId == 0) args.totalRecvTokenNum[0] = 0;
  if (args.curRankNumToken == 0) return;

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  // First copy to staging buffer
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    uint8_t* stagingPtr = args.reg(args.offDispatchStaging)->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset + hiddenDimOffset * sizeof(T),
                               static_cast<const uint8_t*>(args.inpTokenBuf) +
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
          static_cast<index_t>(FlatTokenIndex(config, myPe, tokenId));
  }
  // Slots 12/13 bracket copystaging. It is the ONLY other kernel in the
  // dispatch phase, and the phase's whole regime delta sits outside
  // dispatch_ll, so this is what separates 'the staging copy got slower' from
  // 'the gap between the two launches grew'.
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 13);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpDispatchInterNodeV1KernelLowLatency_body(EpDispatchCombineArgs args,
                                                           const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 0);
  if (blockId < args.rdmaBlockNum) {
    internode::DispatchInterNodeLLSend<kConfig, T>(args, comm);
    internode::DispatchInterNodeLLRecv<kConfig, T>(args);
  } else {
    internode::DispatchIntraNode<kConfig, T>(args);
  }
  internode::DispatchSync<kConfig, T>(args, comm);
  // 6->7 is DispatchSync, the grid-wide barrier that ends the kernel.
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 7);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpCombineInterNodeV1Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */
namespace internode {

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineSync(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineSync);

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, blockNum);
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
      using Fp8T = core::CombineInternalFp8;
      Fp8T* dst = args.reg(args.offCombineInp)->template GetAs<Fp8T*>();
      const T* src = static_cast<const T*>(args.inpTokenBuf);
      const size_t base = tokenId * hiddenDim;
      core::WarpCastBf16ToCombineInternalFp8<T>(dst + base, src + base, hiddenDim, laneId);
    } else {
      core::WarpCopy(args.reg(args.offCombineInp)->template GetAs<T*>() + tokenId * hiddenDim,
                     static_cast<const T*>(args.inpTokenBuf) + tokenId * hiddenDim,
                     hiddenDim);
    }
  }
  if (args.weightsBuf) {
    for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
      core::WarpCopy(args.reg(args.offInpWeights)->template GetAs<float*>() +
                         tokenId * config.numExpertPerToken,
                     args.weightsBuf + tokenId * config.numExpertPerToken,
                     config.numExpertPerToken);
    }
  }
}

namespace combine_impl {

// Gathering a token from its experts reads from up to numExpertPerToken peer
// GPUs over xGMI, and peer-read *latency* -- not bandwidth -- is what caps it.
// WarpAccumLF issues AccumNum*Unroll of those reads before accumulating any of
// them so they overlap; WarpAccum keeps only AccumNum in flight and moves 4B per
// lane. The intra-node combine path (intranode.hpp) has used the 16B load-first
// form for a while; the v1 internode path had not.
//
// Two constraints come with it:
//   - Both ends must be 16B-aligned. A combine staging slot interleaves the
//     hidden payload with the per-token weights, so its stride is only aligned
//     for some topk/dtype combinations; CombineVecAligned() decides per launch
//     and the caller falls back to the 4B path when it cannot.
//   - The vector loop advances CombineVecStep() elements per iteration and drops
//     to a per-lane scalar tail below that. A slice shorter than one step is
//     *slower* than not vectorizing at all, so slices must be a whole multiple
//     of it.
constexpr size_t kCombineVecBytes = 16;

// How many vector steps of a token's hidden dimension go into one warp's slice.
//
// This only sets the split -- CombineVecStep() feeds warpsPerToken below, and
// the slice is rounded up to a whole number of steps so no warp gets less than
// one. It does not reach inside the gather: WarpAccum advances exactly one step
// (warpSize * kCombineVecBytes) per inner iteration regardless of what is set
// here, so a slice of 2 steps simply means each warp runs two iterations.
//
// 2 measures faster than 1 at every token count tried, which is a statement
// about how wide to spread a token, not about the gather's inner loop.
constexpr int kCombineStepsPerWarpSlice = 2;

inline __device__ bool CombineVecAligned(size_t tokHiddenBytes, size_t tokCombXferBytes) {
  return ((tokHiddenBytes % kCombineVecBytes) == 0) && ((tokCombXferBytes % kCombineVecBytes) == 0);
}

template <typename TokT>
inline __device__ size_t CombineVecStep(int warpSizeRt) {
  return static_cast<size_t>(kCombineStepsPerWarpSlice) * warpSizeRt *
         (kCombineVecBytes / sizeof(TokT));
}

template <typename TokT>
inline __device__ void CombineGather(TokT* dest, TokT** srcPtrs, int accumNum, size_t nelems,
                                     bool vecAligned) {
  if (vecAligned) {
    core::WarpAccum<TokT, kCombineVecBytes>(dest, srcPtrs, nullptr, accumNum, nelems);
  } else {
    core::WarpAccum<TokT, 4>(dest, srcPtrs, nullptr, accumNum, nelems);
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeTyped(EpDispatchCombineArgs& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes) {
  DEF_COMMON_VARS;

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
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
        srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim;
        srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
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

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineIntraNodeLLTyped(EpDispatchCombineArgs& args,
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
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes + myNode, 0) * tokCombXferBytes;

  // Slices are snapped to a whole vector step so the gather below stays on
  // WarpAccumLF's vector path instead of its scalar tail.
  MultiWarpIter mwIter(xgmiWarpNum, args.curRankNumToken, hiddenDim,
                       CombineVecStep<TokT>(warpSize));
  const bool vecAligned = CombineVecAligned(tokHiddenBytes, tokCombXferBytes);

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
        srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                          destLocalTokId * hiddenDim + hiddenDimOffset;
        srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    CombineGather<TokT>(
        reinterpret_cast<TokT*>(stagingPtr + tokenId * tokCombXferBytes) + hiddenDimOffset, srcPtrs,
        config.numExpertPerToken, hiddenDimSize, vecAligned);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * tokCombXferBytes + tokHiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeTyped(EpDispatchCombineArgs& args,
                                                      size_t tokHiddenBytes,
                                                      size_t tokCombXferBytes,
                                                      const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  index_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<index_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

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
                  srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                                    destLocalTokId * hiddenDim;
                  srcWeightsPtr[laneId] =
                      args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
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
                    args.reg(args.offChunkFlag)->template GetAs<uint64_t*>() + node * maxChunkNum +
                        k,
                    uint64_t{0});
                core::AtomicStoreRelaxedSystem(
                    args.interNodeChunkFlagCombine + node * maxChunkNum + k, index_t{0});
              }
              int proxyPe = node * config.gpuPerNode + (myPe % config.gpuPerNode);
              int qpId = k % config.numQpPerPe;
              EpInterNodePut(
                  comm, args.reg(args.offStaging),
                  SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                  args.reg(args.offStaging),
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

  // Ensure all prior writes (in particular zeroing the chunk-flag region) are visible
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
          args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if ((laneId < nNodes) &&
        (laneId != myNode)) {  // avoid setting myNode, it will be set in intra node branch
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      for (int i = 0; i < config.numQpPerPe; i++) {
        EpInterNodeAtomicAdd(comm, args.reg(args.offCrossDeviceBarrier),
                             args.rank * sizeof(uint64_t), 1, proxyPe, i);
      }
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    uint64_t* localBarrierPtr = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) !=
             (barrierFlag * config.numQpPerPe)) {
      }
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename TokT, typename T>
__forceinline__ __device__ void CombineInterNodeLLTyped(EpDispatchCombineArgs& args,
                                                        size_t tokHiddenBytes,
                                                        size_t tokCombXferBytes,
                                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.MaxNumTokensToSendPerRank(), warpSize);

  uint64_t* chunkFlag = args.reg(args.offChunkFlag)->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>();

  int rdmaWarpNum = args.rdmaBlockNum * warpNum;
  for (int n = 0; n < (nNodes - 1); n++) {
    int node = (myNode + n + 1) % nNodes;
    uint64_t nodeCount = nodeRecvTokenNum[node];
    if (nodeCount > 0) nodeCount -= 1;
    if (nodeCount == 0) continue;

    // One whole vector step per warp. warpsPerToken was a fixed 4, which for
    // hidden 6144 bf16 gives a 1536-element slice -- one full 1024-element vector
    // step plus a 512-element scalar tail, and that tail costs more than the
    // vector part saves. Sizing the split by the step instead keeps every warp on
    // the vector path.
    //
    // This has to be a static function of the config: chunkFlag is cleared by
    // whichever warp completes a chunk, so anything derived from the live counts
    // can differ between two warps, and they must agree on the completion target.
    const size_t vecStep = CombineVecStep<TokT>(warpSize);
    int warpsPerToken = static_cast<int>(hiddenDim / vecStep);
    if (warpsPerToken < 1) warpsPerToken = 1;
    size_t hiddenDimPerWarp = core::CeilDiv(hiddenDim, static_cast<size_t>(warpsPerToken));
    hiddenDimPerWarp = core::CeilDiv(hiddenDimPerWarp, vecStep) * vecStep;
    const bool vecAligned = CombineVecAligned(tokHiddenBytes, tokCombXferBytes);

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
            srcPtrs[laneId] = args.reg(args.offCombineInp)->template GetAs<TokT*>(destPe) +
                              destLocalTokId * hiddenDim + hiddenDimOffset;
            srcWeightsPtr[laneId] = args.reg(args.offInpWeights)->template GetAs<float*>(destPe) +
                                    destLocalTokId * config.numExpertPerToken;
          }
        }
        CombineGather<TokT>(reinterpret_cast<TokT*>(stagingPtr + globalTokenId * tokCombXferBytes) +
                                hiddenDimOffset,
                            srcPtrs, config.numExpertPerToken, hiddenDimSize, vecAligned);
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
              args.reg(args.offChunkFlag)->template GetAs<uint64_t*>() + node * maxChunkNum + k,
              uint64_t{0});
          core::AtomicStoreRelaxedSystem(args.interNodeChunkFlagCombine + node * maxChunkNum + k,
                                         index_t{0});
        }
        int proxyPe = node * config.gpuPerNode + (myPe % config.gpuPerNode);
        int qpId = k % config.numQpPerPe;
        EpDbgTs(args, 9);
        EpInterNodePut(comm, args.reg(args.offStaging),
                       SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                       args.reg(args.offStaging),
                       SendBufSlotOffset(config, node, startTokenIdx) * tokCombXferBytes,
                       thisChunkTokenNum * tokCombXferBytes, proxyPe, qpId);
      }
    }
  }

  // Ensure all prior writes (in particular zeroing the chunk-flag region) are visible
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
          args.reg(args.offNodeRecvTokenNum)->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
    if ((laneId < nNodes) &&
        (laneId != myNode)) {  // avoid setting myNode, it will be set in intra node branch
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      for (int i = 0; i < config.numQpPerPe; i++) {
        EpInterNodeAtomicAdd(comm, args.reg(args.offCrossDeviceBarrier),
                             args.rank * sizeof(uint64_t), 1, proxyPe, i);
      }
      __threadfence_system();
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    // Wait other nodes
    uint64_t* localBarrierPtr = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
    if ((laneId < nNodes) && (laneId != myNode)) {
      int proxyPe = laneId * config.gpuPerNode + (myPe % config.gpuPerNode);
      // Slots 10/11: combine's only cross-node wait.
      EpDbgTs(args, 10);
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) !=
             (barrierFlag * config.numQpPerPe)) {
      }
      EpDbgTs(args, 11);
    }
  }
}

}  // namespace combine_impl

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineIntraNode(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNode);
  if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineIntraNodeTyped<kConfig, TokT, T>(args, tokHiddenBytes, tokCombXferBytes);
    return;
  }

  combine_impl::CombineIntraNodeTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineIntraNodeLL(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNodeLL);

  if (args.curRankNumToken == 0) return;
  if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineIntraNodeLLTyped<kConfig, TokT, T>(args, tokHiddenBytes, tokCombXferBytes);
    return;
  }
  combine_impl::CombineIntraNodeLLTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineInterNode(EpDispatchCombineArgs& args,
                                        const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNode);

  if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineInterNodeTyped<kConfig, TokT, T>(args, tokHiddenBytes, tokCombXferBytes,
                                                       comm);
    return;
  }
  combine_impl::CombineInterNodeTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes, comm);
}

template <EpInterNodeKernelCfg kConfig, typename T>
inline __device__ void CombineInterNodeLL(EpDispatchCombineArgs& args,
                                          const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL);
  if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
    using TokT = core::CombineInternalFp8;
    const size_t tokHiddenBytes = hiddenDim * sizeof(TokT);
    const size_t tokCombXferBytes =
        (args.weightsBuf == nullptr) ? tokHiddenBytes : tokHiddenBytes + weightBytes;
    combine_impl::CombineInterNodeLLTyped<kConfig, TokT, T>(args, tokHiddenBytes, tokCombXferBytes,
                                                         comm);
    return;
  }
  combine_impl::CombineInterNodeLLTyped<kConfig, T, T>(args, hiddenBytes, combXferBytes, comm);
}
}  // namespace internode

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineInterNodeV1Kernel_body(EpDispatchCombineArgs args,
                                                const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;

  if (blockId < args.rdmaBlockNum) {
    internode::CombineInterNode<kConfig, T>(args, comm);
  } else {
    internode::CombineIntraNode<kConfig, T>(args);
  }
}

namespace combine_all_impl {

template <EpInterNodeKernelCfg kConfig, typename T>
__forceinline__ __device__ void EpCombineAllInternalFp8(EpDispatchCombineArgs& args,
                                                        size_t fp8HiddenBytes,
                                                        size_t fp8CombXferBytes) {
  DEF_COMMON_VARS;
  using Fp8T = core::CombineInternalFp8;

  extern __shared__ char sharedMem[];
  Fp8T** srcPtrs = reinterpret_cast<Fp8T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
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

    T* out =
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset;
    core::WarpAccumCombineInternalFp8ToBf16(out, reinterpret_cast<const Fp8T* const*>(srcPtrs),
                                            nNodes, laneId, hiddenDimSize);

    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(args.reg(args.offCombineOutWeights)->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__forceinline__ __device__ void EpCombineAllGeneric(EpDispatchCombineArgs& args) {
  DEF_COMMON_VARS;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.reg(args.offStaging)->template GetAs<uint8_t*>() +
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
    core::WarpAccum<T, 4>(
        args.reg(args.offCombineOut)->template GetAs<T*>() + tokenId * hiddenDim + hiddenDimOffset,
        srcPtrs, nullptr, nNodes, hiddenDimSize);
    if (args.weightsBuf && (inTokenPartId == mwIter.warpsPerItem - 1)) {
      core::WarpAccum<float, 4>(args.reg(args.offCombineOutWeights)->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

}  // namespace combine_all_impl

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineAll_body(EpDispatchCombineArgs args) {
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
  if constexpr (kConfig.quantType == QuantType::Fp8DirectCast) {
    using Fp8T = core::CombineInternalFp8;
    const size_t fp8HiddenBytes = hiddenDim * sizeof(Fp8T);
    const size_t fp8CombXferBytes =
        (args.weightsBuf == nullptr) ? fp8HiddenBytes : fp8HiddenBytes + weightBytes;
    combine_all_impl::EpCombineAllInternalFp8<kConfig, T>(args, fp8HiddenBytes, fp8CombXferBytes);
    // Slot 14 marks the end of the LAST pass of the combine phase, on both
    // exits. Paired with slot 12 of the NEXT round's dispatch it gives the true
    // GPU-timeline gap between the two phases -- the quantity that the phase
    // events cannot separate from host time.
    if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 14);
    return;
  }
  combine_all_impl::EpCombineAllGeneric<kConfig, T>(args);
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 14);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineInterNodeV1KernelLowLatency_body(EpDispatchCombineArgs args,
                                                          const ::mori::cco::ccoDevComm& comm) {
  DEF_COMMON_VARS;
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 8);

  if (blockId < args.rdmaBlockNum) {
    internode::CombineInterNodeLL<kConfig, T>(args, comm);
  } else {
    internode::CombineIntraNodeLL<kConfig, T>(args);
  }
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSync_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  // 18/19 bracket combinesync, the FIRST pass of the combine phase. With 7->18
  // this splits pre_bar into "everything between the two mori calls" -- the
  // torch convert and the launch path -- and the kernel itself.
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 18);
  internode::CombineSync<kConfig, T>(args);
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 19);
}

template <EpInterNodeKernelCfg kConfig, typename T>
__device__ void EpCombineSyncBarrier_body(EpDispatchCombineArgs args) {
  DEF_COMMON_VARS;
  // 16/17 bracket the cross-device barrier that OPENS the combine phase. It
  // sits between dispatch_ll ending and combine_ll starting, so the phase
  // events charge it to neither -- and the closed per-round accounting put both
  // the largest term and the whole regime delta in exactly that window.
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 16);
  IF_ENABLE_PROFILER(
      INTERNODE_V1_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SPAN(profiler, Slot::EpCombineSyncBarrier);
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  barrierFlag = __shfl(barrierFlag, 0);
  uint64_t* localBarrierPtr = args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>();
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    core::AtomicStoreRelaxedSystem(
        args.reg(args.offCrossDeviceBarrier)->template GetAs<uint64_t*>(destPe) + args.rank,
        barrierFlag);
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + destPe) != barrierFlag) {
    }
  }
  if ((blockId == 0) && (thdId == 0)) EpDbgTs(args, 17);
}

}  // namespace v2
}  // namespace ops
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
namespace ops {
namespace v2 {

}  // namespace v2
}  // namespace ops
}  // namespace mori

// `kConfig` and `TokT` are not macro arguments. The generated TU defines both
// under those names just above the entry, exactly as ep_spec.cpp emits
// `constexpr EpCfg kCfg` / `using TokT` and then instantiates
// `EpDispatchBody<kCfg, TokT>`. A Cfg could not be a macro argument in any case:
// the commas inside its brace initialiser would be taken as argument separators.

// Kernels that reach the network.
#define MORI_EP_INTERNODE_CCO_ENTRY(entry, body)                                                  \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs a) {                       \
    ::mori::ops::v2::body<kConfig, TokT>(a.args, a.devComm); \
  }

// Staging, sync and the final reduction: local or intra-node only, so they take
// no communicator. They still take the same argument struct, so the host has one
// launch path for the whole sequence.
#define MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(entry, body)                                 \
  extern "C" __global__ void entry(::mori::ops::v2::EpInterNodeCcoArgs a) {            \
    ::mori::ops::v2::body<kConfig, TokT>(a.args); \
  }
