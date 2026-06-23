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

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif

namespace mori {
namespace moe {

using core::load;
using core::store;
using core::VecTypeSelector;

constexpr int kMaxGpusPerNode = 8;

#define LDS_BARRIER()                     \
  {                                       \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __builtin_amdgcn_s_barrier();         \
  }

/* ──────────────────────────────────────────────────────────────────────────────
 * ReadLanePtr — broadcast a 64-bit pointer from one lane to all via
 * v_readlane_b32.  Result is SGPR-uniform, so the address stays scalar
 * for downstream stores.
 * ────────────────────────────────────────────────────────────────────────────── */
template <typename T>
inline __device__ T* ReadLanePtr(T* ptr, int srcLane) {
  unsigned long long val = reinterpret_cast<unsigned long long>(ptr);
  uint32_t lo = __builtin_amdgcn_readlane(static_cast<uint32_t>(val), srcLane);
  uint32_t hi = __builtin_amdgcn_readlane(static_cast<uint32_t>(val >> 32), srcLane);
  return reinterpret_cast<T*>((static_cast<unsigned long long>(hi) << 32) | lo);
}

/* ──────────────────────────────────────────────────────────────────────────────
 * WarpLoadBroadcastStore
 *
 * Read `count` elements of type T from `src` once into registers, then store
 * the same register contents to each of `numDests` destination pointers.
 *
 *   HBM reads:    count × sizeof(T)               (once)
 *   Remote writes: count × sizeof(T) × numDests
 *
 * `dstReg`: each lane holds ONE destination pointer in a register.
 *           Lanes 0..numDests-1 hold valid pointers; the rest are unused.
 *           Destination d is broadcast to all lanes via ReadLanePtr from lane d.
 *           No pointer array — everything stays in registers.
 * ────────────────────────────────────────────────────────────────────────────── */
template <typename T, int MaxDests = kMaxGpusPerNode, int Unroll = 1>
inline __device__ void WarpLoadBroadcastStoreImpl(T* dstReg, int numDests,
                                                  const T* __restrict__ src, size_t& offset,
                                                  size_t count) {
  constexpr int kVecBytes = 16;
  constexpr int kElemsPerVec = kVecBytes / sizeof(T);
  using VecT = typename VecTypeSelector<kVecBytes>::dataType;

  const int laneId = threadIdx.x & (warpSize - 1);
  const int elemsPerIter = Unroll * warpSize * kElemsPerVec;
  const size_t numIters = (count - offset) / elemsPerIter;

  for (size_t iter = 0; iter < numIters; ++iter) {
    VecT regs[Unroll];
#pragma unroll Unroll
    for (int u = 0; u < Unroll; ++u) {
      regs[u] = load<kVecBytes>(src + offset + (laneId + u * warpSize) * kElemsPerVec);
    }

    for (int d = 0; d < numDests; ++d) {
      T* dst = ReadLanePtr(dstReg, d);
#pragma unroll Unroll
      for (int u = 0; u < Unroll; ++u) {
        store<kVecBytes>(dst + offset + (laneId + u * warpSize) * kElemsPerVec, regs[u]);
      }
    }
    offset += elemsPerIter;
  }
}

template <typename T, int MaxDests = kMaxGpusPerNode, int Unroll = 1>
inline __device__ void WarpLoadBroadcastStore(T* dstReg, int numDests, const T* __restrict__ src,
                                              size_t count) {
  const int laneId = threadIdx.x & (warpSize - 1);
  size_t offset = 0;

  WarpLoadBroadcastStoreImpl<T, MaxDests, Unroll>(dstReg, numDests, src, offset, count);
  if constexpr (Unroll > 1) {
    WarpLoadBroadcastStoreImpl<T, MaxDests, 1>(dstReg, numDests, src, offset, count);
  }

  // Scalar tail
  for (size_t i = offset + laneId; i < count; i += warpSize) {
    T val = src[i];
    for (int d = 0; d < numDests; ++d) {
      T* dst = ReadLanePtr(dstReg, d);
      dst[i] = val;
    }
  }
}

/* ──────────────────────────────────────────────────────────────────────────────
 * TokenRoute — per-token routing result stored in shared memory
 * ────────────────────────────────────────────────────────────────────────────── */
struct TokenRoute {
  int numDests;
  int srcTokenId;
  int destRank[kMaxGpusPerNode];
  index_t recvSlot[kMaxGpusPerNode];
};

/* ──────────────────────────────────────────────────────────────────────────────
 * ComputeTokenRoute — warp-0-only: PE-indexed dedup + slot allocation
 *
 * Expert-slot lanes (laneId < topK) compute target PE from expertId.
 * PE lanes (laneId < numRanks) use readlane to scan all expert slots
 * and check if any targets them — pure register dedup, no shared memory.
 * ────────────────────────────────────────────────────────────────────────────── */
template <typename T>
__device__ inline void ComputeTokenRoute(EpDispatchCombineArgs<T>& args, int tokenId,
                                         TokenRoute& route, int& outDidAlloc,
                                         int& outExpertSlotForMe, index_t& outMySlot) {
  const EpDispatchCombineConfig& config = args.config;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int topK = config.numExpertPerToken;
  const int numRanks = config.worldSize;

  route.srcTokenId = tokenId;

  // Prefetch remote SHMEM pointer — resolve while dedup runs below
  index_t* dispTokOffset = nullptr;
  if (laneId < numRanks) {
    dispTokOffset = args.dispTokOffsetMemObj->template GetAs<index_t*>(laneId);
  }

  // Expert-slot lanes compute target PE
  int warpTargetPe = -1;
  if (laneId < topK) {
    index_t expertId = args.tokenIndices[tokenId * topK + laneId];
    warpTargetPe = (expertId >= 0) ? (expertId / config.numExpertPerRank) : -1;
  }

  // PE dedup via readlane — each PE lane scans all expert slots
  int expertSlotForMe = -1;
#pragma unroll
  for (int k = 0; k < kMaxGpusPerNode; ++k) {
    int pe_k = __builtin_amdgcn_readlane(warpTargetPe, k);
    expertSlotForMe = (pe_k == laneId) ? k : expertSlotForMe;
  }

  // PE-indexed lanes allocate recv slots
  int didAlloc = 0;
  index_t warpSlot = 0;

  if (laneId < numRanks && expertSlotForMe >= 0) {
    didAlloc = 1;
    warpSlot = atomicAdd(dispTokOffset, 1);
    assert(warpSlot < config.MaxNumTokensToRecv() &&
           "Total recv token overflow: increase maxTotalRecvTokens");
  }

  // Compact active PEs into TokenRoute
  unsigned long long activeMask = __ballot(didAlloc);
  if (didAlloc) {
    int compactPos = __popcll(activeMask & ((1ULL << laneId) - 1));
    route.destRank[compactPos] = laneId;
    route.recvSlot[compactPos] = warpSlot;
  }
  if (laneId == 0) route.numDests = __popcll(activeMask);

  outDidAlloc = didAlloc;
  outExpertSlotForMe = expertSlotForMe;
  outMySlot = warpSlot;
}

/* ──────────────────────────────────────────────────────────────────────────────
 * WriteDispDestTokIdMap — deferred dispDestTokIdMap write, called after
 * __syncthreads() so warp 0 reaches the barrier sooner.
 * ────────────────────────────────────────────────────────────────────────────── */
template <typename T>
__device__ inline void WriteDispDestTokIdMap(EpDispatchCombineArgs<T>& args, int tokenId,
                                             int didAlloc, int expertSlotForMe, index_t warpSlot,
                                             index_t* dispTokIdToSrcBase) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int topK = args.config.numExpertPerToken;
  const int numRanks = args.config.worldSize;

  if (laneId < topK) {
    args.dispDestTokIdMap[tokenId * topK + laneId] = FlatTokenIndex(args.config, numRanks, 0);
  }
  if (didAlloc) {
    atomicAdd(args.destPeTokenCounter + laneId, 1);
    dispTokIdToSrcBase[warpSlot] = FlatTokenIndex(args.config, args.config.rank, tokenId);
    args.dispDestTokIdMap[tokenId * topK + expertSlotForMe] =
        FlatTokenIndex(args.config, laneId, warpSlot);
  }
}

/* ──────────────────────────────────────────────────────────────────────────────
 * EpDispatchIntraNodeCoopKernel
 *
 * Token-centric cooperative dispatch. Compared to the per-(token,expert)
 * EpDispatchIntraNodeKernel and the 2-warp-group EpDispatchIntraNodeLLKernel:
 *
 *   1. Load-once-write-many: each warp loads its hiddenDim slice from HBM
 *      into registers ONCE, then writes the same registers to every dest PE.
 *      Saves (numDests-1)/numDests of HBM read bandwidth.
 *
 *   2. Pipeline: warp 0 (control plane) computes routing for token N+1
 *      while warps 1..N-1 (data plane) broadcast-copy token N.
 *      Double-buffered TokenRoute in shared memory.
 *
 *   3. Block-per-token: one block handles one token's full fan-out,
 *      enabling the load-once pattern without cross-block coordination.
 * ────────────────────────────────────────────────────────────────────────────── */
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeLLKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  const int thdId = threadIdx.x;
  const int laneId = thdId & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int warpRank = config.rank;
  const int numRanks = config.worldSize;
  const int topK = config.numExpertPerToken;
  const size_t hiddenDim = config.HiddenDimSz();
  const int numTokens = args.curRankNumToken;
  const bool hasScales = args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0);

  IF_ENABLE_PROFILER(
      int globalWarpId = blockIdx.x * warpNum + warpId;
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  __shared__ TokenRoute routeBuf[2];  // ping-pong for pipeline
  const bool hasPayload = args.tokenIndices && args.inpTokenBuf;

  if (hasPayload) {
    // ═══════════════════════════════════════════════════════════════════════
    //  Prologue: warp 0 routes the first token this block owns
    // ═══════════════════════════════════════════════════════════════════════
    int prologueDidAlloc = 0, prologueExpertSlot = -1;
    index_t prologueMySlot = 0;

    float* weightsBase = nullptr;
    index_t* indicesBase = nullptr;
    index_t* dispTokIdToSrcBase = nullptr;

    uint8_t* scalesBase = nullptr;
    T* dispOutBase = nullptr;

    if (warpId == 0 && blockIdx.x < numTokens) {
      ComputeTokenRoute(args, blockIdx.x, routeBuf[0], prologueDidAlloc, prologueExpertSlot,
                        prologueMySlot);
      if (laneId < numRanks) {
        if (args.weightsBuf) {
          weightsBase = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(laneId);
        }
        indicesBase = args.shmemOutIndicesMemObj->template GetAs<index_t*>(laneId);
        dispTokIdToSrcBase = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(laneId);
      }
    } else if (warpId == 1 && hasScales && laneId < numRanks) {
      // Warp 1: prefetch scale base pointers
      scalesBase = args.shmemOutScalesMemObj->template GetAs<uint8_t*>(laneId);
    } else if (warpId > 1 && laneId < numRanks) {
      dispOutBase = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(laneId);
    }

    LDS_BARRIER()

    // uint64_t tStart = __builtin_amdgcn_s_memrealtime();
    // uint64_t tEnd = 0;

    if (warpId == 0 && blockIdx.x < numTokens) {
      WriteDispDestTokIdMap(args, blockIdx.x, prologueDidAlloc, prologueExpertSlot, prologueMySlot,
                            dispTokIdToSrcBase);
    }

    //  Main Loop: one token per iteration, block-strided
    for (int tokenIdx = blockIdx.x; tokenIdx < numTokens; tokenIdx += gridDim.x) {
      const int pingPong = (tokenIdx / gridDim.x) & 1;

      // ── Warp 0: route next token + weights/indices for current token ──
      if (warpId == 0) {
        const int nextPingPong = 1 - pingPong;
        const int nextTokenIdx = tokenIdx + gridDim.x;
        const bool hasNextToken = (nextTokenIdx < numTokens);

        int deferDidAlloc = 0, deferExpertSlot = -1;
        index_t deferMySlot = 0;

        if (hasNextToken) {
          ComputeTokenRoute(args, nextTokenIdx, routeBuf[nextPingPong], deferDidAlloc,
                            deferExpertSlot, deferMySlot);
          WriteDispDestTokIdMap(args, nextTokenIdx, deferDidAlloc, deferExpertSlot, deferMySlot,
                                dispTokIdToSrcBase);
        }

        const TokenRoute& route = routeBuf[pingPong];
        if (route.numDests > 0) {
          auto srcTokenId = route.srcTokenId;
          float warpWeight = 0.0f;
          index_t warpIndex = 0;
          if (laneId < topK) {
            if (args.weightsBuf) {
              warpWeight = args.weightsBuf[srcTokenId * topK + laneId];
            }
            warpIndex = args.tokenIndices[srcTokenId * topK + laneId];
          }
          for (int d = 0; d < route.numDests; ++d) {
            int rank = route.destRank[d];
            index_t slot = route.recvSlot[d];
            if (laneId < topK) {
              if (args.weightsBuf) {
                (ReadLanePtr(weightsBase, rank) + slot * topK)[laneId] = warpWeight;
              }
              (ReadLanePtr(indicesBase, rank) + slot * topK)[laneId] = warpIndex;
            }
          }
        }
      }

      // ── Warp 1: scales for current token ─────────────────────────────
      if (warpId == 1 && hasScales) {
        const TokenRoute& route = routeBuf[pingPong];
        if (route.numDests > 0) {
          const size_t scaleSize = config.scaleDim * config.scaleTypeSize;
          uint8_t* warpScaleDstReg = nullptr;
          for (int d = 0; d < route.numDests; ++d) {
            if (laneId == d) {
              warpScaleDstReg = ReadLanePtr(scalesBase, route.destRank[d]) +
                                (size_t)route.recvSlot[d] * scaleSize;
            }
          }
          const uint8_t* srcScale = args.scalesBuf + (size_t)route.srcTokenId * scaleSize;
          WarpLoadBroadcastStore<uint8_t>(warpScaleDstReg, route.numDests, srcScale, scaleSize);
        }
      }

      // ── Data plane: warps 2..N-1 broadcast-copy token payload ────────
      if (warpId > 1) {
        const TokenRoute& route = routeBuf[pingPong];

        if (route.numDests > 0) {
          const int copyWarpRank = warpId - 2;
          const int numCopyWarps = warpNum - 2;

          constexpr int kCopyUnroll = 2;
          constexpr size_t kVecElems = 16 / sizeof(T);
          constexpr size_t kBaseElemsPerWarp = kCopyUnroll * warpSize * kVecElems;
          const size_t warpsNeeded = (hiddenDim + kBaseElemsPerWarp - 1) / kBaseElemsPerWarp;
          const size_t elemsPerWarp = (warpsNeeded <= (size_t)numCopyWarps)
                                          ? kBaseElemsPerWarp
                                          : ((hiddenDim + numCopyWarps - 1) / numCopyWarps);
          const size_t warpElemOffset = (size_t)copyWarpRank * elemsPerWarp;
          const size_t warpElemCount = (warpElemOffset < hiddenDim)
                                           ? min(elemsPerWarp, hiddenDim - warpElemOffset)
                                           : size_t{0};

          if (warpElemCount > 0) {
            const T* srcChunk =
                args.inpTokenBuf + (size_t)route.srcTokenId * hiddenDim + warpElemOffset;

            int warpDestRank = (laneId < route.numDests) ? route.destRank[laneId] : 0;
            index_t warpRecvSlot = (laneId < route.numDests) ? route.recvSlot[laneId] : 0;

            T* warpDstReg = nullptr;
            for (int i = 0; i < route.numDests; ++i) {
              int d = (copyWarpRank & 1) ? (route.numDests - 1 - i) : i;
              int rankD = __builtin_amdgcn_readlane(warpDestRank, d);
              index_t slotD = __builtin_amdgcn_readlane(warpRecvSlot, d);
              if (laneId == i) {
                warpDstReg =
                    ReadLanePtr(dispOutBase, rankD) + (size_t)slotD * hiddenDim + warpElemOffset;
              }
            }
            WarpLoadBroadcastStore<T, kMaxGpusPerNode, kCopyUnroll>(warpDstReg, route.numDests,
                                                                    srcChunk, warpElemCount);
          }
        }
      }

      // tEnd = __builtin_amdgcn_s_memrealtime();
      // if (warpRank == 0 && (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1) && laneId == 0) {
      //   printf("block%d warp%d: %llu cycles hasScales=%d\n", blockIdx.x, warpId, (unsigned long
      //   long)(tEnd - tStart), (int)hasScales);
      // }

      LDS_BARRIER()
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Epilogue: grid barrier → notify peers → wait for incoming tokens
  // ═══════════════════════════════════════════════════════════════════════════
  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  const int globalWarpIdEpi = blockIdx.x * warpNum + warpId;
  if (globalWarpIdEpi == 0) {
    for (int pe = laneId; pe < numRanks; pe += warpSize) {
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      index_t tokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + pe) + 1;
      index_t* signalSlot = args.recvTokenNumMemObj->template GetAs<index_t*>(pe) + warpRank;
      shmem::ShmemInt32WaitUntilEquals(signalSlot, 0);
      core::AtomicStoreRelaxedSystem(signalSlot, tokenSignal);
    }
  }

  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpIdEpi == 0) {
    for (int pe = laneId; pe < numRanks; pe += warpSize) {
      index_t* signalSlot = recvTokenNums + pe;
      index_t recvCount = shmem::ShmemInt32WaitUntilGreaterThan(signalSlot, 0) - 1;
      core::AtomicStoreRelaxedSystem(signalSlot, 0);
      atomicAdd(args.totalRecvTokenNum, recvCount);

      args.destPeTokenCounter[pe] = 0;
    }

    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, warpRank);
  }
#endif
}

}  // namespace moe
}  // namespace mori
