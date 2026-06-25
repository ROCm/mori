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

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint64_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  __syncthreads();
  if (thdId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
    __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp compute token offset on destinition PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destPe;
      index_t destTokId = 0;

      if (!args.replayMode) {
        // Cache routing: decide where this (token, top-k) pair goes via
        // atomicAdd-based slot assignment. Records the routing into dispDestTokIdMap
        // (and the symmetric local view via dispTokIdToSrcTokIdMemObj on the
        // destination PE) so a later replay-routing dispatch / combine can reuse
        // the same layout deterministically.
        index_t destExpert = args.tokenIndices[i];
        // Routing sentinel: a negative expert id means "drop this top-k slot".
        // Skip the dispatch entirely and write the existing combine-side null sentinel
        // (PE == worldSize) into dispDestTokIdMap so combine treats this slot as nullptr.
        if (destExpert < 0) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }
        destPe = destExpert / config.numExpertPerRank;
        // Out-of-range expert id guard: destPe is warp-uniform here (one
        // token-expert per warp) and indexes GetAs(destPe) / destPeTokenCounter
        // below. An out-of-range id (e.g. an EPLB physical id
        // >= worldSize*numExpertPerRank) would index those out of bounds (the
        // assert at dispatch is stripped under NDEBUG) -> HSA page fault. Drop it
        // via the same overflow sentinel the dedup path uses; the whole warp
        // skips coherently.
        if (destPe < 0 || destPe >= config.worldSize) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }

        // Deduplicate
        assert(config.numExpertPerToken < warpSize);
        int condition = 0;
        if (laneId < (i % config.numExpertPerToken)) {
          index_t otherExpert = args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
          condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
        }
        if (__any(condition)) {
          // Indicate that this token is already sent to the destination PE by setting an overflow
          // token index
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }

        if (laneId == 0) {
          // decide token id in dest pe
          destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
          assert(destTokId < config.MaxNumTokensToRecv() &&
                 "Total recv token overflow: increase maxTotalRecvTokens");
          atomicAdd(args.destPeTokenCounter + destPe, 1);
          // In dispDestTokIdMap, record the destination slot for this token-expert pair (flat index
          // into the dest PE's recv buffer) In dispTokIdToSrcTokIdMemObj on the dest PE, record
          // which global source token occupies this slot (for combine-phase routing)
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
              FlatTokenIndex(config, myPe, srcTokId);
        }
        destTokId = __shfl(destTokId, 0);
      } else {
        // Replay routing: caller already supplied a populated dispDestTokIdMap
        // from a matching cache-routing dispatch. Recover (destPe, destTokId) directly
        // and skip CAS / dedup / cross-rank src-id writes. The sentinel slot
        // (destPe == worldSize) means the original cache-routing dispatch dropped or deduped
        // this top-k slot, so we skip transmitting payload as well.
        index_t flat = args.dispDestTokIdMap[i];
        destPe = PeFromFlatTokenIndex(config, flat);
        if (destPe >= config.worldSize) continue;
        destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
      }

      // Write weights and indices
      if (laneId < config.numExpertPerToken) {
        if (args.weightsBuf) {
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
        }
        args.shmemOutIndicesMemObj->template GetAs<index_t*>(
            destPe)[destTokId * config.numExpertPerToken + laneId] =
            args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
      }

      // Write scales
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        size_t destScaleOffset = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
        size_t srcScaleOffset = (size_t)srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }

      size_t srcTokOffset = srcTokId * hiddenDim;
      size_t destTokOffset = destTokId * hiddenDim;

      core::WarpCopy(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
                     args.inpTokenBuf + srcTokOffset, hiddenDim);
    }
  }
  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeLLKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();
  const bool hasScales = args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0);

  // Warp-group coordination: 2 warps per group
  constexpr int kWarpsPerGroup = 2;
  constexpr int kMaxWarpGroups = 8;

  __shared__ uint64_t groupData[kMaxWarpGroups];  // (iteration+1, destTokId)
  __shared__ int groupCounters[kMaxWarpGroups];   // consume counter

  int warpGroupIdInBlock = warpId / kWarpsPerGroup;
  int inGroupWarpId = warpId % kWarpsPerGroup;
  int warpGroupId = globalWarpId / kWarpsPerGroup;
  int warpGroupNum = globalWarpNum / kWarpsPerGroup;

  // Rotate header warp selection so that headers spread evenly across all 4 SIMD units.
  constexpr int kNumSimds = 4;
  int headerWarpIdx;
  if constexpr (kWarpsPerGroup >= kNumSimds) {
    // Each group spans all SIMDs; pick directly.
    headerWarpIdx = warpGroupIdInBlock % kNumSimds;
  } else {
    // Each group spans fewer SIMDs than available; stagger across groups.
    // k=2: groups {0,1}→h=0, {2,3}→h=1 → SIMDs {0,2,1,3}.
    headerWarpIdx = (warpGroupIdInBlock / (kNumSimds / kWarpsPerGroup)) % kWarpsPerGroup;
  }
  bool isHeaderWarp = (inGroupWarpId == headerWarpIdx);

  // clear shared memory
  if (warpId == 0 && laneId < kMaxWarpGroups) {
    groupData[laneId] = 0;
    groupCounters[laneId] = kWarpsPerGroup - 1;
  }

  __syncthreads();

  // Hidden dim split for each warp in group
  size_t dimPerWarp = (hiddenDim + kWarpsPerGroup - 1) / kWarpsPerGroup;
  size_t warpDimOffset = (size_t)inGroupWarpId * dimPerWarp;
  size_t warpDimChunk =
      (warpDimOffset < hiddenDim) ? min(hiddenDim - warpDimOffset, dimPerWarp) : 0;
  assert((warpNum % kWarpsPerGroup == 0) && (warpDimChunk > 0) &&
         "total num of warps must be divisible by the num of warpgroups, "
         "warpDimChunk must be > 0 for warpgroups to be useful");

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp-group processes one token-expert pair
    for (int i = warpGroupId; i < args.curRankNumToken * config.numExpertPerToken;
         i += warpGroupNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t tokExpert = args.tokenIndices[srcTokId * config.numExpertPerToken +
                                            laneId % config.numExpertPerToken];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t destTokId = 0;

      // prefetch remote addr
      auto dispTokOffset = args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe);

      // ALL warps in warp-group do dedup independently (same input = same result)
      assert(config.numExpertPerToken < warpSize);
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        condition = destPe == (tokExpert / config.numExpertPerRank);
      }
      if (__any(condition)) {
        // All 4 warps skip together, only warp 0 writes the skip marker
        if (!isHeaderWarp && laneId == 0) {
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        }
        continue;
      }

      T* dispatchOut = nullptr;

      // Header Warp: atomic allocation + notify via shared memory
      if (isHeaderWarp) {
        if (laneId == 0) {
          // Atomic allocation
          destTokId = atomicAdd(dispTokOffset, 1);
          assert(destTokId < config.MaxNumTokensToRecv() &&
                 "total recv token overflow: increase maxTotalRecvTokens");

          // Wait for all consumers done (counter == N-1), then set to 0
          while (atomicCAS(&groupCounters[warpGroupIdInBlock], kWarpsPerGroup - 1, 0) !=
                 kWarpsPerGroup - 1) {
          }
          // Write to shared mem: use (i+1) to avoid confusion with initial value 0
          __hip_atomic_store((unsigned long long*)&groupData[warpGroupIdInBlock],
                             ((uint64_t)(i + 1) << 32) | (uint32_t)destTokId, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);

          atomicAdd(args.destPeTokenCounter + destPe, 1);
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
              FlatTokenIndex(config, myPe, srcTokId);
        }
        destTokId = __shfl(destTokId, 0);

        // prefetch remote addr
        dispatchOut = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe);

        // Write weights and indices: only warp 0 writes
        if (laneId < config.numExpertPerToken) {
          if (args.weightsBuf) {
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                destPe)[destTokId * config.numExpertPerToken + laneId] =
                args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
          }
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
        }
      } else {
        // prefetch remote addr
        dispatchOut = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe);
        // normal Warps: spin wait for iteration match, then consume
        // only lane 0 spins, then broadcast to other lanes
        if (laneId == 0) {
          uint64_t val;
          do {
            val = __hip_atomic_load(
                reinterpret_cast<unsigned long long*>(&groupData[warpGroupIdInBlock]),
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
          } while ((val >> 32) != (uint32_t)(i + 1));
          atomicAdd(&groupCounters[warpGroupIdInBlock], 1);
          destTokId = (index_t)(val & 0xFFFFFFFF);
        }

        destTokId = __shfl(destTokId, 0);
      }

      // All warps in warpgroup: copy their portion of token data
      size_t srcTokOffset = srcTokId * hiddenDim;
      size_t destTokOffset = destTokId * hiddenDim;

      constexpr int unroll_num = std::is_same_v<T, hip_bfloat16> ? 4 : 2;
      core::WarpCopy<T, unroll_num>(dispatchOut + destTokOffset + warpDimOffset,
                                    args.inpTokenBuf + srcTokOffset + warpDimOffset, warpDimChunk);

      // Write scales: split across 4 warps
      if (hasScales) {
        size_t scaleSize = config.scaleDim * config.scaleTypeSize;
        size_t scalePerWarp = (scaleSize + kWarpsPerGroup - 1) / kWarpsPerGroup;
        size_t myScaleOffset = (size_t)inGroupWarpId * scalePerWarp;
        size_t myScaleChunk =
            (myScaleOffset < scaleSize) ? min(scaleSize - myScaleOffset, scalePerWarp) : 0;

        size_t destScaleOffset = (size_t)destTokId * scaleSize;
        size_t srcScaleOffset = (size_t)srcTokId * scaleSize;
        core::WarpCopy(args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
                           destScaleOffset + myScaleOffset,
                       args.scalesBuf + srcScaleOffset + myScaleOffset, myScaleChunk);
      }
    }
  }

  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, int Vec8AccumNum = 8>
__device__ __forceinline__ void EpCombineIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  using TokT =
      std::conditional_t<UseFp8DirectCast || UseFp8BlockwiseQuant, core::CombineInternalFp8, T>;
  static_assert(!(UseFp8DirectCast && UseFp8BlockwiseQuant),
                "Fp8 direct cast and blockwise quant are mutually exclusive");
  static_assert((!UseFp8DirectCast && !UseFp8BlockwiseQuant) || std::is_same_v<T, hip_bfloat16>,
                "Fp8 combine quant currently only supports bf16 input");
  static_assert((Vec8Top8BlockElems & (Vec8Top8BlockElems - 1)) == 0,
                "Vec8Top8BlockElems must be 0 or a power of two");
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::CombineStageInput);

  const uint64_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // When TokT != T (e.g. fp8 combine), staging layout uses TokT-sized tokens
  const size_t hiddenDim = config.HiddenDimSz();
  const size_t hiddenBytes = hiddenDim * sizeof(TokT);
  const size_t weightBytes =
      (UseWeights && args.weightsBuf != nullptr) ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t scaleBytes =
      UseFp8BlockwiseQuant ? static_cast<size_t>(args.fp8BlockwiseCombineScaleDim) * sizeof(float)
                           : 0;
  const size_t combXferBytes = hiddenBytes + scaleBytes + weightBytes;

  if constexpr (EnableStdMoE) {
#ifdef ENABLE_STANDARD_MOE_ADAPT
    InvokeConvertCombineInput<T, UseP2PRead>(args, myPe);
#endif
  } else if constexpr (UseP2PRead) {
    if (args.config.useExternalInpBuffer) {
      for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
        if constexpr (UseFp8BlockwiseQuant) {
          core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.shmemInpScalesMemObj->template GetAs<float*>() +
                  i * args.fp8BlockwiseCombineScaleDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        } else if constexpr (!std::is_same_v<T, TokT> &&
                             std::is_same_v<TokT, core::CombineInternalFp8>) {
          core::WarpCastBf16ToCombineInternalFp8<T>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, laneId);
        } else {
          core::WarpCopy(args.intraNodeTokBufs.combineInp->template GetAs<T*>() + i * hiddenDim,
                         args.inpTokenBuf + i * hiddenDim, hiddenDim);
        }
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
          core::WarpCopy(
              args.shmemInpWeightsMemObj->template GetAs<float*>() + i * config.numExpertPerToken,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken);
        }
      }
    }
  } else {
    // When the caller passes a routing handle, args.dispTokIdToSrcTokIdLocal
    // holds a per-call snapshot of the symmetric local view. Otherwise fall
    // back to the shared symmetric buffer.
    const index_t* localSrcMap =
        args.dispTokIdToSrcTokIdLocal != nullptr
            ? args.dispTokIdToSrcTokIdLocal
            : args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe);
#ifdef ENABLE_PROFILER
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = localSrcMap[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
          index_t destTokId = localSrcMap[tokenIdx];
          index_t destPe = PeFromFlatTokenIndex(config, destTokId);
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          uint8_t* destStagingPtr =
              args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
              SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#else
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = localSrcMap[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
      if constexpr (UseWeights) {
        if (args.weightsBuf) {
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#endif
  }

  // Make sure copy on all GPUs are finished
  MORI_TRACE_NEXT(seq, Slot::CombineBarrier);
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  // With a routing handle, the caller owns this tensor (it may still be alive in autograd ctx),
  // so we skip the reset. The next dispatch will allocate or replay its own.
  if (args.dispTokIdToSrcTokIdLocal == nullptr) {
    *args.totalRecvTokenNum = 0;
  }
  if (args.curRankNumToken == 0) return;

  MORI_TRACE_NEXT(seq, Slot::CombineAccumSetup);
  extern __shared__ char sharedMem[];
  // Layout: [srcPtrs] [srcWeightsPtr if UseWeights] [srcScalePtrs if UseFp8BlockwiseQuant];
  // host-side combine_shared_mem() must use the same flags.
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = nullptr;
  if constexpr (UseWeights) {
    srcWeightsPtr = reinterpret_cast<float**>(sharedMem) + warpNum * config.numExpertPerToken +
                    warpId * config.numExpertPerToken;
  }
  float** srcScalePtrs = nullptr;
  if constexpr (UseFp8BlockwiseQuant) {
    constexpr int scalePtrArrayOffset = UseWeights ? 2 : 1;
    srcScalePtrs = reinterpret_cast<float**>(sharedMem) +
                   scalePtrArrayOffset * warpNum * config.numExpertPerToken +
                   warpId * config.numExpertPerToken;
  }

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    // Prepare data pointers on different GPUs
    MORI_TRACE_NEXT(seq, Slot::CombinePreparePtrs);
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);

      if (destPe < config.worldSize) {
        if constexpr (UseP2PRead) {
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          srcPtrs[j] = args.intraNodeTokBufs.combineInp->template GetAs<TokT*>(destPe) +
                       destLocalTokId * hiddenDim + hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                               destLocalTokId * config.numExpertPerToken;
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = args.shmemInpScalesMemObj->template GetAs<float*>(destPe) +
                              destLocalTokId * args.fp8BlockwiseCombineScaleDim;
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        } else {
          srcPtrs[j] = reinterpret_cast<TokT*>(
                           args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                           SendBufSlotOffset(config, destPe, tokenId) * combXferBytes) +
                       hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes +
                scaleBytes);
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes);
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (UseWeights) {
          srcWeightsPtr[j] = nullptr;
        }
        if constexpr (UseFp8BlockwiseQuant) {
          srcScalePtrs[j] = nullptr;
        }
      }
    }

    T* outPtr = args.intraNodeTokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
                hiddenDimOffset;

    int validAccumCount = config.numExpertPerToken;
    if (config.worldSize <= 4) {
      {
        int isValid = 0;
        TokT* myTokPtr = nullptr;
        float* myScalePtr = nullptr;
        if (laneId < config.numExpertPerToken) {
          myTokPtr = srcPtrs[laneId];
          if constexpr (UseFp8BlockwiseQuant) {
            myScalePtr = srcScalePtrs[laneId];
          }
          isValid = (myTokPtr != nullptr) ? 1 : 0;
        }
        unsigned long long validMask = __ballot(isValid);
        validAccumCount = __popcll(validMask);
        if (validAccumCount < config.numExpertPerToken && isValid) {
          int myPos = __popcll(validMask & ((1ULL << laneId) - 1));
          srcPtrs[myPos] = myTokPtr;
          if constexpr (UseFp8BlockwiseQuant) {
            srcScalePtrs[myPos] = myScalePtr;
          }
        }
      }
    }

    if constexpr (UseFp8BlockwiseQuant) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      if constexpr (Vec8Top8BlockElems != 0) {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFullBlockVec8Top8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDim);
        } else if ((hiddenDimOffset & 0x7) == 0 && (hiddenDimSize & 0x7) == 0) {
          core::WarpAccumFp8DequantSegmentBlockVec8Top8<T, core::CombineInternalFp8,
                                                        Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        } else {
          // Misaligned segment: vec8 helper would fault on the load. Tiny scalar fallback.
          core::WarpAccumFp8DequantSegmentScalarTop8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        }
      } else {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFull<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDim,
              args.fp8BlockwiseCombineScaleDim);
        } else {
          core::WarpAccumFp8DequantSegment<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDimOffset,
              hiddenDimSize, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        }
      }
    } else if constexpr (!std::is_same_v<T, TokT> &&
                         std::is_same_v<TokT, core::CombineInternalFp8>) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccumCombineInternalFp8ToBf16(outPtr, reinterpret_cast<const TokT* const*>(srcPtrs),
                                              validAccumCount, laneId, hiddenDimSize);
    } else {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccum<T, 4>(outPtr, srcPtrs, nullptr, validAccumCount, hiddenDimSize);
    }

    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineAccumWeights);
      if (args.weightsBuf && inTokenPartId == mwIter.warpsPerItem - 1) {
        core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                      tokenId * config.numExpertPerToken,
                                  srcWeightsPtr, nullptr, config.numExpertPerToken,
                                  config.numExpertPerToken);
      }
    }
  }
}

template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, int Vec8AccumNum = 8>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems, Vec8AccumNum>(args);
}

}  // namespace moe
}  // namespace mori
