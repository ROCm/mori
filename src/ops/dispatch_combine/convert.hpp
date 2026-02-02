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

#include <cstdint>
#include <iterator>

#include <hip/hip_fp16.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

// Profiling macros for timestamp recording
// #define ENABLE_PROFILE 1  // Enable profiling
// #define PROFILE_DISPATCH 1  // Profile ConvertDispatchOutputDevice
// #define PROFILE_COMBINE 1   // Profile ConvertCombineInputKernel

#ifdef ENABLE_PROFILE
#define PROFILE_TS_DECL(kTsMax) \
  uint64_t ts[kTsMax] = {0};    \
  int tsCount = 0

#define PROFILE_TS_RECORD(ts, tsCount, kTsMax, laneId) \
  do {                                                 \
    if ((laneId) == 0 && (tsCount) < (kTsMax)) {       \
      (ts)[(tsCount)++] = wall_clock64();              \
    }                                                  \
  } while (0)

#define PROFILE_TS_PRINT(condition, label, ts, tsCount)            \
  do {                                                             \
    if (condition) {                                               \
      printf("[%s] block=%d warp=%d", label,                       \
             static_cast<int>(blockIdx.x), warpId);                \
      for (int _i = 1; _i < (tsCount); ++_i) {                     \
        printf(" t%d=%.3f", _i, ((ts)[_i] - (ts)[0]) / 100.0f);    \
      }                                                            \
      printf("\n");                                                \
    }                                                              \
  } while (0)

#ifdef PROFILE_DISPATCH
#define PROFILE_DISPATCH_DECL PROFILE_TS_DECL
#define PROFILE_DISPATCH_RECORD PROFILE_TS_RECORD
#define PROFILE_DISPATCH_PRINT PROFILE_TS_PRINT
#else
#define PROFILE_DISPATCH_DECL(kTsMax) do {} while(0)
#define PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId) do {} while(0)
#define PROFILE_DISPATCH_PRINT(condition, label, ts, tsCount) do {} while(0)
#endif

#ifdef PROFILE_COMBINE
#define PROFILE_COMBINE_DECL PROFILE_TS_DECL
#define PROFILE_COMBINE_RECORD PROFILE_TS_RECORD
#define PROFILE_COMBINE_PRINT PROFILE_TS_PRINT
#else
#define PROFILE_COMBINE_DECL(kTsMax) do {} while(0)
#define PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId) do {} while(0)
#define PROFILE_COMBINE_PRINT(condition, label, ts, tsCount) do {} while(0)
#endif

#else  // !ENABLE_PROFILE
#define PROFILE_TS_DECL(kTsMax) do {} while(0)
#define PROFILE_TS_RECORD(ts, tsCount, kTsMax, laneId) do {} while(0)
#define PROFILE_TS_PRINT(condition, label, ts, tsCount) do {} while(0)
#define PROFILE_DISPATCH_DECL(kTsMax) do {} while(0)
#define PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId) do {} while(0)
#define PROFILE_DISPATCH_PRINT(condition, label, ts, tsCount) do {} while(0)
#define PROFILE_COMBINE_DECL(kTsMax) do {} while(0)
#define PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId) do {} while(0)
#define PROFILE_COMBINE_PRINT(condition, label, ts, tsCount) do {} while(0)
#endif

namespace mori {
namespace moe {

struct ConvertDispatchOutputArgs {
  EpDispatchCombineConfig config;
  const void* dispatchOutX{nullptr};
  const void* dispatchOutTopkIdx{nullptr};
  const index_t* dispatchSrcTokenPos{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  uint32_t* dispatchGridBarrier{nullptr};
  void* packedRecvX{nullptr};
  int* packedRecvCount{nullptr};
  int* packedRecvSrcInfo{nullptr};
  int64_t* packedRecvLayoutRange{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
};

struct ConvertCombineInputArgs {
  EpDispatchCombineConfig config;
  const void* packedRecvX{nullptr};
  const void* topkIdx{nullptr};
  const void* topkWeights{nullptr};
  const void* packedRecvSrcInfo{nullptr};
  const void* packedRecvLayoutRange{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  void* combineInput{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
  int* packedRecvCount{nullptr};
  mori::application::SymmMemObjPtr shmemCombineInpTokMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
};

// Grid barrier: synchronize all blocks within a grid
// All threads in all blocks must call this function
template <typename T>
__device__ inline void GridBarrier(T* barrierPtr) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    __hip_atomic_fetch_add(barrierPtr, static_cast<T>(1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (atomicCAS(barrierPtr, static_cast<T>(gridDim.x), static_cast<T>(0)) != 0) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
}

// Forward declarations
template <bool IsStandalone>
__device__ inline void ConvertDispatchOutputDevice(ConvertDispatchOutputArgs args);

template <typename T, bool UseP2PRead>
__device__ inline void ConvertCombineInputDevice(ConvertCombineInputArgs& args);

// Helper to invoke ConvertDispatchOutputDevice from EpDispatchCombineArgs
template <typename T>
__device__ inline void InvokeConvertDispatchOutput(const EpDispatchCombineArgs<T>& args, int myPe) {
  ConvertDispatchOutputArgs convArgs{};
  convArgs.config = args.config;
  convArgs.dispatchOutX = args.shmemDispatchOutTokMemObj->template GetAs<T*>(myPe);
  convArgs.dispatchOutTopkIdx = args.shmemOutIndicesMemObj->template GetAs<index_t*>(myPe);
  convArgs.dispatchSrcTokenPos = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe);
  convArgs.totalRecvTokenNum = args.totalRecvTokenNum;
  convArgs.dispatchGridBarrier = args.dispatchGridBarrier;
  convArgs.packedRecvX = args.standardPackedRecvX;
  convArgs.packedRecvCount = args.standardPackedRecvCount;
  convArgs.packedRecvSrcInfo = args.standardPackedRecvSrcInfo;
  convArgs.packedRecvLayoutRange = args.standardPackedRecvLayoutRange;
  convArgs.dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  ConvertDispatchOutputDevice</*IsStandalone=*/false>(convArgs);
}

// Helper to invoke ConvertCombineInputDevice from EpDispatchCombineArgs
template <typename T, bool UseP2PRead>
__device__ inline void InvokeConvertCombineInput(const EpDispatchCombineArgs<T>& args, int myPe) {
  ConvertCombineInputArgs convArgs{};
  convArgs.config = args.config;
  convArgs.packedRecvX = args.standardPackedRecvX;
  convArgs.topkIdx = args.tokenIndices;
  convArgs.topkWeights = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(myPe);
  convArgs.packedRecvSrcInfo = args.standardPackedRecvSrcInfo;
  convArgs.packedRecvLayoutRange = args.standardPackedRecvLayoutRange;
  convArgs.totalRecvTokenNum = args.totalRecvTokenNum;
  convArgs.combineInput = nullptr;
  convArgs.dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  convArgs.packedRecvCount = args.standardPackedRecvCount;
  convArgs.shmemCombineInpTokMemObj = args.shmemCombineInpTokMemObj;
  convArgs.dispTokIdToSrcTokIdMemObj = args.dispTokIdToSrcTokIdMemObj;
  ConvertCombineInputDevice<T, UseP2PRead>(convArgs);
}

#if 1
// IsStandalone: true if launched as a standalone kernel, false if called from within another kernel
template <bool IsStandalone = true>
__device__ inline void ConvertDispatchOutputDevice(ConvertDispatchOutputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);
  const int warpNum = blockDim.x / warpSize;

  constexpr int kTsMax = 12;
  PROFILE_DISPATCH_DECL(kTsMax);
  PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const int topk = config.numExpertPerToken;

  const int64_t maxNumTokenPerRank = config.maxNumInpTokenPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(config.worldSize) * config.maxNumInpTokenPerRank;
  const size_t hiddenBytes = static_cast<size_t>(config.hiddenDim) * config.maxTokenTypeSize;

  const auto* topkIdx = reinterpret_cast<const index_t*>(args.dispatchOutTopkIdx);
  const auto* dispatchSrcTokenPos = args.dispatchSrcTokenPos;
  auto* packedRecvX = reinterpret_cast<uint8_t*>(args.packedRecvX);
  auto* packedRecvSrcInfo = args.packedRecvSrcInfo;
  auto* packedRecvCount = args.packedRecvCount;
  (void)args.packedRecvLayoutRange;
  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;

  // Only need barrier synchronization when called from within another kernel
  if constexpr (!IsStandalone) {
    GridBarrier(args.dispatchGridBarrier + 1);
  }

  PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);
  for (int i = globalWarpId; i < totalTokens * topk; i += globalWarpNum) {
    auto tokenIdx = i / topk;
    // const index_t expertId = topkIdx[i];
    const index_t expertId = __ldg(topkIdx + i);
    const auto localExpert = expertId - config.rank * config.numExpertPerRank;
    if (localExpert < 0 || localExpert >= config.numExpertPerRank) {
      if (laneId == 0) {
        dispTokToEpSlotMap[i] = static_cast<uint64_t>(-1);
      }
      continue;
    }
    PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

    // const index_t srcTokenPos = dispatchSrcTokenPos[tokenIdx];
    // const int srcRank = static_cast<int>(srcTokenPos / maxNumTokenPerRank);
    // const int srcInfo = static_cast<int>(srcTokenPos - srcRank * maxNumTokenPerRank);

    uint32_t idx = 0;
    if (laneId == 0) {
      idx = atomicAdd(packedRecvCount + localExpert, 1u);
      // idx = __hip_atomic_fetch_add(packedRecvCount + localExpert, 1u, __ATOMIC_RELAXED,
      //                              __HIP_MEMORY_SCOPE_AGENT);
    }
    idx = __shfl(idx, 0);

    const uint64_t linearIndex =
        static_cast<uint64_t>(localExpert) * maxTokensPerExpert + idx;
    if (laneId == 0) {
      // packedRecvSrcInfo[linearIndex] = srcInfo;
      packedRecvSrcInfo[linearIndex] = dispatchSrcTokenPos[tokenIdx];
      dispTokToEpSlotMap[i] = linearIndex;
    }

    PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

    const size_t dstOffset = static_cast<size_t>(linearIndex) * hiddenBytes;
    const size_t srcOffset = static_cast<size_t>(tokenIdx) * hiddenBytes;
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(args.dispatchOutX) + srcOffset;
    auto* dstBytes = packedRecvX + dstOffset;
    core::WarpCopy<uint8_t, 7>(dstBytes, srcBytes, hiddenBytes);
    PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);
  }

  PROFILE_DISPATCH_PRINT(config.rank == 0 && blockIdx.x == 0 && warpId == 0 && laneId == 0,
                         "ConvertDispatchOutputDevice", ts, tsCount);
}
#else
// IsStandalone: true if launched as a standalone kernel, false if called from within another kernel
// Optimized version: blocks are partitioned by expert, each block group handles one local expert
template <bool IsStandalone = true>
__device__ inline void ConvertDispatchOutputDevice(ConvertDispatchOutputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);
  const int warpNum = blockDim.x / warpSize;

  constexpr int kTsMax = 20;
  PROFILE_DISPATCH_DECL(kTsMax);
  PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

  const int topk = config.numExpertPerToken;
  const int numExperts = config.numExpertPerRank;

  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(config.worldSize) * config.maxNumInpTokenPerRank;
  const size_t hiddenBytes = static_cast<size_t>(config.hiddenDim) * config.maxTokenTypeSize;

  const auto* topkIdx = reinterpret_cast<const index_t*>(args.dispatchOutTopkIdx);
  const auto* dispatchSrcTokenPos = args.dispatchSrcTokenPos;
  auto* packedRecvX = reinterpret_cast<uint8_t*>(args.packedRecvX);
  auto* packedRecvSrcInfo = args.packedRecvSrcInfo;
  auto* packedRecvCount = args.packedRecvCount;
  (void)args.packedRecvLayoutRange;
  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;

  // Only need barrier synchronization when called from within another kernel
  if constexpr (!IsStandalone) {
    GridBarrier(args.dispatchGridBarrier + 1);
  }

  PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);

  // Block assignment: divide blocks evenly among local experts
  // Each group of blocks handles one local expert
  const int blocksPerExpert = gridDim.x / numExperts;
  const int localExpertId = blockIdx.x / blocksPerExpert;

  // Skip if this block maps to a non-existent expert (when gridDim.x not divisible by numExperts)
  if (localExpertId >= numExperts || blocksPerExpert == 0) {
    return;
  }

  const int blockInGroup = blockIdx.x % blocksPerExpert;

  // Global expert ID that this block group is responsible for
  const index_t targetExpertId = static_cast<index_t>(config.rank * numExperts + localExpertId);

  // Warp ID within this expert's block group
  const int groupWarpId = blockInGroup * warpNum + warpId;
  const int groupWarpNum = blocksPerExpert * warpNum;

  // Each warp processes tokens (rows of topkIdx) in strided fashion
  for (int64_t tokenIdx = groupWarpId; tokenIdx < totalTokens; tokenIdx += groupWarpNum) {
    // First topk threads read topkIdx for this token in parallel
    index_t myExpertId = laneId < topk ? topkIdx[tokenIdx * topk + laneId] : 0;
    if (laneId < topk) {
      // Check if this is a non-local expert and set -1 (only first group handles)
      if (localExpertId == 0) {
        const auto localExpert = myExpertId - config.rank * numExperts;
        if (localExpert < 0 || localExpert >= numExperts) {
          dispTokToEpSlotMap[tokenIdx * topk + laneId] = static_cast<uint64_t>(-1);
        }
      }
    }

    // Check if this lane's expert matches our target, use ballot to determine if copy needed
    const bool isMatch = laneId < topk && (myExpertId == targetExpertId);
    const uint64_t matchMask = __ballot(isMatch);
    if (matchMask == 0) {
      continue;
    }

    PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);

    // This token goes to our target expert, do the copy
    uint32_t idx = 0;
    if (laneId == 0) {
      idx = atomicAdd(packedRecvCount + localExpertId, 1u);
    }
    idx = __shfl(idx, 0);

    const uint64_t linearIndex = static_cast<uint64_t>(localExpertId) * maxTokensPerExpert + idx;

    // Set dispTokToEpSlotMap for the matching lane(s)
    if (isMatch) {
      dispTokToEpSlotMap[tokenIdx * topk + laneId] = linearIndex;
    }

    // Set packedRecvSrcInfo (only lane 0)
    if (laneId == 0) {
      packedRecvSrcInfo[linearIndex] = dispatchSrcTokenPos[tokenIdx];
    }

    const size_t dstOffset = static_cast<size_t>(linearIndex) * hiddenBytes;
    const size_t srcOffset = static_cast<size_t>(tokenIdx) * hiddenBytes;
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(args.dispatchOutX) + srcOffset;
    auto* dstBytes = packedRecvX + dstOffset;
    core::WarpCopy<uint8_t, 7>(dstBytes, srcBytes, hiddenBytes);

    PROFILE_DISPATCH_RECORD(ts, tsCount, kTsMax, laneId);
  }

  PROFILE_DISPATCH_PRINT(config.rank == 0 && blockIdx.x == 0 && warpId == 0 && laneId == 0,
                         "ConvertDispatchOutputDevice", ts, tsCount);
}
#endif

__global__ void ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}

#if 0
template <typename T>
__global__ void ConvertCombineInputKernel(ConvertCombineInputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);
  const int warpNum = blockDim.x / warpSize;

  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  constexpr int kTsMax = 8;
  PROFILE_COMBINE_DECL(kTsMax);
  PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);

  const int topk = config.numExpertPerToken;
  const int64_t hiddenDim = config.hiddenDim;

  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  const auto* packedRecvX = reinterpret_cast<const T*>(args.packedRecvX);
  auto* combineInput = reinterpret_cast<T*>(args.combineInput);
  const auto* topkWeights = reinterpret_cast<const float*>(args.topkWeights);

  T* srcPtrs[MAX_EXPERTS_PER_TOKEN];

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);

  for (int64_t tokenIdx = globalWarpId; tokenIdx < totalTokens; tokenIdx += globalWarpNum) {
    for (int k = 0; k < topk; ++k) {
      const uint64_t slot = dispTokToEpSlotMap[tokenIdx * topk + k];
      srcPtrs[k] = (slot == static_cast<uint64_t>(-1))
                       ? nullptr
                       : const_cast<T*>(packedRecvX + slot * hiddenDim);
    }
    PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);
    const float* weightRow = topkWeights ? (topkWeights + tokenIdx * topk) : nullptr;
    core::WarpAccum<T, 4>(combineInput + tokenIdx * hiddenDim, srcPtrs, weightRow, topk, hiddenDim);
    PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);
  }

  PROFILE_COMBINE_PRINT(config.rank == 0 && blockIdx.x == 0 && warpId == 0 && laneId == 0,
                        "ConvertCombineInputKernel", ts, tsCount);
}
#else
// Block-per-token implementation: each block processes one token,
// each thread reduces one vector element across all top-k experts
template <typename T, bool UseP2PRead>
__device__ inline void ConvertCombineInputDevice(ConvertCombineInputArgs& args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);

  constexpr int kTsMax = 8;
  PROFILE_COMBINE_DECL(kTsMax);
  PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);

  const int topk = config.numExpertPerToken;
  const int64_t hiddenDim = config.hiddenDim;
  const int64_t hiddenBytes = config.hiddenDim * sizeof(T);

  // Number of T elements per vector load (int4 = 16 bytes)
  constexpr int kElemsPerVec = sizeof(int4) / sizeof(T);
  const int64_t hiddenVecs = hiddenDim / kElemsPerVec;

  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  const auto* packedRecvX = reinterpret_cast<const T*>(args.packedRecvX);
  // auto* convertOutput = reinterpret_cast<T*>(args.combineInput);
  const auto* topkWeights = reinterpret_cast<const float*>(args.topkWeights);

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);
  const int numBlocks = gridDim.x;

  // clear packedRecvCount
  if (thdId < config.numExpertPerRank) {
    args.packedRecvCount[thdId] = 0;
  }

  // Each block handles tokens in strided fashion
  for (int64_t tokenIdx = blockIdx.x; tokenIdx < totalTokens; tokenIdx += numBlocks) {
    // Load top-k slot indices for this token
    uint64_t slots[MAX_EXPERTS_PER_TOKEN];
    float weights[MAX_EXPERTS_PER_TOKEN];
    uint8_t* out;
    if constexpr (UseP2PRead) {
      out = args.shmemCombineInpTokMemObj->template GetAs<uint8_t*>() + tokenIdx * hiddenBytes;
    }
    else {
      index_t destTokId =
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(config.rank)[tokenIdx];
      index_t destPe = destTokId / config.MaxNumTokensToRecvPerRank();
      index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecvPerRank();
      out = args.shmemCombineInpTokMemObj->template GetAs<uint8_t*>(destPe) +
            (config.rank * config.MaxNumTokensToRecvPerRank() + destLocalTokId) * hiddenBytes;
    }
#pragma unroll
    for (int k = 0; k < topk; ++k) {
      slots[k] = dispTokToEpSlotMap[tokenIdx * topk + k];
      weights[k] = topkWeights ? topkWeights[tokenIdx * topk + k] : 1.0f;
    }

    PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);

    // Each thread processes one vector position
    for (int vecIdx = thdId; vecIdx < hiddenVecs; vecIdx += blockDim.x) {
      float accum[kElemsPerVec] = {0.0f};

      // Accumulate contributions from all valid experts
#pragma unroll
      for (int k = 0; k < topk; ++k) {
        if (slots[k] != static_cast<uint64_t>(-1)) {
          const T* srcRow = packedRecvX + slots[k] * hiddenDim;
          int4 srcVec = reinterpret_cast<const int4*>(srcRow)[vecIdx];
          const T* srcElems = reinterpret_cast<const T*>(&srcVec);
#pragma unroll
          for (int e = 0; e < kElemsPerVec; ++e) {
            accum[e] += static_cast<float>(srcElems[e]) * weights[k];
          }
        }
      }

      // Convert back to T and store
      int4 outVec;
      T* outElems = reinterpret_cast<T*>(&outVec);
#pragma unroll
      for (int e = 0; e < kElemsPerVec; ++e) {
        outElems[e] = static_cast<T>(accum[e]);
      }
      // reinterpret_cast<int4*>(convertOutput + tokenIdx * hiddenDim)[vecIdx] = outVec;
      reinterpret_cast<int4*>(out)[vecIdx] = outVec;
    }

    PROFILE_COMBINE_RECORD(ts, tsCount, kTsMax, laneId);
  }

  PROFILE_COMBINE_PRINT(config.rank == 0 && blockIdx.x == 0 && warpId == 0 && laneId == 0,
                        "ConvertCombineInputDevice", ts, tsCount);
}

template <typename T, bool UseP2PRead = true>
__global__ void ConvertCombineInputKernel(ConvertCombineInputArgs args) {
  ConvertCombineInputDevice<T, UseP2PRead>(args);
}
#endif

#if 0
template <typename T>
__global__ void ConvertCombineInputKernel(ConvertCombineInputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;

  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const int topk = config.numExpertPerToken;
  const int64_t hiddenDim = config.hiddenDim;

  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  const auto* packedRecvX = reinterpret_cast<const T*>(args.packedRecvX);
  auto* combineInput = reinterpret_cast<T*>(args.combineInput);
  const auto* topkWeights = reinterpret_cast<const float*>(args.topkWeights);

  T* srcPtrs[MAX_EXPERTS_PER_TOKEN];

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);
  if (totalTokens == 0) return;

  // Even distribution of warps among tokens:
  // - First numTokensWithExtraWarp tokens get (baseWarpsPerToken + 1) warps each
  // - Remaining tokens get baseWarpsPerToken warps each
  const int64_t baseWarpsPerToken = std::max(int64_t{1}, globalWarpNum / totalTokens);
  const int64_t numTokensWithExtraWarp =
      (globalWarpNum >= totalTokens) ? (globalWarpNum % totalTokens) : 0;
  const int64_t largeGroupSize = baseWarpsPerToken + 1;
  const int64_t cutoff = numTokensWithExtraWarp * largeGroupSize;

  // When warps >= tokens: total work = globalWarpNum (each warp does one piece)
  // When warps < tokens: total work = totalTokens (each warp handles multiple tokens)
  const int64_t totalWorkItems = std::max(static_cast<int64_t>(globalWarpNum), totalTokens);

  for (int64_t i = globalWarpId; i < totalWorkItems; i += globalWarpNum) {
    int64_t tokenIdx, inTokenPartId, warpsForThisToken;

    if (globalWarpNum >= totalTokens) {
      // Map work item to (token, part) with even distribution
      if (i < cutoff) {
        tokenIdx = i / largeGroupSize;
        inTokenPartId = i % largeGroupSize;
        warpsForThisToken = largeGroupSize;
      } else {
        tokenIdx = numTokensWithExtraWarp + (i - cutoff) / baseWarpsPerToken;
        inTokenPartId = (i - cutoff) % baseWarpsPerToken;
        warpsForThisToken = baseWarpsPerToken;
      }
    } else {
      // Each warp handles entire token(s), no hiddenDim splitting
      tokenIdx = i;
      inTokenPartId = 0;
      warpsForThisToken = 1;
    }

    if (tokenIdx >= totalTokens) continue;

    const int64_t hiddenDimPerWarp = (hiddenDim + warpsForThisToken - 1) / warpsForThisToken;
    const int64_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    const int64_t hiddenDimSize =
        std::max(int64_t{0}, std::min(hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    if (hiddenDimSize == 0) continue;

    for (int k = 0; k < topk; ++k) {
      const uint64_t slot = dispTokToEpSlotMap[tokenIdx * topk + k];
      srcPtrs[k] = (slot == static_cast<uint64_t>(-1))
                       ? nullptr
                       : const_cast<T*>(packedRecvX + slot * hiddenDim + hiddenDimOffset);
    }

    const float* weightRow = topkWeights ? (topkWeights + tokenIdx * topk) : nullptr;
    core::WarpAccum<T, 4>(combineInput + tokenIdx * hiddenDim + hiddenDimOffset, srcPtrs,
                          weightRow, topk, hiddenDimSize);
  }
}
#endif

}  // namespace moe
}  // namespace mori
