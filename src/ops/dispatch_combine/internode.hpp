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

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

#define DEBUG 0

__device__ void SyncIfDebugEnabled(const char* msg) {
#if DEBUG == 1
  __syncthreads();
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    shmem::ShmemQuietThread();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchInterNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchInterNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = (blockDim.x + warpSize - 1) / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  int myNode = myPe / MAX_GPUS_PER_NODE;

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();
  size_t MaxNumTokensToRecv = config.MaxNumTokensToRecv();

  int numExpertPerToken = config.numExpertPerToken;
  assert(numExpertPerToken < warpSize);

  size_t weightOffset = config.hiddenDim * sizeof(T);
  size_t indicesOffset = weightOffset + sizeof(float) * numExpertPerToken;
  size_t scalesOffset = indicesOffset + sizeof(index_t) * numExpertPerToken;
  size_t stagingOffset = scalesOffset + config.scaleTypeSize * config.scaleDim;

  extern __shared__ char sharedMem[];

  int subWarpNumPerWarp = warpSize / numExpertPerToken;
  int laneInSubWarp = laneId % numExpertPerToken;
  int subWarpId = laneId / numExpertPerToken;
  int globalSubWarpId = globalWarpId * subWarpNumPerWarp + subWarpId;
  int globalSubWarpNum = globalWarpNum * subWarpNumPerWarp;
  if (laneId < subWarpNumPerWarp * numExpertPerToken) {
    for (int tokenId = globalSubWarpId; tokenId < args.curRankNumToken;
         tokenId += globalSubWarpNum) {
      const int expertOffset = tokenId * numExpertPerToken + laneInSubWarp;
      index_t destExpert = args.tokenIndices[expertOffset];
      index_t destPe = destExpert / config.numExpertPerRank;

      unsigned long long subWarpMask = ((1ULL << numExpertPerToken) - 1ULL)
                                       << (subWarpId * numExpertPerToken);
      unsigned long long dupMask = __match_any_sync(subWarpMask, destPe);
      bool dup = false;
      if (laneInSubWarp) {
        unsigned long long lowerMask =
            dupMask & (((1ULL << laneInSubWarp) - 1ULL) << (subWarpId * numExpertPerToken));
        dup = (lowerMask != 0ULL);
      }
      if (dup) {
        args.dispSenderIdxMap[expertOffset] = MaxNumTokensToRecv;
        continue;
      } else {
        index_t destPeTokenIdx = 0, peSortedIdx = 0;
        destPeTokenIdx = atomicAdd(args.destPeTokenCounter + destPe, 1);
        peSortedIdx = destPe * MaxNumTokensToRecvPerRank + destPeTokenIdx;
        args.dispSenderIdxMap[expertOffset] = peSortedIdx;
        args.destPeTokenIdxMap[peSortedIdx] = tokenId;
        __threadfence();
      }
    }
  }

  if (laneId == 0) {
    int old_val = atomicAdd(args.dispatchGridBarrier, 1);
    if (old_val == globalWarpNum - 1) {
      __hip_atomic_store(args.dispatchGridBarrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  if (laneId == 0) {
    shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, 0);
  }

  // TODO: block num should be multiple of npes
  const int numsBlockPerDestPe = gridDim.x / npes;
  const int destPe = blockIdx.x / numsBlockPerDestPe;
  const int destNode = destPe / MAX_GPUS_PER_NODE;
  const int localBlockId = blockIdx.x - destPe * numsBlockPerDestPe;
  const int totalTokens = args.destPeTokenCounter[destPe];
  const int baseChunk = totalTokens / numsBlockPerDestPe;
  const int remainder = totalTokens % numsBlockPerDestPe;

  const int myChunkSize = baseChunk + (localBlockId < remainder);

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;
  if (localBlockId == 0 && warpId == warpNum - 1) {
    shmem::ShmemPutInt32ImmNbiWarp(
        args.recvTokenNumMemObj,
        (myPe + (args.crossDeviceBarrierFlag & 1) * npes) * sizeof(index_t), totalTokens, destPe,
        localBlockId);
  }

  if (destNode == myNode) {
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t mapIdxOffset = mapIdx * stagingOffset;
      const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
      size_t tokenOffset = tokenId * size_t(config.hiddenDim) * sizeof(T);
      const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t peSortedOffset = peSortedId * stagingOffset;
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset,
                     config.hiddenDim * sizeof(T));
      if (args.weightsBuf) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + weightOffset,
            reinterpret_cast<char*>(args.weightsBuf) +
                tokenId * config.numExpertPerToken * sizeof(float),
            config.numExpertPerToken * sizeof(float));
      }
      core::WarpCopy(
          args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + indicesOffset,
          reinterpret_cast<char*>(args.tokenIndices) +
              tokenId * config.numExpertPerToken * sizeof(index_t),
          config.numExpertPerToken * sizeof(index_t));
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + scalesOffset,
            reinterpret_cast<char*>(args.scalesBuf) +
                tokenId * config.scaleDim * config.scaleTypeSize,
            config.scaleDim * config.scaleTypeSize);
      }
      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemDispatchInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, stagingOffset,
                                          destPe, localBlockId);
    }
  } else {
    // inter node use ibgda for transfer
    // last warp for coordinate, other warp for gather token
    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);
    if (warpId == warpNum - 1) {
      const int totalTokenInBlock = endIdx - startIdx;
      int chunkOffset = 0;
      int chunkIdx = 0;
      while (chunkOffset < totalTokenInBlock) {
        int actualTokenNum = totalTokenInBlock - chunkOffset < chunkTokenSize
                                 ? totalTokenInBlock - chunkOffset
                                 : chunkTokenSize;
        if (laneId == 0) {
          while (atomicAdd(&gatherTokenNum[chunkIdx], 0) < actualTokenNum) {
            ;
          }
        }
        // rdma_send
        const index_t srcIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t srcOffset = srcIdx * stagingOffset;
        const index_t dstIdx = myPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t dstOffset = dstIdx * stagingOffset;
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemDispatchInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * stagingOffset, destPe, localBlockId);

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * stagingOffset;
        const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
        size_t tokenOffset = tokenId * size_t(config.hiddenDim) * sizeof(T);
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset,
                       config.hiddenDim * sizeof(T));
        if (args.weightsBuf) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + weightOffset,
              reinterpret_cast<char*>(args.weightsBuf) +
                  tokenId * config.numExpertPerToken * sizeof(float),
              config.numExpertPerToken * sizeof(float));
        }
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + indicesOffset,
            reinterpret_cast<char*>(args.tokenIndices) +
                tokenId * config.numExpertPerToken * sizeof(index_t),
            config.numExpertPerToken * sizeof(index_t));
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + scalesOffset,
              reinterpret_cast<char*>(args.scalesBuf) +
                  tokenId * config.scaleDim * config.scaleTypeSize,
              config.scaleDim * config.scaleTypeSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);
      }
      // if (laneId == 0 && warpTokens) atomicAdd(&gatherTokenNum, warpTokens);
      __threadfence_block();
    }
  }

  __shared__ index_t recvTokenNum;
  __syncthreads();
  if (warpId == warpNum - 1) {
    shmem::ShmemAtomicTypeNonFetchWarp<int64_t>(
        args.sendAtomicSignalMemObj,
        (myPe + (args.crossDeviceBarrierFlag & 1) * npes) * sizeof(int64_t), 1, core::AMO_ADD,
        destPe, localBlockId);
  }
  if (thdId == 0) {
    int64_t* signal = args.sendAtomicSignalMemObj->template GetAs<int64_t*>() + destPe +
                      (args.crossDeviceBarrierFlag & 1) * npes;
    shmem::ShmemInt64WaitUntilGreaterThan(signal, numsBlockPerDestPe - 1);
    recvTokenNum = atomicAdd(
        &args.recvTokenNumMemObj
             ->template GetAs<index_t*>()[destPe + (args.crossDeviceBarrierFlag & 1) * npes],
        0);
    if (localBlockId == 0) {
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[destPe] = 0;
    }
    // if (localBlockId == 0) printf("rank[%d] destPe[%d] recvTokenNum: %d\n", myPe, destPe,
    // recvTokenNum);
  }
  __syncthreads();

  const int baseRecvChunk = recvTokenNum / numsBlockPerDestPe;
  const int recvRemainder = recvTokenNum % numsBlockPerDestPe;
  const int myRecvChunkSize = baseRecvChunk + (localBlockId < recvRemainder);
  // if (localBlockId == 0 && thdId == 0) printf("rank[%d] destPe[%d] myRecvChunkSize: %d\n", myPe,
  // destPe, myRecvChunkSize);
  const int startRecvIdx = localBlockId * baseRecvChunk + min(localBlockId, recvRemainder);
  const int endRecvIdx = startRecvIdx + myRecvChunkSize;
  for (int idx = warpId; idx < myRecvChunkSize; idx += warpNum) {
    index_t localTokenIdx = 0;
    if (laneId == 0) {
      localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
    }
    localTokenIdx = __shfl(localTokenIdx, 0);
    index_t peSortedId = destPe * MaxNumTokensToRecvPerRank + startRecvIdx + idx;

    size_t localTokenOffset = size_t(localTokenIdx) * size_t(config.hiddenDim) * sizeof(T);
    size_t peSortedTokenOffset = size_t(peSortedId) * stagingOffset;

    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<char*>() + localTokenOffset,
                   args.shmemDispatchInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset,
                   config.hiddenDim * sizeof(T));
    core::WarpCopy(args.shmemOutWeightsMemObj->template GetAs<char*>() +
                       localTokenIdx * config.numExpertPerToken * sizeof(float),
                   args.shmemDispatchInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset +
                       weightOffset,
                   config.numExpertPerToken * sizeof(float));
    core::WarpCopy(args.shmemOutIndicesMemObj->template GetAs<char*>() +
                       localTokenIdx * config.numExpertPerToken * sizeof(index_t),
                   args.shmemDispatchInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset +
                       indicesOffset,
                   config.numExpertPerToken * sizeof(index_t));
    if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
      core::WarpCopy(args.shmemOutScalesMemObj->template GetAs<char*>() +
                         localTokenIdx * config.scaleDim * config.scaleTypeSize,
                     args.shmemDispatchInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset +
                         scalesOffset,
                     config.scaleDim * config.scaleTypeSize);
    }
    if (laneId == 0) {
      args.dispReceiverIdxMap[localTokenIdx] = peSortedId;
      args.srcPeTokenIdxMap[peSortedId] = localTokenIdx;
    }
  }
  shmem::ShmemQuietThread();
  __syncthreads();
  // SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args,
                                                         int numQps) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // TODO: still figure out why use multiple threads lost RDMA writes
  if (laneId == 0) {
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
  }

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.config.worldSize) {
    uint64_t currentVal = core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId);
#if DEBUG == 1
    printf("Thread %d: localBarrierPtr[%d] = %lu, expected = %lu\n", thdId, thdId, currentVal,
           (uint64_t)(args.crossDeviceBarrierFlag * numQps));
#endif

    while (currentVal != args.crossDeviceBarrierFlag * numQps) {
      currentVal = core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId);
    }
  }
  __syncthreads();
}
/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernel(EpDispatchCombineArgs<T> args) {
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
  int myNode = myPe / MAX_GPUS_PER_NODE;

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  // Phase 1: send token
  // This phase is symmetric with dispatch recv phase, where tokens are first sent back to its
  // source pe in pe sorted order
  const int numsBlockPerSrcPe = gridDim.x / npes;
  const int srcPe = blockIdx.x / numsBlockPerSrcPe;
  const int srcNode = srcPe / MAX_GPUS_PER_NODE;
  const int localBlockId = blockIdx.x - srcPe * numsBlockPerSrcPe;
  const int srcPeTokenNum = *(args.recvTokenNumMemObj->template GetAs<index_t*>() + srcPe +
                              (args.crossDeviceBarrierFlag & 1) * npes);
  const int baseChunk = srcPeTokenNum / numsBlockPerSrcPe;
  const int remainder = srcPeTokenNum % numsBlockPerSrcPe;

  const int myChunkSize = baseChunk + (localBlockId < remainder);

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;

  const size_t tokenSize = config.hiddenDim * sizeof(T);
  const size_t weightSize = args.weightsBuf ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t tokenPackSize = tokenSize + weightSize;

  if (srcNode == myNode) {
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t mapIdxOffset = mapIdx * tokenPackSize;
      const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];
      size_t tokenOffset = tokenId * tokenSize;
      const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t peSortedOffset = peSortedId * tokenPackSize;
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

      if (args.weightsBuf) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
            reinterpret_cast<char*>(args.weightsBuf) +
                tokenId * config.numExpertPerToken * sizeof(float),
            weightSize);
      }

      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemCombineInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, tokenPackSize,
                                          srcPe);
    }
  } else {
    // inter node use ibgda for transfer
    // last warp for coordinate, other warp for gather token
    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);
    if (warpId == warpNum - 1) {
      const int totalTokenInBlock = endIdx - startIdx;
      int chunkOffset = 0;
      int chunkIdx = 0;
      while (chunkOffset < totalTokenInBlock) {
        int actualTokenNum = totalTokenInBlock - chunkOffset < chunkTokenSize
                                 ? totalTokenInBlock - chunkOffset
                                 : chunkTokenSize;
        if (laneId == 0) {
          while (atomicAdd(&gatherTokenNum[chunkIdx], 0) < actualTokenNum) {
            ;
          }
        }
        // rdma_send
        const index_t srcIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t srcOffset = srcIdx * tokenPackSize;
        const index_t dstIdx = myPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t dstOffset = dstIdx * tokenPackSize;
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemCombineInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * tokenPackSize, srcPe, localBlockId);

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * tokenPackSize;
        const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];
        size_t tokenOffset = tokenId * tokenSize;
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

        if (args.weightsBuf) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
              reinterpret_cast<char*>(args.weightsBuf) +
                  tokenId * config.numExpertPerToken * sizeof(float),
              weightSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);
      }
      // if (laneId == 0 && warpTokens) atomicAdd(&gatherTokenNum, warpTokens);
      __threadfence_block();
    }
  }
  __syncthreads();
  if (warpId == warpNum - 1) {
    shmem::ShmemAtomicTypeNonFetchWarp<uint64_t>(args.crossDeviceBarrierMemObj,
                                                 args.config.rank * sizeof(uint64_t), 1,
                                                 core::AMO_ADD, srcPe, localBlockId);
  }
  SyncIfDebugEnabled("Combine kernel: send token end");

  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierInterNodeKernel(args, numsBlockPerSrcPe);
  shmem::ShmemQuietThread();
  if (globalThdId < npes) {
    args.recvTokenNumMemObj
        ->template GetAs<index_t*>()[globalThdId + (args.crossDeviceBarrierFlag & 1) * npes] = 0;
    args.sendAtomicSignalMemObj
        ->template GetAs<int64_t*>()[globalThdId + (args.crossDeviceBarrierFlag & 1) * npes] = 0;
  }

  if (globalThdId == 0) {
    __hip_atomic_store(args.combineGridBarrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    args.localPeTokenCounter[0] = 0;
    args.totalRecvTokenNum[0] = 0;
  }

  SyncIfDebugEnabled("Dispatch kernel: sync across device end");

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  int warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  size_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    int tokenId = i / warpsPerToken;
    int inTokenPartId = i % warpsPerToken;
    size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    size_t hiddenDimSize = std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t peSortedId = args.dispSenderIdxMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = peSortedId / MaxNumTokensToRecvPerRank;
      size_t byteOffset = size_t(peSortedId) * tokenPackSize + hiddenDimOffset * sizeof(T);
      size_t weightByteOffset = size_t(peSortedId) * tokenPackSize + tokenSize;

      if (destPe < config.worldSize) {
        srcPtrs[j] = reinterpret_cast<T*>(args.shmemCombineInpTokMemObj->template GetAs<char*>() +
                                          byteOffset);
        srcWeightsPtr[j] = reinterpret_cast<float*>(
            args.shmemCombineInpTokMemObj->template GetAs<char*>() + weightByteOffset);
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }
    }

    size_t offset = size_t(tokenId) * size_t(config.hiddenDim) + hiddenDimOffset;
    core::WarpAccum<T, 8>(args.shmemOutTokMemObj->template GetAs<T*>() + offset, srcPtrs, nullptr,
                          config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(
          args.shmemOutWeightsMemObj->template GetAs<float*>() + tokenId * config.numExpertPerToken,
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

}  // namespace moe
}  // namespace mori
