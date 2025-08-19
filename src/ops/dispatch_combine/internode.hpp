#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

#define SIGNAL_PUT_FINISHED 1
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

inline __device__ index_t EncodeValue(index_t val) { return val + 1; }

inline __device__ index_t DecodeValue(index_t val) { return val - 1; }

inline __device__ void LocalBarrier(uint32_t* barrier, int syncNum, int id) {
  if (id == 0) {
    int old_val = atomicAdd(barrier, 1);
    if (old_val == syncNum - 1) {
      __hip_atomic_store(barrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    shmem::ShmemUint32WaitUntilEquals(barrier, 0);
  }
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
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  const int myPe = config.rank;
  const int npes = config.worldSize;
  const int myNode = myPe / MAX_GPUS_PER_NODE;

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();
  size_t MaxNumTokensToRecv = config.MaxNumTokensToRecv();

  int numExpertPerToken = config.numExpertPerToken;
  assert(numExpertPerToken < warpSize);

  const size_t tokenSize = config.hiddenDim * sizeof(T);
  const size_t weightSize = sizeof(float) * numExpertPerToken;
  const size_t indiceSize = sizeof(index_t) * numExpertPerToken;
  const size_t scaleSize = config.scaleTypeSize * config.scaleDim;
  const size_t tokenPackSize = tokenSize + weightSize + indiceSize + scaleSize;

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

  // global warp barrier
  LocalBarrier(&args.dispatchGridBarrier[npes], globalWarpNum, laneId);

  // TODO: block num should be multiple of npes
  const int numsBlockPerDestPe = gridDim.x / npes;
  const int destPe = blockIdx.x / numsBlockPerDestPe;
  const int destNode = destPe / MAX_GPUS_PER_NODE;
  const int localBlockId = blockIdx.x - destPe * numsBlockPerDestPe;
  const int totalTokens = args.destPeTokenCounter[destPe];
  const int baseChunk = totalTokens / numsBlockPerDestPe;
  const int remainder = totalTokens - baseChunk * numsBlockPerDestPe;

  const int myChunkSize = baseChunk + (localBlockId < remainder);

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;

  if (destNode == myNode) {
    if (destPe == myPe && localBlockId == 0 && warpId == 0) {
      // calculate prefix to determine output offset
      if (laneId == 0) {
        // write number of token to send
        for (int peerPe = 0; peerPe < npes; ++peerPe) {
          if (peerPe / MAX_GPUS_PER_NODE == myNode) {
            index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(peerPe) + myPe;
            shmem::ShmemInt32WaitUntilEquals(signal, 0);
            core::AtomicStoreRelaxedSystem(signal, args.destPeTokenCounter[peerPe] + 1);
          } else {
            shmem::ShmemPutInt32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(index_t),
                                             args.destPeTokenCounter[peerPe] + 1, peerPe);
          }
        }
        // obtain number of token to recv
        index_t recvTokenNum = 0, prevRecvTokenNum = 0;
        index_t sum = 0;
        for (int pe = 0; pe < npes; ++pe) {
          index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>() + pe;
          recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
          args.dispReceiverIdxMap[pe] = recvTokenNum;  // TODO 按照实际大小开
          sum += prevRecvTokenNum;
          prevRecvTokenNum = recvTokenNum;
          args.recvTokenOffset[pe] = sum;

          // send prefix back
          if (pe / MAX_GPUS_PER_NODE == myNode) {  // TODO这里要不要优先填本机的，让机内拷贝先开始
            index_t* peerPrefixPtr = args.shmemMetaDataMemObj->template GetAs<index_t*>(pe) + myPe;
            shmem::ShmemInt32WaitUntilEquals(peerPrefixPtr, 0);
            core::AtomicStoreRelaxedSystem(peerPrefixPtr, sum + 1);
          } else {
            shmem::ShmemPutInt32ImmNbiThread(args.shmemMetaDataMemObj, myPe * sizeof(index_t),
                                             sum + 1, pe);
          }
        }
        // args.dispReceiverIdxMap[npes] = sum;

        *args.localPeTokenCounter = npes;  // TODO 和dispReceiverIdxMap一起改一下
        *args.totalRecvTokenNum = sum;     // TODO
      }
    }

    index_t outputBaseIdx;
    if (laneId == 0) {
      index_t* prefix = args.shmemMetaDataMemObj->template GetAs<index_t*>() + destPe;
      outputBaseIdx = shmem::ShmemInt32WaitUntilGreaterThan(prefix, 0) - 1;
      outputBaseIdx += startIdx;
    }
    outputBaseIdx = __shfl(outputBaseIdx, 0);
    // if (myPe != 0 && thdId == 0) {
    //   index_t* prefix = args.shmemMetaDataMemObj->template GetAs<index_t*>(destPe) + (myPe - 1);
    //   outputBaseIdx = shmem::ShmemInt32WaitUntilGreaterThan(prefix, 0) - 1;
    //   outputBaseIdx += startIdx;
    // }
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
      size_t tokenOffset = tokenId * tokenSize;
      const index_t outputIdx = outputBaseIdx + idx;

      core::WarpCopy(args.shmemOutTokMemObj->template GetAs<char*>(destPe) + outputIdx * tokenSize,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);
      core::WarpCopy(
          args.shmemOutWeightsMemObj->template GetAs<char*>(destPe) + outputIdx * weightSize,
          reinterpret_cast<char*>(args.weightsBuf) + tokenId * weightSize, weightSize);
      core::WarpCopy(
          args.shmemOutIndicesMemObj->template GetAs<char*>(destPe) + outputIdx * indiceSize,
          reinterpret_cast<char*>(args.tokenIndices) + tokenId * indiceSize, indiceSize);
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<char*>(destPe) + outputIdx * scaleSize,
            reinterpret_cast<char*>(args.scalesBuf) + tokenId * scaleSize, scaleSize);
      }
    }

    // Same dest PE block barrier
    LocalBarrier(&args.dispatchGridBarrier[destPe], numsBlockPerDestPe, thdId);
    if (localBlockId == 0 && thdId == 0) {
      index_t* sync = args.shmemMetaDataMemObj->template GetAs<index_t*>(destPe) + npes + myPe;
      shmem::ShmemInt32WaitUntilEquals(sync, 0);
      core::AtomicStoreRelaxedSystem(sync, SIGNAL_PUT_FINISHED);
    }
  } else {
    // last warp for coordinate, other warp for gather token
    const size_t stagingBaseOffset = destPe * MaxNumTokensToRecvPerRank * tokenPackSize;
    const size_t stagingWeightBaseOffset = stagingBaseOffset + totalTokens * tokenSize;
    const size_t stagingIndiceBaseOffset = stagingWeightBaseOffset + totalTokens * indiceSize;
    const size_t stagingScaleBaseOffset = stagingIndiceBaseOffset + totalTokens * scaleSize;

    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);
    if (warpId == warpNum - 1) {
      index_t outputBaseIdx = 0;
      if (laneId == 0) {
        index_t* prefix = args.shmemMetaDataMemObj->template GetAs<index_t*>() + destPe;
        outputBaseIdx = shmem::ShmemInt32WaitUntilGreaterThan(prefix, 0) - 1;
      }

      // inter node use ibgda to transfer data
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
        // rdma write to output buffer
        if (laneId == 0) {
          const index_t srcIdx = startIdx + chunkOffset;
          size_t srcOffset = stagingBaseOffset + (startIdx + chunkOffset) * tokenSize;
          const index_t outputIdx = outputBaseIdx + startIdx + chunkOffset;
          size_t dstOffset = outputIdx * tokenSize;
          shmem::ShmemPutTypeNbiThread<uint8_t>(args.shmemOutTokMemObj, dstOffset,
                                                args.shmemStagingTokMemObj, srcOffset,
                                                actualTokenNum * tokenSize, destPe);
          // TODO check perf
          shmem::ShmemPutTypeNbiThread<uint8_t>(
              args.shmemOutWeightsMemObj, outputIdx * weightSize, args.shmemStagingTokMemObj,
              stagingWeightBaseOffset + srcIdx * weightSize, actualTokenNum * weightSize, destPe);
          shmem::ShmemPutTypeNbiThread<uint8_t>(
              args.shmemOutIndicesMemObj, outputIdx * indiceSize, args.shmemStagingTokMemObj,
              stagingIndiceBaseOffset + srcIdx * indiceSize, actualTokenNum * indiceSize, destPe);
          if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
            shmem::ShmemPutTypeNbiThread<uint8_t>(
                args.shmemOutScalesMemObj, outputIdx * scaleSize, args.shmemStagingTokMemObj,
                stagingScaleBaseOffset + srcIdx * scaleSize, actualTokenNum * scaleSize, destPe);
          }
        }

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }

      // Same dest PE block barrier
      LocalBarrier(&args.dispatchGridBarrier[destPe], numsBlockPerDestPe, thdId);
      if (localBlockId == 0 && laneId == 0) {  // TODO 连续跑会hang?
        shmem::ShmemPutInt32ImmNbiThread(args.shmemMetaDataMemObj, (npes + myPe) * sizeof(index_t),
                                         SIGNAL_PUT_FINISHED, destPe);
      }
    } else {
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
        const index_t stagingIdx = startIdx + idx;

        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + stagingBaseOffset +
                           stagingIdx * tokenSize,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenId * tokenSize, tokenSize);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() +
                           stagingWeightBaseOffset + stagingIdx * weightSize,
                       reinterpret_cast<char*>(args.weightsBuf) + tokenId * weightSize, weightSize);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() +
                           stagingIndiceBaseOffset + stagingIdx * indiceSize,
                       reinterpret_cast<char*>(args.tokenIndices) + tokenId * indiceSize,
                       indiceSize);
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() +
                             stagingScaleBaseOffset + stagingIdx * scaleSize,
                         reinterpret_cast<char*>(args.scalesBuf) + tokenId * scaleSize, scaleSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);
      }
      __threadfence_block();
    }
  }

  __syncthreads();
  if (localBlockId == 0 && thdId == 0) {
    index_t* sync = args.shmemMetaDataMemObj->template GetAs<index_t*>() + (npes + destPe);
    shmem::ShmemInt32WaitUntilEquals(sync, SIGNAL_PUT_FINISHED);
    core::AtomicStoreRelaxedSystem(sync, 0);
  }
  // global warp barrier
  LocalBarrier(&args.dispatchGridBarrier[npes], globalWarpNum, laneId);
  // clear meta data
  for (int i = globalThdId; i < npes * 2; ++i) {
    core::AtomicStoreRelaxedSystem(args.shmemMetaDataMemObj->template GetAs<index_t*>() + i, 0);
  }
  for (int i = globalThdId; i < npes; ++i) {
    args.destPeTokenCounter[i] = 0;
  }
  __syncthreads();

  SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // TODO: still figure out why use multiple threads lost RDMA writes
  for (int destPe = globalWarpId; destPe < args.config.worldSize; destPe += globalWarpNum) {
    if (laneId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      shmem::ShmemPutUint32ImmNbiWarp(args.crossDeviceBarrierMemObj,
                                      args.config.rank * sizeof(uint32_t),
                                      args.crossDeviceBarrierFlag, destPe);
    }
  }

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != args.crossDeviceBarrierFlag) {
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
  const int srcPeTokenNum = *(args.recvTokenNumMemObj->template GetAs<index_t*>() + srcPe) - 1;
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
      const index_t tokenId = args.recvTokenOffset[srcPe] + startIdx + idx;
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

      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, tokenPackSize,
                                          srcPe);
      // TODO remove
#if DEBUG==1
      if (fabs(*(float*)(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset) -
               float(srcPe + 1)) > 0.1) {
        int tmp = myPe;
        while (tmp < npes) {
        };
      }
#endif
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
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * tokenPackSize, srcPe);
#if DEBUG == 1
        if (fabs(*(float*)(args.shmemStagingTokMemObj->template GetAs<char*>() + srcOffset) -
                 float(srcPe + 1)) > 0.1) {
          int tmp = myPe;
          while (tmp < npes) {
          };
        }
#endif

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {
      // TODO 多机也可以支持不用external buffer input staging可以改成max recv大小
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * tokenPackSize;
        const index_t tokenId = args.recvTokenOffset[srcPe] + startIdx + idx;
        size_t tokenOffset = tokenId * tokenSize;
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);
#if DEBUG == 1
        if (fabs(*(float*)(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset) -
                 float(srcPe + 1)) > 0.1) {
          int tmp = myPe;
          while (tmp < npes) {
          };
        }
#endif

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
  SyncIfDebugEnabled("Combine kernel: send token end");

  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierInterNodeKernel(args);

  if (globalThdId < npes) {
    args.recvTokenNumMemObj->template GetAs<index_t*>()[globalThdId] = 0;
    args.recvTokenOffset[globalThdId] = 0;
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
        srcPtrs[j] =
            reinterpret_cast<T*>(args.shmemInpTokMemObj->template GetAs<char*>() + byteOffset);
        srcWeightsPtr[j] = reinterpret_cast<float*>(
            args.shmemInpTokMemObj->template GetAs<char*>() + weightByteOffset);
#if DEBUG == 1
        if (fabs(*(float*)(srcPtrs[j]) - float(myPe + 1)) > 0.1) {
          int tmp = myPe;
          while (tmp < npes) {
          };
        }
#endif
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
