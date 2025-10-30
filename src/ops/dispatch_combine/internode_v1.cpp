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

#include "src/ops/dispatch_combine/internode_v1.hpp"

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpDispatchInterNodeV1Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */
#define DEF_COMMON_VARS                                                        \
  const EpDispatchCombineConfig& config = args.config;                         \
  int thdId = threadIdx.x;                                                     \
  int thdNum = blockDim.x;                                                     \
  int laneId = threadIdx.x & (warpSize - 1);                                   \
  int warpId = thdId / warpSize;                                               \
  int warpNum = blockDim.x / warpSize;                                         \
  int blockNum = gridDim.x;                                                    \
  int blockId = blockIdx.x;                                                    \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;                     \
  int globalThdNum = gridDim.x * blockDim.x;                                   \
  int globalWarpId = blockIdx.x * warpNum + warpId;                            \
  int globalWarpNum = gridDim.x * warpNum;                                     \
  int nullTokenId = config.worldSize * config.MaxNumTokensToRecv();            \
  int myPe = config.rank;                                                      \
  int npes = config.worldSize;                                                 \
  int myNode = myPe / config.gpuPerNode;                                       \
  int nNodes = npes / config.gpuPerNode;                                       \
  int numExpertPerToken = config.numExpertPerToken;                            \
  assert(numExpertPerToken < warpSize);                                        \
  size_t hiddenBytes = config.hiddenDim * sizeof(T);                           \
  size_t indexBytes = config.numExpertPerToken * sizeof(index_t);              \
  size_t weightBytes = config.numExpertPerToken * sizeof(float);               \
  size_t srcTokenIdBytes = sizeof(index_t);                                    \
  size_t xferBytes = hiddenBytes + indexBytes + weightBytes + srcTokenIdBytes; \
  size_t combXferBytes = (args.weightsBuf == nullptr) ? hiddenBytes : hiddenBytes + weightBytes;

namespace v1 {
template <typename T>
inline __device__ void DispatchIntraNodeBlock(EpDispatchCombineArgs<T>& args, int tokenId,
                                              int expId, int destPe, int& localPeTokenCounter) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (laneId == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * config.MaxNumTokensToRecv() + destTokId;

    core::AtomicStoreRelaxedSystem(
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destTokId,
        config.rank * config.maxNumInpTokenPerRank + tokenId);
  }
  if (laneId == (destPe % config.gpuPerNode)) localPeTokenCounter++;
  destTokId = __shfl(destTokId, 0);
  size_t srcTokOffset = tokenId * config.hiddenDim;
  size_t destTokOffset = destTokId * config.hiddenDim;

  T* remoteTokenPtr = args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe);
  const T* localTokenPtr = args.inpTokenBuf;
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, config.hiddenDim);

  index_t* remoteIndexPtr = args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe);
  const index_t* localIndexPtr = args.tokenIndices;
  core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
                 localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  float* remoteWeightPtr = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe);
  const float* localWeightPtr = args.weightsBuf;
  core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
                 localWeightPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);
}

template <typename T>
inline __device__ void DispatchIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = config.rdmaBlockNum;
  int xgmiBlockNum = blockNum - config.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  int localPeTokenCounter = 0;

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    int lanePe = -1, laneNode = -1;
    if (laneId < numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    };

    // Send to other pes in myNode
    for (int e = 0; e < config.numExpertPerToken; e++) {
      int tokenExpertId = tokenId * config.numExpertPerToken + e;
      int destPe = __shfl(lanePe, e);
      int destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        if (__any((laneId < e) && (destPe == lanePe))) {
          if (laneId == 0) args.dispDestTokIdMap[tokenExpertId] = nullTokenId;
          continue;
        }
        DispatchIntraNodeBlock(args, tokenId, e, destPe, localPeTokenCounter);
      }
    }
  }
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <typename T>
inline __device__ void DispatchInterNodeSend(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, config.rdmaBlockNum);

  int startTokenIdx = blockChunkNum * blockId * warpSize;
  int endTokenIdx = std::min(startTokenIdx + blockChunkNum * warpSize, args.curRankNumToken);

  // First copy to staging buffer
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy<uint8_t, 8>(stagingPtr + stagingTokOffset,
                               reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
                               hiddenBytes);
    core::WarpCopy(stagingPtr + stagingTokOffset + hiddenBytes,
                   reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes,
                   indexBytes);
    core::WarpCopy(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes,
                   reinterpret_cast<uint8_t*>(args.weightsBuf) + tokenId * weightBytes,
                   weightBytes);
    if (laneId == 0)
      reinterpret_cast<index_t*>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes +
                                 weightBytes)[0] =
          tokenId + config.rank * config.maxNumInpTokenPerRank;
  }
  __syncthreads();

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e] /
                       config.numExpertPerRank / config.gpuPerNode;
        if (destNode == i) {
          shouldSend |= true;
          args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = nullTokenId;
        }
      }
      uint64_t mask = __ballot(shouldSend) & __activemask();
      uint64_t num = __popcll(mask);

      if (num == 0) continue;

      index_t flag = 0;
      index_t flagSlotId = 0;
      if (laneId == 0) {
        // TODO: use block flag counter per node
        flagSlotId = atomicAdd(args.blockFlagCounter, 1);
        // atomicAdd(args.destNodeTokenCounter + i, num);
        flag = num + 1;
      }
      flag = __shfl(flag, 0);
      flagSlotId = __shfl(flagSlotId, 0);

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
        size_t remoteIdx = (myNode * config.MaxNumTokensToRecvPerRank() + destTokId);
        if (count > 0) {
          size_t stagingTokOffset = tokenId * xferBytes;
          shmem::ShmemPutMemNbiSignalThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
                                            args.shmemStagingTokMemObj, stagingTokOffset,
                                            count * xferBytes, args.interNodeChunkFlagMemObj,
                                            (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
                                            flag, core::atomicType::AMO_SET, proxyPe);
        }
        args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.blockFlagCounter) * warpSize + 1;
      shmem::ShmemPutInt32ImmNbiThread(args.nodeRecvTokenNumMemObj, myNode * sizeof(index_t),
                                       numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;
  }
}

template <typename T>
inline __device__ void DispatchInterNodeRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 4;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  index_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();
  uint8_t* stagingPtr = args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;
  int totalChunkNum = 0;

  for (int k = blockId / numRecvBlock; k < maxChunkNum; k += (config.rdmaBlockNum / numRecvBlock)) {
    for (int i = 0; i < nNodes; i++) {
      if (i == myNode) continue;
      int startTokenIdx = k * warpSize;

      // Poll completion flags
      uint64_t thisChunkTokenNum = 0;
      index_t nodeFlag = 0;
      if (laneId == 0) {
        while (1) {
          thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[i * maxChunkNum + k]);
          if (thisChunkTokenNum > 0) break;

          nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[i]);
          if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
            thisChunkTokenNum = 1;
            break;
          }

          // printf("myPe %d block %d chunk %lu node %d\n", myPe, blockId, thisChunkTokenNum,
          // nodeFlag);
        }
      }
      thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;
      nodeFlag = __shfl(nodeFlag, 0) - 1;
      totalChunkNum += thisChunkTokenNum;

      int endTokenIdx = startTokenIdx + thisChunkTokenNum;

      for (int j = startTokenIdx + (blockId % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
           j += numRecvBlock * warpNum) {
        int tokIdx = i * config.MaxNumTokensToRecvPerRank() + j;
        index_t* indices =
            reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
        int lanePe = -1;
        if (laneId < config.numExpertPerToken) {
          lanePe = indices[laneId] / config.numExpertPerRank;
          assert((lanePe < config.worldSize) && (lanePe >= 0));
        }
        index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes +
                                                      hiddenBytes + indexBytes + weightBytes)[0];

        for (int e = 0; e < config.numExpertPerToken; e++) {
          int destPe = __shfl(lanePe, e);
          int destNode = destPe / config.gpuPerNode;

          bool shouldSkip = (destNode != myNode) || __any((laneId < e) && (destPe == lanePe));
          if (shouldSkip) {
            if (laneId == 0)
              args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] = nullTokenId;
            continue;
          }
          int destTokId = 0;
          if (laneId == 0) {
            destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
            args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] =
                destPe * config.MaxNumTokensToRecv() + destTokId;
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
          }
          if ((destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
          destTokId = __shfl(destTokId, 0);
          core::WarpCopy<uint8_t, 8>(
              args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe) +
                  destTokId * hiddenBytes,
              stagingPtr + tokIdx * xferBytes, hiddenBytes);
          core::WarpCopy(
              args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
              stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
          core::WarpCopy(args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>(destPe) +
                             destTokId * weightBytes,
                         stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
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
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

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
    if (laneId == 0) args.dispatchGridBarrier[0] = 0;

    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int destPe = nodePeOffset + laneId; destPe < (nodePeOffset + config.gpuPerNode);
         destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      core::AtomicStoreRelaxed(args.destPeTokenCounter + destPe, 0);
      core::AtomicStoreRelaxed(recvTokenNums + destPe, 0);
    }

    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1);
    }

    if (laneId < nNodes) {
      core::AtomicStoreRelaxedSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<index_t*>() + laneId, 0);
    }
  }

  for (int i = globalWarpId; i < nNodes; i += globalWarpNum) {
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    shmem::ShmemQuietThread(proxyPe);
  }

  // Each warp wait until sender finished by waiting token number signal
  // index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  // if (globalWarpId == 0) {
  //   for (int destPe = nodePeOffset + laneId; destPe < (nodePeOffset + config.gpuPerNode);
  //        destPe += warpSize) {
  //     index_t* signal = recvTokenNums + destPe;
  //     index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
  //     core::AtomicStoreRelaxedSystem(signal, 0);
  //     atomicAdd(args.totalRecvTokenNum, recvTokenNum);

  //     // reset local counter
  //     core::AtomicStoreRelaxed(args.destPeTokenCounter + destPe, 0);
  //     core::AtomicStoreRelaxed(recvTokenNums + destPe, 0);
  //   }

  //   if (laneId == 0) {
  //     args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
  //     atomicAdd(args.crossDeviceBarrierFlag, 1);
  //   }

  //   // if (laneId < nNodes) {
  //   //   core::AtomicStoreRelaxedSystem(
  //   //       args.nodeRecvTokenNumMemObj->template GetAs<index_t*>() + laneId, 0);
  //   //   core::AtomicStoreRelaxedSystem(args.destNodeTokenCounter + laneId, 0);
  //   // }
  // }
}

}  // namespace v1

template <typename T>
__global__ void EpDispatchInterNodeV1Kernel(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < config.rdmaBlockNum) {
    v1::DispatchInterNodeSend(args);
    v1::DispatchInterNodeRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpCombineInterNodeV1Kernel                                   */
/* ---------------------------------------------------------------------------------------------- */
namespace v1 {

template <typename T>
inline __device__ void CombineSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, blockNum);
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    core::WarpCopy(args.shmemCombineInpTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim,
                   args.inpTokenBuf + tokenId * config.hiddenDim, config.hiddenDim);
  }
  if (args.weightsBuf) {
    for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
      core::WarpCopy(
          args.shmemInpWeightsMemObj->template GetAs<float*>() + tokenId * config.numExpertPerToken,
          args.weightsBuf + tokenId * config.numExpertPerToken, config.numExpertPerToken);
    }
  }

  // After all warps copy done, set barrier flag
  uint32_t barrierFlag = 0;
  int finishedWarp = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(args.combineGridBarrier, 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);
  if ((finishedWarp + 1) == (blockNum * warpNum)) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(destPe) + args.config.rank,
          barrierFlag);
    }
    if (laneId == 0) args.combineGridBarrier[0] = 0;
  }
  // Wait other pes to set flag
  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + destPe) != barrierFlag) {
    }
  }
}

template <typename T>
inline __device__ void CombineIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = config.rdmaBlockNum;
  int xgmiBlockNum = blockNum - config.rdmaBlockNum;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        (nNodes + myNode) * config.MaxNumTokensToRecvPerRank() * combXferBytes;

  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = destTokId / config.MaxNumTokensToRecv();
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
        srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                          destLocalTokId * config.hiddenDim;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    core::WarpAccum<T, 4>(reinterpret_cast<T*>(stagingPtr + tokenId * combXferBytes), srcPtrs,
                          nullptr, config.numExpertPerToken, config.hiddenDim);
    if (args.weightsBuf) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * combXferBytes + hiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <typename T>
inline __device__ void CombineInterNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 4;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();

  for (int k = blockId / numRecvBlock; k < maxChunkNum; k += (config.rdmaBlockNum / numRecvBlock)) {
    for (int i = 0; i < nNodes; i++) {
      if (i == myNode) continue;

      uint64_t thisChunkTokenNum = chunkFlag[i * maxChunkNum + k];
      thisChunkTokenNum -= (thisChunkTokenNum > 0) ? 1 : 0;
      int startTokenIdx = k * warpSize;
      int endTokenIdx = startTokenIdx + thisChunkTokenNum;

      for (int j = startTokenIdx + (blockId % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
           j += numRecvBlock * warpNum) {
        int tokIdx = i * config.MaxNumTokensToRecvPerRank() + j;
        if (laneId < config.numExpertPerToken) {
          srcPtrs[laneId] = nullptr;
          srcWeightsPtr[laneId] = nullptr;
          index_t destTokId =
              args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId];
          index_t destPe = destTokId / config.MaxNumTokensToRecv();
          index_t destNode = destPe / config.gpuPerNode;
          if (destNode == myNode) {
            index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
            srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                              destLocalTokId * config.hiddenDim;
            srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                    destLocalTokId * config.numExpertPerToken;
          }
          args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId] = 0;
        }
        core::WarpAccum<T, 4>(reinterpret_cast<T*>(stagingPtr + tokIdx * combXferBytes), srcPtrs,
                              nullptr, config.numExpertPerToken, config.hiddenDim);
        if (args.weightsBuf) {
          core::WarpAccum<float, 4>(
              reinterpret_cast<float*>(stagingPtr + tokIdx * combXferBytes + hiddenBytes),
              srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
        }
      }

      index_t finished = 0;
      if (laneId == 0)
        finished = atomicAdd(&args.interNodeChunkFlagCombine[i * maxChunkNum + k], 1);
      finished = __shfl(finished, 0);
      if ((finished + 1) < (numRecvBlock * warpNum)) continue;

      args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>()[i * maxChunkNum + k] = 0;
      args.interNodeChunkFlagCombine[i * maxChunkNum + k] = 0;
      int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
      shmem::ShmemPutTypeNbiWarp<uint8_t>(
          args.shmemStagingTokMemObj,
          ((myNode + nNodes) * config.MaxNumTokensToRecvPerRank() + startTokenIdx) * combXferBytes,
          args.shmemStagingTokMemObj,
          (i * config.MaxNumTokensToRecvPerRank() + startTokenIdx) * combXferBytes,
          thisChunkTokenNum * combXferBytes, proxyPe);
    }
  }
  int finishedWarp = 0;
  uint32_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
    barrierFlag = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag = __shfl(barrierFlag, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if ((laneId < nNodes) &&
        (laneId != myNode)) {  // avoid setting myNode, it will be set in intra node branch
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      shmem::ShmemPutUint32ImmNbiThread(args.crossDeviceBarrierMemObj,
                                        args.config.rank * sizeof(uint32_t), barrierFlag, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    // Wait other nodes
    uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) != barrierFlag) {
      }
      // printf("myPe %d combine inter finished\n", myPe);
    }
  }
}

template <typename T>
inline __device__ void CombineAll(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Wait all warps
  uint32_t finishedWarps = 0;
  if (laneId == 0) {
    finishedWarps = atomicAdd(&args.combineGridBarrier[1], 1);
    shmem::ShmemUint32WaitUntilEquals(&args.combineGridBarrier[1], globalWarpNum);
  }
  finishedWarps = __shfl(finishedWarps, 0);
  if (((finishedWarps + 1) == globalWarpNum) && (laneId == 0)) {
    args.combineGridBarrier[1] = 0;
  }

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        nNodes * config.MaxNumTokensToRecvPerRank() * combXferBytes;

  int tokenPerBlock = (args.curRankNumToken + blockNum - 1) / blockNum;
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }
    for (int i = 0; i < nNodes; i++) {
      if (__any(laneNode == i) && (laneId == 0)) {
        int mappedId = (i == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + i];
        uint8_t* base =
            stagingPtr + (i * config.MaxNumTokensToRecvPerRank() + mappedId) * combXferBytes;
        srcPtrs[i] = reinterpret_cast<T*>(base);
        srcWeightsPtrs[i] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }
    if (laneId < nNodes) args.interNodeDispSendMap[nNodes * tokenId + laneId] = 0;
    core::WarpAccum<T, 4>(
        args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim, srcPtrs,
        nullptr, nNodes, config.hiddenDim);
    if (args.weightsBuf) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}
}  // namespace v1

template <typename T>
__global__ void EpCombineInterNodeV1Kernel(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  v1::CombineSync(args);
  if (blockId < config.rdmaBlockNum) {
    v1::CombineInterNode(args);
  } else {
    v1::CombineIntraNode(args);
  }
  v1::CombineAll(args);

  // TODO: refactor following state reset code
  if (laneId == 0) {
    args.totalRecvTokenNum[0] = 0;
    args.blockFlagCounter[0] = 0;
  }

  for (int i = globalThdId; i < (config.maxNumInpTokenPerRank * nNodes * config.numExpertPerToken);
       i += globalThdNum) {
    args.interNodeDispDestTokIdMap[i] = 0;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */
template __global__ void EpDispatchInterNodeV1Kernel<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
template __global__ void EpDispatchInterNodeV1Kernel<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
template __global__ void EpDispatchInterNodeV1Kernel<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpCombineInterNodeV1Kernel<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
template __global__ void EpCombineInterNodeV1Kernel<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
template __global__ void EpCombineInterNodeV1Kernel<float>(EpDispatchCombineArgs<float> args);

}  // namespace moe
}  // namespace mori
