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
#define DEF_COMMON_VARS                                                                         \
  const EpDispatchCombineConfig& config = args.config;                                          \
  int thdId = threadIdx.x;                                                                      \
  int thdNum = blockDim.x;                                                                      \
  int laneId = threadIdx.x & (warpSize - 1);                                                    \
  int warpId = thdId / warpSize;                                                                \
  int warpNum = blockDim.x / warpSize;                                                          \
  int blockNum = gridDim.x;                                                                     \
  int blockId = blockIdx.x;                                                                     \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;                                      \
  int globalThdNum = gridDim.x * blockDim.x;                                                    \
  int globalWarpId = blockIdx.x * warpNum + warpId;                                             \
  int globalWarpNum = gridDim.x * warpNum;                                                      \
  int nullTokenId = config.worldSize * config.MaxNumTokensToRecv();                             \
  int myPe = config.rank;                                                                       \
  int npes = config.worldSize;                                                                  \
  int myNode = myPe / config.gpuPerNode;                                                        \
  int nNodes = npes / config.gpuPerNode;                                                        \
  int numExpertPerToken = config.numExpertPerToken;                                             \
  assert(numExpertPerToken < warpSize);                                                         \
  size_t hiddenBytes = config.hiddenDim * sizeof(T);                                            \
  size_t indexBytes = config.numExpertPerToken * sizeof(index_t);                               \
  size_t weightBytes = config.numExpertPerToken * sizeof(float);                                \
  size_t srcTokenIdBytes = sizeof(index_t);                                                     \
  size_t scaleBytes = (args.config.scaleDim == 0) ? 0 : config.scaleDim * config.scaleTypeSize; \
  size_t xferBytes = hiddenBytes + indexBytes + weightBytes + srcTokenIdBytes + scaleBytes;     \
  size_t combScaleBytes = (args.scalesBuf && (scaleBytes > 0)) ? scaleBytes : 0;                \
  size_t combXferBytes =                                                                        \
      hiddenBytes + ((args.weightsBuf == nullptr) ? 0 : weightBytes) + combScaleBytes;

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

  if (args.scalesBuf && (scaleBytes > 0)) {
    core::WarpCopy(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * scaleBytes,
        args.scalesBuf + tokenId * scaleBytes, scaleBytes);
  }
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

template <typename T, bool DEDUP>
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
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset,
                               reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
                               hiddenBytes);
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
          tokenId + config.rank * config.maxNumInpTokenPerRank;
  }
  __syncthreads();

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    if (DEDUP) {
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
          flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
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
            int qpId = (tokenId / warpSize) % config.numQpPerPe;
            shmem::ShmemPutMemNbiSignalThread(
                args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes, args.shmemStagingTokMemObj,
                stagingTokOffset, count * xferBytes, args.interNodeChunkFlagMemObj,
                (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), flag,
                core::atomicType::AMO_ADD, proxyPe, qpId);
          }
          args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
        }
      }
    } else {
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

        index_t flagSlotId = 0;
        if (laneId == 0) {
          flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
        }
        flagSlotId = __shfl(flagSlotId, 0);

        index_t destTokIdOffset = flagSlotId * warpSize;
        index_t destTokId = destTokIdOffset + laneId;

        size_t remoteIdx = (myNode * config.MaxNumTokensToRecvPerRank() + destTokId);
        if (laneId == 0) {
          index_t tokenNum = std::min(tokenId + warpSize, endTokenIdx) - tokenId;
          size_t stagingTokOffset = tokenId * xferBytes;
          int qpId = (tokenId / warpSize) % config.numQpPerPe;
          shmem::ShmemPutMemNbiSignalThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
                                            args.shmemStagingTokMemObj, stagingTokOffset,
                                            tokenNum * xferBytes, args.interNodeChunkFlagMemObj,
                                            (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
                                            tokenNum + 1, core::atomicType::AMO_ADD, proxyPe, qpId);
          // shmem::ShmemPutMemNbiThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
          //                             args.shmemStagingTokMemObj, stagingTokOffset,
          //                             tokenNum * xferBytes, proxyPe, qpId);
          // shmem::ShmemPutUint64ImmNbiThread(args.interNodeChunkFlagMemObj,
          //                                   (myNode * maxChunkNum + flagSlotId) *
          //                                   sizeof(uint64_t), tokenNum + 1, proxyPe, qpId);
        }
        if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(args.nodeRecvTokenNumMemObj,
                                                     myNode * sizeof(uint64_t), numTokenSignal,
                                                     core::AMO_ADD, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;
  }
}

template <typename T>
inline __device__ void DispatchInterNodeLLSend(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks (optimized for LL scenario with small token counts)
  int tokenPerBlock = core::CeilDiv(args.curRankNumToken, config.rdmaBlockNum);

  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  // First copy to staging buffer
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy<uint8_t, 4>(stagingPtr + stagingTokOffset,
                               reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
                               hiddenBytes);
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
          tokenId + config.rank * config.maxNumInpTokenPerRank;
  }
  __syncthreads();

  // sync all rdma blocks
  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId == 0) {
      __hip_atomic_store(&args.interNodeBlocksBarrier[0], 0, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    if (laneId == 0) {
      while (__hip_atomic_load(&args.interNodeBlocksBarrier[0], __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) != 0);
    }
  }

  // Then send to other nodes
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);
  int totalChunkNum = core::CeilDiv(args.curRankNumToken, warpSize);
  int blockChunkNum = core::CeilDiv(totalChunkNum, config.rdmaBlockNum);
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
          args.dispDestTokIdMap[tokenId * numExpertPerToken + e] = nullTokenId;
        }
      }

      index_t flagSlotId = 0;
      if (laneId == 0) {
        flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
      }
      flagSlotId = __shfl(flagSlotId, 0);

      index_t destTokIdOffset = flagSlotId * warpSize;
      index_t destTokId = destTokIdOffset + laneId;

      size_t remoteIdx = (myNode * config.MaxNumTokensToRecvPerRank() + destTokId);
      if (laneId == 0) {
        index_t tokenNum = std::min(tokenId + warpSize, chunkEndTokenIdx) - tokenId;
        size_t stagingTokOffset = tokenId * xferBytes;
        int qpId = (tokenId / warpSize) % config.numQpPerPe;

        // printf(
        //     "BlockId=%d, WarpId=%d, tokenId=%d, remoteIdx=%lu, xferBytes=%lu, "
        //     "stagingTokOffset=%lu, tokenNum=%d, proxyPe=%d, qpId=%d, flagSlotId=%d\n",
        //     blockId, warpId, tokenId, remoteIdx, xferBytes, stagingTokOffset, tokenNum, proxyPe,
        //     qpId, flagSlotId);
        shmem::ShmemPutMemNbiSignalThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
                                          args.shmemStagingTokMemObj, stagingTokOffset,
                                          tokenNum * xferBytes, args.interNodeChunkFlagMemObj,
                                          (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
                                          tokenNum + 1, core::atomicType::AMO_ADD, proxyPe, qpId);
        // shmem::ShmemPutMemNbiThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
        //                             args.shmemStagingTokMemObj, stagingTokOffset,
        //                             tokenNum * xferBytes, proxyPe, qpId);
        // shmem::ShmemPutUint64ImmNbiThread(args.interNodeChunkFlagMemObj,
        //                                   (myNode * maxChunkNum + flagSlotId) *
        //                                   sizeof(uint64_t), tokenNum + 1, proxyPe, qpId);
      }
      if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
    }
  }

  finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[1], 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal =
          core::AtomicLoadRelaxed(args.blockFlagCounter + laneId) * warpSize + 1;
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(args.nodeRecvTokenNumMemObj,
                                                     myNode * sizeof(uint64_t), numTokenSignal,
                                                     core::AMO_ADD, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[1] = 0;
  }
}

template <typename T>
inline __device__ void DispatchInterNodeRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;
  int totalChunkNum = 0;

  // for (int k = blockId / numRecvBlock; k < maxChunkNum; k += (config.rdmaBlockNum /
  // numRecvBlock)) { for (int i = 0; i < (nNodes - 1); i++) {
  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += config.rdmaBlockNum) {
    int k = bid / (numRecvBlock * (nNodes - 1));
    int i = (bid / numRecvBlock) % (nNodes - 1);

    int node = (myNode + 1 + i) % nNodes;
    int startTokenIdx = k * warpSize;

    // Poll completion flags
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
      int tokIdx = node * config.MaxNumTokensToRecvPerRank() + j;
      index_t* indices = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
      int lanePe = -1;
      if (laneId < config.numExpertPerToken) {
        lanePe = indices[laneId] / config.numExpertPerRank;
        assert((lanePe < config.worldSize) && (lanePe >= 0));
      }
      index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes +
                                                    indexBytes + weightBytes + scaleBytes)[0];

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
        core::WarpCopy<uint8_t, 4>(
            args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe) +
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
  // }
  // }

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
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      __threadfence_system();
      // reset local counter
      core::AtomicStoreSeqCstSystem(signal, 0);
      core::AtomicStoreSeqCstSystem(args.destPeTokenCounter + destPe, 0);
    }

    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1);
      args.combineGridBarrier[1] = 0;
    }

    if (laneId < nNodes) {
      core::AtomicStoreSeqCstSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>() + laneId, uint64_t{0});
    }
  }

  for (int i = globalWarpId; i < nNodes; i += globalWarpNum) {
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    shmem::ShmemQuietThread(proxyPe);
  }
}

}  // namespace v1

template <typename T>
__global__ void EpDispatchInterNodeV1Kernel(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < config.rdmaBlockNum) {
    v1::DispatchInterNodeSend<T, true>(args);
    v1::DispatchInterNodeRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args);
}

template <typename T>
__global__ void EpDispatchInterNodeV1KernelLowLatency(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < config.rdmaBlockNum) {
    // v1::DispatchInterNodeSend<T, false>(args);
    v1::DispatchInterNodeLLSend<T>(args);
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
  if (args.scalesBuf && (scaleBytes > 0)) {
    // Copy per-token per-block scale vector to registered buffer.
    for (int i = globalWarpId; i < totalRecvTokenNum * config.scaleDim; i += globalWarpNum) {
      core::WarpCopy<uint8_t, 4>(
          args.shmemInpScalesMemObj->template GetAs<uint8_t*>() + i * config.scaleTypeSize,
          args.scalesBuf + i * config.scaleTypeSize, config.scaleTypeSize);
    }
  }

  // After all warps copy done, set barrier flag
  uint64_t barrierFlag = 0;
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
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(destPe) + args.config.rank,
          barrierFlag);
    }
    if (laneId == 0) args.combineGridBarrier[0] = 0;
  }
  // Wait other pes to set flag
  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
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
  uint8_t** srcScalesPtr = reinterpret_cast<uint8_t**>(sharedMem) +
                           2 * warpNum * config.numExpertPerToken +
                           warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        (nNodes + myNode) * config.MaxNumTokensToRecvPerRank() * combXferBytes;

  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      srcScalesPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = destTokId / config.MaxNumTokensToRecv();
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
        srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                          destLocalTokId * config.hiddenDim;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
        if (combScaleBytes > 0) {
          srcScalesPtr[laneId] = args.shmemInpScalesMemObj->template GetAs<uint8_t*>(destPe) +
                                 destLocalTokId * scaleBytes;
        }
      }
    }
    if constexpr (core::IsFp8Type<T>::value) {
      assert(combScaleBytes > 0);
      T* dstTok = reinterpret_cast<T*>(stagingPtr + tokenId * combXferBytes);
      uint8_t* dstScale = stagingPtr + tokenId * combXferBytes + hiddenBytes +
                          ((args.weightsBuf == nullptr) ? 0 : weightBytes);
      const uint8_t* const* srcScalesConst = reinterpret_cast<const uint8_t* const*>(srcScalesPtr);
      core::WarpAccumFp8Quant<T>(dstTok, dstScale, srcPtrs, srcScalesConst,
                                 config.numExpertPerToken, config.hiddenDim, config.scaleDim,
                                 config.scaleTypeSize);
    } else {
      core::WarpAccum<T, 4>(reinterpret_cast<T*>(stagingPtr + tokenId * combXferBytes), srcPtrs,
                            nullptr, config.numExpertPerToken, config.hiddenDim);
    }
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

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  index_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t** srcScalesPtr = reinterpret_cast<uint8_t**>(sharedMem) +
                           2 * warpNum * config.numExpertPerToken +
                           warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();

  int totalBids = 0;
  for (int bid = blockId; bid < numRecvBlock * maxChunkNum * (nNodes - 1);
       bid += config.rdmaBlockNum) {
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
           bid += config.rdmaBlockNum) {
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
              int tokIdx = node * config.MaxNumTokensToRecvPerRank() + j;
              if (laneId < config.numExpertPerToken) {
                srcPtrs[laneId] = nullptr;
                srcWeightsPtr[laneId] = nullptr;
                srcScalesPtr[laneId] = nullptr;
                index_t destTokId =
                    args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId];
                index_t destPe = destTokId / config.MaxNumTokensToRecv();
                index_t destNode = destPe / config.gpuPerNode;
                if (destNode == myNode) {
                  index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
                  srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                                    destLocalTokId * config.hiddenDim;
                  srcWeightsPtr[laneId] =
                      args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                      destLocalTokId * config.numExpertPerToken;
                  if (combScaleBytes > 0) {
                    srcScalesPtr[laneId] =
                        args.shmemInpScalesMemObj->template GetAs<uint8_t*>(destPe) +
                        destLocalTokId * scaleBytes;
                  }
                }
                args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId] = 0;
              }
              if constexpr (core::IsFp8Type<T>::value) {
                assert(combScaleBytes > 0);
                T* dstTok = reinterpret_cast<T*>(stagingPtr + tokIdx * combXferBytes);
                uint8_t* dstScale = stagingPtr + tokIdx * combXferBytes + hiddenBytes +
                                    ((args.weightsBuf == nullptr) ? 0 : weightBytes);
                const uint8_t* const* srcScalesConst =
                    reinterpret_cast<const uint8_t* const*>(srcScalesPtr);
                core::WarpAccumFp8Quant<T>(dstTok, dstScale, srcPtrs, srcScalesConst,
                                           config.numExpertPerToken, config.hiddenDim,
                                           config.scaleDim, config.scaleTypeSize);
              } else {
                core::WarpAccum<T, 4>(reinterpret_cast<T*>(stagingPtr + tokIdx * combXferBytes),
                                      srcPtrs, nullptr, config.numExpertPerToken, config.hiddenDim);
              }
              if (args.weightsBuf) {
                core::WarpAccum<float, 4>(
                    reinterpret_cast<float*>(stagingPtr + tokIdx * combXferBytes + hiddenBytes),
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
              shmem::ShmemPutTypeNbiWarp<uint8_t>(
                  args.shmemStagingTokMemObj,
                  ((myNode + nNodes) * config.MaxNumTokensToRecvPerRank() + startTokenIdx) *
                      combXferBytes,
                  args.shmemStagingTokMemObj,
                  (node * config.MaxNumTokensToRecvPerRank() + startTokenIdx) * combXferBytes,
                  thisChunkTokenNum * combXferBytes, proxyPe, qpId);
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

  // TODO: this make sure interNodeChunkFlagMemObj is set to zero before sync with other
  // nodes, without this, it may be set by other node first then get override by zero
  __threadfence_system();
  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
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
      for (int i = 0; i < config.numQpPerPe; i++) {
        shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(args.crossDeviceBarrierMemObj,
                                                       args.config.rank * sizeof(uint64_t), 1,
                                                       core::AMO_ADD, proxyPe, i);
      }
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

template <typename T>
inline __device__ void CombineAll(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Wait all warps
  if (laneId == 0) {
    atomicAdd(&args.combineGridBarrier[1], 1);
    shmem::ShmemUint32WaitUntilEquals(&args.combineGridBarrier[1], globalWarpNum);
  }

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t** srcScalesPtrs = reinterpret_cast<uint8_t**>(sharedMem) +
                            2 * warpNum * config.numExpertPerToken +
                            warpId * config.numExpertPerToken;
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
      srcScalesPtrs[laneId] = nullptr;
    }
    for (int i = 0; i < nNodes; i++) {
      if (__any(laneNode == i) && (laneId == 0)) {
        int mappedId = (i == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + i];
        uint8_t* base =
            stagingPtr + (i * config.MaxNumTokensToRecvPerRank() + mappedId) * combXferBytes;
        srcPtrs[i] = reinterpret_cast<T*>(base);
        srcWeightsPtrs[i] = reinterpret_cast<float*>(base + hiddenBytes);
        if (combScaleBytes > 0) {
          srcScalesPtrs[i] = base + hiddenBytes + ((args.weightsBuf == nullptr) ? 0 : weightBytes);
        }
      }
    }
    if (laneId < nNodes) args.interNodeDispSendMap[nNodes * tokenId + laneId] = 0;
    if constexpr (core::IsFp8Type<T>::value) {
      assert(combScaleBytes > 0);
      T* dstTok = args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim;
      uint8_t* dstScale =
          args.shmemOutScalesMemObj->template GetAs<uint8_t*>() + tokenId * scaleBytes;
      const uint8_t* const* srcScalesConst = reinterpret_cast<const uint8_t* const*>(srcScalesPtrs);
      core::WarpAccumFp8Quant<T>(dstTok, dstScale, srcPtrs, srcScalesConst, nNodes,
                                 config.hiddenDim, config.scaleDim, config.scaleTypeSize);
    } else {
      core::WarpAccum<T, 4>(
          args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim, srcPtrs,
          nullptr, nNodes, config.hiddenDim);
    }
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

  if (laneId == 0) {
    args.totalRecvTokenNum[0] = 0;
  }

  if (laneId < nNodes) {
    args.blockFlagCounter[laneId] = 0;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */
template __global__ void EpDispatchInterNodeV1Kernel<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpDispatchInterNodeV1Kernel<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpDispatchInterNodeV1Kernel<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpDispatchInterNodeV1Kernel<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpDispatchInterNodeV1KernelLowLatency<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpDispatchInterNodeV1KernelLowLatency<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpDispatchInterNodeV1KernelLowLatency<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpDispatchInterNodeV1KernelLowLatency<float>(
    EpDispatchCombineArgs<float> args);

template __global__ void EpCombineInterNodeV1Kernel<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template __global__ void EpCombineInterNodeV1Kernel<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template __global__ void EpCombineInterNodeV1Kernel<__hip_fp8_e4m3>(
    EpDispatchCombineArgs<__hip_fp8_e4m3> args);
#endif
template __global__ void EpCombineInterNodeV1Kernel<float>(EpDispatchCombineArgs<float> args);

}  // namespace moe
}  // namespace mori
