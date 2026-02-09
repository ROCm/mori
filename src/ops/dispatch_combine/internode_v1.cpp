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

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#ifdef ENABLE_PROFILER
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/profiler/profiler.hpp"
#endif
#include "src/ops/dispatch_combine/common.hpp"

namespace mori {
namespace moe {

/* ---------------------------------------------------------------------------------------------- */
/*                                   EpDispatchInterNodeV1Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */
namespace v1 {

template <typename T>
__device__ __forceinline__ bool UseCombineInternalFp8(const EpDispatchCombineConfig& config) {
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
  if constexpr (!std::is_same_v<T, hip_bfloat16>) return false;
  const bool isV1 = (config.kernelType == KernelType::InterNodeV1) ||
                    (config.kernelType == KernelType::InterNodeV1LL);
  return config.enableInternalFp8Quant && isV1 && (config.scaleDim > 0) &&
         (config.scaleTypeSize == sizeof(float));
#else
  (void)config;
  return false;
#endif
}

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
  MORI_TRACE_SPAN(profiler, Slot::DispatchIntra);

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  int localPeTokenCounter = 0;

  for (int i = warpId; i < (endTokenIdx - startTokenIdx) * config.numExpertPerToken; i += warpNum) {
    index_t tokenId = i / config.numExpertPerToken + startTokenIdx;
    index_t destPe =
        args.tokenIndices[startTokenIdx * config.numExpertPerToken + i] / config.numExpertPerRank;
    int destNode = destPe / config.gpuPerNode;

    int lanePe = -1, laneNode = -1;
    if (laneId < numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    };

    // Deduplicate
    index_t inTokenExpertId = i % numExpertPerToken;
    if (destNode == myNode) {
      if (__any((laneId < inTokenExpertId) && (destPe == lanePe))) {
        if (laneId == 0)
          args.dispDestTokIdMap[startTokenIdx * config.numExpertPerToken + i] = nullTokenId;
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
inline __device__ void DispatchInterNodeSend(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeSend);

  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);
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
        }
        if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
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
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeLLSend);

  // Then send to other nodes
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);
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

        shmem::ShmemPutMemNbiSignalThread(args.shmemDispatchInpTokMemObj, remoteIdx * xferBytes,
                                          args.shmemStagingTokMemObj, stagingTokOffset,
                                          tokenNum * xferBytes, args.interNodeChunkFlagMemObj,
                                          (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t),
                                          tokenNum + 1, core::atomicType::AMO_ADD, proxyPe, qpId);
      }
      if (shouldSend) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[1], 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (args.rdmaBlockNum * warpNum)) {
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
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeRecv);

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>();

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

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    int counter = atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

template <typename T>
inline __device__ void DispatchInterNodeLLRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::DispatchInterNodeLLRecv);

  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
  uint8_t* stagingPtr = args.shmemDispatchInpTokMemObj->template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;

  // expert -> token -> node
  for (int i = globalWarpId;
       i < config.maxNumInpTokenPerRank * config.numExpertPerToken * (nNodes - 1);
       i += args.rdmaBlockNum * warpNum) {
    int expertId = i % config.numExpertPerToken;
    int tokenId = i / config.numExpertPerToken % config.maxNumInpTokenPerRank;
    int nodeId = i / config.numExpertPerToken / config.maxNumInpTokenPerRank;

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

    int globalTokenId = node * config.MaxNumTokensToRecvPerRank() + tokenId;
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
    bool shouldSkip = (destNode != myNode) || __any((laneId < expertId) && (destPe == lanePe));
    if (shouldSkip) {
      if (laneId == 0)
        args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
            nullTokenId;
      continue;
    }

    int destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
      args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + expertId] =
          destPe * config.MaxNumTokensToRecv() + destTokId;
      args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
    }
    if ((destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
    destTokId = __shfl(destTokId, 0);
    core::WarpCopy<uint8_t, 4>(
        args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
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
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
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
  if (blockId < args.rdmaBlockNum) {
    v1::DispatchInterNodeSend<T, true>(args);
    v1::DispatchInterNodeRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args);
}

template <typename T>
__global__ void EpDispatchCopyToStaging(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::EpDispatchCopyToStaging);
  if (args.curRankNumToken == 0) return;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = ((config.hiddenDim + warpsPerToken - 1) / warpsPerToken + 15) & ~15;

  // First copy to staging buffer
  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
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
          tokenId + config.rank * config.maxNumInpTokenPerRank;
  }
}

template <typename T, bool EnableStdMoE>
__global__ void EpDispatchInterNodeV1KernelLowLatency(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < args.rdmaBlockNum) {
    v1::DispatchInterNodeLLSend<T>(args);
    v1::DispatchInterNodeLLRecv(args);
  } else {
    v1::DispatchIntraNode(args);
  }
  v1::DispatchSync(args);

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
  MORI_TRACE_SPAN(profiler, Slot::CombineSync);

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, blockNum);
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
#ifdef ENABLE_STANDARD_MOE_ADAPT

#else
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    core::WarpCopy(args.shmemCombineInpTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim,
                   args.inpTokenBuf + tokenId * config.hiddenDim, config.hiddenDim);
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

template <typename T>
inline __device__ void CombineIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNode);

  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;

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
inline __device__ void CombineIntraNodeLL(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::CombineIntraNodeLL);

  if (args.curRankNumToken == 0) return;

  // Distribute tokens evenly to all blocks
  int blockOffset = args.rdmaBlockNum;
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  int xgmiWarpNum = xgmiBlockNum * warpNum;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        (nNodes + myNode) * config.MaxNumTokensToRecvPerRank() * combXferBytes;

  index_t warpsPerToken = (xgmiWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = ((config.hiddenDim + warpsPerToken - 1) / warpsPerToken + 15) & ~15;

  for (int i = globalWarpId - blockOffset * warpNum; i < (args.curRankNumToken * warpsPerToken);
       i += xgmiWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtr[laneId] = nullptr;
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + laneId];
      index_t destPe = destTokId / config.MaxNumTokensToRecv();
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
        srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                          destLocalTokId * config.hiddenDim + hiddenDimOffset;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    core::WarpAccum<T, 4>(
        reinterpret_cast<T*>(stagingPtr + tokenId * combXferBytes) + hiddenDimOffset, srcPtrs,
        nullptr, config.numExpertPerToken, hiddenDimSize);
    if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
      core::WarpAccum<float, 4>(
          reinterpret_cast<float*>(stagingPtr + tokenId * combXferBytes + hiddenBytes),
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

template <typename T>
inline __device__ void CombineInterNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNode);

  const bool useInternalFp8 = UseCombineInternalFp8<T>(config);

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  index_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  float** srcScalesPtr = reinterpret_cast<float**>(sharedMem) +
                         2 * warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();

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
                  if (useInternalFp8) {
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
                    srcPtrs[laneId] = reinterpret_cast<T*>(
                        args.shmemCombineInpTokMemObj->template GetAs<core::CombineInternalFp8T*>(
                            destPe) +
                        destLocalTokId * config.hiddenDim);
                    float* tokScales = args.shmemInpScalesMemObj->template GetAs<float*>(destPe) +
                                       destLocalTokId * config.scaleDim;
                    // Negative scale[0] marks tokens that used scaling.
                    if (tokScales[0] < 0.0f) srcScalesPtr[laneId] = tokScales;
#else
                    srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                                      destLocalTokId * config.hiddenDim;
#endif
                  } else {
                    srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                                      destLocalTokId * config.hiddenDim;
                  }
                  srcWeightsPtr[laneId] =
                      args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                      destLocalTokId * config.numExpertPerToken;
                }
                args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId] = 0;
              }

              if (useInternalFp8) {
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
                const core::CombineInternalFp8T* const* srcFp8Ptrs =
                    reinterpret_cast<const core::CombineInternalFp8T* const*>(srcPtrs);
                const float* const* srcScalePtrs =
                    reinterpret_cast<const float* const*>(srcScalesPtr);
                core::WarpAccumFp8DequantFull<T, core::CombineInternalFp8T>(
                    reinterpret_cast<T*>(stagingPtr + tokIdx * combXferBytes), srcFp8Ptrs,
                    srcScalePtrs, config.numExpertPerToken, config.hiddenDim, config.scaleDim);
#endif
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
        shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(args.crossDeviceBarrierMemObj,
                                                       args.config.rank * sizeof(uint64_t), 1,
                                                       core::AMO_ADD, proxyPe, i);
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

template <typename T>
inline __device__ void CombineInterNodeLL(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL);

  const bool useInternalFp8 = UseCombineInternalFp8<T>(config);

  constexpr int numRecvBlock = 8;
  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);

  uint64_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
  size_t fp8HiddenBytes = 0;
  size_t fp8CombXferBytes = 0;
  uint8_t* fp8SendBuf = nullptr;
  if (useInternalFp8) {
    fp8HiddenBytes = static_cast<size_t>(config.hiddenDim) * sizeof(core::CombineInternalFp8T);
    fp8CombXferBytes = (args.weightsBuf == nullptr) ? fp8HiddenBytes : fp8HiddenBytes + weightBytes;
    fp8SendBuf = args.shmemFp8SendTokMemObj->template GetAs<uint8_t*>();
  }
#endif

  int rdmaWarpNum = args.rdmaBlockNum * warpNum;
  for (int n = 0; n < (nNodes - 1); n++) {
    int node = (myNode + n + 1) % nNodes;
    uint64_t nodeCount = nodeRecvTokenNum[node];
    if (nodeCount > 0) nodeCount -= 1;
    if (nodeCount == 0) continue;

    // int warpsPerToken = (rdmaWarpNum + nodeCount - 1) / nodeCount;
    // NOTE: Keep a small fixed value to reduce overhead; use fewer warps for FP8 path.
    int warpsPerToken = useInternalFp8 ? 2 : 4;
    int hiddenDimPerWarp = ((config.hiddenDim + warpsPerToken - 1) / warpsPerToken + 15) & ~15;

    for (int i = globalWarpId; i < (nodeCount * warpsPerToken); i += rdmaWarpNum) {
      int tokenId = i / warpsPerToken;
      int k = tokenId / warpSize;
      int startTokenIdx = k * warpSize;
      uint64_t thisChunkTokenNum = chunkFlag[node * maxChunkNum + k];
      thisChunkTokenNum -= (thisChunkTokenNum > 0) ? 1 : 0;
      if ((tokenId - startTokenIdx) < thisChunkTokenNum) {
        int inTokenPartId = i % warpsPerToken;
        int hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
        int hiddenDimSize =
            std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

        int globalTokenId = node * config.MaxNumTokensToRecvPerRank() + tokenId;
        if (laneId < config.numExpertPerToken) {
          srcPtrs[laneId] = nullptr;
          srcWeightsPtr[laneId] = nullptr;
          index_t destTokId =
              args.interNodeDispDestTokIdMap[globalTokenId * config.numExpertPerToken + laneId];
          index_t destPe = destTokId / config.MaxNumTokensToRecv();
          index_t destNode = destPe / config.gpuPerNode;
          if (destNode == myNode) {
            index_t destLocalTokId = destTokId - destPe * config.MaxNumTokensToRecv();
            srcPtrs[laneId] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                              destLocalTokId * config.hiddenDim + hiddenDimOffset;
            srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                    destLocalTokId * config.numExpertPerToken;
          }
        }
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
        if (useInternalFp8) {
          uint8_t* fp8Rec =
              fp8SendBuf + (static_cast<size_t>(node) * config.MaxNumTokensToRecvPerRank() +
                            static_cast<size_t>(tokenId)) *
                               fp8CombXferBytes;
          core::CombineInternalFp8T* dstPtr =
              reinterpret_cast<core::CombineInternalFp8T*>(fp8Rec) + hiddenDimOffset;
          const hip_bfloat16* const* srcBf16Ptrs =
              reinterpret_cast<const hip_bfloat16* const*>(srcPtrs);
          {
            MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL_Accum);
            core::WarpAccumBf16ToFp8NoScaleAuto<core::CombineInternalFp8T>(
                dstPtr, srcBf16Ptrs, config.numExpertPerToken, hiddenDimSize);
            if (args.weightsBuf && (inTokenPartId == 0)) {
              core::WarpAccum<float, 4>(
                  reinterpret_cast<float*>(stagingPtr + globalTokenId * combXferBytes +
                                           hiddenBytes),
                  srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
              const float* srcW = reinterpret_cast<const float*>(
                  stagingPtr + globalTokenId * combXferBytes + hiddenBytes);
              core::WarpCopyWeightsToFp8Buffer(srcW, fp8Rec, fp8HiddenBytes,
                                               config.numExpertPerToken);
            }
          }
        } else
#endif
        {
          MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL_Accum);
          core::WarpAccum<T, 4>(
              reinterpret_cast<T*>(stagingPtr + globalTokenId * combXferBytes) + hiddenDimOffset,
              srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
          if (args.weightsBuf && (inTokenPartId == 0)) {
            core::WarpAccum<float, 4>(
                reinterpret_cast<float*>(stagingPtr + globalTokenId * combXferBytes + hiddenBytes),
                srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
          }
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
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
        if (useInternalFp8) {
          const size_t dstOffset = static_cast<size_t>(myNode + nNodes) *
                                       config.MaxNumTokensToRecvPerRank() * combXferBytes +
                                   static_cast<size_t>(startTokenIdx) * fp8CombXferBytes;
          const size_t srcOffset = (static_cast<size_t>(node) * config.MaxNumTokensToRecvPerRank() +
                                    static_cast<size_t>(startTokenIdx)) *
                                   fp8CombXferBytes;
          {
            MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL_RdmaPut);
            shmem::ShmemPutTypeNbiWarp<uint8_t>(
                args.shmemStagingTokMemObj, dstOffset, args.shmemFp8SendTokMemObj, srcOffset,
                thisChunkTokenNum * fp8CombXferBytes, proxyPe, qpId);
          }
        } else
#endif
        {
          {
            MORI_TRACE_SPAN(profiler, Slot::CombineInterNodeLL_RdmaPut);
            shmem::ShmemPutTypeNbiWarp<uint8_t>(
                args.shmemStagingTokMemObj,
                ((myNode + nNodes) * config.MaxNumTokensToRecvPerRank() + startTokenIdx) *
                    combXferBytes,
                args.shmemStagingTokMemObj,
                (node * config.MaxNumTokensToRecvPerRank() + startTokenIdx) * combXferBytes,
                thisChunkTokenNum * combXferBytes, proxyPe, qpId);
          }
        }
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
        shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(args.crossDeviceBarrierMemObj,
                                                       args.config.rank * sizeof(uint64_t), 1,
                                                       core::AMO_ADD, proxyPe, i);
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

template <typename T>
inline __device__ void CombineAll(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::CombineAll);

  if (args.curRankNumToken == 0) return;
  // Wait all warps
  if (laneId == 0) {
    atomicAdd(&args.combineGridBarrier[1], 1);
    shmem::ShmemUint32WaitUntilEquals(&args.combineGridBarrier[1], globalWarpNum);
  }

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        nNodes * config.MaxNumTokensToRecvPerRank() * combXferBytes;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = ((config.hiddenDim + warpsPerToken - 1) / warpsPerToken + 15) & ~15;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base =
            stagingPtr + (n * config.MaxNumTokensToRecvPerRank() + mappedId) * combXferBytes;
        srcPtrs[n] = reinterpret_cast<T*>(base) + hiddenDimOffset;
        srcWeightsPtrs[n] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }
    core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                              tokenId * config.hiddenDim + hiddenDimOffset,
                          srcPtrs, nullptr, nNodes, hiddenDimSize);
    if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
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
  if (blockId < args.rdmaBlockNum) {
    v1::CombineInterNode(args);
  } else {
    v1::CombineIntraNode(args);
  }
}

template <typename T>
__global__ void EpCombineAll(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  MORI_TRACE_SPAN(profiler, Slot::EpCombineAll);

  const bool useInternalFp8 = v1::UseCombineInternalFp8<T>(config);

  if (globalWarpId == 0) {
    if (laneId == 0) args.totalRecvTokenNum[0] = 0;
    if (laneId < nNodes) args.blockFlagCounter[laneId] = 0;
  }
  if (args.curRankNumToken == 0) return;

  extern __shared__ char sharedMem[];
  uint8_t** srcPtrs = reinterpret_cast<uint8_t**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtrs = reinterpret_cast<float**>(sharedMem) +
                           warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                        nNodes * config.MaxNumTokensToRecvPerRank() * combXferBytes;
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
  const size_t fp8HiddenBytes =
      static_cast<size_t>(config.hiddenDim) * sizeof(core::CombineInternalFp8T);
  const size_t fp8CombXferBytes =
      (args.weightsBuf == nullptr) ? fp8HiddenBytes : fp8HiddenBytes + weightBytes;
#else
  const size_t fp8HiddenBytes = 0;
  const size_t fp8CombXferBytes = 0;
#endif

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = ((config.hiddenDim + warpsPerToken - 1) / warpsPerToken + 15) & ~15;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    int lanePe = -1, laneNode = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / config.gpuPerNode;
    }

    if (laneId < nNodes) {
      srcPtrs[laneId] = nullptr;
      srcWeightsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        const auto ptrs = core::ResolveMixedTokPtrs<T>(
            stagingPtr, mappedId, n, myNode, useInternalFp8, hiddenDimOffset, combXferBytes,
            hiddenBytes, fp8HiddenBytes, fp8CombXferBytes, config.MaxNumTokensToRecvPerRank());
        srcPtrs[n] = ptrs.tok;
        srcWeightsPtrs[n] = ptrs.weights;
      }
    }
#if (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1) || \
    (defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1)
    if (useInternalFp8) {
      MORI_TRACE_SPAN(profiler, Slot::EpCombineAll_Fp8Mixed);
      T* outTok = args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim +
                  hiddenDimOffset;

      const hip_bfloat16* localTok = nullptr;
      if (srcPtrs[myNode] != nullptr) {
        localTok = reinterpret_cast<const hip_bfloat16*>(srcPtrs[myNode]);
      }
      // ~ 5us
      core::WarpAccumBf16Fp8Mixed<T, core::CombineInternalFp8T>(
          outTok, localTok, reinterpret_cast<const uint8_t* const*>(srcPtrs), nNodes, myNode,
          static_cast<int>(hiddenDimSize));
    } else
#endif
    {
      MORI_TRACE_SPAN(profiler, Slot::EpCombineAll_Bf16);
      core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                                tokenId * config.hiddenDim + hiddenDimOffset,
                            reinterpret_cast<T* const*>(srcPtrs), nullptr, nNodes, hiddenDimSize);
    }
    if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtrs, nullptr, nNodes, config.numExpertPerToken);
    }
  }
}

template <typename T, bool EnableStdMoE>
__global__ void EpCombineInterNodeV1KernelLowLatency(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  // If EnableStdMoE, call ConvertCombineInputDevice first to convert standard MoE format
#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertCombineInput<T>(args, myPe);
  }
#endif
  if (blockId < args.rdmaBlockNum) {
    v1::CombineInterNodeLL(args);
  } else {
    v1::CombineIntraNodeLL(args);
  }
}

template <typename T>
__global__ void EpCombineSync(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v1::CombineSync(args);
}

template <typename T>
__global__ void EpCombineSyncBarrier(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
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

/* ---------------------------------------------------------------------------------------------- */
/*                                     Template Specialization                                    */
/* ---------------------------------------------------------------------------------------------- */

// Helper macros for conditional FP8 compilation
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
#define MORI_FP8_FNUZ(...) __VA_ARGS__
#else
#define MORI_FP8_FNUZ(...)
#endif

#ifdef MORI_FP8_TYPE_OCP_ENABLED
#define MORI_FP8_OCP(...) __VA_ARGS__
#else
#define MORI_FP8_OCP(...)
#endif

// Macro to instantiate a kernel for all data types
#define INSTANTIATE_KERNEL(KernelName)                                                         \
  template __global__ void KernelName<hip_bfloat16>(EpDispatchCombineArgs<hip_bfloat16> args); \
  MORI_FP8_FNUZ(template __global__ void KernelName<__hip_fp8_e4m3_fnuz>(                      \
                    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);)                         \
  MORI_FP8_OCP(template __global__ void KernelName<__hip_fp8_e4m3>(                            \
                   EpDispatchCombineArgs<__hip_fp8_e4m3> args);)                               \
  template __global__ void KernelName<float>(EpDispatchCombineArgs<float> args);

// Macro to instantiate a kernel with EnableStdMoE parameter for all data types
#define INSTANTIATE_LL_KERNEL(KernelName, StdMoE)                                 \
  template __global__ void KernelName<hip_bfloat16, StdMoE>(                      \
      EpDispatchCombineArgs<hip_bfloat16> args);                                  \
  MORI_FP8_FNUZ(template __global__ void KernelName<__hip_fp8_e4m3_fnuz, StdMoE>( \
                    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);)            \
  MORI_FP8_OCP(template __global__ void KernelName<__hip_fp8_e4m3, StdMoE>(       \
                   EpDispatchCombineArgs<__hip_fp8_e4m3> args);)                  \
  template __global__ void KernelName<float, StdMoE>(EpDispatchCombineArgs<float> args);

// Combined macro for kernels with EnableStdMoE template parameter
#ifdef ENABLE_STANDARD_MOE_ADAPT
#define INSTANTIATE_LL_KERNEL_WITH_STDMOE(KernelName) \
  INSTANTIATE_LL_KERNEL(KernelName, false)            \
  INSTANTIATE_LL_KERNEL(KernelName, true)
#else
#define INSTANTIATE_LL_KERNEL_WITH_STDMOE(KernelName) INSTANTIATE_LL_KERNEL(KernelName, false)
#endif

INSTANTIATE_KERNEL(EpCombineSync)
INSTANTIATE_KERNEL(EpCombineSyncBarrier)
INSTANTIATE_KERNEL(EpDispatchInterNodeV1Kernel)
INSTANTIATE_KERNEL(EpCombineInterNodeV1Kernel)
INSTANTIATE_KERNEL(EpCombineAll)
INSTANTIATE_KERNEL(EpDispatchCopyToStaging)
INSTANTIATE_LL_KERNEL_WITH_STDMOE(EpDispatchInterNodeV1KernelLowLatency)
INSTANTIATE_LL_KERNEL_WITH_STDMOE(EpCombineInterNodeV1KernelLowLatency)

#undef MORI_FP8_FNUZ
#undef MORI_FP8_OCP
#undef INSTANTIATE_KERNEL
#undef INSTANTIATE_LL_KERNEL
#undef INSTANTIATE_LL_KERNEL_WITH_STDMOE

}  // namespace moe
}  // namespace mori
