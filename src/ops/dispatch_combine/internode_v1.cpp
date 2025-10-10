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
/*                                   EpDispatchInterNodeKernelV1                                  */
/* ---------------------------------------------------------------------------------------------- */
#define DEF_COMMON_VARS                                                  \
  const EpDispatchCombineConfig& config = args.config;                   \
  int thdId = threadIdx.x;                                               \
  int thdNum = blockDim.x;                                               \
  int laneId = threadIdx.x & (warpSize - 1);                             \
  int warpId = thdId / warpSize;                                         \
  int warpNum = blockDim.x / warpSize;                                   \
  int blockNum = gridDim.x;                                              \
  int blockId = blockIdx.x;                                              \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;               \
  int globalThdNum = gridDim.x * blockDim.x;                             \
  int globalWarpId = blockIdx.x * warpNum + warpId;                      \
  int globalWarpNum = gridDim.x * warpNum;                               \
  int myPe = config.rank;                                                \
  int npes = config.worldSize;                                           \
  int myNode = myPe / config.gpuPerNode;                                 \
  int nNodes = npes / config.gpuPerNode;                                 \
  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank(); \
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank(); \
  size_t MaxNumTokensToRecv = config.MaxNumTokensToRecv();               \
  int numExpertPerToken = config.numExpertPerToken;                      \
  assert(numExpertPerToken < warpSize);                                  \
  size_t hiddenBytes = config.hiddenDim * sizeof(T);                     \
  size_t indexBytes = config.numExpertPerToken * sizeof(index_t);        \
  size_t weightBytes = config.numExpertPerToken * sizeof(float);         \
  size_t srcTokenIdBytes = sizeof(index_t);                              \
  size_t xferBytes = hiddenBytes + indexBytes + weightBytes + srcTokenIdBytes;

namespace v1 {
template <typename T>
inline __device__ void DispatchSendIntraNodeBlock(EpDispatchCombineArgs<T>& args, int tokenId,
                                                  int expId, int destPe) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (laneId == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    atomicAdd(args.destPeTokenCounter + destPe, 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * MaxNumTokensToSendPerRank + destTokId;

    core::AtomicStoreRelaxedSystem(
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destTokId,
        config.rank * config.maxNumInpTokenPerRank + tokenId);
  }
  destTokId = __shfl(destTokId, 0);

  size_t srcTokOffset = tokenId * config.hiddenDim;
  size_t destTokOffset = destTokId * config.hiddenDim;

  T* __restrict__ remoteTokenPtr = args.shmemOutTokMemObj->template GetAs<T*>(destPe);
  const T* __restrict__ localTokenPtr = args.inpTokenBuf;
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, config.hiddenDim);

  //   index_t* __restrict__ remoteIndexPtr =
  //       args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe);
  //   const index_t* __restrict__ localIndexPtr = args.tokenIndices;
  //   core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
  //                  localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  //   float* __restrict__ remoteWeightPtr = args.shmemOutWeightsMemObj->template
  //   GetAs<float*>(destPe); const float* __restrict__ localWeightPtr = args.weightsBuf;
  //   core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
  //                  localWeightPtr + tokenId * config.numExpertPerToken,
  //                  config.numExpertPerToken);
}

template <typename T>
inline __device__ void DispatchSendIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = config.rdmaBlockNum;
  int xgmiBlockNum = blockNum - config.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

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
          if (laneId == 0) args.dispDestTokIdMap[tokenExpertId] = config.MaxNumTokensToRecv();
          continue;
        }
        DispatchSendIntraNodeBlock(args, tokenId, e, destPe);
      }
    }
  }
}

template <typename T>
inline __device__ void DispatchSendInterNodeFinalize(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int finishedWarp = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(args.dispatchGridBarrier, 1);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      index_t proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      shmem::ShmemPutInt32ImmNbiThread(
          args.nodeRecvTokenNumMemObj, myNode * sizeof(index_t),
          core::AtomicLoadRelaxed(args.destNodeTokenCounter + laneId) + 1, proxyPe);
    }
    if (laneId == 0) args.dispatchGridBarrier[0] = 0;
  }
}

template <typename T>
inline __device__ void DispatchSendInterNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int tokenPerBlock = (args.curRankNumToken + config.rdmaBlockNum - 1) / config.rdmaBlockNum;
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  // First copy to staging buffer
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
    size_t stagingTokOffset = tokenId * xferBytes;
    core::WarpCopy(stagingPtr + stagingTokOffset,
                   reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
                   hiddenBytes);
    core::WarpCopy(stagingPtr + stagingTokOffset + hiddenBytes,
                   reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes,
                   indexBytes);
    if (laneId == 0)
      reinterpret_cast<index_t*>(stagingPtr + stagingTokOffset + hiddenBytes + indexBytes)[0] =
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
        shouldSend |= (destNode == i);
      }
      uint64_t mask = __ballot(shouldSend) & __activemask();
      uint64_t num = __popcll(mask);
      index_t destTokIdOffset = 0;
      if (laneId == 0) {
        destTokIdOffset = atomicAdd(args.destNodeTokenCounter + i, num);
      }
      destTokIdOffset = __shfl(destTokIdOffset, 0);

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
          shmem::ShmemPutMemNbiThread(args.shmemInpTokMemObj, remoteIdx * xferBytes,
                                      args.shmemStagingTokMemObj, stagingTokOffset,
                                      count * xferBytes, proxyPe);
        }
      }
    }
  }
}

template <typename T>
inline __device__ void DispatchInterNodeChannel(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int tokenPerBlock = (args.curRankNumToken + config.rdmaBlockNum - 1) / config.rdmaBlockNum;
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

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

  int slotsPerBlock =
      (config.maxNumInpTokenPerRank + config.rdmaBlockNum - 1) / config.rdmaBlockNum;
  int startSlotIdx = blockId * slotsPerBlock;

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    int curSlotIdx = startSlotIdx;
    for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e] /
                       config.numExpertPerRank / config.gpuPerNode;
        shouldSend |= (destNode == i);
      }
      uint64_t mask = __ballot(shouldSend) & __activemask();
      uint64_t num = __popcll(mask);
      index_t destTokIdOffset = curSlotIdx;
      curSlotIdx += num;

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
          shmem::ShmemPutMemNbiThread(args.shmemInpTokMemObj, remoteIdx * xferBytes,
                                      args.shmemStagingTokMemObj, stagingTokOffset,
                                      count * xferBytes, proxyPe);
        }
      }
    }
    shmem::ShmemPutInt32ImmNbiWarp(args.recvTokenFlagMemObj,
                                   (myNode * config.rdmaBlockNum + blockId) * sizeof(index_t),
                                   index_t{curSlotIdx - startSlotIdx + 1}, proxyPe);
  }

  uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>();
  for (int i = 0; i < nNodes; i++) {
    if (i == myNode) continue;
    index_t recvTokenNum = 0;
    if (laneId == 0) {
      recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(
          args.recvTokenFlagMemObj->template GetAs<index_t*>() +
              (i * config.rdmaBlockNum + blockId),
          0);
      // printf("mype %d block %d warp %d recv %d\n", myPe, blockId, warpId, recvTokenNum);
    }
    recvTokenNum = __shfl(recvTokenNum, 0) - 1;
    // 800 us / 1503

    for (int j = startSlotIdx + warpId; j < (startSlotIdx + recvTokenNum); j += warpNum) {
      int tokIdx = i * config.MaxNumTokensToRecvPerRank() + j;
      index_t* indicies = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
      int lanePe = -1;
      if (laneId < config.numExpertPerToken) {
        lanePe = indicies[laneId] / config.numExpertPerRank;
        assert(lanePe < config.worldSize);
      }
      index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes +
                                                    indexBytes + weightBytes)[0];

      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destPe = __shfl(lanePe, e);
        int destNode = destPe / config.gpuPerNode;
        if (destNode != myNode) continue;
        if (__any((laneId < e) && (destPe == lanePe))) {
          continue;
        }
        // 830 us / 1563
        int destTokId = 0;
        if (laneId == 0) {
          destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
          atomicAdd(args.destPeTokenCounter + destPe, 1);
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
        }
        destTokId = __shfl(destTokId, 0);
        // 950 us / 1654
        core::WarpCopy<uint8_t, 8>(
            args.shmemOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
            stagingPtr + tokIdx * xferBytes, hiddenBytes);
        // // 1137 us / 2111us
        core::WarpCopy(
            args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
        core::WarpCopy(
            args.shmemOutWeightsMemObj->template GetAs<uint8_t*>(destPe) + destTokId * weightBytes,
            stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
      }
    }
  }
}

template <typename T>
inline __device__ void DispatchInterNodeChannelOptim1(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int tokenPerBlock = (args.curRankNumToken + config.rdmaBlockNum - 1) / config.rdmaBlockNum;
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

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

  int slotsPerBlock =
      (config.maxNumInpTokenPerRank + config.rdmaBlockNum - 1) / config.rdmaBlockNum;
  int startSlotIdx = blockId * slotsPerBlock;

  // Then send to other nodes
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
    int curSlotIdx = startSlotIdx;
    for (int tokenId = startTokenIdx + laneId; tokenId < endTokenIdx; tokenId += warpSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e] /
                       config.numExpertPerRank / config.gpuPerNode;
        shouldSend |= (destNode == i);
      }
      uint64_t mask = __ballot(shouldSend) & __activemask();
      uint64_t num = __popcll(mask);
      index_t destTokIdOffset = curSlotIdx;
      curSlotIdx += num;

      uint64_t warpOffset = 0;
      if (laneId > 0) warpOffset = __popcll(mask << (warpSize - laneId));
      index_t destTokId = destTokIdOffset + warpOffset;

      uint32_t flag = 0;
      index_t flagSlotId = 0;
      uint16_t tokenNumFlag = 0;
      if (laneId == 0) {
        flagSlotId = atomicAdd(args.blockFlagCounter, 1);
        tokenNumFlag = static_cast<uint16_t>(curSlotIdx - startSlotIdx + 1);
        flag = (uint32_t(blockId) << 16) | tokenNumFlag;
      }
      flag = __shfl(flag, 0);
      flagSlotId = __shfl(flagSlotId, 0);
      tokenNumFlag = __shfl(tokenNumFlag, 0);

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

          // shmem::ShmemPutMemNbiThread(args.shmemInpTokMemObj, remoteIdx * xferBytes,
          //                             args.shmemStagingTokMemObj, stagingTokOffset,
          //                             count * xferBytes, proxyPe);

          shmem::ShmemPutMemNbiThreadKernelImpl<core::ProviderType::MLX5>(
              args.shmemInpTokMemObj, remoteIdx * xferBytes,
              args.shmemStagingTokMemObj->GetRdmaMemoryRegion(shmem::GetGlobalGpuStatesPtr()->rank),
              stagingTokOffset, count * xferBytes, args.recvTokenFlagMemObj,
              (myNode * config.rdmaBlockNum + flagSlotId) * sizeof(uint32_t), &flag,
              sizeof(uint32_t), proxyPe);
          //   printf("myPe %d block %d slot %d token %lu flag %lu\n", myPe, blockId, flagSlotId,
          //  tokenNumFlag, flag);
        }
      }
    }
    // if (laneId == 0) {
    //   index_t flagSlotId = atomicAdd(args.blockFlagCounter, 1);
    //   uint16_t tokenNumFlag = static_cast<uint16_t>(curSlotIdx - startSlotIdx + 1);
    //   uint32_t flag = (uint32_t(blockId) << 16) | tokenNumFlag;
    //   shmem::ShmemPutUint32ImmNbiThread(
    //       args.recvTokenFlagMemObj, (myNode * config.rdmaBlockNum + flagSlotId) *
    //       sizeof(uint32_t), flag, proxyPe);
    // printf("myPe %d block %d slot %d token %lu flag %lu\n", myPe, blockId, flagSlotId,
    //        tokenNumFlag, flag);
    // }
  }

  constexpr int numRecvBlock = 1;
  // if (blockId != 48) return;
  uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>();
  for (int k = blockId; k < config.rdmaBlockNum; k += (config.rdmaBlockNum)) {
    for (int i = 0; i < nNodes; i++) {
      if (i == myNode) continue;
      uint32_t recvFlag = 0;
      if (laneId == 0) {
        recvFlag = shmem::ShmemUint32WaitUntilGreaterThan(
            args.recvTokenFlagMemObj->template GetAs<uint32_t*>() + (i * config.rdmaBlockNum + k),
            0);
        // printf("flag %u\n", recvFlag);
      }
      recvFlag = __shfl(recvFlag, 0);
      uint16_t sendBlockId = recvFlag >> 16;
      uint16_t recvTokenNum = uint16_t(recvFlag) - 1;
      int curStartSlotIdx = sendBlockId * slotsPerBlock;
      // 800 us / 1503

      for (int j = curStartSlotIdx + (blockId % numRecvBlock) * warpNum + warpId;
           j < (curStartSlotIdx + recvTokenNum); j += numRecvBlock * warpNum) {
        int tokIdx = i * config.MaxNumTokensToRecvPerRank() + j;
        index_t* indicies =
            reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
        int lanePe = -1;
        if (laneId < config.numExpertPerToken) {
          lanePe = indicies[laneId] / config.numExpertPerRank;
          assert(lanePe < config.worldSize);
        }
        index_t srcTokId = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes +
                                                      hiddenBytes + indexBytes + weightBytes)[0];

        for (int e = 0; e < config.numExpertPerToken; e++) {
          int destPe = __shfl(lanePe, e);
          int destNode = destPe / config.gpuPerNode;
          if (destNode != myNode) continue;
          if (__any((laneId < e) && (destPe == lanePe))) {
            continue;
          }
          // 830 us / 1563
          int destTokId = 0;
          if (laneId == 0) {
            destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
            atomicAdd(args.destPeTokenCounter + destPe, 1);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
          }
          destTokId = __shfl(destTokId, 0);
          // 950 us / 1654
          core::WarpCopy<uint8_t, 8>(
              args.shmemOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
              stagingPtr + tokIdx * xferBytes, hiddenBytes);
          // // 1137 us / 2111us
          core::WarpCopy(
              args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) + destTokId * indexBytes,
              stagingPtr + tokIdx * xferBytes + hiddenBytes, indexBytes);
          core::WarpCopy(args.shmemOutWeightsMemObj->template GetAs<uint8_t*>(destPe) +
                             destTokId * weightBytes,
                         stagingPtr + tokIdx * xferBytes + hiddenBytes + indexBytes, weightBytes);
        }
      }
    }
  }
}

template <typename T>
inline __device__ void DispatchRecvInterNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  index_t* recvTokenFlags = args.recvTokenFlagMemObj->template GetAs<index_t*>();
  index_t* nodeRecvTokenNums = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();
  uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>();

  int curNode = -1;
  int curNodeRecvTokenNum = -1;
  for (int i = globalWarpId; i < config.MaxNumTokensToRecvPerRank() * nNodes; i += globalWarpNum) {
    int node = i / config.MaxNumTokensToRecvPerRank();
    if (node == myNode) continue;

    if ((curNode == -1) || (curNode != node)) {
      if (laneId == 0) {
        while (true) {
          index_t nodeRecvTokenNum = core::AtomicLoadRelaxedSystem(nodeRecvTokenNums + node);
          if (nodeRecvTokenNum > 0) {
            curNodeRecvTokenNum = nodeRecvTokenNum;
            break;
          }
        }
      }
      curNode = node;
      curNodeRecvTokenNum = __shfl(curNodeRecvTokenNum, 0);
    }
    int tokIdx = i - node * config.MaxNumTokensToRecvPerRank();
    bool shouldRecv = (tokIdx < (curNodeRecvTokenNum - 1));

    if (!shouldRecv) continue;

    index_t* indicies = reinterpret_cast<index_t*>(stagingPtr + i * xferBytes + hiddenBytes);
    int lanePe = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = indicies[laneId] / config.numExpertPerRank;
      assert(lanePe < config.worldSize);
    }
    index_t srcTokId =
        reinterpret_cast<index_t*>(stagingPtr + i * xferBytes + hiddenBytes + indexBytes)[0];

    for (int e = 0; e < config.numExpertPerToken; e++) {
      int destPe = __shfl(lanePe, e);
      int destNode = destPe / config.gpuPerNode;
      if (destNode != myNode) continue;
      if (__any((laneId < e) && (destPe == lanePe))) {
        continue;
      }
      int destTokId = 0;
      if (laneId == 0) {
        destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        core::AtomicStoreRelaxedSystem(
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destTokId, srcTokId);
      }
      destTokId = __shfl(destTokId, 0);
      core::WarpCopy(
          args.shmemOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
          stagingPtr + i * xferBytes, hiddenBytes);
    }
  }
}

template <typename T>
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int nodePeOffset = myNode * config.gpuPerNode;

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.combineGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == globalWarpNum) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
    if (laneId == 0) args.combineGridBarrier[0] = 0;
  }

  // Each warp wait until sender finished by waiting token number signal
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = nodePeOffset + laneId; destPe < (nodePeOffset + config.gpuPerNode);
         destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (core::WarpLaneId1D() == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }

    if (laneId < nNodes) {
      core::AtomicStoreRelaxedSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<index_t*>() + laneId, 0);
      core::AtomicStoreRelaxedSystem(args.destNodeTokenCounter + laneId, 0);
    }
  }
}
}  // namespace v1

template <typename T>
__global__ void EpDispatchInterNodeKernelV1GlobalSync(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < config.rdmaBlockNum) {
    v1::DispatchSendInterNode(args);
    v1::DispatchSendInterNodeFinalize(args);
  } else {
    v1::DispatchSendIntraNode(args);
  }
  v1::DispatchRecvInterNode(args);
  v1::DispatchSync(args);
}

template <typename T>
__global__ void EpDispatchInterNodeKernelV1(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (blockId < config.rdmaBlockNum) {
    // v1::DispatchInterNodeChannel(args);
    v1::DispatchInterNodeChannelOptim1(args);
  } else {
    v1::DispatchSendIntraNode(args);
  }
  v1::DispatchSync(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeDedupKernel(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if (globalThdId == 0) {
    args.totalRecvTokenNum[0] = 0;
    args.blockFlagCounter[0] = 0;
    for (int i = 0; i < config.worldSize; i++) shmem::ShmemQuietThread(i);
  }

  for (int i = globalThdId; i < config.rdmaBlockNum * nNodes; i += globalThdNum)
    args.recvTokenFlagMemObj->template GetAs<uint64_t*>()[i] = 0;
}

template __global__ void EpDispatchInterNodeKernelV1<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
template __global__ void EpDispatchInterNodeKernelV1<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
template __global__ void EpDispatchInterNodeKernelV1<float>(EpDispatchCombineArgs<float> args);

template __global__ void EpCombineInterNodeDedupKernel<hip_bfloat16>(
    EpDispatchCombineArgs<hip_bfloat16> args);
template __global__ void EpCombineInterNodeDedupKernel<__hip_fp8_e4m3_fnuz>(
    EpDispatchCombineArgs<__hip_fp8_e4m3_fnuz> args);
template __global__ void EpCombineInterNodeDedupKernel<float>(EpDispatchCombineArgs<float> args);

}  // namespace moe
}  // namespace mori
