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
#define DEF_COMMON_VARS                                             \
  const EpDispatchCombineConfig& config = args.config;              \
  int thdId = threadIdx.x;                                          \
  int thdNum = blockDim.x;                                          \
  int laneId = threadIdx.x & (warpSize - 1);                        \
  int warpId = thdId / warpSize;                                    \
  int warpNum = blockDim.x / warpSize;                              \
  int blockNum = gridDim.x;                                         \
  int blockId = blockIdx.x;                                         \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;          \
  int globalThdNum = gridDim.x * blockDim.x;                        \
  int globalWarpId = blockIdx.x * warpNum + warpId;                 \
  int globalWarpNum = gridDim.x * warpNum;                          \
  int nullTokenId = config.worldSize * config.MaxNumTokensToRecv(); \
  int myPe = config.rank;                                           \
  int npes = config.worldSize;                                      \
  int myNode = myPe / config.gpuPerNode;                            \
  int nNodes = npes / config.gpuPerNode;                            \
  int numExpertPerToken = config.numExpertPerToken;                 \
  assert(numExpertPerToken < warpSize);                             \
  size_t hiddenBytes = config.hiddenDim * sizeof(T);                \
  size_t indexBytes = config.numExpertPerToken * sizeof(index_t);   \
  size_t weightBytes = config.numExpertPerToken * sizeof(float);    \
  size_t srcTokenIdBytes = sizeof(index_t);                         \
  size_t xferBytes = hiddenBytes + indexBytes + weightBytes + srcTokenIdBytes;

namespace v1 {
template <typename T>
inline __device__ void DispatchIntraNodeBlock(EpDispatchCombineArgs<T>& args, int tokenId,
                                              int expId, int destPe) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (laneId == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    atomicAdd(args.destPeTokenCounter + destPe, 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * config.MaxNumTokensToRecv() + destTokId;
    // printf("send myPe %d tokenExpertId %lu index %lu destPe %lu destTokId %d\n", myPe,
    //        tokenExpertId, destPe * config.MaxNumTokensToRecv() + destTokId, destPe, destTokId);

    core::AtomicStoreRelaxedSystem(
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destTokId,
        config.rank * config.maxNumInpTokenPerRank + tokenId);
  }
  destTokId = __shfl(destTokId, 0);

  size_t srcTokOffset = tokenId * config.hiddenDim;
  size_t destTokOffset = destTokId * config.hiddenDim;

  T* remoteTokenPtr = args.shmemOutTokMemObj->template GetAs<T*>(destPe);
  const T* localTokenPtr = args.inpTokenBuf;
  core::WarpCopy(remoteTokenPtr + destTokOffset, localTokenPtr + srcTokOffset, config.hiddenDim);

  index_t* remoteIndexPtr = args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe);
  const index_t* localIndexPtr = args.tokenIndices;
  core::WarpCopy(remoteIndexPtr + destTokId * config.numExpertPerToken,
                 localIndexPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);

  float* remoteWeightPtr = args.shmemOutWeightsMemObj->template GetAs<float*>(destPe);
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
        DispatchIntraNodeBlock(args, tokenId, e, destPe);
      }
    }
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

      index_t flag = 0;
      index_t flagSlotId = 0;
      if (laneId == 0) {
        flagSlotId = atomicAdd(args.blockFlagCounter, 1);
        atomicAdd(args.destNodeTokenCounter + i, num);
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

          shmem::ShmemPutMemNbiThreadKernelImpl<core::ProviderType::MLX5>(
              args.shmemInpTokMemObj, remoteIdx * xferBytes,
              args.shmemStagingTokMemObj->GetRdmaMemoryRegion(shmem::GetGlobalGpuStatesPtr()->rank),
              stagingTokOffset, count * xferBytes, args.interNodeChunkFlagMemObj,
              (myNode * maxChunkNum + flagSlotId) * sizeof(index_t), &flag, sizeof(index_t),
              proxyPe);
        }
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.interNodeBlocksBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (config.rdmaBlockNum * warpNum)) {
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destNodeTokenCounter + laneId) + 1;
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

  index_t* chunkFlag = args.interNodeChunkFlagMemObj->template GetAs<index_t*>();
  index_t* nodeRecvTokenNum = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();
  uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>();

  for (int k = blockId / numRecvBlock; k < maxChunkNum; k += (config.rdmaBlockNum / numRecvBlock)) {
    for (int i = 0; i < nNodes; i++) {
      if (i == myNode) continue;

      int startTokenIdx = k * warpSize;

      // Poll completion flags
      index_t thisChunkTokenNum = 0;
      if (laneId == 0) {
        while (1) {
          thisChunkTokenNum = core::AtomicLoadRelaxedSystem(&chunkFlag[i * maxChunkNum + k]);
          if (thisChunkTokenNum > 0) break;

          index_t nodeFlag = core::AtomicLoadRelaxedSystem(&nodeRecvTokenNum[i]);
          if ((nodeFlag > 0) && (startTokenIdx >= (nodeFlag - 1))) {
            thisChunkTokenNum = 1;
            break;
          }
        }
      }
      thisChunkTokenNum = __shfl(thisChunkTokenNum, 0) - 1;

      int endTokenIdx = startTokenIdx + thisChunkTokenNum;

      for (int j = startTokenIdx + (blockId % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
           j += numRecvBlock * warpNum) {
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
          int destTokId = 0;
          if (laneId == 0) {
            destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
            atomicAdd(args.destPeTokenCounter + destPe, 1);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
          }
          destTokId = __shfl(destTokId, 0);
          core::WarpCopy<uint8_t, 8>(
              args.shmemOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
              stagingPtr + tokIdx * xferBytes, hiddenBytes);
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
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int nodePeOffset = myNode * config.gpuPerNode;

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.dispatchGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == globalWarpNum) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
    if (laneId == 0) args.dispatchGridBarrier[0] = 0;
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
inline __device__ void CombineIntraNodeSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int blockOffset = config.rdmaBlockNum;
  int xgmiBlockNum = blockNum - config.rdmaBlockNum;

  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  int tokenPerBlock = core::CeilDiv(totalRecvTokenNum, xgmiBlockNum);
  int startTokenIdx = (blockId - blockOffset) * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, totalRecvTokenNum);
  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    core::WarpCopy(args.shmemInpTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim,
                   args.inpTokenBuf + tokenId * config.hiddenDim, config.hiddenDim);
  }

  // After all warps copy done, set barrier flag
  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.combineGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == (xgmiBlockNum * warpNum)) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(destPe) + args.config.rank,
          args.crossDeviceBarrierFlag);
    }
    if (laneId == 0) args.combineGridBarrier[0] = 0;
  }
  // Wait other pes to set flag
  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + destPe) != args.crossDeviceBarrierFlag) {
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
        // printf("recv myPe %d tokenExpertId %lu index %lu destPe %lu destTokId %lu\n", myPe,
        //        tokenId * config.numExpertPerToken + laneId, destTokId, destPe, destLocalTokId);
        srcPtrs[laneId] =
            args.shmemInpTokMemObj->template GetAs<T*>(destPe) + destLocalTokId * config.hiddenDim;
        srcWeightsPtr[laneId] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                                destLocalTokId * config.numExpertPerToken;
      }
    }
    core::WarpAccum<T, 8>(args.shmemOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim,
                          srcPtrs, nullptr, config.numExpertPerToken, config.hiddenDim);
  }
}
}  // namespace v1

template <typename T>
__global__ void EpCombineInterNodeV1Kernel(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  if (blockId < config.rdmaBlockNum) {
  } else {
    // First copy into shmem registered buffer and sync
    v1::CombineIntraNodeSync(args);
    // Then combine
    v1::CombineIntraNode(args);
    if (blockId == config.rdmaBlockNum) {
      if (laneId == 0) {
        if (globalThdId == 0) {
          args.totalRecvTokenNum[0] = 0;
          args.blockFlagCounter[0] = 0;
          for (int i = 0; i < config.worldSize; i++) shmem::ShmemQuietThread(i);
        }
      }
    }
  }

  // TODO: refactor following state reset code
  if (globalThdId < nNodes)
    args.nodeRecvTokenNumMemObj->template GetAs<index_t*>()[globalThdId] = 0;

  int maxChunkNum = core::CeilDiv(config.maxNumInpTokenPerRank, warpSize);
  for (int i = globalThdId; i < (config.maxNumInpTokenPerRank * nNodes); i += globalThdNum)
    args.interNodeChunkFlagMemObj->template GetAs<index_t*>()[i] = 0;
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
