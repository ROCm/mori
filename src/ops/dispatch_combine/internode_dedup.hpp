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

/* ---------------------------------------------------------------------------------------------- */
/*                                 EpDispatchInterNodeDedupKernel                                 */
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
  size_t xferBytes = hiddenBytes + indexBytes;

namespace dedup {
template <typename T>
inline __device__ void DispatchSendIntraNode(EpDispatchCombineArgs<T>& args, int tokenId,
                                             int destPe, int lanePe, int expId) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (core::WarpLaneId1D() == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    atomicAdd(args.destPeTokenCounter + destPe, 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * MaxNumTokensToSendPerRank + destTokId;

    // TODO: use a switch to control the writing of this buffer, should only turn on for testing
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
        config.rank * config.maxNumInpTokenPerRank + tokenId;
  }
  destTokId = __shfl(destTokId, 0);

  size_t srcTokOffset = tokenId * config.hiddenDim;
  size_t destTokOffset = destTokId * config.hiddenDim;
  core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>(destPe) + destTokOffset,
                 args.inpTokenBuf + srcTokOffset, config.hiddenDim);
}

template <typename T>
inline __device__ void DispatchSendInterNode(EpDispatchCombineArgs<T>& args, int tokenId,
                                             int destNode, int laneNode, int expId) {
  DEF_COMMON_VARS;
  index_t proxyPe = (config.rank % config.gpuPerNode) + destNode * config.gpuPerNode;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (core::WarpLaneId1D() == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.destNodeTokenCounter + destNode, 1);
    args.dispDestTokIdMap[tokenExpertId] = proxyPe * config.MaxNumTokensToSendPerRank() + destTokId;
  }
  destTokId = __shfl(destTokId, 0);

  // Copy to local staging buffer
  uint8_t* stagingPtr = args.shmemStagingTokMemObj->template GetAs<uint8_t*>();
  size_t stagingTokOffset = (destNode * config.MaxNumTokensToSendPerRank() + destTokId) * xferBytes;
  core::WarpCopy(stagingPtr + stagingTokOffset,
                 reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes, hiddenBytes);
  core::WarpCopy(stagingPtr + stagingTokOffset + hiddenBytes,
                 reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes, indexBytes);

  // Copy to remote proxy's staging buffer
  size_t remoteIdx = (myNode * config.MaxNumTokensToRecvPerRank() + destTokId);
  size_t remoteOff = remoteIdx * xferBytes;

  shmem::ShmemPutMemNbiWarp(args.shmemInpTokMemObj, remoteIdx * xferBytes,
                            args.shmemStagingTokMemObj, stagingTokOffset, xferBytes, proxyPe);
  shmem::ShmemPutInt32ImmNbiWarp(args.recvTokenFlagMemObj, remoteIdx * sizeof(index_t), index_t{1},
                                 proxyPe);
  // if (laneId == 0)
  // printf("mype %d set remote recv token flag idx %lu proxy pe %d\n", myPe, remoteIdx, proxyPe);
}

template <typename T>
inline __device__ void DispatchSendPhase(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  // Distribute tokens evenly to all blocks
  int tokenPerBlock = (args.curRankNumToken + blockNum - 1) / blockNum;
  int startTokenIdx = blockId * tokenPerBlock;
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
      int destExpert = args.tokenIndices[tokenExpertId];
      int destPe = destExpert / config.numExpertPerRank;
      int destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        if (__any((laneId < e) && (destPe == lanePe))) {
          if (laneId == 0) args.dispDestTokIdMap[tokenExpertId] = config.MaxNumTokensToRecv();
          continue;
        }
        DispatchSendIntraNode(args, tokenId, destPe, lanePe, e);
      } else {
        if (__any((laneId < e) && (destNode == laneNode))) {
          if (laneId == 0) args.dispDestTokIdMap[tokenExpertId] = config.MaxNumTokensToRecv();
          continue;
        }
        DispatchSendInterNode(args, tokenId, destNode, laneNode, e);
      }
    }
  }

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.dispatchGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);
  if ((finishedWarp + 1) == globalWarpNum) {
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
inline __device__ void DispatchRecvPhaseProdConsm(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  constexpr int flagNum = 512;
  __shared__ int tasks[flagNum];
  __shared__ uint32_t slotIdx;
  __shared__ uint32_t producer;
  __shared__ uint32_t consumer;
  __shared__ uint32_t blockFinished;
  for (int i = thdId * 4; i < flagNum; i += thdNum * 4) {
    reinterpret_cast<uint4*>(tasks + i)[0] = {-1, -1, -1, -1};
  }
  if (thdId == 0) {
    slotIdx = -1;
    producer = -1;
    consumer = -1;
    blockFinished = 0;
  }
  __syncthreads();

  // Warp-0 polling on recv token flags and producer tasks
  if (warpId == 0) {
    int totalFlags = nNodes * MaxNumTokensToRecvPerRank;
    index_t* recvTokenFlags = args.recvTokenFlagMemObj->template GetAs<index_t*>();
    for (int i = blockId * warpSize + laneId; i < totalFlags; i += config.blockNum * warpSize) {
      int node = i / MaxNumTokensToRecvPerRank;
      int tokenIdx = i % MaxNumTokensToRecvPerRank;
      bool finished = (node == myNode);
      while (true) {
        index_t nodeRecvTokenNum =
            core::AtomicLoadRelaxed(args.nodeRecvTokenNumMemObj->template GetAs<index_t*>() + node);
        finished |= (nodeRecvTokenNum > 0) && (tokenIdx > (nodeRecvTokenNum - 2));
        if (!finished && core::AtomicLoadRelaxed(recvTokenFlags + i)) {
          finished = true;
          uint32_t mySlot = atomicAdd(&slotIdx, 1);
          core::AtomicStoreRelaxed(tasks + mySlot, i);
          __threadfence();
          int curProd = atomicAdd(&producer, 1);
          printf("recv flag index %d task val %d cur Prod %d\n", i, tasks[mySlot], curProd);
        }
        if (__all(finished)) break;
      }
    }
    core::AtomicStoreRelaxed(&blockFinished, uint32_t{1});
    // if (laneId < nNodes) {
    //   printf("rank %d recv %d tokens from node %d\n", myPe,
    //          args.nodeRecvTokenNumMemObj->template GetAs<index_t*>()[laneId], laneId);
    // }
  }

  // Other warps consume tasks
  while (true) {
    int taskId = -1;
    // Try fetch task with lane-0
    if (laneId == 0) {
      if (core::AtomicLoadRelaxed(&blockFinished)) break;
      int curConsumer = core::AtomicLoadRelaxed(&consumer);
      int curProducer = core::AtomicLoadRelaxed(&producer);
      taskId = curConsumer + 1;
      if ((curConsumer < curProducer) && (curProducer >= 0)) {
        if (atomicCAS(&consumer, curConsumer, taskId) != curConsumer) continue;  // fetch failed
      }
      // printf("cosm %d prod %d task %d\n", curConsumer, curProducer, taskId);
    }
    taskId = __shfl(taskId, 0);

    // Execute task
    int tokenId = core::AtomicLoadRelaxed(tasks + taskId);
    if (laneId == 0) {
      // printf("tokaneId %d\n", tokenId);
    }
    // uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>() + tokenId *
    // xferBytes; index_t* indicies = reinterpret_cast<index_t*>(stagingPtr + hiddenBytes);

    // int lanePe = -1;
    // if (laneId < numExpertPerToken){
    //   lanePe = indicies[tokenId * config.numExpertPerToken + laneId] / config.numExpertPerRank;
    //   printf("lanePE %d\n", lanePe);
    // }
    // for (int i = 0; i < config.numExpertPerToken; i++) {
    //   int destExpt = indicies[tokenId * config.numExpertPerToken + i];
    //   int destPe = destExpt / config.gpuPerNode;
    //   if (__any((laneId < i) && (destPe == lanePe))) {
    //     continue;
    //   }
    //   int destTokId = 0;
    //   if (laneId == 0) {
    //     destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    //     atomicAdd(args.destPeTokenCounter + destPe, 1);
    //     // TODO: use a switch to control the writing of this buffer, should only turn on for
    //     testing
    //   }
    //   destTokId = __shfl(destTokId, 0);
    //   core::WarpCopy(stagingPtr,
    //                  reinterpret_cast<uint8_t*>(args.inpTokenBuf) + tokenId * hiddenBytes,
    //                  hiddenBytes);
    // }
  }
}

template <typename T>
inline __device__ void DispatchRecvPhase(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  index_t* recvTokenFlags = args.recvTokenFlagMemObj->template GetAs<index_t*>();
  index_t* nodeRecvTokenNums = args.nodeRecvTokenNumMemObj->template GetAs<index_t*>();
  uint8_t* stagingPtr = args.shmemInpTokMemObj->template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < config.MaxNumTokensToRecvPerRank() * nNodes; i += globalWarpNum) {
    int node = i / config.MaxNumTokensToRecvPerRank();
    if (node == myNode) continue;

    bool shouldRecv = false;
    if (laneId == 0) {
      int tokIdx = i - node * config.MaxNumTokensToRecvPerRank();
      while (true) {
        index_t nodeRecvTokenNum = core::AtomicLoadRelaxed(nodeRecvTokenNums + node);
        if ((nodeRecvTokenNum > 0) && (tokIdx >= (nodeRecvTokenNum - 1))) {
          break;
        }
        if (core::AtomicLoadRelaxed(recvTokenFlags + i) != 0) {
          shouldRecv = true;
          break;
        }
      }
    }
    shouldRecv = __shfl(shouldRecv, 0);

    // if ((laneId == 0)) printf("myPe %d should recv token %d %d\n", myPe, i, shouldRecv);
    if (!shouldRecv) continue;

    index_t* indicies = reinterpret_cast<index_t*>(stagingPtr + i * xferBytes + hiddenBytes);
    int lanePe = -1;
    if (laneId < config.numExpertPerToken) {
      lanePe = indicies[laneId] / config.numExpertPerRank;
      assert(lanePe < config.worldSize);
    }
    // if ((laneId == 0)) {
    //   printf("myPe %d remote tok %d expert %d %d %d %d %d %d %d %d\n", myPe, i, indicies[0],
    //          indicies[1], indicies[2], indicies[3], indicies[4], indicies[5], indicies[6],
    //          indicies[7]);
    // }
    for (int e = 0; e < config.numExpertPerToken; e++) {
      int destExpt = indicies[e];
      int destPe = destExpt / config.numExpertPerRank;
      int destNode = destPe / config.gpuPerNode;
      if (destNode != myNode) continue;
      if (__any((laneId < e) && (destPe == lanePe))) {
        continue;
      }
      int destTokId = 0;
      if (laneId == 0) {
        destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
        atomicAdd(args.destPeTokenCounter + destPe, 1);
      }
      // destTokId = __shfl(destTokId, 0);
      // core::WarpCopy(
      //     args.shmemOutTokMemObj->template GetAs<uint8_t*>(destPe) + destTokId * hiddenBytes,
      //     stagingPtr + i * xferBytes, hiddenBytes);
    }
  }
}

template <typename T>
inline __device__ void DispatchSyncIntraNode(EpDispatchCombineArgs<T>& args) {
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
  }
}
}  // namespace dedup

template <typename T>
__global__ void EpDispatchInterNodeDedupKernel(EpDispatchCombineArgs<T> args) {
  dedup::DispatchSendPhase(args);
  dedup::DispatchRecvPhase(args);
  dedup::DispatchSyncIntraNode(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
// template <typename T>
// inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args) {}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeDedupKernel(EpDispatchCombineArgs<T> args) {
  args.totalRecvTokenNum[0] = 0;
}

}  // namespace moe
}  // namespace mori
