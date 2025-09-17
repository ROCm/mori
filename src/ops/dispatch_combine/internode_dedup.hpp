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

/*                                 EpDispatchInterNodeDedupKernel                                 */
/* ---------------------------------------------------------------------------------------------- */
#define IMPORT_COMMON_VARIABLES                                          \
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
  int myNode = myPe / MAX_GPUS_PER_NODE;                                 \
  int nNodes = npes / MAX_GPUS_PER_NODE;                                 \
  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank(); \
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank(); \
  size_t MaxNumTokensToRecv = config.MaxNumTokensToRecv();               \
  int numExpertPerToken = config.numExpertPerToken;                      \
  assert(numExpertPerToken < warpSize);

namespace dedup {
template <typename T>
inline __device__ void DispatchSendIntraNode(EpDispatchCombineArgs<T>& args, int tokenId,
                                             int destPe, int lanePe, int expId) {
  const EpDispatchCombineConfig& config = args.config;
  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + destPe;
  int condition = 0;
  if ((core::WarpLaneId1D() < expId) && (destPe == lanePe)) {
    condition = 1;
  }
  if (__any(condition)) {  // Duplicated, skip sending
    if (core::WarpLaneId1D() == 0)
      args.dispDestTokIdMap[tokenExpertId] = config.MaxNumTokensToRecv();
    return;
  }

  index_t destTokId = 0;
  if (core::WarpLaneId1D() == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
    atomicAdd(args.destPeTokenCounter + destPe, 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * config.MaxNumTokensToSendPerRank() + destTokId;

    // TODO: use a switch to control the writing of this buffer, should only turn on for testing
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
        config.rank * config.maxNumInpTokenPerRank + tokenId;
  }
  destTokId = __shfl(destTokId, 0);

  index_t srcTokOffset = tokenId * config.hiddenDim;
  index_t destTokOffset = destTokId * config.hiddenDim;
  core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>(destPe) + destTokOffset,
                 args.inpTokenBuf + srcTokOffset, config.hiddenDim);
}

template <typename T>
inline __device__ void DispatchSendInterNode(EpDispatchCombineArgs<T>& args, int tokenId,
                                             int destNode, int laneNode, int expId) {
  IMPORT_COMMON_VARIABLES;

  index_t tokenExpertId = tokenId * args.config.numExpertPerToken + destPe;
  int condition = 0;
  if ((core::WarpLaneId1D() < expId) && (destNode == laneNode)) {
    condition = 1;
  }
  if (__any(condition)) {  // Duplicated, skip sending
    if (core::WarpLaneId1D() == 0)
      args.dispDestTokIdMap[tokenExpertId] = config.MaxNumTokensToRecv();
    return;
  }

  index_t destTokId = 0;
  index_t destNodeIdx = destNode + MAX_GPUS_PER_NODE;
  if (core::WarpLaneId1D() == 0) {
    // decide token id in dest pe
    destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destNodeIdx), 1);
    atomicAdd(args.destPeTokenCounter + destNodeIdx, 1);
    args.dispDestTokIdMap[tokenExpertId] = destPe * config.MaxNumTokensToSendPerRank() + destTokId;
  }
  destTokId = __shfl(destTokId, 0);

  index_t srcTokOffset = tokenId * config.hiddenDim;
  index_t stagingTokOffset =
      (destNodeIdx * config.MaxNumTokensToSendPerRank() + destTokId) * config.hiddenDim;
  core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<T*>(destNodeIdx) + stagingTokOffset,
                 args.inpTokenBuf + srcTokOffset, config.hiddenDim);
  index_t remoteOffset = stagingTokOffset + (MAX_GPUS_PER_NODE + myNode) *
                                                config.MaxNumTokensToRecvPerRank() *
                                                config.hiddenDim;
  shmem::ShmemPutTypeNbiWarp<T>(args.shmemInpTokMemObj, remoteOffset, args.shmemStagingTokMemObj,
                                stagingTokOffset, config.hiddenDim, destPe);
}

template <typename T>
inline __device__ void DispatchSendPhase(EpDispatchCombineArgs<T>& args) {
  IMPORT_COMMON_VARIABLES;

  // Distribute tokens evenly to all blocks
  int tokenPerBlock = (args.curRankNumToken + blockNum - 1) / blockNum;
  int startTokenIdx = blockId * tokenPerBlock;
  int endTokenIdx = std::min(startTokenIdx + tokenPerBlock, args.curRankNumToken);

  for (int tokenId = startTokenIdx + warpId; tokenId < endTokenIdx; tokenId += warpNum) {
    int lanePe = -1, laneNode = -1;
    if (laneId < numExpertPerToken) {
      lanePe = (args.tokenIndices[tokenId * numExpertPerToken + laneId] / config.numExpertPerRank);
      laneNode = lanePe / MAX_GPUS_PER_NODE;
    };

    // Send to other pes in myNode
    for (int e = 0; e < config.numExpertPerToken; e++) {
      int destExpert = args.tokenIndices[tokenId * numExpertPerToken + e];
      int destPe = destExpert / config.numExpertPerRank;
      int destNode = destPe / MAX_GPUS_PER_NODE;
      if (destNode == myNode)
        dedup::DispatchSendIntraNode(args, tokenId, destPe, lanePe, e);
      else
        dedup::DispatchSendInterNode(args, tokenId, destNode, laneNode, e);
    }
  }
}

template <typename T>
inline __device__ void DispatchRecvPhase(EpDispatchCombineArgs<T>& args) {}

template <typename T>
inline __device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  IMPORT_COMMON_VARIABLES;

  if (core::WarpLaneId1D() == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Each warp wait until sender finished by waiting token number signal
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
  dedup::DispatchSync(args);
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
