#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
// This is a intra-node dispatch kernel that only incurs buffer copy once.
template <typename T>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t maxNumOutTokenPerRank = config.MaxNumTokensToSend();

  // Phase1: send token
  // Each warp compute token offset on destinition PE
  int i = globalWarpId;
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    uint32_t srcTokId = i / config.numExpertPerToken;
    uint32_t destExpert = args.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;
    uint32_t destTokId = 0;

    // Deduplicate
    assert(config.numExpertPerToken < warpSize);
    int condition = 0;
    if (laneId < (i % config.numExpertPerToken)) {
      condition = destPe == (args.tokenIndicies[srcTokId * config.numExpertPerToken + laneId] /
                             config.numExpertPerRank);
    }
    if (__any(condition)) {
      // Indicate that this token is already sent to the destination PE by setting an overflow
      // token index
      if (laneId == 0) args.dispDestTokIdMap[i] = config.worldSize * maxNumOutTokenPerRank;
      continue;
    }

    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<uint32_t*>(destPe), 1);
      atomicAdd(args.peTokenOffset + destPe, 1);
      args.dispDestTokIdMap[i] = destPe * maxNumOutTokenPerRank + destTokId;

      // TODO: use a switch to control the writing of this buffer, should only turn on for testing
      args.dispTokIdToSrcTokIdMemObj->template GetAs<uint32_t*>(destPe)[destTokId] =
          myPe * config.maxNumInpTokenPerRank + srcTokId;
    }
    destTokId = __shfl(destTokId, 0);

    // Write weights and indicies
    if (laneId < config.numExpertPerToken) {
      args.shmemWeightsMemObj->template GetAs<float*>(
          destPe)[destTokId * config.numExpertPerToken + laneId] =
          args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
      args.shmemIndiciesMemObj->template GetAs<uint32_t*>(
          destPe)[destTokId * config.numExpertPerToken + laneId] =
          args.tokenIndicies[srcTokId * config.numExpertPerToken + laneId];
    }

    // Write scales
    uint32_t destScaleOffset = destTokId * config.scaleDim * config.scaleTypeSize;
    uint32_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
    core::WarpCopy(args.shmemScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
                   args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);

    uint32_t srcTokOffset = srcTokId * config.hiddenDim;
    uint32_t destTokOffset = destTokId * config.hiddenDim;
    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>(destPe) + destTokOffset,
                   args.inpTokenBuf + srcTokOffset, config.hiddenDim);
  }
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      uint32_t numTokenSignal = core::AtomicLoadRelaxed(args.peTokenOffset + destPe) + 1;
      core::AtomicStoreRelaxedSystem(
          args.recvTokenNumMemObj->template GetAs<uint32_t*>(destPe) + myPe, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  uint32_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint32_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      uint32_t* signal = recvTokenNums + destPe;
      uint32_t recvTokenNum = shmem::ShmemUint32WaitUntilGreaterThan(signal, 0) - 1;
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
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

  size_t maxNumOutTokenPerRank = config.MaxNumTokensToSend();
  // Copy input to shmem registered buffer so that other GPUs can access directly
  size_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
    core::WarpCopy(args.shmemInpTokMemObj->template GetAs<T*>() + i * config.hiddenDim,
                   args.inpTokenBuf + i * config.hiddenDim, config.hiddenDim);
  }
  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierKernel(args);

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;

  int warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  size_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    int tokenId = i / warpsPerToken;
    int inTokenPartId = i % warpsPerToken;
    size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    size_t hiddenDimSize = std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      uint32_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      uint32_t destPe = destTokId / maxNumOutTokenPerRank;

      if (destPe < config.worldSize) {
        uint32_t destLocalTokId = destTokId - destPe * maxNumOutTokenPerRank;
        srcPtrs[j] = args.shmemInpTokMemObj->template GetAs<T*>(destPe) +
                     destLocalTokId * config.hiddenDim + hiddenDimOffset;
      } else {
        srcPtrs[j] = nullptr;
      }
    }
    core::WarpAccum(
        args.shmemOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim + hiddenDimOffset,
        srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
  }
}

}  // namespace moe
}  // namespace mori