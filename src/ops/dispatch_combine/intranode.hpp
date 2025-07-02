#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(globalThdId) + args.config.rank,
        args.crossDeviceBarrierFlag);
  }

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != args.crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
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
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    index_t srcTokId = i / config.numExpertPerToken;
    index_t destExpert = args.tokenIndices[i];
    index_t destPe = destExpert / config.numExpertPerRank;
    index_t destTokId = 0;

    // Deduplicate
    assert(config.numExpertPerToken < warpSize);
    int condition = 0;
    if (laneId < (i % config.numExpertPerToken)) {
      condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
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
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
      atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = destPe * maxNumOutTokenPerRank + destTokId;

      // TODO: use a switch to control the writing of this buffer, should only turn on for testing
      args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
          myPe * config.maxNumInpTokenPerRank + srcTokId;
    }
    destTokId = __shfl(destTokId, 0);

    // Write weights and indices
    if (laneId < config.numExpertPerToken) {
      args.shmemOutWeightsMemObj->template GetAs<float*>(
          destPe)[destTokId * config.numExpertPerToken + laneId] =
          args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
      args.shmemOutIndicesMemObj->template GetAs<index_t*>(
          destPe)[destTokId * config.numExpertPerToken + laneId] =
          args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
    }

    // Write scales
    if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
      index_t destScaleOffset = destTokId * config.scaleDim * config.scaleTypeSize;
      index_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
      core::WarpCopy(args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
                     args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
    }

    index_t srcTokOffset = srcTokId * config.hiddenDim;
    index_t destTokOffset = destTokId * config.hiddenDim;
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
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      core::AtomicStoreRelaxedSystem(
          args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
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
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  if (args.config.useExternalInpBuffer) {
    for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
      core::WarpCopy(args.shmemInpTokMemObj->template GetAs<T*>() + i * config.hiddenDim,
                     args.inpTokenBuf + i * config.hiddenDim, config.hiddenDim);
    }
  }
  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierIntraNodeKernel(args);
  if (args.curRankNumToken == 0) return;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = destTokId / maxNumOutTokenPerRank;

      if (destPe < config.worldSize) {
        index_t destLocalTokId = destTokId - destPe * maxNumOutTokenPerRank;
        srcPtrs[j] = args.shmemInpTokMemObj->template GetAs<T*>(destPe) +
                     destLocalTokId * config.hiddenDim + hiddenDimOffset;
      } else {
        srcPtrs[j] = nullptr;
      }
    }
    core::WarpAccum<T, 8>(
        args.shmemOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim + hiddenDimOffset,
        srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);
  }
}

}  // namespace moe
}  // namespace mori