#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

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

  size_t maxNumOutTokenPerRank =
      config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken;

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
    if (__any(condition)) continue;

    if (laneId == 0) {
      // decide token id in dest pe
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<uint32_t*>(destPe), 1);
      args.dispTokIdToSrcTokIdMemObj->template GetAs<uint32_t*>(destPe)[destTokId] =
          myPe * config.maxNumInpTokenPerRank + srcTokId;
      args.outTokToExptMapMemObj->template GetAs<uint32_t*>(destPe)[destTokId] = destExpert;
      atomicAdd(args.peTokenOffset + destPe, 1);
      args.dispDestTokIdMap[i] = destPe * maxNumOutTokenPerRank + destTokId;
    }
    destTokId = __shfl(destTokId, 0);

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
      shmem::ShmemPutUint32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                        numTokenSignal, destPe);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  uint32_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<uint32_t*>();
  if (globalWarpId == 0) {
    uint32_t totalRecvTokenNum = 0;
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      uint32_t* signal = recvTokenNums + destPe;
      totalRecvTokenNum = shmem::ShmemUint32WaitUntilGreaterThan(signal, 0) - 1;
      atomicAdd(args.totalRecvTokenNum, totalRecvTokenNum);
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

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t maxNumOutTokenPerRank =
      config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken;

  for (int i = globalWarpId; i < *args.totalRecvTokenNum; i += globalWarpNum) {
    core::WarpCopy(args.shmemInpTokMemObj->template GetAs<T*>() + i * config.hiddenDim,
                   args.inpTokenBuf + i * config.hiddenDim, config.hiddenDim);
  }
  CrossDeviceBarrierKernel(args);

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;

  int warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  int hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  T* outTokenBuf = args.outTokenBuf;
  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    int tokenId = i / warpsPerToken;
    int inTokenPartId = i % warpsPerToken;
    int hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    int hiddenDimSize = std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      uint32_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];

      uint32_t destPe = destTokId / maxNumOutTokenPerRank;
      uint32_t destLocalTokId = destTokId - destPe * maxNumOutTokenPerRank;

      srcPtrs[j] = args.shmemInpTokMemObj->template GetAs<T*>(destPe) +
                   destLocalTokId * config.hiddenDim + hiddenDimOffset;
    }
    core::WarpAccum(args.outTokenBuf + tokenId * config.hiddenDim + hiddenDimOffset, srcPtrs,
                    args.weightsBuf + tokenId * config.numExpertPerToken, config.numExpertPerToken,
                    hiddenDimSize);
  }
}

}  // namespace moe
}  // namespace mori