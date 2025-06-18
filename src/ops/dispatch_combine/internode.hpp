#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

#define DEBUG 1

__device__ void SyncIfDebugEnabled(const char* msg) {
#if DEBUG == 1
  __syncthreads();
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    shmem::ShmemQuietThread();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchInterNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchInterNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  T* inpTokenBuf = args.shmemInpTokMemObj->template GetAs<T*>();
  uint32_t* outTokToExptBuf = args.outTokToExptMapMemObj->template GetAs<uint32_t*>();

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();

  // Send out tokens
  extern __shared__ char sharedMem[];

  // Phase1: send token
  int i = globalWarpId;
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    uint32_t destExpert = args.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;

    uint32_t peTokenIdx = 0;
    if (laneId == 0) {
      peTokenIdx = atomicAdd(args.peTokenOffset + destPe, 1);
      args.tokenIndicesToPeSortedBuf[i] = destPe * MaxNumTokensToSendPerRank + peTokenIdx;
    }
    peTokenIdx = __shfl(peTokenIdx, 0);
    uint32_t tokenId = i / config.numExpertPerToken;
    uint32_t tokenOffset = tokenId * config.hiddenDim;

    // uint32_t peSortedId = myPe * MaxNumTokensToSendPerRank + peTokenIdx;
    // uint32_t peSortedOffset = peSortedId * config.hiddenDim;

    // core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() + tokenOffset,
    //  args.inpTokenBuf + tokenOffset, config.hiddenDim);
    // if (laneId == 0) core::AcquireLock(args.lock);
    // shmem::ShmemPutTypeNbiWarp<T>(args.shmemInpTokMemObj, peSortedOffset, args.shmemOutTokMemObj,
    //                               tokenOffset, config.hiddenDim, destPe);
    // if (laneId == 0) core::ReleaseLock(args.lock);

    // uint32_t localPeSortedOffset = (destPe * MaxNumTokensToSendPerRank + peTokenIdx) *
    // config.hiddenDim; uint32_t remotePeSortedOffset = (myPe * MaxNumTokensToSendPerRank +
    // peTokenIdx) * config.hiddenDim; core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() +
    // localPeSortedOffset,
    //                args.inpTokenBuf + tokenOffset, config.hiddenDim);
    // if (laneId == 0) core::AcquireLock(args.lock);
    // shmem::ShmemPutTypeNbiWarp<T>(args.shmemInpTokMemObj, remotePeSortedOffset,
    // args.shmemOutTokMemObj,
    //                               localPeSortedOffset, config.hiddenDim, destPe);
    // if (laneId == 0) core::ReleaseLock(args.lock);

    uint32_t peSortedId = destPe * MaxNumTokensToSendPerRank + peTokenIdx;
    uint32_t peSortedOffset = peSortedId * config.hiddenDim;
    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() + peSortedOffset,
                   args.inpTokenBuf + tokenOffset, config.hiddenDim);
  }
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);
  SyncIfDebugEnabled("Dispatch kernel: finished send token");

  // Send token num & token to expert mapping to other ranks
  // TODO: we use multiple warps here because using multiple threads in warp 0 lost RDMA writes,
  // don't know why yet
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    if (laneId == 0) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      uint32_t numToken = core::AtomicLoadRelaxed(args.peTokenOffset + destPe);
      uint32_t localPeSortedOffset = destPe * MaxNumTokensToSendPerRank * config.hiddenDim;
      uint32_t remotePeSortedOffset = myPe * MaxNumTokensToSendPerRank * config.hiddenDim;
      shmem::ShmemPutTypeNbiThread<T>(args.shmemInpTokMemObj, remotePeSortedOffset,
                                      args.shmemOutTokMemObj, localPeSortedOffset,
                                      config.hiddenDim * numToken, destPe);
      shmem::ShmemPutUint32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                        numToken + 1, destPe);
      printf("myPe %d send %d tokens to %d\n", myPe, numToken, destPe);
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: finish sending tok2expt mapping & num token signal");

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  uint32_t* recvTokenNum = reinterpret_cast<uint32_t*>(sharedMem) + warpId * npes;
  // TODO: too many blocks han here, debug it
  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = args.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    shmem::ShmemUint32WaitUntilGreaterThan(signal, 0);
    recvTokenNum[destPe] = core::AtomicLoadRelaxedSystem(signal) - 1;
  }
  SyncIfDebugEnabled("Dispatch kernel: finish waiting num token signal");

  for (int i = globalWarpId;; i += globalWarpNum) {
    // find src pe and tok id
    uint32_t srcPe = 0;
    uint32_t accumPeTokOffset = 0;
    for (; srcPe < npes; srcPe++) {
      if ((i >= accumPeTokOffset) && (i < (accumPeTokOffset + recvTokenNum[srcPe]))) break;
      accumPeTokOffset += recvTokenNum[srcPe];
    }
    if (srcPe >= npes) break;

    uint32_t recvTokenOffset = 0;
    if (laneId == 0) {
      recvTokenOffset = atomicAdd(args.exptTokenOffset, 1);
    }
    recvTokenOffset = __shfl(recvTokenOffset, 0);

    // Copy token
    uint32_t peSortedId = srcPe * MaxNumTokensToSendPerRank + i - accumPeTokOffset;
    uint32_t srcTokenOff = peSortedId * config.hiddenDim;

    uint32_t destTokenOff = recvTokenOffset * config.hiddenDim;
    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() + destTokenOff,
                   args.shmemInpTokMemObj->template GetAs<T*>() + srcTokenOff, config.hiddenDim);

    if (laneId == 0) {
      args.exptSortedToPeSortedBuf[recvTokenOffset] = peSortedId;
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernel(EpDispatchCombineArgs<T> args) {}

}  // namespace moe
}  // namespace mori
