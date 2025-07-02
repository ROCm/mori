#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

#define DEBUG 0

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
// TODO: this mode only works correctly with MORI_DISABLE_P2P=1 set, figure out why
#define ENABLE_RDMA_AGGREGATE_WRITE 0

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

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  extern __shared__ char sharedMem[];

  // Phase1: send token
  int i = globalWarpId;
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    index_t destExpert = args.tokenIndices[i];
    index_t destPe = destExpert / config.numExpertPerRank;
    index_t tokenId = i / config.numExpertPerToken;

    // Deduplicate
    assert(config.numExpertPerToken < warpSize);
    int condition = 0;
    if (laneId < (i % config.numExpertPerToken)) {
      condition = destPe == (args.tokenIndices[tokenId * config.numExpertPerToken + laneId] /
                             config.numExpertPerRank);
    }
    if (__any(condition)) {
      // Indicate that this token is already sent to the destination PE by setting an overflowed
      // token index
      if (laneId == 0) args.dispSenderIdxMap[i] = config.worldSize * MaxNumTokensToRecvPerRank;
      continue;
    }

    // Allocate a remote token slot
    index_t destPeTokenIdx = 0, peSortedIdx = 0;
    if (laneId == 0) {
      destPeTokenIdx = atomicAdd(args.destPeTokenCounter + destPe, 1);
      peSortedIdx = destPe * MaxNumTokensToRecvPerRank + destPeTokenIdx;
      args.dispSenderIdxMap[i] = peSortedIdx;
    }
    destPeTokenIdx = __shfl(destPeTokenIdx, 0);
    peSortedIdx = __shfl(peSortedIdx, 0);

    index_t tokenOffset = tokenId * config.hiddenDim;

#if ENABLE_RDMA_AGGREGATE_WRITE == 1
    // For aggregated write, first copy data into a contiguous buffer
    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() + peSortedIdx * config.hiddenDim,
                   args.inpTokenBuf + tokenOffset, config.hiddenDim);
    core::WarpCopy(args.shmemOutWeightsMemObj->template GetAs<float*>() +
                       peSortedIdx * config.numExpertPerToken,
                   args.weightsBuf + tokenId * config.numExpertPerToken, config.numExpertPerToken);
    core::WarpCopy(args.shmemOutIndicesMemObj->template GetAs<index_t*>() +
                       peSortedIdx * config.numExpertPerToken,
                   args.tokenIndices + tokenId * config.numExpertPerToken,
                   config.numExpertPerToken);
#else
    // For disaggregated write, write data to remote directly
    // TODO: copy weight and indicies
    index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + destPeTokenIdx;
    index_t peSortedOffset = peSortedId * config.hiddenDim;

    core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<T*>() + tokenOffset,
                   args.inpTokenBuf + tokenOffset, config.hiddenDim);
    shmem::ShmemPutTypeNbiWarp<T>(args.shmemInpTokMemObj, peSortedOffset,
                                  args.shmemStagingTokMemObj, tokenOffset, config.hiddenDim,
                                  destPe);

#endif
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
      index_t numToken = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
#if ENABLE_RDMA_AGGREGATE_WRITE == 1
      index_t localPeSortedOffset = destPe * MaxNumTokensToRecvPerRank;
      index_t remotePeSortedOffset = myPe * MaxNumTokensToRecvPerRank;
      shmem::ShmemPutTypeNbiThread<T>(
          args.shmemInpTokMemObj, remotePeSortedOffset * config.hiddenDim, args.shmemOutTokMemObj,
          localPeSortedOffset * config.hiddenDim, config.hiddenDim * numToken, destPe);
      shmem::ShmemPutTypeNbiThread<float>(
          args.shmemInpWeightsMemObj, remotePeSortedOffset * config.numExpertPerToken,
          args.shmemOutWeightsMemObj, localPeSortedOffset * config.numExpertPerToken,
          config.numExpertPerToken * numToken, destPe);
      shmem::ShmemPutTypeNbiThread<index_t>(
          args.shmemInpIndicesMemObj, remotePeSortedOffset * config.numExpertPerToken,
          args.shmemOutIndicesMemObj, localPeSortedOffset * config.numExpertPerToken,
          config.numExpertPerToken * numToken, destPe);
#endif
      shmem::ShmemPutInt32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(index_t),
                                       numToken + 1, destPe);
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: finish sending tok2expt mapping & num token signal");

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  index_t* recvTokenNumArr = reinterpret_cast<index_t*>(sharedMem) + warpId * npes;
  // TODO: kernel hangs here when launch too many blocks
  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>() + destPe;
    shmem::ShmemInt32WaitUntilGreaterThan(signal, 0);
    index_t recvTokenNum = core::AtomicLoadRelaxedSystem(signal) - 1;
    recvTokenNumArr[destPe] = recvTokenNum;
    if (globalWarpId == 0) atomicAdd(args.totalRecvTokenNum, recvTokenNum);
  }
  SyncIfDebugEnabled("Dispatch kernel: finish waiting num token signal");

  for (int i = globalWarpId;; i += globalWarpNum) {
    // find src pe and tok id
    index_t srcPe = 0;
    index_t accumPeTokOffset = 0;
    for (; srcPe < npes; srcPe++) {
      if ((i >= accumPeTokOffset) && (i < (accumPeTokOffset + recvTokenNumArr[srcPe]))) break;
      accumPeTokOffset += recvTokenNumArr[srcPe];
    }
    if (srcPe >= npes) break;

    index_t localTokenIdx = 0;
    if (laneId == 0) {
      localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);
    }
    localTokenIdx = __shfl(localTokenIdx, 0);

    // Copy token
    index_t peSortedId = srcPe * MaxNumTokensToRecvPerRank + i - accumPeTokOffset;

    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>() + localTokenIdx * config.hiddenDim,
                   args.shmemInpTokMemObj->template GetAs<T*>() + peSortedId * config.hiddenDim,
                   config.hiddenDim);
    core::WarpCopy(args.shmemOutWeightsMemObj->template GetAs<float*>() +
                       localTokenIdx * config.numExpertPerToken,
                   args.shmemInpWeightsMemObj->template GetAs<float*>() +
                       peSortedId * config.numExpertPerToken,
                   config.numExpertPerToken);
    core::WarpCopy(args.shmemOutIndicesMemObj->template GetAs<index_t*>() +
                       localTokenIdx * config.numExpertPerToken,
                   args.shmemInpIndicesMemObj->template GetAs<index_t*>() +
                       peSortedId * config.numExpertPerToken,
                   config.numExpertPerToken);
    if (laneId == 0) {
      args.dispReceiverIdxMap[localTokenIdx] = peSortedId;
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // TODO: still figure out why use multiple threads lost RDMA writes
  for (int destPe = globalWarpId; destPe < args.config.worldSize; destPe += globalWarpNum) {
    if (laneId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      shmem::ShmemPutUint32ImmNbiWarp(args.crossDeviceBarrierMemObj,
                                      args.config.rank * sizeof(uint32_t),
                                      args.crossDeviceBarrierFlag, destPe);
    }
  }

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != args.crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernel(EpDispatchCombineArgs<T> args) {
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

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];

  // Phase 1: send token
  // This phase is symmetric with dispatch recv phase, where tokens are first sent back to its
  // source pe in pe sorted order
  for (int localTokenIdx = globalWarpId; localTokenIdx < totalRecvTokenNum;
       localTokenIdx += globalWarpNum) {
    index_t peSortedId = args.dispReceiverIdxMap[localTokenIdx];
    index_t srcPe = peSortedId / MaxNumTokensToRecvPerRank;
    peSortedId = peSortedId - srcPe * MaxNumTokensToRecvPerRank + myPe * MaxNumTokensToRecvPerRank;

    index_t peSortedOffset = peSortedId * config.hiddenDim;
    index_t tokenOffset = localTokenIdx * config.hiddenDim;

    core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<T*>() + tokenOffset,
                   args.inpTokenBuf + tokenOffset, config.hiddenDim);
    shmem::ShmemPutTypeNbiWarp<T>(args.shmemInpTokMemObj, peSortedOffset, args.shmemStagingTokMemObj,
                                  tokenOffset, config.hiddenDim, srcPe);
  }
  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierInterNodeKernel(args);

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
      index_t peSortedId = args.dispSenderIdxMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = peSortedId / MaxNumTokensToRecvPerRank;
      if (destPe < config.worldSize) {
        srcPtrs[j] = args.shmemInpTokMemObj->template GetAs<T*>() + peSortedId * config.hiddenDim +
                     hiddenDimOffset;
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
