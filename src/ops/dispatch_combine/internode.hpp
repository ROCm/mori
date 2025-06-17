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
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  T* inpTokenBuf = args.shmemInpTokMemObj->template GetAs<T*>();
  uint32_t* outTokToExptBuf = args.outTokToExptMapMemObj->template GetAs<uint32_t*>();

  size_t maxNumOutTokenPerRank = config.MaxNumTokensToRecvPerRank();

  // Send out tokens
  extern __shared__ char sharedMem[];

  // Phase1: send token
  // TODO: finish sender code
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);
  SyncIfDebugEnabled("Dispatch kernel: finished send token");

  // Send token num & token to expert mapping to other ranks
  // TODO: we use multiple warps here because using multiple threads in warp 0 lost RDMA writes,
  // don't know why
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    if (laneId == 0) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      uint32_t numTokenSignal = core::AtomicLoadRelaxed(args.peTokenOffset + destPe) + 1;
      shmem::ShmemPutUint32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                        numTokenSignal, destPe);
      printf("myPe %d send %d tokens to %d\n", myPe, numTokenSignal - 1, destPe);
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: finish sending tok2expt mapping & num token signal");

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      uint32_t* signal = args.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
      uint32_t recvTokenNum = shmem::ShmemUint32WaitUntilGreaterThan(signal, 0);
      printf("myPe %d recv %d tokens from %d\n", myPe, recvTokenNum, destPe);
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: finish waiting num token signal");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernel(EpDispatchCombineArgs<T> args) {}

}  // namespace moe
}  // namespace mori