#pragma once

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

inline __device__ index_t EncodeValue(index_t val) { return val + 1; }
inline __device__ index_t DecodeValue(index_t val) { return val - 1; }

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

}  // namespace moe
}  // namespace mori