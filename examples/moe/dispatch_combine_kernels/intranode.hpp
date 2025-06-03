
#include "dispatch_combine_kernels/dispatch_combine.hpp"
#include "mori/core/core.hpp"
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

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Send out tokens
  extern __shared__ char sharedMem[];

  // Phase1: send token
  // Each warp compute token offset on destinition PE
  int i = globalWarpId;
  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    uint32_t destExpert = args.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;

    uint32_t destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<uint32_t*>(destPe), 1);
      args.dispTokIdToSrcTokIdMemObj->template GetAs<uint32_t*>(destPe)[destTokId] =
          myPe * config.maxNumInpTokenPerRank + i;
      args.outTokToExptMapMemObj->template GetAs<uint32_t*>(destPe)[destTokId] = destExpert;
    }
    destTokId = __shfl(destTokId, 0);

    uint32_t srcTokOffset = i / config.numExpertPerToken * config.hiddenDim;
    uint32_t destTokOffset = destTokId * config.hiddenDim;
    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>(destPe) + destTokOffset,
                   args.inpTokenBuf + srcTokOffset, config.hiddenDim);
  }
  if (laneId == 0) atomicAdd(args.dispatchGridCopyTokenBarrier, 1);
  // 95 us

  // Send token num & token to expert mapping to other ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridCopyTokenBarrier, globalWarpNum);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      uint32_t numTokenSignal = core::AtomicLoadRelaxed(args.peTokenOffset + destPe) + 1;
      shmem::ShmemPutUint32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(uint32_t),
                                        numTokenSignal, destPe);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  for (int destPe = thdId; destPe < npes; destPe += thdNum) {
    uint32_t* signal = args.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    shmem::ShmemUint32WaitUntilGreaterThan(signal, 0);
  }
  __syncthreads();
}

}  // namespace moe
}  // namespace mori