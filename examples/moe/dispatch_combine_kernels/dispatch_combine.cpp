#include "dispatch_combine_kernels/dispatch_combine.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

#define DEBUG 0

__device__ void SyncIfDebugEnabled(const char* msg) {
#if DEBUG == 1
  __syncthreads();
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    ShmemQuietThread();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}

template <typename T>
struct EpDispatchCombineArgs {
  EpDispatchCombineConfig config;
  int curRankNumToken{-1};
  uint32_t* tokenIndicies{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  SymmMemObjPtr shmemInpTokMemObj;
  SymmMemObjPtr shmemOutTokMemObj;
  SymmMemObjPtr recvTokenNumMemObj;
  SymmMemObjPtr sendTokenNumMemObj;
  uint32_t* gridCopyTokenBarrier{nullptr};
  SymmMemObjPtr inpTokToExptMapMemObj;
  SymmMemObjPtr outTokToExptMapMemObj;
  uint32_t* peTokenOffset{nullptr};
  uint32_t* exptTokenOffset{nullptr};
  uint32_t* exptSortedToPeSortedBuf{nullptr};
  uint32_t* tokenIndicesToPeSortedBuf{nullptr};
};

template <typename T>

EpDispatchCombineArgs<T> GetEpDispatchCombineArgs(const EpDispatchCombineHandle<T>& handle) {
  EpDispatchCombineArgs<T> args;
  args.config = handle.config;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndicies = handle.tokenIndicies;
  args.inpTokenBuf = handle.inpTokenBuf;
  args.outTokenBuf = handle.outTokenBuf;
  args.weightsBuf = handle.weightsBuf;
  args.peTokenOffset = handle.peTokenOffset;
  args.exptTokenOffset = handle.exptTokenOffset;
  args.shmemInpTokMemObj = handle.shmemInpTokMemObj;
  args.shmemOutTokMemObj = handle.shmemOutTokMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.gridCopyTokenBarrier = handle.gridCopyTokenBarrier;
  args.inpTokToExptMapMemObj = handle.inpTokToExptMapMemObj;
  args.outTokToExptMapMemObj = handle.outTokToExptMapMemObj;
  args.exptSortedToPeSortedBuf = handle.exptSortedToPeSortedBuf;
  args.tokenIndicesToPeSortedBuf = handle.tokenIndicesToPeSortedBuf;
  return args;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        EpDispatchKernel                                        */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchKernel(EpDispatchCombineArgs<T> args) {
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
  uint32_t* inpTokToExptBuf = args.inpTokToExptMapMemObj->template GetAs<uint32_t*>();
  uint32_t* outTokToExptBuf = args.outTokToExptMapMemObj->template GetAs<uint32_t*>();

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Send out tokens
  extern __shared__ char sharedMem[];

  // Phase1: send token
  // Each warp compute token offset on destinition PE

  for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
       i += globalWarpNum) {
    uint32_t destExpert = args.tokenIndicies[i];
    uint32_t destPe = destExpert / config.numExpertPerRank;

    uint32_t peTokenIdx = 0;
    if (laneId == 0) {
      peTokenIdx = atomicAdd(args.peTokenOffset + destPe, 1);
      args.tokenIndicesToPeSortedBuf[i] = destPe * maxNumOutTokenPerRank + peTokenIdx;
      inpTokToExptBuf[destPe * maxNumOutTokenPerRank + peTokenIdx] = destExpert;
    }
    peTokenIdx = __shfl(peTokenIdx, 0);

    uint32_t tokenId = i / config.numExpertPerToken;
    uint32_t tokenOffset = tokenId * config.hiddenDim;

    uint32_t peSortedId = myPe * maxNumOutTokenPerRank + peTokenIdx;
    uint32_t peSortedOffset = peSortedId * config.hiddenDim;

    // WarpCopy(args.shmemInpTokMemObj->template GetAs<T*>() + tokenOffset,
    //          args.inpTokenBuf + tokenOffset, config.hiddenDim);
    // ShmemPutTypeNbiWarp<T>(args.shmemOutTokMemObj, peSortedOffset, args.shmemInpTokMemObj,
    //                        tokenOffset, config.hiddenDim, destPe);
    WarpCopy(args.shmemOutTokMemObj->template GetAs<T*>(destPe) + peSortedOffset,
             args.inpTokenBuf + tokenOffset, config.hiddenDim);
  }
  if (laneId == 0) atomicAdd(args.gridCopyTokenBarrier, 1);
  SyncIfDebugEnabled("Dispatch kernel: finished send token");

  // Send token num & token to expert mapping to other ranks
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    // Wait until all tokens are sent
    ShmemUint32WaitUntilEquals(args.gridCopyTokenBarrier, globalWarpNum);

    // TODO: reset grid copy token barrier!
    // AtomicStoreRelaxed(args.gridCopyTokenBarrier, uint32_t{0});  // reset for next inference

    // Add 1 so that when token number == 0, receiver side still know the signal is sent
    uint32_t recvTokenNum = AtomicLoadRelaxed(args.peTokenOffset + destPe);

    ShmemPutUint32NbiWarp(args.outTokToExptMapMemObj, myPe * maxNumOutTokenPerRank,
                          args.inpTokToExptMapMemObj, destPe * maxNumOutTokenPerRank, recvTokenNum,
                          destPe);

    uint32_t numTokenSignal = recvTokenNum + 1;
    ShmemPutUint32ImmNbiWarp(args.recvTokenNumMemObj, myPe * sizeof(uint32_t), numTokenSignal,
                             destPe);
  }
  SyncIfDebugEnabled("Dispatch kernel: finish sending tok2expt mapping & num token signal");

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  uint32_t* recvTokenNum = reinterpret_cast<uint32_t*>(sharedMem) + warpId * npes;
  uint32_t* accumExpertTokOffsets = reinterpret_cast<uint32_t*>(sharedMem) + warpNum * npes;
  uint32_t* expertTokOff = reinterpret_cast<uint32_t*>(sharedMem) + warpNum * npes +
                           config.numExpertPerRank + warpId * config.numExpertPerRank;
  for (int i = thdId; i < config.numExpertPerRank; i += thdNum) {
    accumExpertTokOffsets[i] = 0;
  }
  __syncthreads();

  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = args.recvTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    ShmemUint32WaitUntilGreaterThan(signal, 0);
    recvTokenNum[destPe] = AtomicLoadRelaxedSystem(signal) - 1;
  }
  SyncIfDebugEnabled("Dispatch kernel: finish waiting num token signal");

  // Compute token number for each expert
  for (int srcPe = warpId; srcPe < npes; srcPe += warpNum) {
    for (int tokId = laneId; tokId < recvTokenNum[srcPe]; tokId += warpSize) {
      uint32_t expertId =
          AtomicLoadRelaxed(outTokToExptBuf + srcPe * maxNumOutTokenPerRank + tokId);
      uint32_t localExpertId = expertId % config.numExpertPerRank;
      atomicAdd(accumExpertTokOffsets + localExpertId, 1);
    }
  }
  __syncthreads();

  // Calculate prefix sum of expert token offset
  uint32_t accumOffset = 0;
  if (config.numExpertPerRank <= warpSize) {
    accumOffset = WarpPrefixSum(accumExpertTokOffsets[thdId], config.numExpertPerRank);
  } else {
    assert(config.numExpertPerRank <= FlatBlockSize());
    accumOffset = BlockPrefixSum(accumExpertTokOffsets[thdId], config.numExpertPerRank);
  }
  if (thdId < config.numExpertPerRank) {
    accumExpertTokOffsets[thdId] = accumOffset;
  }
  __syncthreads();

  T* outTokenBuf = args.outTokenBuf;
  T* shmemOutTokenBuf = args.shmemOutTokMemObj->template GetAs<T*>();

  for (int i = globalWarpId;; i += globalWarpNum) {
    // find src pe and tok id
    uint32_t srcPe = 0;
    uint32_t accumPeTokOffset = 0;
    for (; srcPe < npes; srcPe++) {
      if ((i >= accumPeTokOffset) && (i < (accumPeTokOffset + recvTokenNum[srcPe]))) break;
      accumPeTokOffset += recvTokenNum[srcPe];
    }
    if (srcPe >= npes) break;
    uint32_t srcTokId = i - accumPeTokOffset;

    uint32_t localExpertId =
        outTokToExptBuf[srcPe * maxNumOutTokenPerRank + srcTokId] % config.numExpertPerRank;

    uint32_t exptTokId = 0;
    if (laneId == 0) {
      exptTokId = atomicAdd(args.exptTokenOffset + localExpertId, 1);
    }
    exptTokId = __shfl(exptTokId, 0);

    // Copy token
    uint32_t peSortedId = srcPe * maxNumOutTokenPerRank + srcTokId;
    uint32_t srcTokenOff = peSortedId * config.hiddenDim;

    uint32_t exptSortedId = accumExpertTokOffsets[localExpertId] + exptTokId;
    uint32_t destTokenOff = exptSortedId * config.hiddenDim;
    WarpCopy(outTokenBuf + destTokenOff, shmemOutTokenBuf + srcTokenOff, config.hiddenDim);

    if (laneId == 0) {
      args.exptSortedToPeSortedBuf[exptSortedId] = peSortedId;
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         EpCombineKernel                                        */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineKernel(EpDispatchCombineArgs<T> args) {
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

  T* inpTokenBuf = args.inpTokenBuf;
  T* shmemInpTokenBuf = args.shmemInpTokMemObj->template GetAs<T*>();
  T* shmemOutTokenBuf = args.shmemOutTokMemObj->template GetAs<T*>();

  size_t maxNumOutTokenPerRank = config.maxNumInpTokenPerRank * config.numExpertPerToken;

  // Phase 1: recover tokens from expert sorted order to pe sorted order and send token back
  // Each warp compute total number of recveid tokens
  uint32_t* recvTokenNumBuf = args.recvTokenNumMemObj->template GetAs<uint32_t*>();
  uint32_t totalNumRecvToken = 0;
  for (int i = laneId; i < npes; i += warpSize) {
    totalNumRecvToken += recvTokenNumBuf[i] - 1;
  }
  totalNumRecvToken = WarpReduceSum(totalNumRecvToken);
  totalNumRecvToken = __shfl(totalNumRecvToken, 0);

  // Recover pe sorted order and send back
  for (int exptSortedId = 0; exptSortedId < totalNumRecvToken; exptSortedId++) {
    if ((exptSortedId % globalWarpNum) != globalWarpId) continue;

    uint32_t peSortedId = args.exptSortedToPeSortedBuf[exptSortedId];
    uint32_t peSortedOffset = peSortedId * config.hiddenDim;

    uint32_t destPe = peSortedId / maxNumOutTokenPerRank;
    uint32_t peerPeSortedOffset =
        (peSortedId - destPe * maxNumOutTokenPerRank + myPe * maxNumOutTokenPerRank) *
        config.hiddenDim;

    uint32_t exptSortedOffset = exptSortedId * config.hiddenDim;

    WarpCopy(shmemInpTokenBuf + peSortedOffset, inpTokenBuf + exptSortedOffset, config.hiddenDim);
    ShmemPutTypeNbiWarp<T>(args.shmemOutTokMemObj, peerPeSortedOffset, args.shmemInpTokMemObj,
                           peSortedOffset, config.hiddenDim, destPe);
    if (laneId == 0) {
      uint32_t destPeCopyCnt = atomicAdd(args.gridCopyTokenBarrier + destPe, 1);
    }
  }
  SyncIfDebugEnabled("Combine kernel: finish recovering from expert sorted to pe sorted");

  // TODO: since we don't have atomic yet, we have to wait untill all tokens are sent, then set
  // the remote flag; once we have atomic operation, we can send an atomic rdma op after each
  // token and the remote peer polling the flag to know if the token is finished sent
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    uint32_t numTokenSignal = recvTokenNumBuf[destPe];
    uint32_t recvTokenNum = numTokenSignal - 1;

    ShmemUint32WaitUntilEquals(args.gridCopyTokenBarrier + destPe, recvTokenNum);
    AtomicStoreRelaxed(args.gridCopyTokenBarrier + destPe, uint32_t{0});
    ShmemPutUint32ImmNbiWarp(args.sendTokenNumMemObj, myPe * sizeof(uint32_t), numTokenSignal,
                             destPe);
  }
  SyncIfDebugEnabled("Combine kernel: finish sending tokens");
  __threadfence_system();

  // Phase 2: recv pe sorted token, reduce accross expert and recover original order
  for (int destPe = laneId; destPe < npes; destPe += warpSize) {
    uint32_t* signal = args.sendTokenNumMemObj->template GetAs<uint32_t*>() + destPe;
    ShmemUint32WaitUntilGreaterThan(signal, 0);
  }
  SyncIfDebugEnabled("Combine kernel: finish waiting num token signal");

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;

  T* outTokenBuf = args.outTokenBuf;
  for (int i = 0; i < args.curRankNumToken; i++) {
    if ((i % globalWarpNum) != globalWarpId) continue;

    for (int j = 0; j < config.numExpertPerToken; j++) {
      uint32_t peSortedId = args.tokenIndicesToPeSortedBuf[i * config.numExpertPerToken + j];
      uint32_t peSortedOffset = peSortedId * config.hiddenDim;
      srcPtrs[j] = shmemOutTokenBuf + peSortedOffset;
    }

    WarpAccum(args.outTokenBuf + i * config.hiddenDim, srcPtrs,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken,
              config.hiddenDim);
  }
}

template <typename T>
__global__ void EpDispatchCombineResetKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  for (int destPe = thdId; destPe < args.config.worldSize; destPe += blockDim.x) {
    args.recvTokenNumMemObj->template GetAs<uint32_t*>()[destPe] = 0;
    args.sendTokenNumMemObj->template GetAs<uint32_t*>()[destPe] = 0;
    args.peTokenOffset[destPe] = 0;
    args.gridCopyTokenBarrier[destPe] = 0;
  }
  for (int exptId = thdId; exptId < args.config.numExpertPerRank; exptId += blockDim.x) {
    args.exptTokenOffset[exptId] = 0;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
EpDispatchCombineHandle<T>::EpDispatchCombineHandle(EpDispatchCombineConfig config)
    : config(config) {
  IntializeShmemInpOutTokBuf();
  IntializeTokenNumSignalBuf();
  IntializeTokToExptBuf();
  IntializeOrderMapBuf();
}

template <typename T>
EpDispatchCombineHandle<T>::~EpDispatchCombineHandle() {
  FinalizeShmemInpOutTokBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeTokToExptBuf();
  FinalizeOrderMapBuf();
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeShmemInpOutTokBuf() {
  int maxTokenSize = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken *
                     config.hiddenDim * sizeof(T);

  void* shmemInpTokBuf = ShmemExtMallocWithFlags(maxTokenSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(shmemInpTokBuf, 0, maxTokenSize));
  shmemInpTokMemObj = ShmemQueryMemObjPtr(shmemInpTokBuf);
  assert(shmemInpTokMemObj.IsValid());

  void* shmemOutTokBuf = ShmemExtMallocWithFlags(maxTokenSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(shmemOutTokBuf, 0, maxTokenSize));
  shmemOutTokMemObj = ShmemQueryMemObjPtr(shmemOutTokBuf);
  assert(shmemOutTokMemObj.IsValid());
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeShmemInpOutTokBuf() {
  ShmemFree(shmemInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeTokenNumSignalBuf() {
  int tokenNumSignalSize = config.worldSize * sizeof(uint32_t);

  void* recvTokenNumBuf = ShmemExtMallocWithFlags(tokenNumSignalSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(recvTokenNumBuf, 0, tokenNumSignalSize));
  recvTokenNumMemObj = ShmemQueryMemObjPtr(recvTokenNumBuf);
  assert(recvTokenNumMemObj.IsValid());

  void* sendTokenNumBuf = ShmemExtMallocWithFlags(tokenNumSignalSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(sendTokenNumBuf, 0, tokenNumSignalSize));
  sendTokenNumMemObj = ShmemQueryMemObjPtr(sendTokenNumBuf);
  assert(sendTokenNumMemObj.IsValid());

  HIP_RUNTIME_CHECK(hipMalloc(&gridCopyTokenBarrier, tokenNumSignalSize));
  HIP_RUNTIME_CHECK(hipMemset(gridCopyTokenBarrier, 0, tokenNumSignalSize));
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(gridCopyTokenBarrier));
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeTokToExptBuf() {
  int tokToExptMapSize =
      config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerToken * sizeof(uint32_t);
  void* inpTokToExptMapBuf = ShmemExtMallocWithFlags(tokToExptMapSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(inpTokToExptMapBuf, 0, tokToExptMapSize));
  inpTokToExptMapMemObj = ShmemQueryMemObjPtr(inpTokToExptMapBuf);
  assert(inpTokToExptMapMemObj.IsValid());

  void* outTokToExptMapBuf = ShmemExtMallocWithFlags(tokToExptMapSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(outTokToExptMapBuf, 0, tokToExptMapSize));
  outTokToExptMapMemObj = ShmemQueryMemObjPtr(outTokToExptMapBuf);
  assert(outTokToExptMapMemObj.IsValid());
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeTokToExptBuf() {
  ShmemFree(inpTokToExptMapMemObj->localPtr);
  ShmemFree(outTokToExptMapMemObj->localPtr);
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeOrderMapBuf() {
  int maxNumOutToken = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerRank;
  HIP_RUNTIME_CHECK(hipMalloc(&exptSortedToPeSortedBuf, maxNumOutToken * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(exptSortedToPeSortedBuf, 0, maxNumOutToken * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&tokenIndicesToPeSortedBuf, maxNumOutToken * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(tokenIndicesToPeSortedBuf, 0, maxNumOutToken * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&peTokenOffset, config.worldSize * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(peTokenOffset, 0, config.worldSize * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&exptTokenOffset, config.numExpertPerRank * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(exptTokenOffset, 0, config.numExpertPerRank * sizeof(uint32_t)));
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(exptSortedToPeSortedBuf));
  HIP_RUNTIME_CHECK(hipFree(tokenIndicesToPeSortedBuf));
  HIP_RUNTIME_CHECK(hipFree(peTokenOffset));
  HIP_RUNTIME_CHECK(hipFree(exptTokenOffset));
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchDispatch(hipStream_t stream) {
  dim3 grid(config.blockNum);
  dim3 block(warpSize * config.warpNumPerBlock);
  size_t sharedMemSize =
      (config.worldSize * config.warpNumPerBlock +
       config.numExpertPerRank * config.warpNumPerBlock + config.numExpertPerRank) *
      sizeof(uint32_t);
  EpDispatchKernel<<<grid, block, sharedMemSize, stream>>>(GetEpDispatchCombineArgs(*this));
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchCombine(hipStream_t stream) {
  dim3 grid(config.blockNum);
  dim3 block(warpSize * config.warpNumPerBlock);
  size_t sharedMemSize = config.warpNumPerBlock * config.numExpertPerToken * sizeof(T**);
  EpCombineKernel<<<grid, block, sharedMemSize, stream>>>(GetEpDispatchCombineArgs(*this));
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchReset(hipStream_t stream) {
  EpDispatchCombineResetKernel<<<1, config.numExpertPerRank, 0, stream>>>(
      GetEpDispatchCombineArgs(*this));
}

template class EpDispatchCombineHandle<float>;
template class EpDispatchCombineHandle<hip_bfloat16>;
template class EpDispatchCombineHandle<__hip_fp8_e4m3_fnuz>;

}  // namespace moe
}  // namespace mori