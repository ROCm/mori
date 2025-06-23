#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/internode.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

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

    uint32_t peSortedId = args.dispReceiverIdxMap[exptSortedId];
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
      uint32_t destPeCopyCnt = atomicAdd(args.combineGridBarrier + destPe, 1);
    }
  }
  SyncIfDebugEnabled("Combine kernel: finish recovering from expert sorted to pe sorted");

  // TODO: since we don't have atomic yet, we have to wait untill all tokens are sent, then set
  // the remote flag; once we have atomic operation, we can send an atomic rdma op after each
  // token and the remote peer polling the flag to know if the token is finished sent
  for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
    uint32_t numTokenSignal = recvTokenNumBuf[destPe];
    uint32_t recvTokenNum = numTokenSignal - 1;

    ShmemUint32WaitUntilEquals(args.combineGridBarrier + destPe, recvTokenNum);
    AtomicStoreRelaxed(args.combineGridBarrier + destPe, uint32_t{0});
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
      uint32_t peSortedId = args.dispSenderIdxMap[i * config.numExpertPerToken + j];
      uint32_t peSortedOffset = peSortedId * config.hiddenDim;
      srcPtrs[j] = shmemOutTokenBuf + peSortedOffset;
    }

    WarpAccum(args.outTokenBuf + i * config.hiddenDim, srcPtrs,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken,
              config.hiddenDim);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           ResetKernel                                          */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchCombineResetKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  for (int destPe = thdId; destPe < args.config.worldSize; destPe += blockDim.x) {
    args.recvTokenNumMemObj->template GetAs<uint32_t*>()[destPe] = 0;
    args.sendTokenNumMemObj->template GetAs<uint32_t*>()[destPe] = 0;
    args.destPeTokenCounter[destPe] = 0;
    args.dispatchGridBarrier[destPe] = 0;
    args.combineGridBarrier[destPe] = 0;
  }
  for (int exptId = thdId; exptId < args.config.numExpertPerRank; exptId += blockDim.x) {
    args.localPeTokenCounter[exptId] = 0;
  }
  if (thdId == 0) {
    args.dispTokOffsetMemObj->template GetAs<uint32_t*>()[0] = 0;
    core::AtomicStoreRelaxedSystem(args.totalRecvTokenNum, size_t{0});
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
EpDispatchCombineHandle<T>::EpDispatchCombineHandle(EpDispatchCombineConfig config)
    : config(config) {
  IntializeShmemBuf();
  IntializeTokenNumSignalBuf();
  IntializeOrderMapBuf();
  IntializeBarrier();
}

template <typename T>
EpDispatchCombineHandle<T>::~EpDispatchCombineHandle() {
  FinalizeShmemBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeOrderMapBuf();
  FinalizeBarrier();
}

mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags) {
  void* buf = ShmemExtMallocWithFlags(size, flags);
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
  mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
  assert(obj.IsValid());
  return obj;
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeShmemBuf() {
  size_t maxTokenSize = config.MaxNumTokensToRecv() * config.hiddenDim * sizeof(T);
  shmemInpTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);

  size_t maxWeightSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemOutWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  size_t maxIndiciesSize =
      config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(uint32_t);
  shmemInpIndiciesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndiciesSize, hipDeviceMallocUncached);
  shmemOutIndiciesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndiciesSize, hipDeviceMallocUncached);
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeShmemBuf() {
  ShmemFree(shmemInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemOutWeightsMemObj->localPtr);
  ShmemFree(shmemInpIndiciesMemObj->localPtr);
  ShmemFree(shmemOutIndiciesMemObj->localPtr);
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(uint32_t);
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(
      hipExtMallocWithFlags((void**)&lock, sizeof(uint32_t), hipDeviceMallocUncached));
  HIP_RUNTIME_CHECK(hipMemset(lock, 0, sizeof(uint32_t)));
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
  HIP_RUNTIME_CHECK(hipFree(lock));
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeOrderMapBuf() {
  size_t maxNumOutToken = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerRank;
  HIP_RUNTIME_CHECK(hipMalloc(&dispReceiverIdxMap, maxNumOutToken * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, maxNumOutToken * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&dispSenderIdxMap, maxNumOutToken * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispSenderIdxMap, 0, maxNumOutToken * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&destPeTokenCounter, config.worldSize * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenCounter, 0, config.worldSize * sizeof(uint32_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&localPeTokenCounter, config.numExpertPerRank * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(localPeTokenCounter, 0, config.numExpertPerRank * sizeof(uint32_t)));

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(uint32_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(uint32_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(uint32_t)));
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(dispReceiverIdxMap));
  HIP_RUNTIME_CHECK(hipFree(dispSenderIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispatchGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&combineGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(combineGridBarrier, 0, barrierSize));
  crossDeviceBarrierMemObj = ShmemMallocAndReturnMemObjPtr(barrierSize, hipDeviceMallocUncached);
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeBarrier() {
  HIP_RUNTIME_CHECK(hipFree(dispatchGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(combineGridBarrier));
  ShmemFree(crossDeviceBarrierMemObj->localPtr);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchIntraNodeDispatch(hipStream_t stream) {
  LaunchDispatch(KernelType::IntraNode, stream);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchInterNodeDispatch(hipStream_t stream) {
  LaunchDispatch(KernelType::InterNode, stream);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchIntraNodeCombine(hipStream_t stream) {
  LaunchCombine(KernelType::IntraNode, stream);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchInterNodeCombine(hipStream_t stream) {
  LaunchCombine(KernelType::InterNode, stream);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchDispatch(KernelType kernelType, hipStream_t stream) {
  dim3 grid(config.blockNum);
  dim3 block(warpSize * config.warpNumPerBlock);
  size_t sharedMemSize =
      (config.worldSize * config.warpNumPerBlock +
       config.numExpertPerRank * config.warpNumPerBlock + config.numExpertPerRank) *
      sizeof(uint32_t);
  if (kernelType == KernelType::InterNode)
    EpDispatchInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(
        GetEpDispatchCombineArgs(*this));
  else if (kernelType == KernelType::IntraNode) {
    EpDispatchIntraNodeKernel<T>
        <<<grid, block, sharedMemSize, stream>>>(GetEpDispatchCombineArgs(*this));
  } else
    assert(false);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchCombine(KernelType kernelType, hipStream_t stream) {
  dim3 grid(config.blockNum);
  dim3 block(warpSize * config.warpNumPerBlock);
  size_t sharedMemSize = config.warpNumPerBlock * config.numExpertPerToken * sizeof(T**);

  if (kernelType == KernelType::InterNode)
    EpCombineInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(
        GetEpDispatchCombineArgs(*this));
  else if (kernelType == KernelType::IntraNode) {
    EpCombineIntraNodeKernel<T>
        <<<grid, block, sharedMemSize, stream>>>(GetEpDispatchCombineArgs(*this));
  } else
    assert(false);
}

template <typename T>
void EpDispatchCombineHandle<T>::LaunchReset(hipStream_t stream) {
  dim3 block(std::max(config.numExpertPerRank, config.worldSize));
  EpDispatchCombineResetKernel<<<1, config.numExpertPerRank, 0, stream>>>(
      GetEpDispatchCombineArgs(*this));
  crossDeviceBarrierFlag++;
}

template class EpDispatchCombineHandle<float>;
template class EpDispatchCombineHandle<hip_bfloat16>;
template class EpDispatchCombineHandle<__hip_fp8_e4m3_fnuz>;

}  // namespace moe
}  // namespace mori