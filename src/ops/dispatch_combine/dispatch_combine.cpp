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
/*                                           ResetKernel                                          */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchCombineResetKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  for (int destPe = thdId; destPe < args.config.worldSize; destPe += blockDim.x) {
    args.recvTokenNumMemObj->template GetAs<index_t*>()[destPe] = 0;
    args.sendTokenNumMemObj->template GetAs<index_t*>()[destPe] = 0;
    args.destPeTokenCounter[destPe] = 0;
    args.dispatchGridBarrier[destPe] = 0;
    args.combineGridBarrier[destPe] = 0;
  }
  for (int exptId = thdId; exptId < args.config.numExpertPerRank; exptId += blockDim.x) {
    args.localPeTokenCounter[exptId] = 0;
  }
  if (thdId == 0) {
    args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    core::AtomicStoreRelaxedSystem(args.totalRecvTokenNum, index_t{0});
  }
  // TODO: this should be one in wqe post API
  if (globalThdId == 0) {
    shmem::ShmemQuietThread();
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

  size_t maxScaleSize = config.MaxNumTokensToRecvPerRank() * config.scaleDim * config.scaleTypeSize;
  shmemScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);

  size_t maxIndiciesSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndiciesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndiciesSize, hipDeviceMallocUncached);
  shmemOutIndiciesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndiciesSize, hipDeviceMallocUncached);
}

template <typename T>
void EpDispatchCombineHandle<T>::FinalizeShmemBuf() {
  ShmemFree(shmemInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemOutWeightsMemObj->localPtr);
  ShmemFree(shmemScalesMemObj->localPtr);
  ShmemFree(shmemInpIndiciesMemObj->localPtr);
  ShmemFree(shmemOutIndiciesMemObj->localPtr);
}

template <typename T>
void EpDispatchCombineHandle<T>::IntializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t);
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));

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
  HIP_RUNTIME_CHECK(hipMalloc(&dispReceiverIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&dispSenderIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispSenderIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&destPeTokenCounter, config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenCounter, 0, config.worldSize * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&localPeTokenCounter, config.numExpertPerRank * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(localPeTokenCounter, 0, config.numExpertPerRank * sizeof(index_t)));

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(index_t)));
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
      sizeof(index_t);
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