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
  __threadfence_system();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config)
    : config(config) {
  InitializeShmemBuf();
  IntializeTokenNumSignalBuf();
  IntializeOrderMapBuf();
  IntializeBarrier();
}

EpDispatchCombineHandle::~EpDispatchCombineHandle() {
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

void EpDispatchCombineHandle::InitializeShmemBuf() {
  size_t maxTokenSize = config.MaxNumTokensToRecv() * config.hiddenDim * config.maxTokenTypeSize;
  shmemInpTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemStagingTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);

  size_t maxWeightSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemOutWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  if (config.scaleDim > 0 && config.scaleTypeSize > 0) {
    size_t maxScaleSize = config.MaxNumTokensToRecv() * config.scaleDim * config.scaleTypeSize;
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  ShmemFree(shmemInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
  ShmemFree(shmemStagingTokMemObj->localPtr);
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemOutWeightsMemObj->localPtr);
  if (shmemInpScalesMemObj.IsValid()) ShmemFree(shmemInpScalesMemObj->localPtr);
  if (shmemOutScalesMemObj.IsValid()) ShmemFree(shmemOutScalesMemObj->localPtr);
  ShmemFree(shmemInpIndicesMemObj->localPtr);
  ShmemFree(shmemOutIndicesMemObj->localPtr);
}

void EpDispatchCombineHandle::IntializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t);
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
}

void EpDispatchCombineHandle::IntializeOrderMapBuf() {
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

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(dispReceiverIdxMap));
  HIP_RUNTIME_CHECK(hipFree(dispSenderIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
}

void EpDispatchCombineHandle::IntializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispatchGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&combineGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(combineGridBarrier, 0, barrierSize));
  crossDeviceBarrierMemObj = ShmemMallocAndReturnMemObjPtr(barrierSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeBarrier() {
  HIP_RUNTIME_CHECK(hipFree(dispatchGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(combineGridBarrier));
  ShmemFree(crossDeviceBarrierMemObj->localPtr);
}

void EpDispatchCombineHandle::LaunchIntraNodeDispatch(int blockNum, int warpPerBlock,
                                                      hipStream_t stream) {
  LaunchDispatch(KernelType::IntraNode, blockNum, warpPerBlock, stream);
}

void EpDispatchCombineHandle::LaunchInterNodeDispatch(int blockNum, int warpPerBlock,
                                                      hipStream_t stream) {
  LaunchDispatch(KernelType::InterNode, blockNum, warpPerBlock, stream);
}

void EpDispatchCombineHandle::LaunchIntraNodeCombine(int blockNum, int warpPerBlock,
                                                     hipStream_t stream) {
  LaunchCombine(KernelType::IntraNode, blockNum, warpPerBlock, stream);
}

void EpDispatchCombineHandle::LaunchInterNodeCombine(int blockNum, int warpPerBlock,
                                                     hipStream_t stream) {
  LaunchCombine(KernelType::InterNode, blockNum, warpPerBlock, stream);
}

void EpDispatchCombineHandle::LaunchDispatch(KernelType kernelType, int blockNum, int warpPerBlock,
                                             hipStream_t stream) {
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * ((warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock));

  size_t sharedMemSize =
      (config.worldSize * config.warpNumPerBlock +
       config.numExpertPerRank * config.warpNumPerBlock + config.numExpertPerRank) *
      sizeof(index_t);
  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        if (kernelType == KernelType::InterNode) {
          assert(config.useExternalInpBuffer);
          EpDispatchInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::IntraNode) {
          EpDispatchIntraNodeKernel<DataT><<<grid, block, sharedMemSize, stream>>>(args);
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchCombine(KernelType kernelType, int blockNum, int warpPerBlock,
                                            hipStream_t stream) {
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * ((warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock));

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        size_t sharedMemSize = config.warpNumPerBlock * config.numExpertPerToken * sizeof(DataT**);
        if (kernelType == KernelType::InterNode) {
          assert(!config.useExternalInpBuffer);
          EpCombineInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::IntraNode) {
          EpCombineIntraNodeKernel<DataT><<<grid, block, sharedMemSize, stream>>>(args);
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {
  dim3 block(std::max(config.numExpertPerRank, config.worldSize));

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        EpDispatchCombineResetKernel<<<1, config.numExpertPerRank, 0, stream>>>(args);
      },
      argsVariant);
  crossDeviceBarrierFlag++;
}

}  // namespace moe
}  // namespace mori