#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/internode.hpp"
#include "src/ops/dispatch_combine/internode_normal.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

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

mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags,
                                                               const char* file = __FILE__,
                                                               int line = __LINE__) {
  void* buf = ShmemExtMallocWithFlags(size, flags);
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
  mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
  assert(obj.IsValid());
#if 0
  int dev = 0;
  HIP_RUNTIME_CHECK(hipGetDevice(&dev));
  if (dev == 0) {
    printf("[ShmemMalloc] %s:%d device=%d size=%zu bytes\n", file, line, dev, size);
  }
#endif
  return obj;
}

void EpDispatchCombineHandle::InitializeShmemBuf() {
  const size_t maxNumTokensToRecv = config.MaxNumTokensToRecv();
  size_t maxTokenSize = maxNumTokensToRecv * config.hiddenDim * config.maxTokenTypeSize;
  size_t maxStagingSize =
      maxNumTokensToRecv * (config.hiddenDim * config.maxTokenTypeSize +
                            (sizeof(float) + sizeof(index_t)) * config.numExpertPerToken +
                            config.scaleDim * config.scaleTypeSize);
  // printf("MaxNumTokensToRecv=%d\n", config.MaxNumTokensToRecv());
  shmemInpTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
  shmemOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemStagingTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
  const size_t prefixSize = config.worldSize * sizeof(index_t);
  shmemMetaDataMemObj =
      ShmemMallocAndReturnMemObjPtr(prefixSize, hipDeviceMallocUncached);
  const size_t syncSize = config.worldSize * sizeof(index_t);
  shmemSyncDataMemObj =
      ShmemMallocAndReturnMemObjPtr(syncSize, hipDeviceMallocUncached);

  size_t maxWeightSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemOutWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  if (config.scaleDim > 0 && config.scaleTypeSize > 0) {
    size_t maxScaleSize = maxNumTokensToRecv * config.scaleDim * config.scaleTypeSize;
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize = maxNumTokensToRecv * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  ShmemFree(shmemInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
  ShmemFree(shmemStagingTokMemObj->localPtr);
  if (shmemMetaDataMemObj.IsValid()) ShmemFree(shmemMetaDataMemObj->localPtr);
  if (shmemSyncDataMemObj.IsValid()) ShmemFree(shmemSyncDataMemObj->localPtr);
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

  HIP_RUNTIME_CHECK(
      HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&totalRecvTokenNum), sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
}

void EpDispatchCombineHandle::IntializeOrderMapBuf() {
  size_t maxNumOutToken = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerRank;
  // HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&dispReceiverIdxMap),
  //                                    maxNumOutToken * sizeof(index_t)));
  // HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&dispReceiverIdxMap),
                                        config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, config.worldSize * sizeof(index_t)));

  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&dispSenderIdxMap),
                                     maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispSenderIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&destPeTokenIdxMap),
                                     maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenIdxMap, -1, maxNumOutToken * sizeof(index_t)));

  if (config.kernelType == KernelType::InterNodeNormal) {
    HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&recvTokenOffset),
                                          config.worldSize * sizeof(index_t)));
    HIP_RUNTIME_CHECK(hipMemset(recvTokenOffset, 0, config.worldSize * sizeof(index_t)));
  } else {
    HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&srcPeTokenIdxMap),
                                          maxNumOutToken * sizeof(index_t)));
    HIP_RUNTIME_CHECK(hipMemset(srcPeTokenIdxMap, 0, maxNumOutToken * sizeof(index_t)));
  }

  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&destPeTokenCounter),
                                     config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenCounter, 0, config.worldSize * sizeof(index_t)));

  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&localPeTokenCounter),
                                     config.numExpertPerRank * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(localPeTokenCounter, 0, config.numExpertPerRank * sizeof(index_t)));

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&dispDestTokIdMap),
                                     maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(dispReceiverIdxMap));
  HIP_RUNTIME_CHECK(hipFree(dispSenderIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(srcPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(recvTokenOffset));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
}

void EpDispatchCombineHandle::IntializeBarrier() {
  size_t barrierSize = (config.worldSize + 1) * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&dispatchGridBarrier), barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(HIP_MALLOC_WITH_LOG(reinterpret_cast<void**>(&combineGridBarrier), barrierSize));
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
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  size_t sharedMemSize =
      (config.worldSize * actualWarpNumPerBlock + config.numExpertPerRank * actualWarpNumPerBlock +
       config.numExpertPerRank) *
      sizeof(index_t);
  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        if (kernelType == KernelType::InterNode) {
          assert(config.useExternalInpBuffer);
          EpDispatchInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::InterNodeNormal) {
          EpDispatchInterNodeNormalKernel<DataT><<<grid, block, sharedMemSize, stream>>>(args);
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
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        size_t sharedMemSize = actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
        if (kernelType == KernelType::InterNode) {
          assert(config.useExternalInpBuffer);
          EpCombineInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::InterNodeNormal) {
          EpCombineIntraNodeNormalKernel<DataT><<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::IntraNode) {
          EpCombineIntraNodeKernel<DataT><<<grid, block, sharedMemSize, stream>>>(args);
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) { crossDeviceBarrierFlag++; }

}  // namespace moe
}  // namespace mori