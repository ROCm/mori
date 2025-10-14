// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config) : config(config) {
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();
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
  size_t maxTokenSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) * config.hiddenDim *
                        config.maxTokenTypeSize;
  size_t maxStagingTokSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) *
                             (config.hiddenDim * config.maxTokenTypeSize +
                              (sizeof(float) + sizeof(index_t)) * config.numExpertPerToken +
                              config.scaleDim * config.scaleTypeSize);
  shmemDispatchInpTokMemObj =
      ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);
  shmemCombineInpTokMemObj =
      ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);
  shmemOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemStagingTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);

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
  ShmemFree(shmemDispatchInpTokMemObj->localPtr);
  ShmemFree(shmemCombineInpTokMemObj->localPtr);
  ShmemFree(shmemOutTokMemObj->localPtr);
  ShmemFree(shmemStagingTokMemObj->localPtr);
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemOutWeightsMemObj->localPtr);
  if (shmemInpScalesMemObj.IsValid()) ShmemFree(shmemInpScalesMemObj->localPtr);
  if (shmemOutScalesMemObj.IsValid()) ShmemFree(shmemOutScalesMemObj->localPtr);
  ShmemFree(shmemInpIndicesMemObj->localPtr);
  ShmemFree(shmemOutIndicesMemObj->localPtr);
}

void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2;
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  // The extra *2 is for the laddr.
  sendAtomicSignalMemObj = ShmemMallocAndReturnMemObjPtr(
      (config.worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  ShmemFree(sendAtomicSignalMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
}

void EpDispatchCombineHandle::InitializeOrderMapBuf() {
  size_t maxNumOutToken = config.worldSize * config.maxNumInpTokenPerRank * config.numExpertPerRank;
  HIP_RUNTIME_CHECK(hipMalloc(&dispReceiverIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&dispSenderIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispSenderIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&destPeTokenIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenIdxMap, -1, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&srcPeTokenIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(srcPeTokenIdxMap, -1, maxNumOutToken * sizeof(index_t)));

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
  HIP_RUNTIME_CHECK(hipFree(destPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(srcPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
}

void EpDispatchCombineHandle::InitializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispatchGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&combineGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(combineGridBarrier, 0, barrierSize));
  crossDeviceBarrierMemObj = ShmemMallocAndReturnMemObjPtr(
      barrierSize * 2 * sizeof(uint64_t) / sizeof(uint32_t), hipDeviceMallocUncached);
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

        size_t sharedMemSize =
            actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
        if (kernelType == KernelType::InterNode) {
          assert(config.useExternalInpBuffer);
          EpCombineInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
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
