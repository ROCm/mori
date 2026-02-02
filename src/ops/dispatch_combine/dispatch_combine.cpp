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
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#include "src/ops/dispatch_combine/internode.hpp"
#include "src/ops/dispatch_combine/internode_v1.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_)
    : config(config_) {
  assert(IsPowerOf2(config.gpuPerNode) && (config.worldSize % config.gpuPerNode == 0));
  int shmemNumQpPerPe = ShmemNumQpPerPe();
  if (config.numQpPerPe > shmemNumQpPerPe) {
    config.numQpPerPe = shmemNumQpPerPe;
    MORI_OPS_INFO("numQpPerPe %d larger than shmem numQpPerPe %d, set to %d", config.numQpPerPe,
                  shmemNumQpPerPe, shmemNumQpPerPe);
  }
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();

  this->multiProcessorCount = GetCurDeviceMultiProcessorCount();
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
  shmemDispatchOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemCombineOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
  shmemStagingTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);

  size_t maxWeightSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemDispatchOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemCombineOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  if ((config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
    size_t maxScaleSize = config.MaxNumTokensToRecv() * config.scaleDim * config.scaleTypeSize;
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);

#ifdef ENABLE_PROFILER
  size_t debugBufSize = MAX_DEBUG_TIME_SLOTS * sizeof(int64_t);
  HIP_RUNTIME_CHECK(hipMalloc(&profilerConfig.debugTimeBuf, debugBufSize));
  HIP_RUNTIME_CHECK(hipMemset(profilerConfig.debugTimeBuf, 0, debugBufSize));

  size_t offsetBufSize = PROFILER_WARPS_PER_RANK * sizeof(unsigned int);
  HIP_RUNTIME_CHECK(hipMalloc(&profilerConfig.debugTimeOffset, offsetBufSize));
  HIP_RUNTIME_CHECK(hipMemset(profilerConfig.debugTimeOffset, 0, offsetBufSize));
#endif
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  ShmemFree(shmemDispatchInpTokMemObj->localPtr);
  ShmemFree(shmemCombineInpTokMemObj->localPtr);
  ShmemFree(shmemDispatchOutTokMemObj->localPtr);
  ShmemFree(shmemCombineOutTokMemObj->localPtr);
  ShmemFree(shmemStagingTokMemObj->localPtr);
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemDispatchOutWeightsMemObj->localPtr);
  ShmemFree(shmemCombineOutWeightsMemObj->localPtr);
  if (shmemInpScalesMemObj.IsValid()) ShmemFree(shmemInpScalesMemObj->localPtr);
  if (shmemOutScalesMemObj.IsValid()) ShmemFree(shmemOutScalesMemObj->localPtr);
  ShmemFree(shmemInpIndicesMemObj->localPtr);
  ShmemFree(shmemOutIndicesMemObj->localPtr);
#ifdef ENABLE_PROFILER
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeBuf));
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeOffset));
#endif
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

  size_t nodeTokenNumSignalSize = config.worldSize / config.gpuPerNode * sizeof(uint64_t);
  nodeRecvTokenNumMemObj =
      ShmemMallocAndReturnMemObjPtr(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  ShmemFree(sendAtomicSignalMemObj->localPtr);
  ShmemFree(nodeRecvTokenNumMemObj->localPtr);
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

  HIP_RUNTIME_CHECK(
      hipMalloc(&destNodeTokenCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(destNodeTokenCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&localPeTokenCounter, config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(localPeTokenCounter, 0, config.worldSize * sizeof(index_t)));

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(index_t)));

  size_t maxNumInterNodeToken = config.worldSize / config.gpuPerNode *
                                config.maxNumInpTokenPerRank * config.numExpertPerToken;
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(interNodeDispDestTokIdMap, 0, maxNumInterNodeToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(
      hipMalloc(&blockFlagCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(blockFlagCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  size_t interNodeDispSendMapSize =
      config.worldSize / config.gpuPerNode * config.maxNumInpTokenPerRank * sizeof(index_t);
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispSendMap, interNodeDispSendMapSize));
  HIP_RUNTIME_CHECK(hipMemset(interNodeDispSendMap, 0, interNodeDispSendMapSize));

#ifdef ENABLE_STANDARD_MOE_ADAPT
  const size_t maxDispatchTokens = static_cast<size_t>(config.MaxNumTokensToRecv());
  const size_t mapSize = maxDispatchTokens * config.numExpertPerToken * sizeof(uint64_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispTokToEpSlotMap, mapSize));
  HIP_RUNTIME_CHECK(hipMemset(dispTokToEpSlotMap, 0, mapSize));

  // Allocate standard MoE output buffers
  HIP_RUNTIME_CHECK(hipMalloc(&standardPackedRecvCount, config.numExpertPerRank * sizeof(int)));
  HIP_RUNTIME_CHECK(hipMemset(standardPackedRecvCount, 0, config.numExpertPerRank * sizeof(int)));
#endif
}

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(dispReceiverIdxMap));
  HIP_RUNTIME_CHECK(hipFree(dispSenderIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(srcPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(destNodeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
  HIP_RUNTIME_CHECK(hipFree(interNodeDispDestTokIdMap));
  HIP_RUNTIME_CHECK(hipFree(blockFlagCounter));
  HIP_RUNTIME_CHECK(hipFree(interNodeDispSendMap));
#ifdef ENABLE_STANDARD_MOE_ADAPT
  HIP_RUNTIME_CHECK(hipFree(dispTokToEpSlotMap));
  HIP_RUNTIME_CHECK(hipFree(standardPackedRecvCount));
#endif
}

void EpDispatchCombineHandle::InitializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispatchGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&combineGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(combineGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&crossDeviceBarrierFlag, sizeof(uint64_t)));
  crossDeviceBarrierFlag[0] = ((config.kernelType == KernelType::InterNodeV1) ||
                               (config.kernelType == KernelType::InterNodeV1LL))
                                  ? 0
                                  : 1;
  // HIP_RUNTIME_CHECK(hipMemset(crossDeviceBarrierFlag, 1, 1));
  crossDeviceBarrierMemObj =
      ShmemMallocAndReturnMemObjPtr(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

  // We allocate one flag for each token, this ensure that we can use all chunk size(>=1)
  size_t interNodeChunkFlagSize =
      config.worldSize / config.gpuPerNode * config.MaxNumTokensToRecvPerRank() * sizeof(uint64_t);
  interNodeChunkFlagMemObj =
      ShmemMallocAndReturnMemObjPtr(interNodeChunkFlagSize, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&interNodeChunkFlagCombine, interNodeChunkFlagSize));
  HIP_RUNTIME_CHECK(hipMemset(interNodeChunkFlagCombine, 0, interNodeChunkFlagSize));

  HIP_RUNTIME_CHECK(hipMalloc(&interNodeBlocksBarrier, 4 * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(interNodeBlocksBarrier, 0, 4 * sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeBarrier() {
  HIP_RUNTIME_CHECK(hipFree(dispatchGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(combineGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(crossDeviceBarrierFlag));
  HIP_RUNTIME_CHECK(hipFree(interNodeChunkFlagCombine));
  HIP_RUNTIME_CHECK(hipFree(interNodeBlocksBarrier));
  ShmemFree(crossDeviceBarrierMemObj->localPtr);
  ShmemFree(interNodeChunkFlagMemObj->localPtr);
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
                                             hipStream_t stream, bool enableStandardMoeOutput) {
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
        } else if (kernelType == KernelType::InterNodeV1) {
          EpDispatchCopyToStaging<<<this->multiProcessorCount, block, 0, stream>>>(args);
          EpDispatchInterNodeV1Kernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::InterNodeV1LL) {
          EpDispatchCopyToStaging<<<this->multiProcessorCount, block, 0, stream>>>(args);
          EpDispatchInterNodeV1KernelLowLatency<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::IntraNode) {
          if (enableStandardMoeOutput) {
            EpDispatchIntraNodeKernel<DataT, true><<<grid, block, sharedMemSize, stream>>>(args);
          } else {
            EpDispatchIntraNodeKernel<DataT, false><<<grid, block, sharedMemSize, stream>>>(args);
          }
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchCombine(KernelType kernelType, int blockNum, int warpPerBlock,
                                            hipStream_t stream, bool enableStandardMoeInput,
                                            int useExternalInpBuf) {
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  // Determine useExternalInpBuffer: use parameter if >= 0, otherwise use config
  const bool actualUseExternalInpBuffer =
      (useExternalInpBuf >= 0) ? static_cast<bool>(useExternalInpBuf) : config.useExternalInpBuffer;

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        // Override args.config.useExternalInpBuffer with the actual value
        args.config.useExternalInpBuffer = actualUseExternalInpBuffer;

        size_t sharedMemSize =
            actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
        if (kernelType == KernelType::InterNode) {
          assert(actualUseExternalInpBuffer);
          EpCombineInterNodeKernel<<<grid, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::InterNodeV1) {
          assert(actualUseExternalInpBuffer);
          EpCombineInterNodeV1Kernel<<<grid, block, sharedMemSize, stream>>>(args);
          EpCombineAll<<<this->multiProcessorCount, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::InterNodeV1LL) {
          assert(actualUseExternalInpBuffer);
          EpCombineInterNodeV1KernelLowLatency<<<grid, block, sharedMemSize, stream>>>(args);
          EpCombineAll<<<this->multiProcessorCount, block, sharedMemSize, stream>>>(args);
        } else if (kernelType == KernelType::IntraNode) {
#ifndef ENABLE_STANDARD_MOE_ADAPT
          if (actualUseExternalInpBuffer) {
            // UseP2PRead=false: does not support zero-copy and provides better bandwidth and lower
            // latency
            EpCombineIntraNodeKernel<DataT, /*UseP2PRead=*/false>
                <<<grid, block, sharedMemSize, stream>>>(args);
          } else {  // zero-copy mode (requires UseP2PRead=true)
            EpCombineIntraNodeKernel<DataT, /*UseP2PRead=*/true>
                <<<grid, block, sharedMemSize, stream>>>(args);
          }
#else
          if (enableStandardMoeInput) {
            // Standard MoE mode: convert packed expert output to shmem format first
            EpCombineIntraNodeKernel<DataT, /*UseP2PRead=*/true, /*EnableStdMoE=*/true>
                <<<grid, block, sharedMemSize, stream>>>(args);
          } else {
            EpCombineIntraNodeKernel<DataT, /*UseP2PRead=*/true, /*EnableStdMoE=*/false>
                <<<grid, block, sharedMemSize, stream>>>(args);
          }
#endif
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchConvertDispatchOutputKernel(const void* dispatchOutX,
                                                                const void* dispatchOutTopkIdx,
                                                                void* packedRecvX,
                                                                int* packedRecvCount,
                                                                int* packedRecvSrcInfo,
                                                                int64_t* packedRecvLayoutRange,
                                                                int blockNum, int warpPerBlock,
                                                                hipStream_t stream) {
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  ConvertDispatchOutputArgs args{};
  args.config = config;
  args.dispatchOutX = dispatchOutX;
  args.dispatchOutTopkIdx = dispatchOutTopkIdx;
  args.dispatchSrcTokenPos = dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>();
  args.totalRecvTokenNum = totalRecvTokenNum;
  args.packedRecvX = packedRecvX;
  args.packedRecvCount = packedRecvCount;
  args.packedRecvSrcInfo = packedRecvSrcInfo;
  args.packedRecvLayoutRange = packedRecvLayoutRange;
  args.dispTokToEpSlotMap = dispTokToEpSlotMap;
  args.dispatchGridBarrier = dispatchGridBarrier;

  ConvertDispatchOutputKernel<<<grid, block, 0, stream>>>(args);
}

void EpDispatchCombineHandle::LaunchConvertCombineInputKernel(
    const void* packedRecvX, const void* packedRecvSrcInfo, const void* packedRecvLayoutRange,
    void* combineInput, mori::application::SymmMemObjPtr shmemCombineInpTokMemObj, int blockNum,
    int warpPerBlock, hipStream_t stream) {
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  ConvertCombineInputArgs args{};
  args.config = config;
  args.packedRecvX = packedRecvX;
  args.topkIdx = shmemOutIndicesMemObj->Get();
  args.topkWeights = shmemDispatchOutWeightsMemObj->Get();
  args.packedRecvSrcInfo = packedRecvSrcInfo;
  args.packedRecvLayoutRange = packedRecvLayoutRange;
  args.totalRecvTokenNum = totalRecvTokenNum;
  args.combineInput = combineInput;
  args.shmemCombineInpTokMemObj = shmemCombineInpTokMemObj;
  args.dispTokIdToSrcTokIdMemObj = dispTokIdToSrcTokIdMemObj;
  args.dispTokToEpSlotMap = dispTokToEpSlotMap;
  args.packedRecvCount = standardPackedRecvCount;

  switch (inputType) {
    case HIP_R_32F:
      ConvertCombineInputKernel<float><<<grid, block, 0, stream>>>(args);
      break;
    case HIP_R_16BF:
      ConvertCombineInputKernel<hip_bfloat16><<<grid, block, 0, stream>>>(args);
      break;
#ifdef MORI_FP8_TYPE_OCP_ENABLED
    case HIP_R_8F_E4M3:
      ConvertCombineInputKernel<__hip_fp8_e4m3><<<grid, block, 0, stream>>>(args);
      break;
#endif
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
    case HIP_R_8F_E4M3_FNUZ:
      ConvertCombineInputKernel<__hip_fp8_e4m3_fnuz><<<grid, block, 0, stream>>>(args);
      break;
#endif
    default:
      assert(false);
      break;
  }
}

// no need for a separate reset kernel now
void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}

}  // namespace moe
}  // namespace mori
