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

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <stdexcept>

#include "mori/core/core.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

static constexpr int32_t EP_CONFIG_I32_VERSION = 1;

std::vector<int32_t> EpDispatchCombineConfig::ToPackedI32Array() const {
  return {
      EP_CONFIG_I32_VERSION,
      rank,
      worldSize,
      hiddenDim,
      scaleDim,
      scaleTypeSize,
      maxTokenTypeSize,
      maxNumInpTokenPerRank,
      numExpertPerRank,
      numExpertPerToken,
      warpNumPerBlock,
      blockNum,
      static_cast<int32_t>(useExternalInpBuffer),
      static_cast<int32_t>(kernelType),
      gpuPerNode,
      rdmaBlockNum,
      numQpPerPe,
      static_cast<int32_t>(quantType),
      static_cast<int32_t>(enableSdma),
  };
}

EpDispatchCombineConfig EpDispatchCombineConfig::FromPackedI32Array(const int32_t* packed,
                                                                    size_t size) {
  // Runtime check to ensure the size of the packed array is correct
  if (size - 1 != kPackedI32Len) {
    throw std::runtime_error("EpDispatchCombineConfig i32 decode failed: invalid size");
  }
  if (packed == nullptr || packed[0] != EP_CONFIG_I32_VERSION) {
    throw std::runtime_error("EpDispatchCombineConfig i32 decode failed: unsupported version");
  }

  EpDispatchCombineConfig cfg;
  cfg.rank = packed[1];
  cfg.worldSize = packed[2];
  cfg.hiddenDim = packed[3];
  cfg.scaleDim = packed[4];
  cfg.scaleTypeSize = packed[5];
  cfg.maxTokenTypeSize = packed[6];
  cfg.maxNumInpTokenPerRank = packed[7];
  cfg.numExpertPerRank = packed[8];
  cfg.numExpertPerToken = packed[9];
  cfg.warpNumPerBlock = packed[10];
  cfg.blockNum = packed[11];
  cfg.useExternalInpBuffer = (packed[12] != 0);
  cfg.kernelType = static_cast<KernelType>(packed[13]);
  cfg.gpuPerNode = packed[14];
  cfg.rdmaBlockNum = packed[15];
  cfg.numQpPerPe = packed[16];
  cfg.quantType = static_cast<QuantType>(packed[17]);
  cfg.enableSdma = (packed[18] != 0);
  return cfg;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_)
    : config(config_) {
  assert(IsPowerOf2(config.gpuPerNode) && (config.worldSize % config.gpuPerNode == 0));
  int shmemNumQpPerPe = ShmemNumQpPerPe();
  if (config.numQpPerPe > shmemNumQpPerPe) {
    config.numQpPerPe = shmemNumQpPerPe;
    MORI_OPS_INFO("numQpPerPe {} larger than shmem numQpPerPe {}, set to {}", config.numQpPerPe,
                  shmemNumQpPerPe, shmemNumQpPerPe);
  }
  config.enableSdma = env::IsEnvVarEnabled("MORI_ENABLE_SDMA");
  MORI_OPS_INFO("EpDispatchCombine SDMA {} (currently only effective for AsyncLL kernel type)",
                config.enableSdma ? "enabled" : "disabled");
  if (config.maxTotalRecvTokens > 0) {
    int worstCase = config.worldSize * config.maxNumInpTokenPerRank;
    if (config.maxTotalRecvTokens > worstCase) {
      MORI_OPS_INFO("maxTotalRecvTokens={} exceeds worst case {}, clamping to worst case",
                    config.maxTotalRecvTokens, worstCase);
      config.maxTotalRecvTokens = worstCase;
    }
    MORI_OPS_INFO(
        "maxTotalRecvTokens={}, effective MaxNumTokensToRecvPerRank={}, "
        "buffer MaxNumTokensToRecv={} (original worst case={})",
        config.maxTotalRecvTokens, config.MaxNumTokensToRecvPerRank(), config.MaxNumTokensToRecv(),
        worstCase);
  }
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();

  this->multiProcessorCount = GetCurDeviceMultiProcessorCount();
  this->maxThreads = std::min(GetCurDeviceMaxThreads(), 1024);
  MORI_OPS_INFO("Device capability: multiProcessorCount={}, maxThreads={}",
                static_cast<int>(this->multiProcessorCount), static_cast<int>(this->maxThreads));
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
  size_t combineOutSize = static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) *
                          config.hiddenDim * config.maxTokenTypeSize;
  size_t dispatchOutSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) * config.hiddenDim *
                           config.maxTokenTypeSize;
  size_t maxStagingSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) *
                          (config.hiddenDim * config.maxTokenTypeSize +
                           (sizeof(float) + sizeof(index_t)) * config.numExpertPerToken +
                           config.scaleDim * config.scaleTypeSize);

  if (config.kernelType == KernelType::IntraNode) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsIntraNode>();
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  } else if (config.kernelType == KernelType::InterNodeV1 ||
             config.kernelType == KernelType::InterNodeV1LL) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNodeV1>();
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  } else {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNode>();
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  }

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
  if (config.kernelType == KernelType::IntraNode) {
    auto& bufs = std::get<ShmemBufsIntraNode>(shmemTokBufs);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
  } else if (config.kernelType == KernelType::InterNodeV1 ||
             config.kernelType == KernelType::InterNodeV1LL) {
    auto& bufs = std::get<ShmemBufsInterNodeV1>(shmemTokBufs);
    ShmemFree(bufs.dispatchInp->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
    ShmemFree(bufs.staging->localPtr);
  } else {
    auto& bufs = std::get<ShmemBufsInterNode>(shmemTokBufs);
    ShmemFree(bufs.dispatchInp->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
    ShmemFree(bufs.staging->localPtr);
  }
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
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
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
  size_t maxNumOutToken =
      config.worldSize * config.MaxNumTokensToRecvPerRank() * config.numExpertPerRank;
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
                                config.MaxNumTokensToRecvPerRank() * config.numExpertPerToken;
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(interNodeDispDestTokIdMap, 0, maxNumInterNodeToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(
      hipMalloc(&blockFlagCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(blockFlagCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  size_t interNodeDispSendMapSize =
      config.worldSize / config.gpuPerNode * config.MaxNumTokensToRecvPerRank() * sizeof(index_t);
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispSendMap, interNodeDispSendMapSize));
  HIP_RUNTIME_CHECK(hipMemset(interNodeDispSendMap, 0, interNodeDispSendMapSize));

#ifdef ENABLE_STANDARD_MOE_ADAPT
  const size_t maxDispatchTokens = static_cast<size_t>(config.MaxNumTokensToRecv());
  const size_t mapSize = maxDispatchTokens * config.numExpertPerToken * sizeof(uint64_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispTokToEpSlotMap, mapSize));
  HIP_RUNTIME_CHECK(hipMemset(dispTokToEpSlotMap, 0, mapSize));

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
                               (config.kernelType == KernelType::InterNodeV1LL) ||
                               (config.kernelType == KernelType::AsyncLL))
                                  ? 0
                                  : 1;
  crossDeviceBarrierMemObj =
      ShmemMallocAndReturnMemObjPtr(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

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

void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}

/* ---------------------------------------------------------------------------------------------- */
/*                              Args construction for Python launch                               */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum) {
  EpDispatchCombineArgsRaw args;
  args.config = handle.config;
  args.rdmaBlockNum = rdmaBlockNum;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndices = handle.tokenIndices;
  args.inpTokenBuf = handle.inpTokenBuf;
  args.outTokenBuf = handle.outTokenBuf;
  args.weightsBuf = handle.weightsBuf;
  args.scalesBuf = handle.scalesBuf;
  args.destPeTokenCounter = handle.destPeTokenCounter;
  args.localPeTokenCounter = handle.localPeTokenCounter;
  if (handle.config.kernelType == KernelType::IntraNode) {
    args.intraNodeTokBufs = std::get<ShmemBufsIntraNode>(handle.shmemTokBufs);
  } else if (handle.config.kernelType == KernelType::InterNodeV1 ||
             handle.config.kernelType == KernelType::InterNodeV1LL) {
    args.interNodeV1TokBufs = std::get<ShmemBufsInterNodeV1>(handle.shmemTokBufs);
  } else {
    args.interNodeTokBufs = std::get<ShmemBufsInterNode>(handle.shmemTokBufs);
  }
  args.shmemInpWeightsMemObj = handle.shmemInpWeightsMemObj;
  args.shmemDispatchOutWeightsMemObj = handle.shmemDispatchOutWeightsMemObj;
  args.shmemCombineOutWeightsMemObj = handle.shmemCombineOutWeightsMemObj;
  args.shmemInpScalesMemObj = handle.shmemInpScalesMemObj;
  args.shmemOutScalesMemObj = handle.shmemOutScalesMemObj;
  args.shmemInpIndicesMemObj = handle.shmemInpIndicesMemObj;
  args.shmemOutIndicesMemObj = handle.shmemOutIndicesMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.sendAtomicSignalMemObj = handle.sendAtomicSignalMemObj;
  args.dispatchGridBarrier = handle.dispatchGridBarrier;
  args.combineGridBarrier = handle.combineGridBarrier;
  args.dispReceiverIdxMap = handle.dispReceiverIdxMap;
  args.dispSenderIdxMap = handle.dispSenderIdxMap;
  args.destPeTokenIdxMap = handle.destPeTokenIdxMap;
  args.srcPeTokenIdxMap = handle.srcPeTokenIdxMap;
  args.dispTokOffsetMemObj = handle.dispTokOffsetMemObj;
  args.dispTokIdToSrcTokIdMemObj = handle.dispTokIdToSrcTokIdMemObj;
  args.dispDestTokIdMap = handle.dispDestTokIdMap;
  args.totalRecvTokenNum = handle.totalRecvTokenNum;
  args.crossDeviceBarrierMemObj = handle.crossDeviceBarrierMemObj;
  args.crossDeviceBarrierFlag = handle.crossDeviceBarrierFlag;
  args.interNodeChunkFlagMemObj = handle.interNodeChunkFlagMemObj;
  args.destNodeTokenCounter = handle.destNodeTokenCounter;
  args.nodeRecvTokenNumMemObj = handle.nodeRecvTokenNumMemObj;
  args.blockFlagCounter = handle.blockFlagCounter;
  args.interNodeBlocksBarrier = handle.interNodeBlocksBarrier;
  args.interNodeDispDestTokIdMap = handle.interNodeDispDestTokIdMap;
  args.interNodeChunkFlagCombine = handle.interNodeChunkFlagCombine;
  args.interNodeDispSendMap = handle.interNodeDispSendMap;
#ifdef ENABLE_PROFILER
  args.profilerConfig = handle.profilerConfig;
#endif
#ifdef ENABLE_STANDARD_MOE_ADAPT
  args.enableStandardMoeOutput = handle.enableStandardMoeOutput;
  args.standardPackedRecvX = handle.standardPackedRecvX;
  args.standardPackedRecvCount = handle.standardPackedRecvCount;
  args.standardPackedRecvSrcInfo = handle.standardPackedRecvSrcInfo;
  args.standardPackedRecvLayoutRange = handle.standardPackedRecvLayoutRange;
  args.dispTokToEpSlotMap = handle.dispTokToEpSlotMap;
#endif
  return args;
}

}  // namespace moe
}  // namespace mori
