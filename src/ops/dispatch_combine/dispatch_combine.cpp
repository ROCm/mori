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
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#include "mori/core/core.hpp"
#include "mori/shmem/internal.hpp"
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

// 56 → block_elems = 7168/56 = 128, matching the AccumNum=8 + VecBytes=8 dequant specialization.
static constexpr int kDefaultFp8BlockwiseScaleDim = 56;
static constexpr const char* kFp8BlockwiseScaleDimEnv = "MORI_FP8_COMBINE_SCALE_DIM";

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
      maxTotalRecvTokens,
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
  cfg.maxTotalRecvTokens = packed[10];
  cfg.warpNumPerBlock = packed[11];
  cfg.blockNum = packed[12];
  cfg.useExternalInpBuffer = (packed[13] != 0);
  cfg.kernelType = static_cast<KernelType>(packed[14]);
  cfg.gpuPerNode = packed[15];
  cfg.rdmaBlockNum = packed[16];
  cfg.numQpPerPe = packed[17];
  cfg.quantType = static_cast<QuantType>(packed[18]);
  cfg.enableSdma = (packed[19] != 0);
  return cfg;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
namespace {

// Test-only allocation fault injection: MORI_TEST_FAIL_ALLOC=<buffer name>, and
// MORI_TEST_FAIL_ALLOC_TIMES=<n> (default 1, negative means every time). Reconfigure's
// rollback and group-fatal branches are otherwise reachable only by provoking a
// real OOM, which no test can do on demand.
bool ShouldInjectAllocFailure(const char* what) {
  const char* target = std::getenv("MORI_TEST_FAIL_ALLOC");
  std::string requested = target ? target : "";
  // Re-arm on every change of the variable, including clearing it, so one
  // long-lived worker process can run several injection cases in a row.
  static std::string armed;
  static int budget = 0;
  if (requested != armed) {
    armed = requested;
    const char* times = std::getenv("MORI_TEST_FAIL_ALLOC_TIMES");
    budget = times ? std::atoi(times) : 1;
  }
  if (requested.empty() || requested != what || budget == 0) return false;
  if (budget > 0) --budget;
  return true;
}

// Allocate, optionally fill (std::nullopt leaves the bytes alone), throwing
// instead of exiting. HIP_RUNTIME_CHECK is fprintf + exit(-1), which kills the
// rank before Reconfigure can roll back and before the python layer can reduce a
// group verdict -- so the resize path cannot use it.
template <typename T>
void HipMallocOrThrow(T** ptr, size_t bytes, std::optional<int> fill, const char* what) {
  hipError_t err = ShouldInjectAllocFailure(what) ? hipErrorOutOfMemory
                                                  : hipMalloc(reinterpret_cast<void**>(ptr), bytes);
  if (err == hipSuccess && fill.has_value()) err = hipMemset(*ptr, *fill, bytes);
  if (err != hipSuccess) {
    *ptr = nullptr;
    (void)hipGetLastError();
    throw std::runtime_error(std::string("failed to allocate ") + what + ": " +
                             hipGetErrorString(err));
  }
}

// Freeing must tolerate a handle that InitializeAll() only partly built, and must
// not double-free when a failed Reconfigure cleans up and then rolls back.
template <typename T>
void FreeDeviceBuf(T*& ptr) {
  if (ptr == nullptr) return;
  (void)hipFree(ptr);
  ptr = nullptr;
}

void FreeSymmBuf(mori::application::SymmMemObjPtr& obj) {
  if (!obj.IsValid()) return;
  ShmemFree(obj->localPtr);
  obj = {};
}

}  // namespace

EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_)
    : config(config_) {
  NormalizeConfig();
  InitializeAll();
  buffersInitialized = true;
  allocatedConfig = config;

  this->multiProcessorCount = GetCurDeviceMultiProcessorCount();
  this->maxThreads = std::min(GetCurDeviceMaxThreads(), 1024);
  MORI_OPS_INFO("Device capability: multiProcessorCount={}, maxThreads={}",
                static_cast<int>(this->multiProcessorCount), static_cast<int>(this->maxThreads));
}

void EpDispatchCombineHandle::NormalizeConfig() {
  assert(IsPowerOf2(config.gpuPerNode) && (config.worldSize % config.gpuPerNode == 0));
  int shmemNumQpPerPe = ShmemNumQpPerPe();
  if (config.numQpPerPe > shmemNumQpPerPe) {
    config.numQpPerPe = shmemNumQpPerPe;
    MORI_OPS_INFO("numQpPerPe {} larger than shmem numQpPerPe {}, set to {}", config.numQpPerPe,
                  shmemNumQpPerPe, shmemNumQpPerPe);
  }

  if (IsBlockwiseCombineQuant(config.quantType)) {
    fp8BlockwiseCombineScaleDim =
        env::GetPositiveIntOr(kFp8BlockwiseScaleDimEnv, kDefaultFp8BlockwiseScaleDim);
    fp8BlockwiseCombineScaleTypeSize = static_cast<int>(sizeof(float));
    if (config.rank == 0) {
      MORI_OPS_INFO("Blockwise combine ({}) scale_dim={} (override via {})",
                    config.quantType == QuantType::Fp4BlockwiseQuant ? "FP4" : "FP8",
                    fp8BlockwiseCombineScaleDim, kFp8BlockwiseScaleDimEnv);
    }
  }

  // Read the SDMA flag from the Context-cached snapshot (set once at Context
  // construction). Reading getenv directly here would race with the
  // SymmMemManager / Context decisions made at shmem init time -- the symptom
  // was tests that set MORI_ENABLE_SDMA inside the test function deadlocking
  // because Malloc started returning uncached buffers while Context still
  // believed the transport was P2P.
  config.enableSdma = ShmemSdmaEnabled();
  MORI_OPS_INFO("EpDispatchCombine SDMA {} (currently only effective for AsyncLL kernel type)",
                config.enableSdma ? "enabled" : "disabled");
  if (config.kernelType == KernelType::AsyncLL && !config.enableSdma && config.rank == 0) {
    MORI_OPS_WARN(
        "Mori AsyncLL is selected but SDMA is disabled. AsyncLL without SDMA uses compute units "
        "for communication, which provides little overlap benefit and can severely degrade "
        "performance. Use a non-AsyncLL kernel path or set MORI_ENABLE_SDMA=1.");
  }
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
}

void EpDispatchCombineHandle::InitializeAll() {
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();
}

void EpDispatchCombineHandle::FinalizeAll() {
  FinalizeShmemBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeOrderMapBuf();
  FinalizeBarrier();
}

EpDispatchCombineHandle::~EpDispatchCombineHandle() { Finalize(); }

void EpDispatchCombineHandle::Finalize() {
  if (!buffersInitialized) return;
  buffersInitialized = false;
  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
    // The symmetric heap is already gone; there is nothing left to give back.
    return;
  }
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  FinalizeAll();
}

void EpDispatchCombineHandle::ValidateReconfigurable(
    const EpDispatchCombineConfig& newConfig) const {
  auto reject = [](const char* field) {
    throw std::invalid_argument(std::string("reconfigure cannot change layout-defining field '") +
                                field + "'");
  };
  if (newConfig.rank != config.rank) reject("rank");
  if (newConfig.worldSize != config.worldSize) reject("worldSize");
  if (newConfig.hiddenDim != config.hiddenDim) reject("hiddenDim");
  if (newConfig.scaleDim != config.scaleDim) reject("scaleDim");
  if (newConfig.scaleTypeSize != config.scaleTypeSize) reject("scaleTypeSize");
  if (newConfig.maxTokenTypeSize != config.maxTokenTypeSize) reject("maxTokenTypeSize");
  if (newConfig.numExpertPerRank != config.numExpertPerRank) reject("numExpertPerRank");
  if (newConfig.numExpertPerToken != config.numExpertPerToken) reject("numExpertPerToken");
  if (newConfig.kernelType != config.kernelType) reject("kernelType");
  if (newConfig.gpuPerNode != config.gpuPerNode) reject("gpuPerNode");
  if (newConfig.quantType != config.quantType) reject("quantType");
  if (newConfig.maxNumInpTokenPerRank <= 0)
    throw std::invalid_argument("reconfigure needs a positive maxNumInpTokenPerRank");
}

void EpDispatchCombineHandle::Reconfigure(const EpDispatchCombineConfig& newConfig,
                                          bool releaseCapacity) {
  ValidateReconfigurable(newConfig);
  if (!buffersInitialized) {
    throw std::runtime_error("reconfigure on a handle that owns no buffers");
  }

  EpDispatchCombineConfig oldConfig = config;
  config = newConfig;
  NormalizeConfig();
  if (!releaseCapacity && config.MaxNumTokensToSend() <= allocatedConfig.MaxNumTokensToSend() &&
      config.MaxNumTokensToRecv() <= allocatedConfig.MaxNumTokensToRecv()) {
    return;
  }

  Finalize();
  try {
    InitializeAll();
  } catch (const std::exception& err) {
    // Drop whatever the failed attempt did allocate before retrying at the old
    // size, or the rollback allocates on top of it and is likelier to OOM too.
    FinalizeAll();
    config = allocatedConfig;
    try {
      InitializeAll();
    } catch (const std::exception& rollbackErr) {
      FinalizeAll();
      throw std::runtime_error(std::string("reconfigure failed and could not roll back: ") +
                               err.what() + " / " + rollbackErr.what());
    }
    config = oldConfig;
    buffersInitialized = true;
    throw std::runtime_error(std::string("reconfigure failed, rolled back to the old capacity: ") +
                             err.what());
  }
  buffersInitialized = true;
  allocatedConfig = config;
}

mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags) {
  void* buf = ShmemExtMallocWithFlags(size, flags);
  mori::application::SymmMemObjPtr obj;
  if (buf != nullptr) {
    HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
    obj = ShmemQueryMemObjPtr(buf);
  }
  // Throw rather than assert: a symmetric-heap exhaustion during a resize has to
  // reach Reconfigure's rollback, and NDEBUG would drop the assert entirely.
  if (!obj.IsValid()) {
    throw std::runtime_error("shmem allocation of " + std::to_string(size) + " bytes failed");
  }
  return obj;
}

void EpDispatchCombineHandle::InitializeShmemBuf() {
  size_t combineOutSize = static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) *
                          config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t dispatchOutSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) *
                           config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t maxStagingSize =
      static_cast<ssize_t>(config.MaxNumTokensToRecv()) * config.MaxXferBytesPerToken();
  if (config.kernelType == KernelType::IntraNode && IsBlockwiseCombineQuant(config.quantType)) {
    size_t blockwiseScaleBytes =
        (fp8BlockwiseCombineScaleDim > 0)
            ? static_cast<size_t>(fp8BlockwiseCombineScaleDim) * fp8BlockwiseCombineScaleTypeSize
            : 0;
    // FP4 packs the token region at 0.5 byte/elem (CombineTokenRegionBytes()), so its staging slot
    // is half the FP8 one -- no FP8-sized over-allocation for FP4.
    maxStagingSize = static_cast<size_t>(config.MaxNumTokensToRecv()) *
                     (config.CombineTokenRegionBytes() + config.IndexBytes() +
                      config.WeightBytes() + config.SrcTokenIdBytes() + blockwiseScaleBytes);
  }

  if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsIntraNode>();
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  } else if (config.kernelType == KernelType::InterNodeV1 ||
             config.kernelType == KernelType::InterNodeV1LL) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNodeV1>();
    const int nNodes = config.worldSize / config.gpuPerNode;
    size_t dispatchInpSize = static_cast<ssize_t>(nNodes) * config.MaxNumTokensToSendPerRank() *
                             config.MaxXferBytesPerToken();
    size_t stagingSize = static_cast<ssize_t>(2 * nNodes) * config.MaxNumTokensToSendPerRank() *
                         config.MaxXferBytesPerToken();
    size_t dispatchStagingSize =
        static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) * config.MaxXferBytesPerToken();
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(dispatchInpSize, hipDeviceMallocUncached);
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(stagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
    bufs.dispatchStaging =
        ShmemMallocAndReturnMemObjPtr(dispatchStagingSize, hipDeviceMallocUncached);
  } else {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNode>();
    // NOTE(ditian12): no overflow protection for dispatchInp/combinInp/staging in async kernel,
    // hence have to allocate to max size we need to either implement compact layout or add
    // pre-assertion to prevent silent memory access fault
    size_t maxStagingSize =
        static_cast<ssize_t>(config.MaxNumTokensToSend()) * config.MaxXferBytesPerToken();
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  }

  size_t maxWeightSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemDispatchOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemCombineOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  size_t userScaleSize = 0;
  if (config.scaleDim > 0 && config.scaleTypeSize > 0) {
    userScaleSize =
        static_cast<size_t>(config.MaxNumTokensToRecv()) * config.scaleDim * config.scaleTypeSize;
  }
  size_t fp8BlockwiseScaleSize = 0;
  if (IsBlockwiseCombineQuant(config.quantType) && fp8BlockwiseCombineScaleDim > 0) {
    fp8BlockwiseScaleSize = static_cast<size_t>(config.MaxNumTokensToRecv()) *
                            fp8BlockwiseCombineScaleDim * fp8BlockwiseCombineScaleTypeSize;
  }
  size_t inpScaleSize = std::max(userScaleSize, fp8BlockwiseScaleSize);
  if (inpScaleSize > 0) {
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(inpScaleSize, hipDeviceMallocUncached);
  }
  if (userScaleSize > 0) {
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(userScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);

#ifdef ENABLE_PROFILER
  HipMallocOrThrow(&profilerConfig.debugTimeBuf, MAX_DEBUG_TIME_SLOTS * sizeof(int64_t), 0,
                   "debugTimeBuf");
  HipMallocOrThrow(&profilerConfig.debugTimeOffset, PROFILER_WARPS_PER_RANK * sizeof(unsigned int),
                   0, "debugTimeOffset");
#endif
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  // By the engaged alternative, so that cleaning up after a partly-built
  // InitializeAll() frees what exists instead of dereferencing what does not.
  if (auto* bufs = std::get_if<ShmemBufsIntraNode>(&shmemTokBufs)) {
    FreeSymmBuf(bufs->dispatchOut);
    FreeSymmBuf(bufs->combineInp);
    FreeSymmBuf(bufs->combineOut);
  } else if (auto* bufs = std::get_if<ShmemBufsInterNodeV1>(&shmemTokBufs)) {
    FreeSymmBuf(bufs->dispatchInp);
    FreeSymmBuf(bufs->combineInp);
    FreeSymmBuf(bufs->dispatchOut);
    FreeSymmBuf(bufs->combineOut);
    FreeSymmBuf(bufs->staging);
    FreeSymmBuf(bufs->dispatchStaging);
  } else if (auto* bufs = std::get_if<ShmemBufsInterNode>(&shmemTokBufs)) {
    FreeSymmBuf(bufs->dispatchInp);
    FreeSymmBuf(bufs->combineInp);
    FreeSymmBuf(bufs->dispatchOut);
    FreeSymmBuf(bufs->combineOut);
    FreeSymmBuf(bufs->staging);
  }
  FreeSymmBuf(shmemInpWeightsMemObj);
  FreeSymmBuf(shmemDispatchOutWeightsMemObj);
  FreeSymmBuf(shmemCombineOutWeightsMemObj);
  FreeSymmBuf(shmemInpScalesMemObj);
  FreeSymmBuf(shmemOutScalesMemObj);
  FreeSymmBuf(shmemInpIndicesMemObj);
  FreeSymmBuf(shmemOutIndicesMemObj);
#ifdef ENABLE_PROFILER
  FreeDeviceBuf(profilerConfig.debugTimeBuf);
  FreeDeviceBuf(profilerConfig.debugTimeOffset);
#endif
}

void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendAtomicSignalMemObj = ShmemMallocAndReturnMemObjPtr(
      (config.worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);

  HipMallocOrThrow(&totalRecvTokenNum, sizeof(index_t), 0, "totalRecvTokenNum");

  size_t nodeTokenNumSignalSize = config.worldSize / config.gpuPerNode * sizeof(uint64_t);
  nodeRecvTokenNumMemObj =
      ShmemMallocAndReturnMemObjPtr(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  FreeSymmBuf(recvTokenNumMemObj);
  FreeSymmBuf(sendTokenNumMemObj);
  FreeSymmBuf(sendAtomicSignalMemObj);
  FreeSymmBuf(nodeRecvTokenNumMemObj);
  FreeDeviceBuf(totalRecvTokenNum);
}

void EpDispatchCombineHandle::InitializeOrderMapBuf() {
  size_t maxNumOutToken =
      static_cast<size_t>(config.MaxNumTokensToSend()) * config.numExpertPerRank;
  size_t orderMapSize = maxNumOutToken * sizeof(index_t);
  size_t perRankSize = config.worldSize * sizeof(index_t);
  size_t perNodeSize = config.worldSize / config.gpuPerNode * sizeof(index_t);
  HipMallocOrThrow(&dispReceiverIdxMap, orderMapSize, 0, "dispReceiverIdxMap");
  HipMallocOrThrow(&dispSenderIdxMap, orderMapSize, 0, "dispSenderIdxMap");
  HipMallocOrThrow(&destPeTokenIdxMap, orderMapSize, -1, "destPeTokenIdxMap");
  HipMallocOrThrow(&srcPeTokenIdxMap, orderMapSize, -1, "srcPeTokenIdxMap");
  HipMallocOrThrow(&destPeTokenCounter, perRankSize, 0, "destPeTokenCounter");
  HipMallocOrThrow(&destNodeTokenCounter, perNodeSize, 0, "destNodeTokenCounter");
  HipMallocOrThrow(&localPeTokenCounter, perRankSize, 0, "localPeTokenCounter");

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HipMallocOrThrow(&dispDestTokIdMap, orderMapSize, 0, "dispDestTokIdMap");

  size_t maxNumInterNodeToken = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                config.MaxNumTokensToSendPerRank() * config.numExpertPerToken;
  HipMallocOrThrow(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t), 0,
                   "interNodeDispDestTokIdMap");
  HipMallocOrThrow(&blockFlagCounter, perNodeSize, 0, "blockFlagCounter");

  size_t interNodeDispSendMapSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                    config.MaxNumTokensToSendPerRank() * sizeof(index_t);
  HipMallocOrThrow(&interNodeDispSendMap, interNodeDispSendMapSize, 0, "interNodeDispSendMap");

#ifdef ENABLE_STANDARD_MOE_ADAPT
  const size_t maxDispatchTokens = static_cast<size_t>(config.MaxNumTokensToRecv());
  const size_t mapSize = maxDispatchTokens * config.numExpertPerToken * sizeof(uint64_t);
  HipMallocOrThrow(&dispTokToEpSlotMap, mapSize, 0, "dispTokToEpSlotMap");
  HipMallocOrThrow(&standardPackedRecvCount, config.numExpertPerRank * sizeof(int), 0,
                   "standardPackedRecvCount");
#endif
}

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  FreeDeviceBuf(dispReceiverIdxMap);
  FreeDeviceBuf(dispSenderIdxMap);
  FreeDeviceBuf(destPeTokenIdxMap);
  FreeDeviceBuf(srcPeTokenIdxMap);
  FreeDeviceBuf(destPeTokenCounter);
  FreeDeviceBuf(destNodeTokenCounter);
  FreeDeviceBuf(localPeTokenCounter);
  FreeSymmBuf(dispTokOffsetMemObj);
  FreeSymmBuf(dispTokIdToSrcTokIdMemObj);
  FreeDeviceBuf(dispDestTokIdMap);
  FreeDeviceBuf(interNodeDispDestTokIdMap);
  FreeDeviceBuf(blockFlagCounter);
  FreeDeviceBuf(interNodeDispSendMap);
#ifdef ENABLE_STANDARD_MOE_ADAPT
  FreeDeviceBuf(dispTokToEpSlotMap);
  FreeDeviceBuf(standardPackedRecvCount);
#endif
}

void EpDispatchCombineHandle::InitializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HipMallocOrThrow(&dispatchGridBarrier, barrierSize, 0, "dispatchGridBarrier");
  HipMallocOrThrow(&combineGridBarrier, barrierSize, 0, "combineGridBarrier");
  // Host-initialised below, so it must not be filled: only IntraNode wants a
  // non-zero flag, and a fill that lands after the store leaves the intra
  // combine barrier matching peer slots that are still zero -- it stops waiting.
  HipMallocOrThrow(&crossDeviceBarrierFlag, sizeof(uint64_t), std::nullopt,
                   "crossDeviceBarrierFlag");
  crossDeviceBarrierFlag[0] = ((config.kernelType == KernelType::InterNodeV1) ||
                               (config.kernelType == KernelType::InterNodeV1LL) ||
                               (config.kernelType == KernelType::AsyncLL))
                                  ? 0
                                  : 1;
  crossDeviceBarrierMemObj =
      ShmemMallocAndReturnMemObjPtr(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

  size_t interNodeChunkFlagSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                  config.MaxNumTokensToSendPerRank() * sizeof(uint64_t);
  interNodeChunkFlagMemObj =
      ShmemMallocAndReturnMemObjPtr(interNodeChunkFlagSize, hipDeviceMallocUncached);

  HipMallocOrThrow(&interNodeChunkFlagCombine, interNodeChunkFlagSize, 0,
                   "interNodeChunkFlagCombine");
  HipMallocOrThrow(&interNodeBlocksBarrier, 4 * sizeof(index_t), 0, "interNodeBlocksBarrier");
}

void EpDispatchCombineHandle::FinalizeBarrier() {
  FreeDeviceBuf(dispatchGridBarrier);
  FreeDeviceBuf(combineGridBarrier);
  FreeDeviceBuf(crossDeviceBarrierFlag);
  FreeDeviceBuf(interNodeChunkFlagCombine);
  FreeDeviceBuf(interNodeBlocksBarrier);
  FreeSymmBuf(crossDeviceBarrierMemObj);
  FreeSymmBuf(interNodeChunkFlagMemObj);
}

void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}

/* ---------------------------------------------------------------------------------------------- */
/*                              Args construction for Python launch                               */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum) {
  EpDispatchCombineArgsRaw args;
  args.config = handle.config;
  args.fp8BlockwiseCombineScaleDim = handle.fp8BlockwiseCombineScaleDim;
  args.rdmaBlockNum = rdmaBlockNum;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndices = handle.tokenIndices;
  args.inpTokenBuf = handle.inpTokenBuf;
  args.outTokenBuf = handle.outTokenBuf;
  args.weightsBuf = handle.weightsBuf;
  args.scalesBuf = handle.scalesBuf;
  args.destPeTokenCounter = handle.destPeTokenCounter;
  args.localPeTokenCounter = handle.localPeTokenCounter;
  if (handle.config.kernelType == KernelType::IntraNode ||
      handle.config.kernelType == KernelType::IntraNodeLL) {
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

void EpDispatchCombineRoutingPtrs::Validate() const {
  if (IsValid()) return;
  std::string missing;
  auto append = [&](const char* name, const index_t* ptr) {
    if (ptr == nullptr) {
      if (!missing.empty()) missing += ", ";
      missing += name;
    }
  };
  append("dispDestTokIdMap", dispDestTokIdMap);
  append("interNodeDispDestTokIdMap", interNodeDispDestTokIdMap);
  append("interNodeDispSendMap", interNodeDispSendMap);
  append("totalRecvTokenNum", totalRecvTokenNum);
  append("dispTokIdToSrcTokIdLocal", dispTokIdToSrcTokIdLocal);
  throw std::invalid_argument(
      "EpDispatchCombineRoutingPtrs: missing required routing pointer(s): " + missing);
}

EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum,
                                                     const EpDispatchCombineRoutingPtrs* routing,
                                                     bool replayMode) {
  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  args.replayMode = replayMode;
  if (routing != nullptr) {
    routing->Validate();
    args.dispDestTokIdMap = routing->dispDestTokIdMap;
    args.interNodeDispDestTokIdMap = routing->interNodeDispDestTokIdMap;
    args.interNodeDispSendMap = routing->interNodeDispSendMap;
    args.totalRecvTokenNum = routing->totalRecvTokenNum;
    args.dispTokIdToSrcTokIdLocal = routing->dispTokIdToSrcTokIdLocal;
  }
  return args;
}

}  // namespace moe
}  // namespace mori
