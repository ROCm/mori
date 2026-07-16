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
#include <string>

#include "mori/cco/cco.hpp"
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
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_,
                                                 uintptr_t ccoCommPtr_)
    : config(config_) {
  ccoCommPtr = reinterpret_cast<void*>(ccoCommPtr_);
  useCcoComm = (ccoCommPtr != nullptr);
  assert(IsPowerOf2(config.gpuPerNode) && (config.worldSize % config.gpuPerNode == 0));
  if (!useCcoComm) {
    int shmemNumQpPerPe = ShmemNumQpPerPe();
    if (config.numQpPerPe > shmemNumQpPerPe) {
      config.numQpPerPe = shmemNumQpPerPe;
      MORI_OPS_INFO("numQpPerPe {} larger than shmem numQpPerPe {}, set to {}", config.numQpPerPe,
                    shmemNumQpPerPe, shmemNumQpPerPe);
    }
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
  config.enableSdma = useCcoComm ? false : ShmemSdmaEnabled();
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
  if (!useCcoComm) {
    auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
    if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
      return;
    }
  }
  hipDeviceSynchronize();
  (void)hipGetLastError();
  FinalizeShmemBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeOrderMapBuf();
  FinalizeBarrier();
  // Release any cco symmetric resources not already freed by the Finalize*Buf
  // routines (all symmetric frees route through FreeSymm; this is a safety net).
  if (useCcoComm) FinalizeCcoAllocs();
}

mori::application::SymmMemObjPtr EpDispatchCombineHandle::MallocSymm(size_t size,
                                                                     unsigned int flags) {
  if (!useCcoComm) {
    void* buf = ShmemExtMallocWithFlags(size, flags);
    HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
    mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
    assert(obj.IsValid());
    return obj;
  }

  // CCO backend: carve every symmetric buffer out of ONE LSA window (bump
  // allocator). gfx942/ROCm fails hipMemSetAccess after a few *separate* VMM
  // allocations, so we mirror FlyDSL's SymmArena: a single ccoWindowRegister,
  // then sub-regions addressed by offset. Peer flat-VA of a sub-region is just
  // ccoGetPeerPtr(subLocalPtr, pe), so kernels' GetAs<T*>(pe) work unchanged.
  auto* comm = reinterpret_cast<mori::cco::ccoComm*>(ccoCommPtr);
  if (ccoArenaLocalPtr == nullptr) {
    ccoArenaSize = 6ull * static_cast<size_t>(config.MaxNumTokensToRecv()) * config.HiddenDimSz() *
                       static_cast<size_t>(config.maxTokenTypeSize) +
                   (static_cast<size_t>(256) << 20);
    mori::cco::ccoWindow_t awin = nullptr;
    void* aptr = nullptr;
    int rc = mori::cco::ccoWindowRegister(comm, ccoArenaSize, &awin, &aptr);
    if (rc != 0 || aptr == nullptr) {
      throw std::runtime_error("MallocSymm: cco arena register failed (rc=" + std::to_string(rc) +
                               ")");
    }
    ccoArenaWin = reinterpret_cast<void*>(awin);
    ccoArenaLocalPtr = aptr;
    ccoArenaBump = 0;
    HIP_RUNTIME_CHECK(hipMemset(aptr, 0, ccoArenaSize));
  }

  const size_t kAlign = 256;
  const size_t aligned = (size + kAlign - 1) & ~(kAlign - 1);
  if (ccoArenaBump + aligned > ccoArenaSize) {
    throw std::runtime_error("MallocSymm: cco arena exhausted (need " + std::to_string(aligned) +
                             ", have " + std::to_string(ccoArenaSize - ccoArenaBump) + ")");
  }
  void* localPtr = static_cast<char*>(ccoArenaLocalPtr) + ccoArenaBump;
  ccoArenaBump += aligned;  // sub-region already zeroed by the arena memset

  const int ws = config.worldSize;
  std::vector<uintptr_t> peers(ws);
  for (int pe = 0; pe < ws; ++pe) {
    peers[pe] = reinterpret_cast<uintptr_t>(mori::cco::ccoGetPeerPtr(comm, localPtr, pe));
  }
  uintptr_t* p2pDev = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&p2pDev, ws * sizeof(uintptr_t)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(p2pDev, peers.data(), ws * sizeof(uintptr_t), hipMemcpyHostToDevice));

  auto* cpuObj = new mori::application::SymmMemObj();
  cpuObj->localPtr = localPtr;
  cpuObj->p2pPeerPtrs = p2pDev;  // device array; only dereferenced on device
  cpuObj->size = size;
  cpuObj->worldSize = ws;

  mori::application::SymmMemObj* gpuObj = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuObj, sizeof(mori::application::SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuObj, cpuObj, sizeof(mori::application::SymmMemObj),
                              hipMemcpyHostToDevice));

  // win=nullptr marks a sub-region (arena window is freed once in FinalizeCcoAllocs).
  ccoAllocs.push_back(CcoSymmAlloc{nullptr, localPtr, cpuObj, gpuObj, p2pDev});

  mori::application::SymmMemObjPtr obj;
  obj.cpu = cpuObj;
  obj.gpu = gpuObj;
  return obj;
}

void EpDispatchCombineHandle::FreeSymmByLocalPtr(void* localPtr) {
  if (localPtr == nullptr) return;
  if (!useCcoComm) {
    mori::shmem::ShmemFree(localPtr);
    return;
  }
  // Sub-region: drop only its shadow objects; the shared arena window is released
  // once in FinalizeCcoAllocs.
  for (size_t i = 0; i < ccoAllocs.size(); ++i) {
    if (ccoAllocs[i].localPtr != localPtr) continue;
    CcoSymmAlloc a = ccoAllocs[i];
    if (a.gpuObj) (void)hipFree(a.gpuObj);
    if (a.p2pDev) (void)hipFree(a.p2pDev);
    delete a.cpuObj;
    ccoAllocs.erase(ccoAllocs.begin() + i);
    break;
  }
}

void EpDispatchCombineHandle::FinalizeCcoAllocs() {
  if (!useCcoComm) return;
  auto* comm = reinterpret_cast<mori::cco::ccoComm*>(ccoCommPtr);
  for (auto& a : ccoAllocs) {
    if (a.gpuObj) (void)hipFree(a.gpuObj);
    if (a.p2pDev) (void)hipFree(a.p2pDev);
    delete a.cpuObj;
  }
  ccoAllocs.clear();
  if (ccoArenaWin != nullptr) {
    mori::cco::ccoWindowDeregister(comm, reinterpret_cast<mori::cco::ccoWindow_t>(ccoArenaWin));
    if (ccoArenaLocalPtr) mori::cco::ccoMemFree(comm, ccoArenaLocalPtr);
    ccoArenaWin = nullptr;
    ccoArenaLocalPtr = nullptr;
  }
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
    bufs.combineInp = MallocSymm(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = MallocSymm(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = MallocSymm(combineOutSize, hipDeviceMallocUncached);
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
    bufs.dispatchInp = MallocSymm(dispatchInpSize, hipDeviceMallocUncached);
    bufs.combineInp = MallocSymm(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = MallocSymm(stagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = MallocSymm(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = MallocSymm(combineOutSize, hipDeviceMallocUncached);
    bufs.dispatchStaging =
        MallocSymm(dispatchStagingSize, hipDeviceMallocUncached);
  } else {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNode>();
    // NOTE(ditian12): no overflow protection for dispatchInp/combinInp/staging in async kernel,
    // hence have to allocate to max size we need to either implement compact layout or add
    // pre-assertion to prevent silent memory access fault
    size_t maxStagingSize =
        static_cast<ssize_t>(config.MaxNumTokensToSend()) * config.MaxXferBytesPerToken();
    bufs.dispatchInp = MallocSymm(maxStagingSize, hipDeviceMallocUncached);
    bufs.combineInp = MallocSymm(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = MallocSymm(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = MallocSymm(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = MallocSymm(combineOutSize, hipDeviceMallocUncached);
  }

  size_t maxWeightSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(float);
  shmemInpWeightsMemObj = MallocSymm(maxWeightSize, hipDeviceMallocUncached);
  shmemDispatchOutWeightsMemObj =
      MallocSymm(maxWeightSize, hipDeviceMallocUncached);
  shmemCombineOutWeightsMemObj =
      MallocSymm(maxWeightSize, hipDeviceMallocUncached);

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
    shmemInpScalesMemObj = MallocSymm(inpScaleSize, hipDeviceMallocUncached);
  }
  if (userScaleSize > 0) {
    shmemOutScalesMemObj = MallocSymm(userScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = MallocSymm(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = MallocSymm(maxIndicesSize, hipDeviceMallocUncached);

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
  if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL) {
    auto& bufs = std::get<ShmemBufsIntraNode>(shmemTokBufs);
    FreeSymmByLocalPtr(bufs.dispatchOut->localPtr);
    FreeSymmByLocalPtr(bufs.combineInp->localPtr);
    FreeSymmByLocalPtr(bufs.combineOut->localPtr);
  } else if (config.kernelType == KernelType::InterNodeV1 ||
             config.kernelType == KernelType::InterNodeV1LL) {
    auto& bufs = std::get<ShmemBufsInterNodeV1>(shmemTokBufs);
    FreeSymmByLocalPtr(bufs.dispatchInp->localPtr);
    FreeSymmByLocalPtr(bufs.combineInp->localPtr);
    FreeSymmByLocalPtr(bufs.dispatchOut->localPtr);
    FreeSymmByLocalPtr(bufs.combineOut->localPtr);
    FreeSymmByLocalPtr(bufs.staging->localPtr);
    FreeSymmByLocalPtr(bufs.dispatchStaging->localPtr);
  } else {
    auto& bufs = std::get<ShmemBufsInterNode>(shmemTokBufs);
    FreeSymmByLocalPtr(bufs.dispatchInp->localPtr);
    FreeSymmByLocalPtr(bufs.combineInp->localPtr);
    FreeSymmByLocalPtr(bufs.dispatchOut->localPtr);
    FreeSymmByLocalPtr(bufs.combineOut->localPtr);
    FreeSymmByLocalPtr(bufs.staging->localPtr);
  }
  FreeSymmByLocalPtr(shmemInpWeightsMemObj->localPtr);
  FreeSymmByLocalPtr(shmemDispatchOutWeightsMemObj->localPtr);
  FreeSymmByLocalPtr(shmemCombineOutWeightsMemObj->localPtr);
  if (shmemInpScalesMemObj.IsValid()) FreeSymmByLocalPtr(shmemInpScalesMemObj->localPtr);
  if (shmemOutScalesMemObj.IsValid()) FreeSymmByLocalPtr(shmemOutScalesMemObj->localPtr);
  FreeSymmByLocalPtr(shmemInpIndicesMemObj->localPtr);
  FreeSymmByLocalPtr(shmemOutIndicesMemObj->localPtr);
#ifdef ENABLE_PROFILER
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeBuf));
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeOffset));
#endif
}

void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
  recvTokenNumMemObj = MallocSymm(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = MallocSymm(tokenNumSignalSize, hipDeviceMallocUncached);
  sendAtomicSignalMemObj = MallocSymm(
      (config.worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));

  size_t nodeTokenNumSignalSize = config.worldSize / config.gpuPerNode * sizeof(uint64_t);
  nodeRecvTokenNumMemObj =
      MallocSymm(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  FreeSymmByLocalPtr(recvTokenNumMemObj->localPtr);
  FreeSymmByLocalPtr(sendTokenNumMemObj->localPtr);
  FreeSymmByLocalPtr(sendAtomicSignalMemObj->localPtr);
  FreeSymmByLocalPtr(nodeRecvTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
}

void EpDispatchCombineHandle::InitializeOrderMapBuf() {
  size_t maxNumOutToken =
      static_cast<size_t>(config.MaxNumTokensToSend()) * config.numExpertPerRank;
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

  dispTokOffsetMemObj = MallocSymm(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      MallocSymm(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(index_t)));

  size_t maxNumInterNodeToken = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                config.MaxNumTokensToSendPerRank() * config.numExpertPerToken;
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(interNodeDispDestTokIdMap, 0, maxNumInterNodeToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(
      hipMalloc(&blockFlagCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(blockFlagCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  size_t interNodeDispSendMapSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                    config.MaxNumTokensToSendPerRank() * sizeof(index_t);
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
  FreeSymmByLocalPtr(dispTokOffsetMemObj->localPtr);
  FreeSymmByLocalPtr(dispTokIdToSrcTokIdMemObj->localPtr);
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
      MallocSymm(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

  size_t interNodeChunkFlagSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                  config.MaxNumTokensToSendPerRank() * sizeof(uint64_t);
  interNodeChunkFlagMemObj =
      MallocSymm(interNodeChunkFlagSize, hipDeviceMallocUncached);

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
  FreeSymmByLocalPtr(crossDeviceBarrierMemObj->localPtr);
  FreeSymmByLocalPtr(interNodeChunkFlagMemObj->localPtr);
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
  // BW diagnostic knob: 0 = off (baseline). N>0 injects N dummy remote atomics
  // per sent token in intra-node dispatch to probe the remote-atomic bottleneck.
  args.bwDummyAtomics = env::GetPositiveIntOr("MORI_BW_DUMMY_ATOMICS", 0);
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
