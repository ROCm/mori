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

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
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
#include "src/ops/dispatch_combine/low_latency_async.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

// ---------------------------------------------------------------------------
// JIT kernel module loader
// ---------------------------------------------------------------------------
class KernelModule {
 public:
  void load(const char* hsaco_path) {
    std::string path(hsaco_path);
    if (loaded_paths_.count(path)) return;

    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoad(&mod, hsaco_path);
    if (err != hipSuccess) {
      MORI_OPS_ERROR("Failed to load kernel module from %s: %d", hsaco_path, err);
      return;
    }
    modules_.push_back(mod);
    loaded_paths_.insert(path);
    gpu_states_initialized_ = false;
  }

  void ensure_gpu_states() {
    if (gpu_states_initialized_ || modules_.empty()) return;
    for (auto mod : modules_) {
      mori::shmem::ShmemModuleInit(static_cast<void*>(mod));
    }
    gpu_states_initialized_ = true;
  }

  bool is_loaded() const { return !modules_.empty(); }

  hipFunction_t get(const std::string& name) {
    auto it = funcs_.find(name);
    if (it != funcs_.end()) return it->second;
    for (auto mod : modules_) {
      hipFunction_t func = nullptr;
      hipError_t err = hipModuleGetFunction(&func, mod, name.c_str());
      if (err == hipSuccess && func) {
        funcs_[name] = func;
        return func;
      }
      (void)hipGetLastError();
    }
    MORI_OPS_ERROR("Kernel function '%s' not found in any loaded module", name.c_str());
    return nullptr;
  }

  ~KernelModule() {
    for (auto mod : modules_) {
      hipModuleUnload(mod);
    }
    modules_.clear();
  }

 private:
  std::vector<hipModule_t> modules_;
  std::set<std::string> loaded_paths_;
  bool gpu_states_initialized_ = false;
  std::unordered_map<std::string, hipFunction_t> funcs_;
};

static KernelModule s_jit_module;

template <typename T> const char* type_suffix();
template <> const char* type_suffix<hip_bfloat16>() { return "bf16"; }
template <> const char* type_suffix<float>() { return "f32"; }
template <> const char* type_suffix<mori_fp4x2_e2m1>() { return "fp4"; }
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
template <> const char* type_suffix<__hip_fp8_e4m3_fnuz>() { return "fp8_fnuz"; }
#endif
#ifdef MORI_FP8_TYPE_OCP_ENABLED
template <> const char* type_suffix<__hip_fp8_e4m3>() { return "fp8_ocp"; }
#endif

template <typename ArgsT>
void jit_launch(const std::string& func_name, dim3 grid, dim3 block,
                size_t shared_mem, hipStream_t stream, ArgsT& args) {
  if (!s_jit_module.is_loaded()) {
    MORI_OPS_ERROR("JIT kernel module not loaded. Call load_ops_kernels() first.");
    assert(false && "JIT kernel module not loaded");
    return;
  }
  s_jit_module.ensure_gpu_states();
  hipFunction_t func = s_jit_module.get(func_name);
  if (!func) {
    MORI_OPS_ERROR("Kernel function '%s' not found", func_name.c_str());
    assert(false && "Kernel function not found");
    return;
  }
  void* params[] = {&args};
  hipError_t err = hipModuleLaunchKernel(
      func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      shared_mem, stream, params, nullptr);
  if (err != hipSuccess) {
    MORI_OPS_ERROR("hipModuleLaunchKernel(%s) failed: %d", func_name.c_str(), err);
    assert(false && "hipModuleLaunchKernel failed");
  }
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
    MORI_OPS_INFO("numQpPerPe %d larger than shmem numQpPerPe %d, set to %d", config.numQpPerPe,
                  shmemNumQpPerPe, shmemNumQpPerPe);
  }
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();

  this->multiProcessorCount = GetCurDeviceMultiProcessorCount();
  this->maxThreads = std::min(GetCurDeviceMaxThreads(), 1024);
  MORI_OPS_INFO("Device capability: multiProcessorCount=%d, maxThreads=%d",
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
  // NOTE: config.numQpPerPe is for async kernel's multi-qp optimization
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
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
                               (config.kernelType == KernelType::InterNodeV1LL) ||
                               (config.kernelType == KernelType::AsyncLL))
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

void EpDispatchCombineHandle::LaunchIntraNodeDispatch(int blockNum, int rdmaBlockNum,
                                                      int warpPerBlock, hipStream_t stream,
                                                      int hiddenDim) {
  LaunchDispatch(KernelType::IntraNode, blockNum, rdmaBlockNum, warpPerBlock, stream, hiddenDim);
}

void EpDispatchCombineHandle::LaunchInterNodeDispatch(int blockNum, int rdmaBlockNum,
                                                      int warpPerBlock, hipStream_t stream,
                                                      int hiddenDim) {
  LaunchDispatch(KernelType::InterNode, blockNum, rdmaBlockNum, warpPerBlock, stream, hiddenDim);
}

void EpDispatchCombineHandle::LaunchIntraNodeCombine(int blockNum, int rdmaBlockNum,
                                                     int warpPerBlock, int useExternalInpBuf,
                                                     hipStream_t stream, int hiddenDim) {
  LaunchCombine(KernelType::IntraNode, blockNum, rdmaBlockNum, warpPerBlock, useExternalInpBuf,
                stream, hiddenDim);
}

void EpDispatchCombineHandle::LaunchInterNodeCombine(int blockNum, int rdmaBlockNum,
                                                     int warpPerBlock, int useExternalInpBuf,
                                                     hipStream_t stream, int hiddenDim) {
  LaunchCombine(KernelType::InterNode, blockNum, rdmaBlockNum, warpPerBlock, useExternalInpBuf,
                stream, hiddenDim);
}

void EpDispatchCombineHandle::LaunchDispatch(KernelType kernelType, int blockNum, int rdmaBlockNum,
                                             int warpPerBlock, hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  size_t actualRdmaBlockNum = (rdmaBlockNum <= 0) ? config.rdmaBlockNum : rdmaBlockNum;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  size_t sharedMemSize =
      (config.worldSize * actualWarpNumPerBlock + config.numExpertPerRank * actualWarpNumPerBlock +
       config.numExpertPerRank) *
      sizeof(index_t);
  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, actualRdmaBlockNum);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;
        args.config.hiddenDim = actualHiddenDim;

        std::string sfx = type_suffix<DataT>();
        if (kernelType == KernelType::InterNode) {
          assert(config.useExternalInpBuffer);
          jit_launch("EpDispatchInterNodeKernel_" + sfx, grid, block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::InterNodeV1) {
          jit_launch("EpDispatchCopyToStaging_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
          jit_launch("EpDispatchInterNodeV1Kernel_" + sfx, grid, block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::InterNodeV1LL) {
          jit_launch("EpDispatchCopyToStaging_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
          jit_launch("EpDispatchInterNodeV1KernelLowLatency_" + sfx, grid, block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::IntraNode) {
          jit_launch("EpDispatchIntraNodeKernel_" + sfx, grid, block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::AsyncLL) {
          assert(config.useExternalInpBuffer);
          jit_launch("EpDispatchLowLatencyAsyncSend_" + sfx, grid, block, sharedMemSize, stream, args);
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchDispatchRecv(KernelType kernelType, int blockNum,
                                                 int warpPerBlock, hipStream_t stream) {
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  size_t actualBlockNum = (blockNum <= 0) ? config.blockNum : blockNum;
  dim3 grid(actualBlockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  size_t sharedMemSize =
      (config.worldSize * actualWarpNumPerBlock + config.numExpertPerRank * actualWarpNumPerBlock +
       config.numExpertPerRank) *
      sizeof(index_t);
  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, 0);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;
        std::string sfx = type_suffix<DataT>();
        if (kernelType == KernelType::AsyncLL) {
          assert(config.useExternalInpBuffer);
          assert((actualBlockNum % config.worldSize) == 0);
          jit_launch("EpDispatchLowLatencyAsyncRecv_" + sfx, grid, block, sharedMemSize, stream, args);
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchCombine(KernelType kernelType, int blockNum, int rdmaBlockNum,
                                            int warpPerBlock, int useExternalInpBuf,
                                            hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  // Determine actual values: use parameter if >= 0, otherwise use config
  const size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  const size_t actualRdmaBlockNum = (rdmaBlockNum <= 0) ? config.rdmaBlockNum : rdmaBlockNum;
  const bool actualUseExternalInpBuffer =
      (useExternalInpBuf >= 0) ? static_cast<bool>(useExternalInpBuf) : config.useExternalInpBuffer;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, actualRdmaBlockNum);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;

        // Override args.config.useExternalInpBuffer with the actual value
        args.config.useExternalInpBuffer = actualUseExternalInpBuffer;
        args.config.hiddenDim = actualHiddenDim;

        std::string sfx = type_suffix<DataT>();
        size_t sharedMemSize =
            actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
        if (kernelType == KernelType::InterNode) {
          assert(actualUseExternalInpBuffer);
          jit_launch("EpCombineInterNodeKernel_" + sfx, grid, block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::InterNodeV1) {
          assert(actualUseExternalInpBuffer);
          jit_launch("EpCombineSync_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
          jit_launch("EpCombineSyncBarrier_" + sfx, dim3(1), dim3(warpSize), 0, stream, args);
          jit_launch("EpCombineInterNodeV1Kernel_" + sfx, grid, block, sharedMemSize, stream, args);
          jit_launch("EpCombineAll_" + sfx, dim3(this->multiProcessorCount), block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::InterNodeV1LL) {
          assert(actualUseExternalInpBuffer);
          jit_launch("EpCombineSync_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
          jit_launch("EpCombineSyncBarrier_" + sfx, dim3(1), dim3(warpSize), 0, stream, args);
          jit_launch("EpCombineInterNodeV1KernelLowLatency_" + sfx, grid, block, sharedMemSize, stream, args);
          jit_launch("EpCombineAll_" + sfx, dim3(this->multiProcessorCount), block, sharedMemSize, stream, args);
        } else if (kernelType == KernelType::IntraNode) {
#ifdef ENABLE_STANDARD_MOE_ADAPT
          jit_launch("EpCombineIntraNodeKernel_" + sfx + "_p2p", grid, block, sharedMemSize, stream, args);
#else
          if (actualUseExternalInpBuffer) {
            if constexpr (std::is_same_v<DataT, hip_bfloat16>) {
              const bool useFp8DirectCast = (config.quantType == QuantType::Fp8DirectCast);
              if (useFp8DirectCast) {
                jit_launch(std::string("EpCombineIntraNodeKernel_bf16_nop2p_fp8cast"), grid, block, sharedMemSize, stream, args);
              } else {
                jit_launch("EpCombineIntraNodeKernel_" + sfx + "_nop2p", grid, block, sharedMemSize, stream, args);
              }
            } else {
              jit_launch("EpCombineIntraNodeKernel_" + sfx + "_nop2p", grid, block, sharedMemSize, stream, args);
            }
          } else {
            assert(config.quantType != QuantType::Fp8DirectCast &&
                   "Fp8DirectCast is not supported in zero-copy mode");
            jit_launch("EpCombineIntraNodeKernel_" + sfx + "_p2p", grid, block, sharedMemSize, stream, args);
          }
#endif
        } else if (kernelType == KernelType::AsyncLL) {
          assert(config.useExternalInpBuffer);
          if constexpr (std::is_same_v<DataT, hip_bfloat16>) {
            const bool useFp8DirectCast = (config.quantType == QuantType::Fp8DirectCast);
            if (useFp8DirectCast) {
#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
              jit_launch(std::string("EpCombineLowLatencyAsyncSend_bf16_fp8cast"), grid, block, sharedMemSize, stream, args);
#else
              assert(false && "Fp8DirectCast requires FP8 type support in this build");
#endif
            } else {
              jit_launch("EpCombineLowLatencyAsyncSend_" + sfx, grid, block, sharedMemSize, stream, args);
            }
          } else {
            assert(config.quantType != QuantType::Fp8DirectCast &&
                   "Fp8DirectCast combine only supports bf16 input for AsyncLL");
            jit_launch("EpCombineLowLatencyAsyncSend_" + sfx, grid, block, sharedMemSize, stream, args);
          }
        } else {
          assert(false);
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchCombineRecv(KernelType kernelType, int blockNum,
                                                int warpPerBlock, hipStream_t stream) {
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  size_t actualBlockNum = (blockNum <= 0) ? config.blockNum : blockNum;
  dim3 grid(actualBlockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, 0);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;
        std::string sfx = type_suffix<DataT>();
        size_t sharedMemSize =
            actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
        if (kernelType == KernelType::AsyncLL) {
          assert(config.useExternalInpBuffer);
          assert((actualBlockNum % config.worldSize) == 0);
          if constexpr (std::is_same_v<DataT, hip_bfloat16>) {
            const bool useFp8DirectCast = (config.quantType == QuantType::Fp8DirectCast);
            if (useFp8DirectCast) {
#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
              jit_launch(std::string("EpCombineLowLatencyAsyncRecv_bf16_fp8cast"), grid, block, sharedMemSize, stream, args);
#else
              assert(false && "Fp8DirectCast requires FP8 type support in this build");
#endif
            } else {
              jit_launch("EpCombineLowLatencyAsyncRecv_" + sfx, grid, block, sharedMemSize, stream, args);
            }
          } else {
            assert(config.quantType != QuantType::Fp8DirectCast &&
                   "Fp8DirectCast combine only supports bf16 input for AsyncLL");
            jit_launch("EpCombineLowLatencyAsyncRecv_" + sfx, grid, block, sharedMemSize, stream, args);
          }
        } else {
          assert(false);
        }
      },
      argsVariant);
}

#ifdef ENABLE_STANDARD_MOE_ADAPT
void EpDispatchCombineHandle::LaunchDispatchForStandardMoE(KernelType kernelType, int blockNum,
                                                           int rdmaBlockNum, int warpPerBlock,
                                                           hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  size_t actualRdmaBlockNum = (rdmaBlockNum <= 0) ? config.rdmaBlockNum : rdmaBlockNum;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  size_t sharedMemSize =
      (config.worldSize * actualWarpNumPerBlock + config.numExpertPerRank * actualWarpNumPerBlock +
       config.numExpertPerRank) *
      sizeof(index_t);
  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, actualRdmaBlockNum);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;
        if constexpr (std::is_same_v<DataT, mori_fp4x2_e2m1>) {
          assert(false && "fp4x2 is not supported for standard MoE dispatch");
        } else {
          args.config.hiddenDim = actualHiddenDim;
          std::string sfx = type_suffix<DataT>();
          if (kernelType == KernelType::InterNodeV1LL) {
            jit_launch("EpDispatchCopyToStaging_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
            jit_launch("EpDispatchInterNodeV1KernelLowLatency_" + sfx + "_stdmoe", grid, block, sharedMemSize, stream, args);
          } else if (kernelType == KernelType::IntraNode) {
            jit_launch("EpDispatchIntraNodeKernel_" + sfx + "_stdmoe", grid, block, sharedMemSize, stream, args);
          } else {
            assert(false &&
                   "LaunchDispatchForStandardMoE only supports IntraNode/InterNodeV1LL kernel type");
          }
        }
      },
      argsVariant);
}

void EpDispatchCombineHandle::LaunchCombineForStandardMoE(KernelType kernelType, int blockNum,
                                                          int rdmaBlockNum, int warpPerBlock,
                                                          hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  size_t actualRdmaBlockNum = (rdmaBlockNum <= 0) ? config.rdmaBlockNum : rdmaBlockNum;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  auto argsVariant = GetEpDispatchCombineArgsByInputType(*this, actualRdmaBlockNum);
  std::visit(
      [&](auto&& args) {
        using ArgsT = std::decay_t<decltype(args)>;
        using DataT = typename ArgsT::data_type;
        if constexpr (std::is_same_v<DataT, mori_fp4x2_e2m1>) {
          assert(false && "fp4x2 is not supported for standard MoE combine");
        } else {
          args.config.hiddenDim = actualHiddenDim;
          std::string sfx = type_suffix<DataT>();
          size_t sharedMemSize =
              actualWarpNumPerBlock * config.numExpertPerToken * (sizeof(DataT**) + sizeof(float**));
          if (kernelType == KernelType::InterNodeV1LL) {
            jit_launch("EpCombineSync_" + sfx, dim3(this->multiProcessorCount), block, 0, stream, args);
            jit_launch("EpCombineSyncBarrier_" + sfx, dim3(1), dim3(warpSize), 0, stream, args);
            jit_launch("EpCombineInterNodeV1KernelLowLatency_" + sfx + "_stdmoe", grid, block, sharedMemSize, stream, args);
            jit_launch("EpCombineAll_" + sfx, dim3(this->multiProcessorCount), block, sharedMemSize, stream, args);
          } else if (kernelType == KernelType::IntraNode) {
            jit_launch("EpCombineIntraNodeKernel_" + sfx + "_p2p_stdmoe", grid, block, sharedMemSize, stream, args);
          } else {
            assert(false &&
                   "LaunchCombineForStandardMoE only supports IntraNode/InterNodeV1LL kernel type");
          }
        }
      },
      argsVariant);
}
#endif  // ENABLE_STANDARD_MOE_ADAPT

#ifdef ENABLE_STANDARD_MOE_ADAPT
__device__ void ConvertDispatchOutputKernel_body(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}

__global__ void ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputKernel_body(args);
}

void EpDispatchCombineHandle::LaunchConvertDispatchOutputKernel(
    const void* dispatchOutX, const void* dispatchOutTopkIdx, void* packedRecvX,
    int* packedRecvCount, int* packedRecvSrcInfo, int64_t* packedRecvLayoutRange, int blockNum,
    int warpPerBlock, hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  ConvertDispatchOutputArgs args{};
  args.config = config;
  args.config.hiddenDim = actualHiddenDim;
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

  jit_launch(std::string("mori_ConvertDispatchOutputKernel"), grid, block, 0, stream, args);
}

void EpDispatchCombineHandle::LaunchConvertCombineInputKernel(
    const void* packedRecvX, const void* packedRecvSrcInfo, const void* packedRecvLayoutRange,
    void* combineInput, mori::application::SymmMemObjPtr shmemCombineInpTokMemObj, int blockNum,
    int warpPerBlock, hipStream_t stream, int hiddenDim) {
  const int actualHiddenDim = (hiddenDim > 0) ? hiddenDim : config.hiddenDim;
  assert(actualHiddenDim > 0 && actualHiddenDim <= config.hiddenDim);
  size_t actualWarpNumPerBlock = (warpPerBlock <= 0) ? config.warpNumPerBlock : warpPerBlock;
  dim3 grid((blockNum <= 0) ? config.blockNum : blockNum);
  dim3 block(warpSize * actualWarpNumPerBlock);

  ConvertCombineInputArgs args{};
  args.config = config;
  args.config.hiddenDim = actualHiddenDim;
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
      jit_launch(std::string("ConvertCombineInputKernel_f32"), grid, block, 0, stream, args);
      break;
    case HIP_R_16BF:
      jit_launch(std::string("ConvertCombineInputKernel_bf16"), grid, block, 0, stream, args);
      break;
#ifdef MORI_FP8_TYPE_OCP_ENABLED
    case HIP_R_8F_E4M3:
      jit_launch(std::string("ConvertCombineInputKernel_fp8_ocp"), grid, block, 0, stream, args);
      break;
#endif
#ifdef MORI_FP8_TYPE_FNUZ_ENABLED
    case HIP_R_8F_E4M3_FNUZ:
      jit_launch(std::string("ConvertCombineInputKernel_fp8_fnuz"), grid, block, 0, stream, args);
      break;
#endif
    default:
      assert(false);
      break;
  }
}
#endif  // ENABLE_STANDARD_MOE_ADAPT

// no need for a separate reset kernel now
void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}

void LoadJitKernelModule(const char* hsaco_path) {
  s_jit_module.load(hsaco_path);
}

}  // namespace moe
}  // namespace mori
