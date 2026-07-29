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
#include <cstring>
#include <new>
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
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_)
    : config(config_) {
  NormalizeConfig();
  try {
    InitializeAll();
  } catch (...) {
    // A throw out of a constructor skips the destructor, so nothing would ever
    // release what InitializeAll() managed to allocate before it failed. Now
    // that a failed allocation raises instead of exit(-1)ing, that is a
    // reachable leak of symmetric-heap space that no later call can reclaim
    // (the handle never becomes an object). Tear the partial build down here.
    FinalizeAll();
    throw;
  }

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

  if (config.quantType == QuantType::Fp8BlockwiseQuant) {
    fp8BlockwiseCombineScaleDim =
        env::GetPositiveIntOr(kFp8BlockwiseScaleDimEnv, kDefaultFp8BlockwiseScaleDim);
    fp8BlockwiseCombineScaleTypeSize = static_cast<int>(sizeof(float));
    if (config.rank == 0) {
      MORI_OPS_INFO("Fp8BlockwiseQuant combine scale_dim={} (override via {})",
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
  // Set the flag BEFORE allocating, not after. FinalizeAll() early-returns on
  // !buffersInitialized, so if this were set last a throw partway through would
  // leave a partially-built handle that FinalizeAll() refuses to touch: the
  // rollback in Reconfigure() would free nothing and then overwrite every
  // pointer, leaking whatever the failed attempt had already taken from the
  // symmetric heap. That leak is permanent and cumulative -- each failed D->P
  // flip would make the next one likelier to fail. Every Finalize*() body is
  // null/invalidate based, so tearing down a half-built handle is safe.
  buffersInitialized = true;
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();
}

void EpDispatchCombineHandle::FinalizeAll() {
  if (!buffersInitialized) return;
  FinalizeShmemBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeOrderMapBuf();
  FinalizeBarrier();
  // Every Finalize*() above nulls / invalidates what it released, so a second
  // pass is a no-op rather than a double free.
  buffersInitialized = false;
}

void EpDispatchCombineHandle::Finalize() {
  if (!buffersInitialized) return;
  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
    // The shmem context is already gone; the symmetric heap went with it, so
    // the symmetric buffers cannot (and need not) be handed back. The plain
    // hipMalloc scratch (order maps, barriers, totalRecvTokenNum) is NOT owned
    // by shmem though -- returning early here would leak it while
    // IsInitialized() started reporting false. Run the normal teardown;
    // ShmemFreeAndInvalidate skips the ShmemFree on a dead context.
    FinalizeAll();
    return;
  }
  // The op must be idle. Drain any in-flight kernel before releasing buffers
  // the kernels may still be writing to.
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  FinalizeAll();
}

// Stable marker on every ValidateReconfigurable rejection.
//
// A rejection is categorically different from a resize FAILURE: it is raised
// before anything is freed, so the op is bit-for-bit untouched and the group is
// not in a partial state at all. The python wrapper reduces a severity across
// the EP group and rewrites the message to the group's worst outcome, and
// without this marker a rejection was indistinguishable from an OOM -- asking
// for max_num_inp_token_per_rank=0 reported "could not grow the a2a buffers ...
// every rank rolled back", which is false in both halves and would send sglang
// down its retry path for a caller bug that will fail identically forever.
//
// A marker rather than a distinct exception type because the public contract is
// already "reconfigure raises RuntimeError" and sglang matches on the message;
// changing the type would break callers for a diagnostic improvement. The tag
// is stripped before the message reaches the user.
static constexpr const char* kReconfigureRejectedTag = "[reconfigure-rejected]";

void EpDispatchCombineHandle::ValidateReconfigurable(
    const EpDispatchCombineConfig& newConfig) const {
  auto require = [](const char* field, long long oldV, long long newV) {
    if (oldV != newV) {
      throw std::runtime_error(std::string(kReconfigureRejectedTag) +
                               " EpDispatchCombineHandle::Reconfigure: field '" + field +
                               "' must not change (" + std::to_string(oldV) + " -> " +
                               std::to_string(newV) + ")");
    }
  };
  // Peer-visible / layout-defining fields: changing any of them would make this
  // rank's symmetric buffers disagree with its peers'.
  require("rank", config.rank, newConfig.rank);
  require("worldSize", config.worldSize, newConfig.worldSize);
  require("gpuPerNode", config.gpuPerNode, newConfig.gpuPerNode);
  require("kernelType", static_cast<long long>(config.kernelType),
          static_cast<long long>(newConfig.kernelType));
  require("hiddenDim", config.hiddenDim, newConfig.hiddenDim);
  require("numExpertPerRank", config.numExpertPerRank, newConfig.numExpertPerRank);
  require("numExpertPerToken", config.numExpertPerToken, newConfig.numExpertPerToken);
  require("quantType", static_cast<long long>(config.quantType),
          static_cast<long long>(newConfig.quantType));
  require("scaleDim", config.scaleDim, newConfig.scaleDim);
  require("scaleTypeSize", config.scaleTypeSize, newConfig.scaleTypeSize);
  require("maxTokenTypeSize", config.maxTokenTypeSize, newConfig.maxTokenTypeSize);
  // numQpPerPe is owned by the shmem context (NormalizeConfig only clamps it to
  // ShmemNumQpPerPe) and Reconfigure does NOT rebuild that context, so it is as
  // peer-visible as kernelType. useExternalInpBuffer decides whether the
  // dispatch input token buffer is allocated at all -- changing it across a
  // reconfigure would silently change which symmetric buffers exist.
  // Compare the CLAMPED form: `config` has already been through NormalizeConfig
  // (which caps numQpPerPe at ShmemNumQpPerPe()) but `newConfig` has not, so a
  // raw compare would spuriously reject a caller that just passed its original
  // config back in.
  require("numQpPerPe", std::min(config.numQpPerPe, ShmemNumQpPerPe()),
          std::min(newConfig.numQpPerPe, ShmemNumQpPerPe()));
  require("useExternalInpBuffer", static_cast<long long>(config.useExternalInpBuffer),
          static_cast<long long>(newConfig.useExternalInpBuffer));
  if (newConfig.maxNumInpTokenPerRank <= 0) {
    throw std::runtime_error(
        std::string(kReconfigureRejectedTag) +
        " EpDispatchCombineHandle::Reconfigure: maxNumInpTokenPerRank must be > 0");
  }
}

void EpDispatchCombineHandle::Reconfigure(const EpDispatchCombineConfig& newConfig) {
  ValidateReconfigurable(newConfig);

  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
    throw std::runtime_error(
        "EpDispatchCombineHandle::Reconfigure: shmem context is not initialized");
  }

  MORI_OPS_INFO("Reconfigure: maxNumInpTokenPerRank {} -> {}, maxTotalRecvTokens {} -> {}",
                config.maxNumInpTokenPerRank, newConfig.maxNumInpTokenPerRank,
                config.maxTotalRecvTokens, newConfig.maxTotalRecvTokens);

  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  // Keep the normalized old config so a failed re-allocation can roll back to a
  // capacity we know fits.
  const EpDispatchCombineConfig oldConfig = config;
  FinalizeAll();

  config = newConfig;
  // Per-inference state refers to the buffers we just freed; drop it so a stale
  // pointer cannot be handed to a kernel before the next PrepareInference.
  curRankNumToken = 0;
  curHiddenDim = -1;
  inpTokenBuf = nullptr;
  outTokenBuf = nullptr;
  weightsBuf = nullptr;
  scalesBuf = nullptr;
  tokenIndices = nullptr;

  NormalizeConfig();
  try {
    InitializeAll();
  } catch (const std::exception& e) {
    // The new (larger) capacity did not fit. Without this the op would be left
    // holding half-built buffers with buffersInitialized=false -- strictly worse
    // than before the flip, and unrecoverable. Roll back to the previous
    // capacity so the caller can keep serving in its old role and surface the
    // failure. Every Finalize*() nulls what it released, so tearing down the
    // partial allocation is safe.
    MORI_OPS_WARN("Reconfigure to maxNumInpTokenPerRank={} failed ({}); rolling back to {}",
                  newConfig.maxNumInpTokenPerRank, e.what(), oldConfig.maxNumInpTokenPerRank);
    FinalizeAll();
    config = oldConfig;
    try {
      InitializeAll();
    } catch (const std::exception& e2) {
      // The rollback itself failed: the heap is in a state we cannot reason
      // about, so do not pretend the op is usable.
      FinalizeAll();
      throw std::runtime_error(
          std::string("EpDispatchCombineHandle::Reconfigure: allocation for the new capacity "
                      "failed (") +
          e.what() + ") AND the rollback to the previous capacity also failed (" + e2.what() +
          "); the op is now finalized and must be rebuilt");
    }
    throw std::runtime_error(
        std::string("EpDispatchCombineHandle::Reconfigure: could not grow the a2a buffers to "
                    "maxNumInpTokenPerRank=") +
        std::to_string(newConfig.maxNumInpTokenPerRank) + " (" + e.what() +
        "); rolled back to " + std::to_string(oldConfig.maxNumInpTokenPerRank) +
        ", the op is still usable at the old capacity");
  }
}

EpDispatchCombineHandle::~EpDispatchCombineHandle() {
  if (!buffersInitialized) {
    return;
  }
  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
    return;
  }
  hipDeviceSynchronize();
  (void)hipGetLastError();
  FinalizeAll();
}

mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags) {
  void* buf = ShmemExtMallocWithFlags(size, flags);
  if (buf == nullptr) {
    // Symmetric heap exhausted. This is a REACHABLE path now that buffers get
    // re-allocated at a larger capacity on a role switch (sglang D->P grows
    // maxNumInpTokenPerRank), so it must be a recoverable exception rather than
    // the hipMemset-on-null / assert it used to be. Reconfigure() catches this
    // and rolls back to the previous capacity.
    throw std::bad_alloc();
  }
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
  mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
  if (!obj.IsValid()) {
    ShmemFree(buf);
    throw std::runtime_error(
        "EpDispatchCombine: symmetric allocation of " + std::to_string(size) +
        " bytes is not registered in the SymmMemObj pool");
  }
  return obj;
}

// hipMalloc + hipMemset that THROWS instead of exit(-1)ing the process.
//
// The plain-device scratch buffers (order maps, barriers, counters) scale with
// maxNumOutToken = MaxNumTokensToSend() * numExpertPerRank, i.e. with the very
// capacity a role switch changes. Under HIP_RUNTIME_CHECK an allocation failure
// here calls exit(-1) (application/utils/check.hpp:39), so a D->P flip that
// could not grow this scratch would SIGKILL the whole inference server instead
// of raising -- and plain device memory is exactly what sglang's KV cache has
// already eaten, which makes this the MORE likely OOM of the two, not the rare
// one. Reconfigure() catches std::bad_alloc and rolls back to the previous
// capacity, so raising keeps the all-or-nothing contract that sglang relies on.
//
// `what` names the buffer so the rollback message can say which one ran out.
//
// kNoMemset leaves the allocation uninitialized. That is NOT a micro-
// optimization: hipMemset is asynchronous with respect to the host for device
// memory, so for a buffer the host initializes itself on the next line (only
// crossDeviceBarrierFlag does) an added memset can land AFTER the host store
// and silently clobber it. Doing that turned the barrier flag into 0 for the
// IntraNode kernels and produced partial peer contributions in combine -- the
// same "some peers' data missing" corruption the correctness bar exists to
// prevent. Call sites must match the original init exactly.
static constexpr int kNoMemset = -1000;

// Fault injection, test-only. Set MORI_TEST_FAIL_HIPMALLOC to the `what` name of
// a plain-device buffer (e.g. "dispSenderIdxMap") and the next allocation of it
// fails as if the device were out of memory.
//
// This exists because the failure it simulates cannot otherwise be provoked on
// demand: the symmetric heap can be exhausted by asking for a capacity larger
// than MORI_SHMEM_HEAP_SIZE, but plain device memory is whatever the GPU has
// free, so "a D->P flip that cannot grow the order maps" is untestable without a
// hook -- and that path (raise + roll back, rather than exit(-1)) is precisely
// the contract sglang's role switch depends on. Untested error paths are how the
// exit(-1) survived this long.
//
// Unset => never fires. Fires MORI_TEST_FAIL_HIPMALLOC_TIMES times (default 1),
// then disarms itself.
//
// It used to re-read the env on every call with no limit, which sabotaged the
// very path it exists to test. Reconfigure()'s rollback re-runs InitializeAll()
// at the OLD config, so a permanently-armed hook failed the rollback too --
// turning every injected "grow failed, rolled back, still usable" into "rollback
// also failed, the op is finalized". A test cannot disarm between the two: they
// are both inside the single reconfigure() call.
//
// So the count is the knob, and it selects which of the two contracts is under
// test:
//   TIMES=1 (default) -- the grow fails, the smaller rollback allocation fits.
//     Models transient device-memory pressure at the larger size. Yields
//     "could not grow ... still usable at the old capacity" (SEVERITY_ROLLED_BACK).
//   TIMES=2 -- the rollback fails too, so the handle ends finalized. Yields
//     "the op is now finalized and must be rebuilt" (SEVERITY_UNRECOVERABLE).
//     That is the state RequireInitialized() in the pybind layer exists to
//     protect, and the state sglang's MoriA2AResizeUnrecoverable escalation
//     keys on; with a hard one-shot it was unreachable from any test, i.e. the
//     SIGSEGV fixed in 85c34c18 could not be proven gone.
static bool ShouldInjectAllocFailure(const char* what) {
  const char* target = env::Get("MORI_TEST_FAIL_HIPMALLOC");
  if ((target == nullptr) || (target[0] == '\0') || (what == nullptr)) return false;
  if (std::strcmp(target, what) != 0) return false;

  // Re-arming (a new test setting the var again, possibly to a different site)
  // resets the count. Keyed on the target string so consecutive tests in one
  // worker process do not inherit each other's remaining fires.
  static std::string armedFor;
  static int firesLeft = 0;
  if (armedFor != target) {
    armedFor = target;
    const char* timesStr = env::Get("MORI_TEST_FAIL_HIPMALLOC_TIMES");
    firesLeft = 1;
    if ((timesStr != nullptr) && (timesStr[0] != '\0')) {
      firesLeft = std::atoi(timesStr);
    }
    if (firesLeft < 1) firesLeft = 1;
  }

  firesLeft--;
  if (firesLeft <= 0) {
    // Disarm, and forget the target so a later re-arm at the same site works.
    ::unsetenv("MORI_TEST_FAIL_HIPMALLOC");
    armedFor.clear();
  }
  return true;
}

static void HipMallocOrThrow(void** ptr, size_t size, const char* what, int memsetValue = 0) {
  *ptr = nullptr;
  if (ShouldInjectAllocFailure(what)) {
    MORI_OPS_WARN("MORI_TEST_FAIL_HIPMALLOC: failing allocation of '{}' ({} bytes) on purpose",
                  what, size);
    throw std::bad_alloc();
  }
  hipError_t err = hipMalloc(ptr, size);
  if (err != hipSuccess || *ptr == nullptr) {
    // Clear the sticky error so a later HIP_RUNTIME_CHECK on the rollback path
    // does not trip on this failure and exit(-1) after we have recovered.
    (void)hipGetLastError();
    *ptr = nullptr;
    throw std::bad_alloc();
  }
  if (memsetValue == kNoMemset) return;
  err = hipMemset(*ptr, memsetValue, size);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    (void)hipFree(*ptr);
    *ptr = nullptr;
    throw std::bad_alloc();
  }
}

// Typed wrapper so call sites keep their natural pointer types.
template <typename T>
static void HipMallocOrThrow(T** ptr, size_t size, const char* what, int memsetValue = 0) {
  HipMallocOrThrow(reinterpret_cast<void**>(ptr), size, what, memsetValue);
}

void EpDispatchCombineHandle::InitializeShmemBuf() {
  size_t combineOutSize = static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) *
                          config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t dispatchOutSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) *
                           config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t maxStagingSize =
      static_cast<ssize_t>(config.MaxNumTokensToRecv()) * config.MaxXferBytesPerToken();
  if (config.kernelType == KernelType::IntraNode &&
      config.quantType == QuantType::Fp8BlockwiseQuant) {
    size_t blockwiseScaleBytes =
        (fp8BlockwiseCombineScaleDim > 0)
            ? static_cast<size_t>(fp8BlockwiseCombineScaleDim) * fp8BlockwiseCombineScaleTypeSize
            : 0;
    maxStagingSize = static_cast<size_t>(config.MaxNumTokensToRecv()) *
                     (config.HiddenBytes(config.maxTokenTypeSize) + config.IndexBytes() +
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

  size_t maxWeightSize = config.MaxNumTokensToRecv() * config.numExpertPerToken * sizeof(float);
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
  if (config.quantType == QuantType::Fp8BlockwiseQuant && fp8BlockwiseCombineScaleDim > 0) {
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

// Free a symmetric buffer and invalidate the handle so a repeated Finalize
// (or a use-after-free from a stale kernel arg) is caught rather than silently
// double-freeing. Safe on an already-invalid / never-allocated obj.
//
// If the shmem context has already been torn down the symmetric heap went with
// it, so there is nothing to give back: just invalidate. This lets the plain
// hipMalloc scratch in the same Finalize*() body still be released instead of
// leaking (which is what an early return from Finalize() used to do).
static void ShmemFreeAndInvalidate(mori::application::SymmMemObjPtr& obj) {
  if (!obj.IsValid()) return;
  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status == mori::shmem::ShmemStatesStatus::Initialized) {
    ShmemFree(obj->localPtr);
  }
  obj = mori::application::SymmMemObjPtr{nullptr, nullptr};
}

// hipFree + null, so FinalizeAll() is idempotent for plain device scratch too.
template <typename T>
static void HipFreeAndNull(T*& ptr) {
  if (ptr == nullptr) return;
  HIP_RUNTIME_CHECK(hipFree(ptr));
  ptr = nullptr;
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL) {
    auto& bufs = std::get<ShmemBufsIntraNode>(shmemTokBufs);
    ShmemFreeAndInvalidate(bufs.dispatchOut);
    ShmemFreeAndInvalidate(bufs.combineInp);
    ShmemFreeAndInvalidate(bufs.combineOut);
  } else if (config.kernelType == KernelType::InterNodeV1 ||
             config.kernelType == KernelType::InterNodeV1LL) {
    auto& bufs = std::get<ShmemBufsInterNodeV1>(shmemTokBufs);
    ShmemFreeAndInvalidate(bufs.dispatchInp);
    ShmemFreeAndInvalidate(bufs.combineInp);
    ShmemFreeAndInvalidate(bufs.dispatchOut);
    ShmemFreeAndInvalidate(bufs.combineOut);
    ShmemFreeAndInvalidate(bufs.staging);
    ShmemFreeAndInvalidate(bufs.dispatchStaging);
  } else {
    auto& bufs = std::get<ShmemBufsInterNode>(shmemTokBufs);
    ShmemFreeAndInvalidate(bufs.dispatchInp);
    ShmemFreeAndInvalidate(bufs.combineInp);
    ShmemFreeAndInvalidate(bufs.dispatchOut);
    ShmemFreeAndInvalidate(bufs.combineOut);
    ShmemFreeAndInvalidate(bufs.staging);
  }
  ShmemFreeAndInvalidate(shmemInpWeightsMemObj);
  ShmemFreeAndInvalidate(shmemDispatchOutWeightsMemObj);
  ShmemFreeAndInvalidate(shmemCombineOutWeightsMemObj);
  ShmemFreeAndInvalidate(shmemInpScalesMemObj);
  ShmemFreeAndInvalidate(shmemOutScalesMemObj);
  ShmemFreeAndInvalidate(shmemInpIndicesMemObj);
  ShmemFreeAndInvalidate(shmemOutIndicesMemObj);
#ifdef ENABLE_PROFILER
  HipFreeAndNull(profilerConfig.debugTimeBuf);
  HipFreeAndNull(profilerConfig.debugTimeOffset);
#endif
}

void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendAtomicSignalMemObj = ShmemMallocAndReturnMemObjPtr(
      (config.worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);

  HipMallocOrThrow(&totalRecvTokenNum, sizeof(index_t), "totalRecvTokenNum");

  size_t nodeTokenNumSignalSize = config.worldSize / config.gpuPerNode * sizeof(uint64_t);
  nodeRecvTokenNumMemObj =
      ShmemMallocAndReturnMemObjPtr(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFreeAndInvalidate(recvTokenNumMemObj);
  ShmemFreeAndInvalidate(sendTokenNumMemObj);
  ShmemFreeAndInvalidate(sendAtomicSignalMemObj);
  ShmemFreeAndInvalidate(nodeRecvTokenNumMemObj);
  HipFreeAndNull(totalRecvTokenNum);
}

void EpDispatchCombineHandle::InitializeOrderMapBuf() {
  size_t maxNumOutToken = config.MaxNumTokensToSend() * config.numExpertPerRank;
  // These scale with the capacity a role switch changes, so they must raise
  // rather than exit(-1) on OOM -- see HipMallocOrThrow.
  HipMallocOrThrow(&dispReceiverIdxMap, maxNumOutToken * sizeof(index_t), "dispReceiverIdxMap");
  HipMallocOrThrow(&dispSenderIdxMap, maxNumOutToken * sizeof(index_t), "dispSenderIdxMap");
  HipMallocOrThrow(&destPeTokenIdxMap, maxNumOutToken * sizeof(index_t), "destPeTokenIdxMap", -1);
  HipMallocOrThrow(&srcPeTokenIdxMap, maxNumOutToken * sizeof(index_t), "srcPeTokenIdxMap", -1);
  HipMallocOrThrow(&destPeTokenCounter, config.worldSize * sizeof(index_t), "destPeTokenCounter");

  HipMallocOrThrow(&destNodeTokenCounter, config.worldSize / config.gpuPerNode * sizeof(index_t),
                   "destNodeTokenCounter");
  HipMallocOrThrow(&localPeTokenCounter, config.worldSize * sizeof(index_t), "localPeTokenCounter");

  dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HipMallocOrThrow(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t), "dispDestTokIdMap");

  size_t maxNumInterNodeToken = config.worldSize / config.gpuPerNode *
                                config.MaxNumTokensToSendPerRank() * config.numExpertPerToken;
  HipMallocOrThrow(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t),
                   "interNodeDispDestTokIdMap");
  HipMallocOrThrow(&blockFlagCounter, config.worldSize / config.gpuPerNode * sizeof(index_t),
                   "blockFlagCounter");

  size_t interNodeDispSendMapSize =
      config.worldSize / config.gpuPerNode * config.MaxNumTokensToSendPerRank() * sizeof(index_t);
  HipMallocOrThrow(&interNodeDispSendMap, interNodeDispSendMapSize, "interNodeDispSendMap");

#ifdef ENABLE_STANDARD_MOE_ADAPT
  const size_t maxDispatchTokens = static_cast<size_t>(config.MaxNumTokensToRecv());
  const size_t mapSize = maxDispatchTokens * config.numExpertPerToken * sizeof(uint64_t);
  HipMallocOrThrow(&dispTokToEpSlotMap, mapSize, "dispTokToEpSlotMap");
  HipMallocOrThrow(&standardPackedRecvCount, config.numExpertPerRank * sizeof(int),
                   "standardPackedRecvCount");
#endif
}

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  HipFreeAndNull(dispReceiverIdxMap);
  HipFreeAndNull(dispSenderIdxMap);
  HipFreeAndNull(destPeTokenIdxMap);
  HipFreeAndNull(srcPeTokenIdxMap);
  HipFreeAndNull(destPeTokenCounter);
  HipFreeAndNull(destNodeTokenCounter);
  HipFreeAndNull(localPeTokenCounter);
  ShmemFreeAndInvalidate(dispTokOffsetMemObj);
  ShmemFreeAndInvalidate(dispTokIdToSrcTokIdMemObj);
  HipFreeAndNull(dispDestTokIdMap);
  HipFreeAndNull(interNodeDispDestTokIdMap);
  HipFreeAndNull(blockFlagCounter);
  HipFreeAndNull(interNodeDispSendMap);
#ifdef ENABLE_STANDARD_MOE_ADAPT
  HipFreeAndNull(dispTokToEpSlotMap);
  HipFreeAndNull(standardPackedRecvCount);
#endif
}

void EpDispatchCombineHandle::InitializeBarrier() {
  size_t barrierSize = config.worldSize * sizeof(uint32_t);
  HipMallocOrThrow(&dispatchGridBarrier, barrierSize, "dispatchGridBarrier");
  HipMallocOrThrow(&combineGridBarrier, barrierSize, "combineGridBarrier");
  // No memset: the host writes this flag on the very next line, and an async
  // hipMemset can land after that store and clobber it. See kNoMemset.
  HipMallocOrThrow(&crossDeviceBarrierFlag, sizeof(uint64_t), "crossDeviceBarrierFlag", kNoMemset);
  crossDeviceBarrierFlag[0] = ((config.kernelType == KernelType::InterNodeV1) ||
                               (config.kernelType == KernelType::InterNodeV1LL) ||
                               (config.kernelType == KernelType::AsyncLL))
                                  ? 0
                                  : 1;
  crossDeviceBarrierMemObj =
      ShmemMallocAndReturnMemObjPtr(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

  size_t interNodeChunkFlagSize =
      config.worldSize / config.gpuPerNode * config.MaxNumTokensToSendPerRank() * sizeof(uint64_t);
  interNodeChunkFlagMemObj =
      ShmemMallocAndReturnMemObjPtr(interNodeChunkFlagSize, hipDeviceMallocUncached);

  HipMallocOrThrow(&interNodeChunkFlagCombine, interNodeChunkFlagSize, "interNodeChunkFlagCombine");
  HipMallocOrThrow(&interNodeBlocksBarrier, 4 * sizeof(index_t), "interNodeBlocksBarrier");
}

void EpDispatchCombineHandle::FinalizeBarrier() {
  HipFreeAndNull(dispatchGridBarrier);
  HipFreeAndNull(combineGridBarrier);
  HipFreeAndNull(crossDeviceBarrierFlag);
  HipFreeAndNull(interNodeChunkFlagCombine);
  HipFreeAndNull(interNodeBlocksBarrier);
  ShmemFreeAndInvalidate(crossDeviceBarrierMemObj);
  ShmemFreeAndInvalidate(interNodeChunkFlagMemObj);
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

}  // namespace moe
}  // namespace mori
