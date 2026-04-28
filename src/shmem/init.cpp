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
#include <arpa/inet.h>
#ifdef MORI_WITH_MPI
#include <mpi.h>
#endif
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#include "hip/hip_runtime_api.h"
#include "mori/application/application.hpp"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Helper Functions                                     */
/* ---------------------------------------------------------------------------------------------- */

// Parse size strings with various suffixes (G/GB/GiB, M/MB/MiB, K/KB/KiB)
static size_t ParseSizeString(const std::string& sizeStr) {
  if (sizeStr.empty()) {
    return 0;
  }

  std::string numStr = sizeStr;
  size_t multiplier = 1;

  // Try three-character suffixes first (GiB, MiB, KiB)
  if (numStr.size() >= 3) {
    std::string suffix = numStr.substr(numStr.size() - 3);
    if (suffix == "GiB" || suffix == "gib") {
      multiplier = 1024ULL * 1024ULL * 1024ULL;
      numStr.erase(numStr.size() - 3);
    } else if (suffix == "MiB" || suffix == "mib") {
      multiplier = 1024ULL * 1024ULL;
      numStr.erase(numStr.size() - 3);
    } else if (suffix == "KiB" || suffix == "kib") {
      multiplier = 1024ULL;
      numStr.erase(numStr.size() - 3);
    }
  }

  // Try two-character suffixes (GB, MB, KB)
  if (multiplier == 1 && numStr.size() >= 2) {
    std::string suffix = numStr.substr(numStr.size() - 2);
    if (suffix == "GB" || suffix == "gb" || suffix == "Gb") {
      multiplier = 1024ULL * 1024ULL * 1024ULL;
      numStr.erase(numStr.size() - 2);
    } else if (suffix == "MB" || suffix == "mb" || suffix == "Mb") {
      multiplier = 1024ULL * 1024ULL;
      numStr.erase(numStr.size() - 2);
    } else if (suffix == "KB" || suffix == "kb" || suffix == "Kb") {
      multiplier = 1024ULL;
      numStr.erase(numStr.size() - 2);
    }
  }

  // Fallback to single-character suffixes (G, M, K)
  if (multiplier == 1 && !numStr.empty()) {
    char lastChar = numStr.back();
    if (lastChar == 'G' || lastChar == 'g') {
      multiplier = 1024ULL * 1024ULL * 1024ULL;
      numStr.pop_back();
    } else if (lastChar == 'M' || lastChar == 'm') {
      multiplier = 1024ULL * 1024ULL;
      numStr.pop_back();
    } else if (lastChar == 'K' || lastChar == 'k') {
      multiplier = 1024ULL;
      numStr.pop_back();
    }
  }

  return std::stoull(numStr) * multiplier;
}

// Check if ROCm version is >= 7.0 (required for VMM support)
static bool IsROCmVersionGreaterThan7() {
  int hipVersion;
  hipError_t result = hipRuntimeGetVersion(&hipVersion);
  if (result != hipSuccess) {
    MORI_SHMEM_WARN("Failed to get HIP runtime version");
    return false;
  }

  int hip_major = hipVersion / 10000000;
  int hip_minor = (hipVersion / 100000) % 100;

  MORI_SHMEM_TRACE("Detected HIP version: {}.{} (version code: {})", hip_major, hip_minor,
                   hipVersion);

  return hip_major >= 7;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      RDMA States Initialization                               */
/* ---------------------------------------------------------------------------------------------- */

void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;
  rdmaStates->commContext = new application::Context(*states->bootStates->bootNet);
  MORI_SHMEM_TRACE("RdmaStatesInit: rank {}, worldSize {}", rank, worldSize);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                 Heap Initialization Helpers                                    */
/* ---------------------------------------------------------------------------------------------- */

// Configure heap type from environment variable
static application::HeapType ConfigureHeapType() {
  const char* heapTypeEnv = std::getenv("MORI_SHMEM_HEAP_TYPE");
  application::HeapType heapType = application::HeapType::Uncached;  // Default to uncached

  if (heapTypeEnv) {
    std::string heapTypeStr(heapTypeEnv);
    if (heapTypeStr == "normal" || heapTypeStr == "NORMAL") {
      heapType = application::HeapType::Normal;
      MORI_SHMEM_INFO("Heap type: Normal (cached)");
    } else if (heapTypeStr == "uncached" || heapTypeStr == "UNCACHED") {
      heapType = application::HeapType::Uncached;
      MORI_SHMEM_INFO("Heap type: Uncached");
    } else {
      MORI_SHMEM_WARN("Unknown MORI_SHMEM_HEAP_TYPE '{}', defaulting to uncached", heapTypeStr);
      heapType = application::HeapType::Uncached;
    }
  } else {
    MORI_SHMEM_INFO("MORI_SHMEM_HEAP_TYPE not set, defaulting to uncached");
  }

  return heapType;
}

// Initialize static heap mode
static void InitializeStaticHeap(ShmemStates* states, application::HeapType heapType) {
  MORI_SHMEM_TRACE("Initializing static symmetric heap");

  // Parse heap size from environment variable
  const char* heapSizeEnv = std::getenv("MORI_SHMEM_HEAP_SIZE");
  size_t heapSize = DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE;
  if (heapSizeEnv) {
    heapSize = ParseSizeString(heapSizeEnv);
  }

  MORI_SHMEM_TRACE("Static heap size: {} bytes ({} MB)", heapSize, heapSize / (1024 * 1024));

  // Allocate GPU memory based on heap type
  void* staticHeapPtr = nullptr;
  if (heapType == application::HeapType::Uncached) {
    HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&staticHeapPtr, heapSize, hipDeviceMallocUncached));
  } else {
    HIP_RUNTIME_CHECK(hipMalloc(&staticHeapPtr, heapSize));
  }

  // Initialize memory
  HIP_RUNTIME_CHECK(hipMemset(staticHeapPtr, 0, heapSize));

  // Register with symmetric memory manager
  application::SymmMemObjPtr heapObj =
      states->memoryStates->symmMemMgr->RegisterSymmMemObj(staticHeapPtr, heapSize, true);

  if (!heapObj.IsValid()) {
    MORI_SHMEM_ERROR("Failed to allocate static symmetric heap!");
    throw std::runtime_error("Failed to allocate static symmetric heap");
  }

  // Store heap metadata
  states->memoryStates->staticHeapBasePtr = heapObj.cpu->localPtr;
  states->memoryStates->staticHeapSize = heapSize;
  states->memoryStates->staticHeapObj = heapObj;

  // IMPORTANT: Start with a small offset to avoid collision between heap base address
  // and first ShmemMalloc allocation. Without this, when staticHeapUsed == 0,
  // the first ShmemMalloc would return staticHeapBasePtr, which is the same address
  // as the heap itself in memObjPool, causing the heap's SymmMemObj to be overwritten.
  constexpr size_t HEAP_INITIAL_OFFSET = 256;
  states->memoryStates->staticHeapUsed = HEAP_INITIAL_OFFSET;

  // Initialize VA manager for static heap to enable memory reuse
  states->memoryStates->symmMemMgr->InitHeapVAManager(
      reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr), heapSize);

  MORI_SHMEM_TRACE("Static heap allocated at {} (local), size {} bytes, initial offset {} bytes",
                   states->memoryStates->staticHeapBasePtr, heapSize, HEAP_INITIAL_OFFSET);
  MORI_SHMEM_INFO("Static heap initialized successfully");
}

// Initialize VMM heap mode with fallback to static heap
static bool TryInitializeVMMHeap(ShmemStates* states, application::HeapType heapType) {
  MORI_SHMEM_TRACE("Initializing VMM-based dynamic heap");

  // Check ROCm and hardware VMM support
  bool rocmSupportsVMM = IsROCmVersionGreaterThan7();
  bool hardwareSupportsVMM = states->memoryStates->symmMemMgr->IsVMMSupported();
  MORI_SHMEM_INFO("VMM support: ROCm >= 7.0: {}, Hardware: {}", rocmSupportsVMM,
                  hardwareSupportsVMM);

  if (!rocmSupportsVMM || !hardwareSupportsVMM) {
    MORI_SHMEM_INFO("VMM not supported, will fallback to static heap");
    return false;
  }

  // Parse VMM configuration from environment variables
  const char* chunkSizeEnv = std::getenv("MORI_SHMEM_VMM_CHUNK_SIZE");
  const char* vmmHeapSizeEnv = std::getenv("MORI_SHMEM_HEAP_SIZE");

  size_t chunkSize = 0;  // 0 means auto-detect
  size_t vmmHeapSize = DEFAULT_VMM_SYMMETRIC_HEAP_SIZE;

  if (chunkSizeEnv) {
    chunkSize = std::max(ParseSizeString(chunkSizeEnv), DEFAULT_VMM_MIN_CHUNK_SIZE);
  }
  if (vmmHeapSizeEnv) {
    vmmHeapSize = ParseSizeString(vmmHeapSizeEnv);
  }

  MORI_SHMEM_TRACE("VMM heap config: virtual size {} bytes ({} MB), chunk size {} bytes ({} KB)",
                   vmmHeapSize, vmmHeapSize / (1024 * 1024), chunkSize, chunkSize / 1024);

  // Try to initialize VMM heap
  bool success =
      states->memoryStates->symmMemMgr->InitializeVMMHeap(vmmHeapSize, chunkSize, heapType);

  if (success) {
    // Store VMM heap metadata
    states->memoryStates->useVMMHeap = true;
    states->memoryStates->vmmHeapInitialized = true;
    states->memoryStates->vmmHeapVirtualSize = vmmHeapSize;
    states->memoryStates->vmmHeapChunkSize = states->memoryStates->symmMemMgr->GetVMMChunkSize();
    states->memoryStates->vmmHeapObj = states->memoryStates->symmMemMgr->GetVMMHeapObj();
    states->memoryStates->vmmHeapBaseAddr = states->memoryStates->vmmHeapObj.cpu->localPtr;

    // Initialize VA Manager for VMM heap to enable memory reuse
    // Pass granularity (chunkSize) to ensure VA blocks don't cross physical memory boundaries
    states->memoryStates->symmMemMgr->InitHeapVAManager(
        reinterpret_cast<uintptr_t>(states->memoryStates->vmmHeapBaseAddr), vmmHeapSize,
        states->memoryStates->vmmHeapChunkSize);

    MORI_SHMEM_TRACE(
        "VMM heap VA Manager initialized: base={}, size={} bytes, granularity={} bytes",
        states->memoryStates->vmmHeapBaseAddr, vmmHeapSize, states->memoryStates->vmmHeapChunkSize);
    MORI_SHMEM_INFO("VMM heap initialized successfully");
    return true;
  } else {
    MORI_SHMEM_INFO("Failed to initialize VMM heap, will fallback to static heap");
    return false;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   Memory States Initialization                                */
/* ---------------------------------------------------------------------------------------------- */

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::Context* context = states->rdmaStates->commContext;

  // Create memory management objects
  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr =
      new application::SymmMemManager(*states->bootStates->bootNet, *context);
  states->memoryStates->mrMgr =
      new application::RdmaMemoryRegionManager(*context->GetRdmaDeviceContext());

  // Handle Isolation mode (no heap allocation needed)
  if (states->mode == ShmemMode::Isolation) {
    MORI_SHMEM_INFO("Running in isolation mode (no heap allocation)");
    return;
  }

  // Configure heap type (applies to both VMM and static heap)
  application::HeapType heapType = ConfigureHeapType();
  states->memoryStates->heapType = heapType;

  // Initialize heap based on mode
  switch (states->mode) {
    case ShmemMode::VMHeap: {
      // Try to initialize VMM heap
      bool vmmSuccess = TryInitializeVMMHeap(states, heapType);
      if (vmmSuccess) {
        return;  // VMM heap initialized successfully
      }
      // Fallback to static heap if VMM initialization failed
      MORI_SHMEM_INFO("Falling back to static heap mode");
      states->mode = ShmemMode::StaticHeap;
      // Fall through to StaticHeap case
      [[fallthrough]];
    }

    case ShmemMode::StaticHeap: {
      InitializeStaticHeap(states, heapType);
      break;
    }

    default: {
      MORI_SHMEM_ERROR("Unknown heap mode: {}", static_cast<int>(states->mode));
      throw std::runtime_error("Unknown heap mode");
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      GPU States Initialization                                */
/* ---------------------------------------------------------------------------------------------- */

// Copy transport types to GPU device memory
static void CopyTransportTypesToGpu(GpuStates* gpuStates, const ShmemStates* states) {
  int worldSize = states->bootStates->worldSize;

  HIP_RUNTIME_CHECK(
      hipMalloc(&gpuStates->transportTypes, sizeof(application::TransportType) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(
      gpuStates->transportTypes, states->rdmaStates->commContext->GetTransportTypes().data(),
      sizeof(application::TransportType) * worldSize, hipMemcpyHostToDevice));
}

// Copy RDMA endpoints to GPU device memory
static void CopyRdmaEndpointsToGpu(GpuStates* gpuStates, const ShmemStates* states) {
  if (!states->rdmaStates->commContext->RdmaTransportEnabled()) {
    return;
  }

  size_t numEndpoints = gpuStates->worldSize * gpuStates->numQpPerPe;

  // Allocate and copy endpoints.
  // Convert from host-side application::RdmaEndpoint (which contains ibverbs handles and QP
  // connection parameters unused by device kernels) to the GPU-side ShmemRdmaEndpoint that
  // only contains the fields device kernels actually access.
  HIP_RUNTIME_CHECK(hipMalloc(&gpuStates->rdmaEndpoints, sizeof(ShmemRdmaEndpoint) * numEndpoints));
  {
    const auto& hostEndpoints = states->rdmaStates->commContext->GetRdmaEndpoints();
    std::vector<ShmemRdmaEndpoint> shmemEndpoints(numEndpoints);
    for (size_t i = 0; i < numEndpoints; i++) {
      shmemEndpoints[i].vendorId = hostEndpoints[i].vendorId;
      shmemEndpoints[i].qpn = hostEndpoints[i].handle.qpn;
      shmemEndpoints[i].wqHandle = hostEndpoints[i].wqHandle;
      shmemEndpoints[i].cqHandle = hostEndpoints[i].cqHandle;
      shmemEndpoints[i].atomicIbuf = hostEndpoints[i].atomicIbuf;
    }
    HIP_RUNTIME_CHECK(hipMemcpy(gpuStates->rdmaEndpoints, shmemEndpoints.data(),
                                sizeof(ShmemRdmaEndpoint) * numEndpoints, hipMemcpyHostToDevice));
  }

  // Allocate and initialize endpoint locks
  size_t lockSize = numEndpoints * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&gpuStates->endpointLock, lockSize));
  HIP_RUNTIME_CHECK(hipMemset(gpuStates->endpointLock, 0, lockSize));
}

// Configure heap information for GPU based on current heap mode
static void ConfigureHeapInfoForGpu(GpuStates* gpuStates, const ShmemStates* states) {
  gpuStates->useVMMHeap = states->memoryStates->useVMMHeap;

  switch (states->mode) {
    case ShmemMode::Isolation: {
      // No heap in isolation mode
      gpuStates->heapBaseAddr = 0;
      gpuStates->heapEndAddr = 0;
      gpuStates->heapObj = nullptr;
      gpuStates->vmmChunkSizeShift = 0;
      MORI_SHMEM_TRACE("Isolation mode: no heap info for GPU");
      break;
    }

    case ShmemMode::VMHeap: {
      // VMM heap mode (check if actually initialized)
      if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
        uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->vmmHeapBaseAddr);
        gpuStates->heapBaseAddr = heapBase;
        gpuStates->heapEndAddr = heapBase + states->memoryStates->vmmHeapVirtualSize;
        gpuStates->heapObj = states->memoryStates->vmmHeapObj.gpu;
        gpuStates->vmmChunkSizeShift =
            static_cast<uint8_t>(__builtin_ctzll(states->memoryStates->vmmHeapChunkSize));

        MORI_SHMEM_TRACE(
            "VMM heap configured for GPU: base=0x{:x}, end=0x{:x}, size={} bytes, "
            "chunkSize={} (shift={}), heapObj=0x{:x}",
            gpuStates->heapBaseAddr, gpuStates->heapEndAddr,
            gpuStates->heapEndAddr - gpuStates->heapBaseAddr,
            states->memoryStates->vmmHeapChunkSize, gpuStates->vmmChunkSizeShift,
            reinterpret_cast<uintptr_t>(gpuStates->heapObj));
      } else {
        MORI_SHMEM_ERROR("VMM heap mode but heap not properly initialized");
      }
      break;
    }

    case ShmemMode::StaticHeap: {
      // Static heap mode (single contiguous allocation)
      if (states->memoryStates->staticHeapObj.IsValid()) {
        uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
        gpuStates->heapBaseAddr = heapBase;
        gpuStates->heapEndAddr = heapBase + states->memoryStates->staticHeapSize;
        gpuStates->heapObj = states->memoryStates->staticHeapObj.gpu;
        gpuStates->vmmChunkSizeShift = 0;  // No chunking

        MORI_SHMEM_TRACE(
            "Static heap configured for GPU: base=0x{:x}, end=0x{:x}, size={} bytes, "
            "heapObj=0x{:x}",
            gpuStates->heapBaseAddr, gpuStates->heapEndAddr,
            gpuStates->heapEndAddr - gpuStates->heapBaseAddr,
            reinterpret_cast<uintptr_t>(gpuStates->heapObj));
      } else {
        MORI_SHMEM_ERROR("Static heap mode but heap object is invalid");
      }
      break;
    }

    default: {
      MORI_SHMEM_ERROR("Unknown heap mode: {}", static_cast<int>(states->mode));
      break;
    }
  }
}

// Allocate internal synchronization memory for device barriers
static void AllocateInternalSync(GpuStates* gpuStates, const ShmemStates* states) {
  constexpr size_t MORI_INTERNAL_SYNC_SIZE = 128 * sizeof(uint64_t);
  constexpr size_t ALIGNMENT = 256;
  void* syncPtr = nullptr;

  switch (states->mode) {
    case ShmemMode::StaticHeap: {
      uintptr_t allocAddr = states->memoryStates->symmMemMgr->GetHeapVAManager()->Allocate(
          MORI_INTERNAL_SYNC_SIZE, ALIGNMENT);
      if (allocAddr == 0) {
        MORI_SHMEM_ERROR("Out of static heap memory for internal sync buffer!");
      } else {
        syncPtr = reinterpret_cast<void*>(allocAddr);
        states->memoryStates->symmMemMgr->RegisterStaticHeapSubRegion(
            syncPtr, MORI_INTERNAL_SYNC_SIZE, &states->memoryStates->staticHeapObj);
      }
      break;
    }

    case ShmemMode::VMHeap: {
      application::SymmMemObjPtr syncObj = states->memoryStates->symmMemMgr->VMMAllocChunk(
          MORI_INTERNAL_SYNC_SIZE, states->memoryStates->heapType);
      if (syncObj.IsValid()) {
        syncPtr = syncObj.cpu->localPtr;
      } else {
        MORI_SHMEM_ERROR("Failed to allocate internal sync buffer from VMM heap!");
      }
      break;
    }

    case ShmemMode::Isolation: {
      application::SymmMemObjPtr syncObj =
          states->memoryStates->symmMemMgr->Malloc(MORI_INTERNAL_SYNC_SIZE);
      if (syncObj.IsValid()) {
        syncPtr = syncObj.cpu->localPtr;
      } else {
        MORI_SHMEM_ERROR("Failed to allocate internal sync buffer in isolation mode!");
      }
      break;
    }

    default:
      MORI_SHMEM_ERROR("Unknown ShmemMode for internal sync allocation: {}",
                       static_cast<int>(states->mode));
      break;
  }

  if (syncPtr != nullptr) {
    gpuStates->internalSyncPtr = reinterpret_cast<uint64_t*>(syncPtr);
    HIP_RUNTIME_CHECK(hipMemset(gpuStates->internalSyncPtr, 0, MORI_INTERNAL_SYNC_SIZE));
    MORI_SHMEM_TRACE("Internal sync pointer allocated at 0x{:x} (mode={})",
                     reinterpret_cast<uintptr_t>(gpuStates->internalSyncPtr),
                     static_cast<int>(states->mode));
  }
}

static void FinalizeInternalSync(const ShmemStates* states) {
  if (s_hostGpuStatesCopy.internalSyncPtr == nullptr) {
    return;
  }

  void* syncPtr = reinterpret_cast<void*>(s_hostGpuStatesCopy.internalSyncPtr);
  switch (states->mode) {
    case ShmemMode::StaticHeap: {
      states->memoryStates->symmMemMgr->DeregisterStaticHeapSubRegion(syncPtr);
      states->memoryStates->symmMemMgr->GetHeapVAManager()->Free(
          reinterpret_cast<uintptr_t>(syncPtr));
      break;
    }
    case ShmemMode::VMHeap: {
      states->memoryStates->symmMemMgr->VMMFreeChunk(syncPtr);
      break;
    }
    case ShmemMode::Isolation: {
      states->memoryStates->symmMemMgr->Free(syncPtr);
      break;
    }
    default:
      break;
  }
  MORI_SHMEM_TRACE("Internal sync memory freed (mode={})", static_cast<int>(states->mode));
}

// CopyGpuStatesToDevice is in runtime.cpp

void GpuStateInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  // Initialize basic GPU states
  GpuStates gpuStates;
  gpuStates.rank = states->bootStates->rank;
  gpuStates.worldSize = states->bootStates->worldSize;
  gpuStates.numQpPerPe = states->rdmaStates->commContext->GetNumQpPerPe();

  // Copy communication metadata to GPU
  CopyTransportTypesToGpu(&gpuStates, states);
  CopyRdmaEndpointsToGpu(&gpuStates, states);

  // Configure heap information for GPU access
  ConfigureHeapInfoForGpu(&gpuStates, states);

  // Allocate internal synchronization memory for device barriers
  AllocateInternalSync(&gpuStates, states);

  // Copy complete state to device
  CopyGpuStatesToDevice(&gpuStates);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Shmem Mode Configuration                                 */
/* ---------------------------------------------------------------------------------------------- */

// Determine and configure shmem mode from environment variable
static ShmemMode ConfigureShmemMode() {
  const char* modeEnv = std::getenv("MORI_SHMEM_MODE");

  if (!modeEnv) {
    MORI_SHMEM_INFO("MORI_SHMEM_MODE not set, defaulting to static heap mode");
    return ShmemMode::StaticHeap;
  }

  std::string modeStr(modeEnv);

  if (modeStr == "isolation" || modeStr == "ISOLATION") {
    MORI_SHMEM_INFO("Shmem mode: Isolation");
    return ShmemMode::Isolation;
  } else if (modeStr == "static_heap" || modeStr == "STATIC_HEAP") {
    MORI_SHMEM_INFO("Shmem mode: Static Heap");
    return ShmemMode::StaticHeap;
  } else if (modeStr == "vmm_heap" || modeStr == "VMM_HEAP") {
    MORI_SHMEM_INFO("Shmem mode: VMM Heap");
    return ShmemMode::VMHeap;
  } else {
    MORI_SHMEM_WARN("Unknown MORI_SHMEM_MODE '{}', defaulting to static heap", modeStr);
    return ShmemMode::StaticHeap;
  }
}

// Initialize bootstrap states from bootstrap network
static void InitializeBootStates(ShmemStates* states, application::BootstrapNetwork* bootNet) {
  states->bootStates = new BootStates();
  states->bootStates->bootNet = bootNet;
  states->bootStates->bootNet->Initialize();
  states->bootStates->rank = states->bootStates->bootNet->GetLocalRank();
  states->bootStates->worldSize = states->bootStates->bootNet->GetWorldSize();

  MORI_SHMEM_TRACE("Bootstrap initialized: rank={}, worldSize={}", states->bootStates->rank,
                   states->bootStates->worldSize);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Main Initialization                                      */
/* ---------------------------------------------------------------------------------------------- */

int ShmemInit(application::BootstrapNetwork* bootNet) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  if (states->status == ShmemStatesStatus::Initialized) {
    MORI_SHMEM_INFO("Shmem already initialized, skipping");
    delete bootNet;
    return 0;
  }
  if (states->status == ShmemStatesStatus::Finalized) {
    MORI_SHMEM_ERROR("Shmem has been finalized, cannot re-initialize");
    delete bootNet;
    return -1;
  }

  // Configure shmem mode
  states->mode = ConfigureShmemMode();

  // Initialize all subsystems
  InitializeBootStates(states, bootNet);
  RdmaStatesInit();
  MemoryStatesInit();
  GpuStateInit();

  states->status = ShmemStatesStatus::Initialized;
  MORI_SHMEM_INFO("Shmem initialization completed");
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Finalization Helpers                                     */
/* ---------------------------------------------------------------------------------------------- */

static void FinalizeGpuStates() {
  HIP_RUNTIME_CHECK(hipFree(s_hostGpuStatesCopy.transportTypes));
  HIP_RUNTIME_CHECK(hipFree(s_hostGpuStatesCopy.rdmaEndpoints));
  FinalizeRuntime();
  MORI_SHMEM_TRACE("GPU states finalized");
}

// Clean up VMM heap
static void FinalizeVMMHeap(ShmemStates* states) {
  MORI_SHMEM_TRACE("Finalizing VMM heap");
  states->memoryStates->symmMemMgr->FinalizeVMMHeap();
}

// Clean up static heap
static void FinalizeStaticHeap(ShmemStates* states) {
  MORI_SHMEM_TRACE("Finalizing static heap");

  // Free CPU-side metadata
  free(states->memoryStates->staticHeapObj.cpu->peerPtrs);
  free(states->memoryStates->staticHeapObj.cpu->peerRkeys);
  free(states->memoryStates->staticHeapObj.cpu->ipcMemHandles);

  // Deregister RDMA memory region
  application::RdmaDeviceContext* rdmaDeviceContext =
      states->rdmaStates->commContext->GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    rdmaDeviceContext->DeregisterRdmaMemoryRegion(states->memoryStates->staticHeapBasePtr);
  }

  free(states->memoryStates->staticHeapObj.cpu);

  // Clean up GPU side metadata
  HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu));

  // Free the actual heap memory
  HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapBasePtr));
}

// Clean up heap based on mode
static void FinalizeHeap(ShmemStates* states) {
  if (states->mode == ShmemMode::Isolation) {
    MORI_SHMEM_TRACE("Isolation mode: no heap to finalize");
    return;
  }

  if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
    FinalizeVMMHeap(states);
  } else if (states->memoryStates->staticHeapObj.IsValid()) {
    FinalizeStaticHeap(states);
  }
}

// Clean up all shmem states
static void FinalizeAllStates(ShmemStates* states) {
  // Memory states
  delete states->memoryStates->symmMemMgr;
  delete states->memoryStates->mrMgr;
  delete states->memoryStates;

  // RDMA states
  delete states->rdmaStates->commContext;
  delete states->rdmaStates;

  // Bootstrap states
  states->bootStates->bootNet->Finalize();
  delete states->bootStates->bootNet;
  delete states->bootStates;

  MORI_SHMEM_TRACE("All states finalized");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Main Finalization                                        */
/* ---------------------------------------------------------------------------------------------- */

int ShmemFinalize() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  MORI_SHMEM_TRACE("Starting shmem finalization");

  // Clean up in reverse order of initialization
  FinalizeGpuStates();

  // Clean up internal sync memory
  FinalizeInternalSync(states);

  FinalizeHeap(states);
  FinalizeAllStates(states);

  states->status = ShmemStatesStatus::Finalized;
  MORI_SHMEM_INFO("Shmem finalization completed");
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Other Initialization APIs                                */
/* ---------------------------------------------------------------------------------------------- */

#ifdef MORI_WITH_MPI
int ShmemMpiInit(MPI_Comm mpiComm) {
  return ShmemInit(new application::MpiBootstrapNetwork(mpiComm));
}

int ShmemInit() { return ShmemMpiInit(MPI_COMM_WORLD); }
#endif

int ShmemTorchProcessGroupInit(const std::string& groupName) {
#ifdef MORI_HAS_TORCH
  return ShmemInit(new application::TorchBootstrapNetwork(groupName));
#else
  (void)groupName;
  fprintf(stderr,
          "[mori] Error: Torch bootstrap not available (built without PyTorch).\n"
          "[mori] Use ShmemMpiInit() or rebuild with PyTorch installed.\n");
  return -1;
#endif
}

// Query APIs, Module Init, Barriers are in runtime.cpp

/* ---------------------------------------------------------------------------------------------- */
/*                                      UniqueId-based Initialization                            */
/* ---------------------------------------------------------------------------------------------- */

// Generate a unique ID for socket-based bootstrap
int ShmemGetUniqueId(mori_shmem_uniqueid_t* uid) {
  if (uid == nullptr) {
    MORI_SHMEM_ERROR("ShmemGetUniqueId - invalid input argument");
    return -1;
  }

  try {
    const char* ifname = std::getenv("MORI_SOCKET_IFNAME");
    application::UniqueId socket_uid;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> port_dis(10000, 60000);

    constexpr int kMaxPortRetries = 20;
    bool port_found = false;

    for (int attempt = 0; attempt < kMaxPortRetries; attempt++) {
      int random_port = port_dis(gen);

      int probe_fd = socket(AF_INET, SOCK_STREAM, 0);
      if (probe_fd < 0) continue;

      int opt = 1;
      setsockopt(probe_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      struct sockaddr_in probe_addr{};
      probe_addr.sin_family = AF_INET;
      probe_addr.sin_port = htons(random_port);
      probe_addr.sin_addr.s_addr = htonl(INADDR_ANY);

      if (bind(probe_fd, reinterpret_cast<struct sockaddr*>(&probe_addr), sizeof(probe_addr)) ==
          0) {
        close(probe_fd);

        if (ifname) {
          socket_uid = application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(
              ifname, random_port);
          MORI_SHMEM_TRACE("Generated UniqueId with specified interface: {} (port {})", ifname,
                           random_port);
        } else {
          socket_uid =
              application::SocketBootstrapNetwork::GenerateUniqueIdWithLocalAddr(random_port);
          std::string localAddr = application::SocketBootstrapNetwork::GetLocalNonLoopbackAddress();
          MORI_SHMEM_TRACE("Generated UniqueId with auto-detected interface: {} (port {})",
                           localAddr, random_port);
        }
        port_found = true;
        break;
      }

      close(probe_fd);
      MORI_SHMEM_TRACE("Port {} in use, retrying ({}/{})", random_port, attempt + 1,
                       kMaxPortRetries);
    }

    if (!port_found) {
      MORI_SHMEM_ERROR(
          "Failed to find available port after {} attempts. "
          "Try setting MORI_SOCKET_IFNAME=<interface> (e.g. eth0, eno1) "
          "to specify the network interface.",
          kMaxPortRetries);
      return -1;
    }

    static_assert(sizeof(socket_uid) == sizeof(mori_shmem_uniqueid_t),
                  "UniqueId size mismatch between Socket Bootstrap and mori SHMEM");

    std::memcpy(uid->data(), &socket_uid, sizeof(socket_uid));

    return 0;

  } catch (const std::exception& e) {
    MORI_SHMEM_ERROR(
        "ShmemGetUniqueId failed: {}. "
        "Try setting MORI_SOCKET_IFNAME=<interface> (e.g. eth0, eno1) "
        "to specify the network interface for bootstrap communication.",
        e.what());
    return -1;
  }
}

// Configure initialization attributes for UniqueId-based initialization
int ShmemSetAttrUniqueIdArgs(int rank, int nranks, mori_shmem_uniqueid_t* uid,
                             mori_shmem_init_attr_t* attr) {
  if (uid == nullptr || attr == nullptr) {
    MORI_SHMEM_ERROR("Invalid arguments: uid or attr is null");
    return -1;
  }

  if (rank < 0 || nranks <= 0 || rank >= nranks) {
    MORI_SHMEM_ERROR("Invalid rank={} or nranks={}", rank, nranks);
    return -1;
  }

  // Configure attributes
  attr->rank = rank;
  attr->nranks = nranks;
  attr->uid = *uid;
  attr->mpi_comm = nullptr;  // Not using MPI

  return 0;
}

// Initialize shmem with custom attributes (UniqueId or MPI)
int ShmemInitAttr(unsigned int flags, mori_shmem_init_attr_t* attr) {
  // Validate arguments
  if (attr == nullptr ||
      ((flags != MORI_SHMEM_INIT_WITH_UNIQUEID) && (flags != MORI_SHMEM_INIT_WITH_MPI_COMM))) {
    MORI_SHMEM_ERROR("Invalid arguments");
    return -1;
  }

#ifdef MORI_WITH_MPI
  // MPI-based initialization
  if (flags == MORI_SHMEM_INIT_WITH_MPI_COMM) {
    if (attr->mpi_comm == nullptr) {
      MORI_SHMEM_ERROR("MPI_Comm is null");
      return -1;
    }
    int result = ShmemMpiInit(*reinterpret_cast<MPI_Comm*>(attr->mpi_comm));
    return (result == 0) ? 0 : -1;
  }
#else
  if (flags == MORI_SHMEM_INIT_WITH_MPI_COMM) {
    MORI_SHMEM_ERROR("MPI support is not enabled. Rebuild with -DWITH_MPI=ON.");
    return -1;
  }
#endif

  // UniqueId-based initialization
  if (flags == MORI_SHMEM_INIT_WITH_UNIQUEID) {
    // Validate rank parameters
    if (attr->nranks <= 0 || attr->rank < 0 || attr->rank >= attr->nranks) {
      MORI_SHMEM_ERROR("Invalid rank={} or nranks={}", attr->rank, attr->nranks);
      return -1;
    }

    try {
      // Convert UniqueId and create bootstrap network
      application::UniqueId socket_uid;
      std::memcpy(&socket_uid, attr->uid.data(), sizeof(socket_uid));

      auto socket_bootstrap = std::make_unique<application::SocketBootstrapNetwork>(
          socket_uid, attr->rank, attr->nranks);

      MORI_SHMEM_TRACE("Socket Bootstrap created - rank={}, nranks={}", attr->rank, attr->nranks);

      // Initialize shmem
      int result = ShmemInit(socket_bootstrap.release());
      if (result != 0) {
        MORI_SHMEM_ERROR("ShmemInit failed with code {}", result);
        return -1;
      }

      MORI_SHMEM_TRACE("Successfully initialized with UniqueId");
      return 0;

    } catch (const std::exception& e) {
      MORI_SHMEM_ERROR(
          "UniqueId initialization failed: {}. "
          "Try setting MORI_SOCKET_IFNAME=<interface> (e.g. eth0, eno1) "
          "to specify the network interface for bootstrap communication.",
          e.what());
      return -1;
    }
  }

  return -1;
}

}  // namespace shmem
}  // namespace mori
