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
#include <mpi.h>

#include <cstdlib>

#include "hip/hip_runtime.h"
#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Initialization */
/* ---------------------------------------------------------------------------------------------- */
__constant__ GpuStates globalGpuStates;

bool IsROCmVersionGreaterThan7() {
  // Check HIP version which corresponds to ROCm version
  int hipVersion;
  hipError_t result = hipRuntimeGetVersion(&hipVersion);
  if (result != hipSuccess) {
    MORI_SHMEM_WARN("Failed to get HIP runtime version, using static heap as fallback");
    return false;
  }

  int hip_major = hipVersion / 10000000;
  int hip_minor = (hipVersion / 100000) % 100;

  MORI_SHMEM_INFO("Detected HIP version: {}.{} (version code: {})", hip_major, hip_minor,
                  hipVersion);

  return hip_major >= 6;
}

void RdmaStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->rdmaStates = new RdmaStates();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;

  rdmaStates->commContext = new application::Context(*states->bootStates->bootNet);
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::Context* context = states->rdmaStates->commContext;

  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr =
      new application::SymmMemManager(*states->bootStates->bootNet, *context);
  states->memoryStates->mrMgr =
      new application::RdmaMemoryRegionManager(*context->GetRdmaDeviceContext());

  // Auto-select heap type based on ROCm version and other factors
  bool useVMM = false;

  // Check if ROCm version supports VMM (>= 7.0)
  bool rocmSupportsVMM = IsROCmVersionGreaterThan7();

  // Check hardware VMM support
  bool hardwareSupportsVMM = states->memoryStates->symmMemMgr->IsVMMSupported();

  // Check environment variable override
  const char* forceVMMEnv = std::getenv("MORI_SHMEM_USE_VMM");
  if (forceVMMEnv) {
    useVMM = (std::string(forceVMMEnv) == "1" || std::string(forceVMMEnv) == "true" ||
              std::string(forceVMMEnv) == "TRUE");
    MORI_SHMEM_INFO("VMM usage forced by environment variable: {}",
                    useVMM ? "enabled" : "disabled");
  } else {
    // Auto-select based on ROCm version and hardware support
    useVMM = rocmSupportsVMM && hardwareSupportsVMM;
    MORI_SHMEM_INFO(
        "Auto-selecting heap type: ROCm >= 7.0: {}, Hardware VMM support: {}, Using VMM: {}",
        rocmSupportsVMM, hardwareSupportsVMM, useVMM);
  }

  if (useVMM) {
    // Initialize VMM-based dynamic heap
    const char* chunkSizeEnv = std::getenv("MORI_SHMEM_VMM_CHUNK_SIZE");
    size_t chunkSize = 0;
    const char* vmmHeapSizeEnv = std::getenv("MORI_SHMEM_VMM_HEAP_SIZE");
    size_t vmmHeapSize = DEFAULT_VMM_SYMMETRIC_HEAP_SIZE;
    size_t vmmMultiplier = 1;
    if (chunkSizeEnv) {
      std::string chunkSizeStr(chunkSizeEnv);

      if (!chunkSizeStr.empty()) {
        char lastChar = chunkSizeStr.back();
        if (lastChar == 'M' || lastChar == 'm') {
          vmmMultiplier = 1024ULL * 1024ULL;  // MiB
          chunkSizeStr.pop_back();
        } else if (lastChar == 'K' || lastChar == 'k') {
          vmmMultiplier = 1024ULL;  // KiB
          chunkSizeStr.pop_back();
        }
      }

      chunkSize = std::stoull(chunkSizeStr) * vmmMultiplier;
    }

    if (vmmHeapSizeEnv) {
      std::string vmmHeapSizeStr(vmmHeapSizeEnv);

      // Check for suffix
      if (!vmmHeapSizeStr.empty()) {
        char lastChar = vmmHeapSizeStr.back();
        if (lastChar == 'G' || lastChar == 'g') {
          vmmMultiplier = 1024ULL * 1024ULL * 1024ULL;  // GiB
          vmmHeapSizeStr.pop_back();
        } else if (lastChar == 'M' || lastChar == 'm') {
          vmmMultiplier = 1024ULL * 1024ULL;  // MiB
          vmmHeapSizeStr.pop_back();
        }
      }

      vmmHeapSize = std::stoull(vmmHeapSizeStr) * vmmMultiplier;
    }

    MORI_SHMEM_INFO(
        "Initializing VMM-based dynamic heap: virtual size {} bytes ({} MB), chunk size {} bytes "
        "({} KB)",
        vmmHeapSize, vmmHeapSize / (1024 * 1024), chunkSize, chunkSize / 1024);

    bool vmmSuccess = states->memoryStates->symmMemMgr->InitializeVMMHeap(vmmHeapSize, chunkSize);
    if (vmmSuccess) {
      states->memoryStates->useVMMHeap = true;
      states->memoryStates->vmmHeapInitialized = true;
      states->memoryStates->vmmHeapVirtualSize = vmmHeapSize;
      states->memoryStates->vmmHeapChunkSize = chunkSize;
      states->memoryStates->vmmHeapObj = states->memoryStates->symmMemMgr->GetVMMHeapObj();
      states->memoryStates->vmmHeapBaseAddr = states->memoryStates->vmmHeapObj.cpu->localPtr;

      MORI_SHMEM_INFO("VMM-based dynamic heap initialized successfully");
      return;
    } else {
      MORI_SHMEM_WARN("Failed to initialize VMM heap, falling back to static heap");
    }
  } else {
    MORI_SHMEM_INFO("VMM not supported or disabled, using static heap");
  }

  // Fallback to static heap allocation
  // Configure heap size
  const char* heapSizeEnv = std::getenv("MORI_SHMEM_STATIC_HEAP_SIZE");
  size_t heapSize = DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE;

  if (heapSizeEnv) {
    std::string heapSizeStr(heapSizeEnv);
    size_t multiplier = 1;

    // Check for suffix
    if (!heapSizeStr.empty()) {
      char lastChar = heapSizeStr.back();
      if (lastChar == 'G' || lastChar == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;  // GiB
        heapSizeStr.pop_back();
      } else if (lastChar == 'M' || lastChar == 'm') {
        multiplier = 1024ULL * 1024ULL;  // MiB
        heapSizeStr.pop_back();
      }
    }

    heapSize = std::stoull(heapSizeStr) * multiplier;
  }
  MORI_SHMEM_INFO("Allocating static symmetric heap of size {} bytes ({} MB)", heapSize,
                  heapSize / (1024 * 1024));

  // Allocate the symmetric heap using the SymmMemManager
  void* staticHeapPtr = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&staticHeapPtr, heapSize, hipDeviceMallocUncached));
  HIP_RUNTIME_CHECK(hipMemset(staticHeapPtr, 0, heapSize));
  application::SymmMemObjPtr heapObj =
      states->memoryStates->symmMemMgr->RegisterSymmMemObj(staticHeapPtr, heapSize, true);

  if (!heapObj.IsValid()) {
    MORI_SHMEM_ERROR("Failed to allocate static symmetric heap!");
    throw std::runtime_error("Failed to allocate static symmetric heap");
  }

  states->memoryStates->staticHeapBasePtr = heapObj.cpu->localPtr;
  states->memoryStates->staticHeapSize = heapSize;
  // IMPORTANT: Start with a small offset to avoid collision between heap base address
  // and first ShmemMalloc allocation. Without this, when staticHeapUsed == 0,
  // the first ShmemMalloc would return staticHeapBasePtr, which is the same address
  // as the heap itself in memObjPool, causing the heap's SymmMemObj to be overwritten.
  constexpr size_t HEAP_INITIAL_OFFSET = 256;
  states->memoryStates->staticHeapUsed = HEAP_INITIAL_OFFSET;
  states->memoryStates->staticHeapObj = heapObj;

  MORI_SHMEM_INFO(
      "Static symmetric heap allocated at {} (local), size {} bytes, initial offset {} bytes",
      states->memoryStates->staticHeapBasePtr, heapSize, HEAP_INITIAL_OFFSET);
}

void GpuStateInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  RdmaStates* rdmaStates = states->rdmaStates;

  int rank = states->bootStates->rank;
  int worldSize = states->bootStates->worldSize;

  // Copy to gpu constance memory
  GpuStates gpuStates;
  gpuStates.rank = rank;
  gpuStates.worldSize = worldSize;
  gpuStates.numQpPerPe = rdmaStates->commContext->GetNumQpPerPe();

  // Copy transport types to GPU
  HIP_RUNTIME_CHECK(
      hipMalloc(&gpuStates.transportTypes, sizeof(application::TransportType) * worldSize));
  HIP_RUNTIME_CHECK(
      hipMemcpy(gpuStates.transportTypes, rdmaStates->commContext->GetTransportTypes().data(),
                sizeof(application::TransportType) * worldSize, hipMemcpyHostToDevice));

  // Copy endpoints to GPU
  if (rdmaStates->commContext->RdmaTransportEnabled()) {
    size_t numEndpoints = gpuStates.worldSize * gpuStates.numQpPerPe;
    HIP_RUNTIME_CHECK(
        hipMalloc(&gpuStates.rdmaEndpoints, sizeof(application::RdmaEndpoint) * numEndpoints));
    HIP_RUNTIME_CHECK(
        hipMemcpy(gpuStates.rdmaEndpoints, rdmaStates->commContext->GetRdmaEndpoints().data(),
                  sizeof(application::RdmaEndpoint) * numEndpoints, hipMemcpyHostToDevice));

    size_t lockSize = numEndpoints * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMalloc(&gpuStates.endpointLock, lockSize));
    HIP_RUNTIME_CHECK(hipMemset(gpuStates.endpointLock, 0, lockSize));
  }

  // Copy symmetric heap info to GPU
  gpuStates.useVMMHeap = states->memoryStates->useVMMHeap;

  if (states->memoryStates->useVMMHeap) {
    uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->vmmHeapBaseAddr);
    gpuStates.heapBaseAddr = heapBase;
    gpuStates.heapEndAddr = heapBase + states->memoryStates->vmmHeapVirtualSize;
    gpuStates.heapObj = states->memoryStates->vmmHeapObj.gpu;
  } else {
    // Traditional static heap
    uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
    gpuStates.heapBaseAddr = heapBase;
    gpuStates.heapEndAddr = heapBase + states->memoryStates->staticHeapSize;
    gpuStates.heapObj = states->memoryStates->staticHeapObj.gpu;
  }

  MORI_SHMEM_INFO("Heap info copied to GPU: base=0x{:x}, end=0x{:x}, size={} bytes, heapObj=0x{:x}",
                  gpuStates.heapBaseAddr, gpuStates.heapEndAddr,
                  gpuStates.heapEndAddr - gpuStates.heapBaseAddr,
                  reinterpret_cast<uintptr_t>(gpuStates.heapObj));

  // Copy gpu states to constant memory
  HIP_RUNTIME_CHECK(
      hipMemcpyToSymbol(globalGpuStates, &gpuStates, sizeof(GpuStates), 0, hipMemcpyHostToDevice));
}

int ShmemInit(application::BootstrapNetwork* bootNet) {
  int status;

  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  states->bootStates = new BootStates();
  states->bootStates->bootNet = bootNet;
  states->bootStates->bootNet->Initialize();
  states->bootStates->rank = states->bootStates->bootNet->GetLocalRank();
  states->bootStates->worldSize = states->bootStates->bootNet->GetWorldSize();

  RdmaStatesInit();
  MemoryStatesInit();
  GpuStateInit();
  states->status = ShmemStatesStatus::Initialized;
  return 0;
}

int ShmemFinalize() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.transportTypes));
  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.rdmaEndpoints));

  // Clean up heap (VMM or static)
  if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
    // Finalize VMM heap
    states->memoryStates->symmMemMgr->FinalizeVMMHeap();
  } else if (states->memoryStates->staticHeapObj.IsValid()) {
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
    
    // Clean up GPU side
    HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu->peerPtrs));
    HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu->peerRkeys));
    HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapObj.gpu));
    
    // Free the actual heap memory
    HIP_RUNTIME_CHECK(hipFree(states->memoryStates->staticHeapBasePtr));
  }

  delete states->memoryStates->symmMemMgr;
  delete states->memoryStates->mrMgr;
  delete states->memoryStates;

  delete states->rdmaStates->commContext;
  delete states->rdmaStates;

  states->bootStates->bootNet->Finalize();
  delete states->bootStates->bootNet;

  states->status = ShmemStatesStatus::Finalized;
  return 0;
}

int ShmemMpiInit(MPI_Comm mpiComm) {
  return ShmemInit(new application::MpiBootstrapNetwork(mpiComm));
}

int ShmemInit() { return ShmemMpiInit(MPI_COMM_WORLD); }

int ShmemTorchProcessGroupInit(const std::string& groupName) {
  return ShmemInit(new application::TorchBootstrapNetwork(groupName));
}

int ShmemMyPe() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->rank;
}

int ShmemNPes() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->worldSize;
}

// int ShmemTeamMyPe(ShmemTeamType);
// int ShmemTeamNPes(ShmemTeamType);

}  // namespace shmem
}  // namespace mori
