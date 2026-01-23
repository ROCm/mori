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
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#include "hip/hip_runtime.h"
#include "mori/application/application.hpp"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                      UniqueId Support                                         */
/* ---------------------------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------------------------- */
/*                                          Initialization                                       */
/* ---------------------------------------------------------------------------------------------- */
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;

// Helper function to parse size strings with various suffixes (G/GB/GiB, M/MB/MiB, K/KB/KiB)
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
  MORI_SHMEM_TRACE("RdmaStatesInit: rank {}, worldSize {}", rank, worldSize);
}

void MemoryStatesInit() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  application::Context* context = states->rdmaStates->commContext;

  states->memoryStates = new MemoryStates();
  states->memoryStates->symmMemMgr =
      new application::SymmMemManager(*states->bootStates->bootNet, *context);
  states->memoryStates->mrMgr =
      new application::RdmaMemoryRegionManager(*context->GetRdmaDeviceContext());

  // Handle different heap modes
  if (states->mode == ShmemMode::Isolation) {
    MORI_SHMEM_INFO("Running in isolation mode (no heap allocation)");
    return;
  }

  // For VMHeap mode, check VMM support first
  if (states->mode == ShmemMode::VMHeap) {
    MORI_SHMEM_INFO("VMM heap mode selected, checking VMM support...");

    // Check if ROCm version supports VMM (>= 7.0) and hardware VMM support
    bool rocmSupportsVMM = IsROCmVersionGreaterThan7();
    bool hardwareSupportsVMM = states->memoryStates->symmMemMgr->IsVMMSupported();
    MORI_SHMEM_INFO("VMM support check: ROCm >= 7.0: {}, Hardware VMM: {}", rocmSupportsVMM,
                    hardwareSupportsVMM);

    if (rocmSupportsVMM && hardwareSupportsVMM) {
      // VMM is supported, initialize VMM heap
      const char* chunkSizeEnv = std::getenv("MORI_SHMEM_VMM_CHUNK_SIZE");
      size_t chunkSize = 0;
      const char* vmmHeapSizeEnv = std::getenv("MORI_SHMEM_HEAP_SIZE");
      size_t vmmHeapSize = DEFAULT_VMM_SYMMETRIC_HEAP_SIZE;

      if (chunkSizeEnv) {
        chunkSize = std::max(ParseSizeString(chunkSizeEnv), DEFAULT_VMM_MIN_CHUNK_SIZE);
      }

      if (vmmHeapSizeEnv) {
        vmmHeapSize = ParseSizeString(vmmHeapSizeEnv);
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
        states->memoryStates->vmmHeapChunkSize = states->memoryStates->symmMemMgr->GetVMMChunkSize(); 
        states->memoryStates->vmmHeapObj = states->memoryStates->symmMemMgr->GetVMMHeapObj();
        states->memoryStates->vmmHeapBaseAddr = states->memoryStates->vmmHeapObj.cpu->localPtr;

        MORI_SHMEM_INFO("VMM-based dynamic heap initialized successfully");
        return;
      } else {
        states->mode = ShmemMode::StaticHeap;
        MORI_SHMEM_WARN("Failed to initialize VMM heap, falling back to static heap");
      }
    } else {
      states->mode = ShmemMode::StaticHeap;
      MORI_SHMEM_WARN("VMM not supported (ROCm: {}, Hardware: {}), falling back to static heap",
                      rocmSupportsVMM, hardwareSupportsVMM);
    }
  }

  // StaticHeap mode or fallback from VMHeap mode
  MORI_SHMEM_INFO("Allocating static symmetric heap");

  // Configure heap size
  const char* heapSizeEnv = std::getenv("MORI_SHMEM_HEAP_SIZE");
  size_t heapSize = DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE;

  if (heapSizeEnv) {
    heapSize = ParseSizeString(heapSizeEnv);
  }

  MORI_SHMEM_INFO("Allocating static symmetric heap of size {} bytes ({} MB)", heapSize,
                  heapSize / (1024 * 1024));

  // Allocate the symmetric heap
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

  // Initialize VA manager for static heap to enable memory reuse
  states->memoryStates->symmMemMgr->InitHeapVAManager(
      reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr), heapSize);

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

  // Copy gpu states to device memory (using hipGetSymbolAddress + hipMemcpy)
  GpuStates* globalGpuStatesAddr = nullptr;
  HIP_RUNTIME_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&globalGpuStatesAddr),
                                        HIP_SYMBOL(globalGpuStates)));

  MORI_SHMEM_INFO("globalGpuStates device address: 0x{:x}",
                  reinterpret_cast<uintptr_t>(globalGpuStatesAddr));

  // Copy symmetric heap info to GPU
  gpuStates.useVMMHeap = states->memoryStates->useVMMHeap;

  if (states->mode == ShmemMode::Isolation) {
    // In isolation mode, no heap info needed
    gpuStates.heapBaseAddr = 0;
    gpuStates.heapEndAddr = 0;
    gpuStates.heapObj = nullptr;
    gpuStates.vmmChunkSizeShift = 0;
    MORI_SHMEM_INFO("Isolation mode: no heap info copied to GPU");
  } else if (states->mode == ShmemMode::VMHeap && states->memoryStates->useVMMHeap &&
             states->memoryStates->vmmHeapInitialized) {
    // VMM heap mode
    uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->vmmHeapBaseAddr);
    gpuStates.heapBaseAddr = heapBase;
    gpuStates.heapEndAddr = heapBase + states->memoryStates->vmmHeapVirtualSize;
    gpuStates.heapObj = states->memoryStates->vmmHeapObj.gpu;
    gpuStates.vmmChunkSizeShift = static_cast<uint8_t>(__builtin_ctzll(states->memoryStates->vmmHeapChunkSize));
    MORI_SHMEM_INFO(
        "VMM heap info copied to GPU: base=0x{:x}, end=0x{:x}, size={} bytes, chunkSize={} (shift={}), heapObj=0x{:x}",
        gpuStates.heapBaseAddr, gpuStates.heapEndAddr,
        gpuStates.heapEndAddr - gpuStates.heapBaseAddr,
        states->memoryStates->vmmHeapChunkSize, gpuStates.vmmChunkSizeShift,
        reinterpret_cast<uintptr_t>(gpuStates.heapObj));
  } else if (states->mode == ShmemMode::StaticHeap && states->memoryStates->staticHeapObj.IsValid()) {
    // Static heap mode (no chunking, single RDMA registration)
    uintptr_t heapBase = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
    gpuStates.heapBaseAddr = heapBase;
    gpuStates.heapEndAddr = heapBase + states->memoryStates->staticHeapSize;
    gpuStates.heapObj = states->memoryStates->staticHeapObj.gpu;
    gpuStates.vmmChunkSizeShift = 0;  // No chunking for static heap
    MORI_SHMEM_INFO(
        "Static heap info copied to GPU: base=0x{:x}, end=0x{:x}, size={} bytes, heapObj=0x{:x}",
        gpuStates.heapBaseAddr, gpuStates.heapEndAddr,
        gpuStates.heapEndAddr - gpuStates.heapBaseAddr,
        reinterpret_cast<uintptr_t>(gpuStates.heapObj));
  } else {
    // Mode/heap mismatch or invalid configuration
    gpuStates.heapBaseAddr = 0;
    gpuStates.heapEndAddr = 0;
    gpuStates.heapObj = nullptr;
    gpuStates.vmmChunkSizeShift = 0;
    MORI_SHMEM_ERROR("Invalid heap configuration for mode {}: mode={}, useVMMHeap={}, "
                     "vmmHeapInitialized={}, staticHeapValid={}",
                     static_cast<int>(states->mode), static_cast<int>(states->mode),
                     states->memoryStates->useVMMHeap, 
                     states->memoryStates->vmmHeapInitialized,
                     states->memoryStates->staticHeapObj.IsValid());
  }

  // Copy gpu states to constant memory

  HIP_RUNTIME_CHECK(
      hipMemcpy(globalGpuStatesAddr, &gpuStates, sizeof(GpuStates), hipMemcpyDefault));

  MORI_SHMEM_INFO("Successfully copied GpuStates to device (rank={}, worldSize={})", gpuStates.rank,
                  gpuStates.worldSize);
}

int ShmemInit(application::BootstrapNetwork* bootNet) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();

  // Determine mode from environment variable
  const char* modeEnv = std::getenv("MORI_SHMEM_MODE");
  if (modeEnv) {
    std::string modeStr(modeEnv);
    if (modeStr == "isolation" || modeStr == "ISOLATION") {
      states->mode = ShmemMode::Isolation;
      MORI_SHMEM_INFO("Running in isolation mode");
    } else if (modeStr == "static_heap" || modeStr == "STATIC_HEAP") {
      states->mode = ShmemMode::StaticHeap;
      MORI_SHMEM_INFO("Running in static heap mode");
    } else if (modeStr == "vmm_heap" || modeStr == "VMM_HEAP") {
      states->mode = ShmemMode::VMHeap;
      MORI_SHMEM_INFO("Running in VMM heap mode");
    } else {
      MORI_SHMEM_WARN("Unknown MORI_SHMEM_MODE '{}', defaulting to static_heap", modeStr);
      states->mode = ShmemMode::StaticHeap;
    }
  } else {
    // Default to static heap mode
    states->mode = ShmemMode::StaticHeap;
    MORI_SHMEM_INFO("MORI_SHMEM_MODE not set, defaulting to static heap mode");
  }

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
  states->CheckStatusValid();

  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.transportTypes));
  HIP_RUNTIME_CHECK(hipFree(globalGpuStates.rdmaEndpoints));

  // Clean up heap based on what was actually allocated
  if (states->mode != ShmemMode::Isolation) {
    if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
      // Clean up VMM heap
      MORI_SHMEM_INFO("Finalizing VMM heap");
      states->memoryStates->symmMemMgr->FinalizeVMMHeap();
    } else if (states->memoryStates->staticHeapObj.IsValid()) {
      // Clean up static heap
      MORI_SHMEM_INFO("Finalizing static heap");

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
  } else {
    MORI_SHMEM_INFO("Isolation mode: no heap to finalize");
  }

  delete states->memoryStates->symmMemMgr;
  delete states->memoryStates->mrMgr;
  delete states->memoryStates;

  delete states->rdmaStates->commContext;
  delete states->rdmaStates;

  states->bootStates->bootNet->Finalize();
  delete states->bootStates->bootNet;
  delete states->bootStates;

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

int ShmemModuleInit(void* hipModule) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  GpuStates* hostGlobalGpuStatesAddr = nullptr;
  HIP_RUNTIME_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&hostGlobalGpuStatesAddr),
                                        HIP_SYMBOL(globalGpuStates)));

  // Read the current values from device
  GpuStates gpuStates;
  HIP_RUNTIME_CHECK(
      hipMemcpy(&gpuStates, hostGlobalGpuStatesAddr, sizeof(GpuStates), hipMemcpyDeviceToHost));

  // Get the symbol address from the specific module
  hipModule_t module = static_cast<hipModule_t>(hipModule);
  GpuStates* moduleGlobalGpuStatesAddr = nullptr;

  hipError_t err = hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&moduleGlobalGpuStatesAddr),
                                      nullptr, module, "_ZN4mori5shmem15globalGpuStatesE");

  if (err != hipSuccess) {
    MORI_SHMEM_WARN("Failed to get globalGpuStates symbol from module: {} (error code: {})",
                    hipGetErrorString(err), err);
    return -1;
  }

  MORI_SHMEM_INFO("Module globalGpuStates address: 0x{:x} (host lib address: 0x{:x})",
                  reinterpret_cast<uintptr_t>(moduleGlobalGpuStatesAddr),
                  reinterpret_cast<uintptr_t>(hostGlobalGpuStatesAddr));

  // Copy the GpuStates to the module's globalGpuStates
  HIP_RUNTIME_CHECK(
      hipMemcpy(moduleGlobalGpuStatesAddr, &gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));

  MORI_SHMEM_INFO("Successfully initialized globalGpuStates in module (rank={}, worldSize={})",
                  gpuStates.rank, gpuStates.worldSize);

  return 0;
}

void ShmemBarrierAll() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  MORI_SHMEM_TRACE("ShmemBarrierAll: PE {} entering barrier", states->bootStates->rank);
  states->bootStates->bootNet->Barrier();
  MORI_SHMEM_TRACE("ShmemBarrierAll: PE {} exiting barrier", states->bootStates->rank);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      UniqueId APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
int ShmemGetUniqueId(mori_shmem_uniqueid_t* uid) {
  if (uid == nullptr) {
    MORI_SHMEM_ERROR("ShmemGetUniqueId - invalid input argument");
    return -1;
  }

  try {
    const char* ifname = std::getenv("MORI_SOCKET_IFNAME");
    application::UniqueId socket_uid;

    // Generate random port for UniqueId
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> port_dis(25000, 35000);
    int random_port = port_dis(gen);

    if (ifname) {
      socket_uid =
          application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(ifname, random_port);
      MORI_SHMEM_INFO("Generated UniqueId with specified interface: {} (port {})", ifname,
                      random_port);
    } else {
      socket_uid = application::SocketBootstrapNetwork::GenerateUniqueIdWithLocalAddr(random_port);
      std::string localAddr = application::SocketBootstrapNetwork::GetLocalNonLoopbackAddress();
      MORI_SHMEM_INFO("Generated UniqueId with auto-detected interface: {} (port {})", localAddr,
                      random_port);
    }
    static_assert(sizeof(socket_uid) == sizeof(mori_shmem_uniqueid_t),
                  "UniqueId size mismatch between Socket Bootstrap and mori SHMEM");

    // Copy to mori_shmem_uniqueid_t
    std::memcpy(uid->data(), &socket_uid, sizeof(socket_uid));

    return 0;

  } catch (const std::exception& e) {
    MORI_SHMEM_ERROR("ShmemGetUniqueId failed: {}", e.what());
    return -1;
  }
}

int ShmemSetAttrUniqueIdArgs(int rank, int nranks, mori_shmem_uniqueid_t* uid,
                             mori_shmem_init_attr_t* attr) {
  if (uid == nullptr || attr == nullptr) {
    MORI_SHMEM_ERROR("ShmemSetAttrUniqueIdArgs - invalid input argument");
    return -1;
  }

  if (rank < 0 || nranks <= 0 || rank >= nranks) {
    MORI_SHMEM_ERROR("ShmemSetAttrUniqueIdArgs - invalid rank={} or nranks={}", rank, nranks);
    return -1;
  }

  // Set attributes
  attr->rank = rank;
  attr->nranks = nranks;
  attr->uid = *uid;
  attr->mpi_comm = nullptr;  // Not using MPI for UniqueId-based initialization

  return 0;
}

int ShmemInitAttr(unsigned int flags, mori_shmem_init_attr_t* attr) {
  if (attr == nullptr ||
      ((flags != MORI_SHMEM_INIT_WITH_UNIQUEID) && (flags != MORI_SHMEM_INIT_WITH_MPI_COMM))) {
    MORI_SHMEM_ERROR("ShmemInitAttr - invalid input argument");
    return -1;
  }

  if (flags == MORI_SHMEM_INIT_WITH_MPI_COMM) {
    // Handle MPI-based initialization (delegate to existing ShmemMpiInit)
    if (attr->mpi_comm == nullptr) {
      MORI_SHMEM_ERROR("ShmemInitAttr - MPI_Comm is null");
      return -1;
    }

    int result = ShmemMpiInit(*reinterpret_cast<MPI_Comm*>(attr->mpi_comm));
    return (result == 0) ? 0 : -1;
  }

  if (flags == MORI_SHMEM_INIT_WITH_UNIQUEID) {
    // Validate UniqueId-based initialization parameters
    if (attr->nranks <= 0 || attr->rank < 0 || attr->rank >= attr->nranks) {
      MORI_SHMEM_ERROR("ShmemInitAttr - invalid rank={} or nranks={}", attr->rank, attr->nranks);
      return -1;
    }

    try {
      // Convert mori_shmem_uniqueid_t back to Socket Bootstrap UniqueId
      application::UniqueId socket_uid;
      std::memcpy(&socket_uid, attr->uid.data(), sizeof(socket_uid));

      // Create Socket Bootstrap Network
      auto socket_bootstrap = std::make_unique<application::SocketBootstrapNetwork>(
          socket_uid, attr->rank, attr->nranks);

      MORI_SHMEM_INFO("Initialized Socket Bootstrap - rank={}, nranks={}", attr->rank,
                      attr->nranks);

      // Initialize mori SHMEM using the bootstrap network
      int result = ShmemInit(socket_bootstrap.release());

      if (result != 0) {
        MORI_SHMEM_ERROR("ShmemInitAttr - ShmemInit failed with code {}", result);
        return -1;
      }

      MORI_SHMEM_INFO("Successfully initialized with UniqueId");
      return 0;

    } catch (const std::exception& e) {
      MORI_SHMEM_ERROR("ShmemInitAttr failed: {}", e.what());
      return -1;
    }
  }

  return -1;
}

int ShmemNumQpPerPe() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->rdmaStates->commContext->GetNumQpPerPe();
}

// int ShmemTeamMyPe(ShmemTeamType);
// int ShmemTeamNPes(ShmemTeamType);

}  // namespace shmem
}  // namespace mori
