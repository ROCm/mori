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
#include "mori/application/memory/symmetric_memory.hpp"

#include <map>
#include <vector>
#include <fcntl.h>
#include <cerrno>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {

namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                    Constructor / Destructor                                   */
/* ---------------------------------------------------------------------------------------------- */

SymmMemManager::SymmMemManager(BootstrapNetwork& bootNet, Context& context)
    : bootNet(bootNet), context(context) {}

SymmMemManager::~SymmMemManager() {
  while (!memObjPool.empty()) {
    DeregisterSymmMemObj(memObjPool.begin()->first);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Basic Memory Allocation                                    */
/* ---------------------------------------------------------------------------------------------- */

SymmMemObjPtr SymmMemManager::HostMalloc(size_t size, size_t alignment) {
  void* ptr = nullptr;
  int status = posix_memalign(&ptr, alignment, size);
  assert(!status);
  memset(ptr, 0, size);
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::HostFree(void* localPtr) {
  free(localPtr);
  DeregisterSymmMemObj(localPtr);
}

SymmMemObjPtr SymmMemManager::Malloc(size_t size) {
  void* ptr = nullptr;
  // HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, hipDeviceMallocUncached));
  HIP_RUNTIME_CHECK(hipMalloc(&ptr, size));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

SymmMemObjPtr SymmMemManager::ExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, flags));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, size));
  return RegisterSymmMemObj(ptr, size);
}

void SymmMemManager::Free(void* localPtr) {
  HIP_RUNTIME_CHECK(hipFree(localPtr));
  DeregisterSymmMemObj(localPtr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    SymmMemObj Registration                                    */
/* ---------------------------------------------------------------------------------------------- */

SymmMemObjPtr SymmMemManager::RegisterSymmMemObj(void* localPtr, size_t size, bool heap_begin) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Exchange pointers
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&localPtr, cpuMemObj->peerPtrs, sizeof(uintptr_t));
  // cpuMemObj->peerPtrs[rank] = reinterpret_cast<uintptr_t>(cpuMemObj->localPtr);

  // P2P context: exchange ipc mem handles
  hipIpcMemHandle_t handle;
  HIP_RUNTIME_CHECK(hipIpcGetMemHandle(&handle, localPtr));
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  bootNet.Allgather(&handle, cpuMemObj->ipcMemHandles, sizeof(hipIpcMemHandle_t));
  for (int i = 0; i < worldSize; i++) {
    if ((context.GetTransportType(i) != TransportType::P2P) &&
        (context.GetTransportType(i) != TransportType::SDMA))
      continue;
    if (i == rank) continue;

    HIP_RUNTIME_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&cpuMemObj->peerPtrs[i]),
                                          cpuMemObj->ipcMemHandles[i],
                                          hipIpcMemLazyEnablePeerAccess));
  }

  // Rdma context: set lkey and exchange rkeys
  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  cpuMemObj->peerRkeys[rank] = 0;
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    application::RdmaMemoryRegion mr = rdmaDeviceContext->RegisterRdmaMemoryRegion(localPtr, size);
    cpuMemObj->lkey = mr.lkey;
    cpuMemObj->peerRkeys[rank] = mr.rkey;
  }
  bootNet.Allgather(&cpuMemObj->peerRkeys[rank], cpuMemObj->peerRkeys, sizeof(uint32_t));

  // Copy memory object to GPU memory, we need to access it from GPU directly
  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  std::vector<int> dstDeviceIds;
  for (int i = 0; i < worldSize; i++) {
    if (context.GetTransportType(i) != TransportType::SDMA) continue;
    if (i == rank) continue;
    dstDeviceIds.push_back(i % 8);  // should be intra devices count
  }
  if (dstDeviceIds.size() != 0) {
    int srcDeviceId = rank % 8;
    int numOfQueuesPerDevice = gpuMemObj->sdmaNumQueue;  // all sdma queues are inited
    HIP_RUNTIME_CHECK(hipMalloc(
        &gpuMemObj->deviceHandles_d,
        dstDeviceIds.size() * numOfQueuesPerDevice * sizeof(anvil::SdmaQueueDeviceHandle*)));

    for (auto& dstDeviceId : dstDeviceIds) {
      for (size_t q = 0; q < numOfQueuesPerDevice; q++) {
        gpuMemObj->deviceHandles_d[dstDeviceId * numOfQueuesPerDevice + q] =
            anvil::anvil.getSdmaQueue(srcDeviceId, dstDeviceId, q)->deviceHandle();
      }
    }

    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->signalPtrs,
                                sizeof(HSAuint64) * dstDeviceIds.size() * numOfQueuesPerDevice));
    HIP_RUNTIME_CHECK(hipMemset(gpuMemObj->signalPtrs, 0,
                                sizeof(HSAuint64) * dstDeviceIds.size() * numOfQueuesPerDevice));
    HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->expectSignalsPtr,
                                sizeof(HSAuint64) * dstDeviceIds.size() * numOfQueuesPerDevice));
    HIP_RUNTIME_CHECK(hipMemset(gpuMemObj->expectSignalsPtr, 0,
                                sizeof(HSAuint64) * dstDeviceIds.size() * numOfQueuesPerDevice));
  }
  SymmMemObjPtr result{cpuMemObj, gpuMemObj};
  if (!heap_begin) {
    memObjPool.insert({localPtr, result});
    return memObjPool.at(localPtr);
  } else {
    return result;
  }
}

void SymmMemManager::DeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) rdmaDeviceContext->DeregisterRdmaMemoryRegion(localPtr);

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);
  free(memObjPtr.cpu->ipcMemHandles);
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

SymmMemObjPtr SymmMemManager::RegisterStaticHeapSubRegion(void* localPtr, size_t size,
                                                          SymmMemObjPtr* heapObj) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Calculate offset from heap base
  uintptr_t heapBase = reinterpret_cast<uintptr_t>(heapObj->cpu->localPtr);
  uintptr_t localAddr = reinterpret_cast<uintptr_t>(localPtr);
  size_t offset = localAddr - heapBase;

  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  for (int i = 0; i < worldSize; i++) {
    cpuMemObj->peerPtrs[i] = heapObj->cpu->peerPtrs[i] + offset;
  }

  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  memcpy(cpuMemObj->ipcMemHandles, heapObj->cpu->ipcMemHandles,
         sizeof(hipIpcMemHandle_t) * worldSize);

  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));
  memcpy(cpuMemObj->peerRkeys, heapObj->cpu->peerRkeys, sizeof(uint32_t) * worldSize);
  cpuMemObj->lkey = heapObj->cpu->lkey;
  cpuMemObj->sdmaNumQueue = heapObj->cpu->sdmaNumQueue;

  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  // Copy SDMA resources from heap object (shared across all heap allocations)
  if (heapObj->gpu->deviceHandles_d != nullptr) {
    std::vector<int> dstDeviceIds;
    for (int i = 0; i < worldSize; i++) {
      if (context.GetTransportType(i) != TransportType::SDMA) continue;
      if (i == rank) continue;
      dstDeviceIds.push_back(i % 8);  // should be intra devices count
    }

    if (dstDeviceIds.size() != 0) {
      int numOfQueuesPerDevice = cpuMemObj->sdmaNumQueue;
      gpuMemObj->deviceHandles_d = heapObj->gpu->deviceHandles_d;
      gpuMemObj->signalPtrs = heapObj->gpu->signalPtrs;
      gpuMemObj->expectSignalsPtr = heapObj->gpu->expectSignalsPtr;
    }
  }

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  return memObjPool.at(localPtr);
}

void SymmMemManager::DeregisterStaticHeapSubRegion(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  // No need to deregister RDMA memory region - this is a sub-region of the static heap

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);
  free(memObjPtr.cpu->ipcMemHandles);
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

void SymmMemManager::VMMDeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  // No need to deregister RDMA memory region - this is a sub-region of the VMM heap

  SymmMemObjPtr memObjPtr = memObjPool.at(localPtr);
  free(memObjPtr.cpu->peerPtrs);
  free(memObjPtr.cpu->peerRkeys);  // nullptr for VMM allocations (safe to free)
  free(memObjPtr.cpu->ipcMemHandles);
  // Note: vmmLkeyInfo and vmmRkeyInfo are NOT freed here - they point to shared vmmHeapObj arrays
  free(memObjPtr.cpu);
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerPtrs));
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu->peerRkeys));  // nullptr for VMM allocations (safe to free)
  HIP_RUNTIME_CHECK(hipFree(memObjPtr.gpu));

  memObjPool.erase(localPtr);
}

SymmMemObjPtr SymmMemManager::Get(void* localPtr) const {
  if (memObjPool.find(localPtr) == memObjPool.end()) return SymmMemObjPtr{};
  return memObjPool.at(localPtr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Heap Support Check                                     */
/* ---------------------------------------------------------------------------------------------- */

bool SymmMemManager::IsVMMSupported() const {
  int currentDev = 0;
  if (hipGetDevice(&currentDev) != hipSuccess) {
    return false;
  }

  int vmm = 0;
  return (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported,
                                currentDev) == hipSuccess &&
          vmm != 0);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                VMM Heap Initialization Helpers                                */
/* ---------------------------------------------------------------------------------------------- */

// Step 1: Determine optimal chunk size based on HIP granularity
size_t SymmMemManager::DetermineVMMChunkSize(size_t userChunkSize, HeapType heapType) {
  if (userChunkSize != 0) {
    // User provided chunk size, ensure it's not too small
    return std::max(userChunkSize, shmem::DEFAULT_VMM_MIN_CHUNK_SIZE);
  }

  // Auto-detect optimal chunk size based on HIP granularity
  int currentDev = 0;
  if (hipGetDevice(&currentDev) != hipSuccess) {
    return shmem::DEFAULT_VMM_MIN_CHUNK_SIZE;
  }

  hipMemAllocationProp allocProp = {};
#if HIP_VERSION >= 70000000  // ROCm 7.0+
  allocProp.type = (heapType == HeapType::Normal) ? hipMemAllocationTypePinned 
                                                   : hipMemAllocationTypeUncached;
#else
  allocProp.type = hipMemAllocationTypePinned;
  if (heapType == HeapType::Uncached) {
    MORI_APP_WARN("Uncached heap type requested but ROCm version < 7.0 does not support "
                  "hipMemAllocationTypeUncached, falling back to Pinned memory");
  }
#endif
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = currentDev;

  // Try to get recommended granularity first
  size_t granularity = 0;
  hipError_t result = hipMemGetAllocationGranularity(&granularity, &allocProp,
                                                     hipMemAllocationGranularityRecommended);
  if (result == hipSuccess && granularity > 0) {
    size_t chunkSize = std::max(granularity, shmem::DEFAULT_VMM_MIN_CHUNK_SIZE);
    
    // Get minimum granularity for vmmMinChunkSize
    size_t minGranularity = 0;
    if (hipMemGetAllocationGranularity(&minGranularity, &allocProp,
                                       hipMemAllocationGranularityMinimum) == hipSuccess &&
        minGranularity > 0) {
      vmmMinChunkSize = minGranularity;
    } else {
      vmmMinChunkSize = 4 * 1024;  // 4KB fallback
    }
    return chunkSize;
  }

  // Fallback: try minimum granularity
  if (hipMemGetAllocationGranularity(&granularity, &allocProp,
                                     hipMemAllocationGranularityMinimum) == hipSuccess &&
      granularity > 0) {
    vmmMinChunkSize = granularity;
    return std::max(granularity, shmem::DEFAULT_VMM_MIN_CHUNK_SIZE);
  }

  // Final fallback
  vmmMinChunkSize = 4 * 1024;  // 4KB
  return shmem::DEFAULT_VMM_MIN_CHUNK_SIZE;
}

// Step 2: Calculate total virtual address space size
size_t SymmMemManager::CalculateTotalVirtualSize(size_t perPeerSize, int worldSize) {
  int myPe = bootNet.GetLocalRank();
  std::vector<TransportType> transportTypes = context.GetTransportTypes();
  
  vmmPerPeerSize = perPeerSize;
  size_t p2pPeCount = 0;
  
  for (int pe = 0; pe < worldSize; ++pe) {
    if (transportTypes[pe] == TransportType::P2P && myPe != pe) {
      p2pPeCount++;
    }
  }

  // Only allocate virtual space for P2P accessible PEs
  size_t totalSize = vmmPerPeerSize * (p2pPeCount + 1);
  
  MORI_APP_TRACE("VMM Heap: world={} p2pPeers={} totalVA={}", worldSize, p2pPeCount, totalSize);
  return totalSize;
}

// Step 3: Reserve virtual address space
bool SymmMemManager::ReserveVirtualAddressSpace(size_t totalSize, size_t chunkSize) {
  hipError_t result = hipMemAddressReserve(&vmmVirtualBasePtr, totalSize, chunkSize, nullptr, 0);
  if (result != hipSuccess) {
    MORI_APP_ERROR("VMM Init failed: hipMemAddressReserve size={} align={} err={}",
                   totalSize, chunkSize, result);
    return false;
  }

  // Verify alignment
  if (reinterpret_cast<uintptr_t>(vmmVirtualBasePtr) % chunkSize != 0) {
    MORI_APP_WARN("VMM Init: vmmVirtualBasePtr {:p} is NOT aligned to chunkSize={}", 
                  vmmVirtualBasePtr, chunkSize);
  }

  int myPe = bootNet.GetLocalRank();
  MORI_APP_TRACE("VMM Init: rank={} vmmVirtualBasePtr={:p} (aligned to {} bytes)", 
                 myPe, vmmVirtualBasePtr, chunkSize);
  return true;
}

// Step 4: Setup peer base pointers for each PE
void SymmMemManager::SetupPeerBasePointers(int worldSize, int myPe) {
  std::vector<TransportType> transportTypes = context.GetTransportTypes();
  vmmPeerBasePtrs.resize(worldSize);
  
  size_t virtualOffset = 0;
  for (int i = 0; i < worldSize; ++i) {
    int pe = (myPe + i) % worldSize;

    if (pe == myPe || transportTypes[pe] == TransportType::P2P) {
      vmmPeerBasePtrs[pe] =
          static_cast<void*>(static_cast<char*>(vmmVirtualBasePtr) + virtualOffset);
      virtualOffset += vmmPerPeerSize;
    } else {
      vmmPeerBasePtrs[pe] = nullptr;
    }
  }
}

// Step 5: Initialize chunk tracking structures
void SymmMemManager::InitializeChunkTracking() {
  vmmChunks.resize(vmmMaxChunks);
  
  int worldSize = bootNet.GetWorldSize();
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    vmmChunks[i].peerRkeys.resize(worldSize, 0);
  }
}

// Step 6: Create SymmMemObj for VMM heap
SymmMemObjPtr SymmMemManager::CreateVMMHeapObject(size_t virtualSize, int worldSize, int myPe) {
  SymmMemObj* cpuHeapObj = new SymmMemObj();
  cpuHeapObj->localPtr = vmmVirtualBasePtr;
  cpuHeapObj->size = virtualSize;

  // Exchange virtual base pointers among all PEs
  cpuHeapObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  bootNet.Allgather(&vmmVirtualBasePtr, cpuHeapObj->peerPtrs, sizeof(uintptr_t));
  for (int pe = 0; pe < worldSize; ++pe) {
    if (vmmPeerBasePtrs[pe] != nullptr) {
      cpuHeapObj->peerPtrs[pe] = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[pe]);
    }
  }

  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess
  cpuHeapObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // Initialize VMMChunkKey arrays (nvshmem-style: key + next_addr per chunk)
  uintptr_t heapBase = reinterpret_cast<uintptr_t>(vmmVirtualBasePtr);
  
  cpuHeapObj->vmmLkeyInfo = static_cast<VMMChunkKey*>(calloc(vmmMaxChunks, sizeof(VMMChunkKey)));
  cpuHeapObj->vmmRkeyInfo = static_cast<VMMChunkKey*>(calloc(worldSize * vmmMaxChunks, sizeof(VMMChunkKey)));
  
  // Initialize VMMChunkKey arrays with next_addr values
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    uintptr_t chunkEnd = heapBase + (i + 1) * vmmChunkSize;
    uintptr_t heapEnd = heapBase + virtualSize;
    cpuHeapObj->vmmLkeyInfo[i].next_addr = (chunkEnd < heapEnd) ? chunkEnd : heapEnd;
    cpuHeapObj->vmmLkeyInfo[i].key = 0;  // Will be set when chunk is allocated
    
    // Initialize rkey info for all PEs
    for (int pe = 0; pe < worldSize; ++pe) {
      cpuHeapObj->vmmRkeyInfo[i * worldSize + pe].next_addr = cpuHeapObj->vmmLkeyInfo[i].next_addr;
      cpuHeapObj->vmmRkeyInfo[i * worldSize + pe].key = 0;  // Will be set when chunk is allocated
    }
  }
  
  cpuHeapObj->vmmNumChunks = vmmMaxChunks;
  cpuHeapObj->worldSize = worldSize;
  
  // Keep lkey and peerRkeys as nullptr for VMM heap to distinguish from static heap
  cpuHeapObj->lkey = 0;
  cpuHeapObj->peerRkeys = nullptr;

  // Copy heap object to GPU memory
  SymmMemObj* gpuHeapObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj, cpuHeapObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->peerPtrs, cpuHeapObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  // Allocate and copy VMM-specific RDMA key info to GPU
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->vmmLkeyInfo, sizeof(VMMChunkKey) * vmmMaxChunks));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->vmmLkeyInfo, cpuHeapObj->vmmLkeyInfo,
                              sizeof(VMMChunkKey) * vmmMaxChunks, hipMemcpyHostToDevice));
  
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->vmmRkeyInfo, sizeof(VMMChunkKey) * worldSize * vmmMaxChunks));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->vmmRkeyInfo, cpuHeapObj->vmmRkeyInfo,
                              sizeof(VMMChunkKey) * worldSize * vmmMaxChunks, hipMemcpyHostToDevice));
  
  gpuHeapObj->vmmNumChunks = vmmMaxChunks;
  gpuHeapObj->worldSize = worldSize;
  
  // Set lkey and peerRkeys to 0/nullptr for VMM heap
  gpuHeapObj->lkey = 0;
  gpuHeapObj->peerRkeys = nullptr;

  return SymmMemObjPtr{cpuHeapObj, gpuHeapObj};
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Heap Initialization                                    */
/* ---------------------------------------------------------------------------------------------- */

bool SymmMemManager::InitializeVMMHeap(size_t virtualSize, size_t chunkSize, HeapType heapType) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (vmmInitialized) {
    return true;  // Already initialized
  }

  int worldSize = bootNet.GetWorldSize();
  int myPe = bootNet.GetLocalRank();

  // Step 1: Determine optimal chunk size
  chunkSize = DetermineVMMChunkSize(chunkSize, heapType);
  vmmChunkSize = chunkSize;
  vmmVirtualSize = virtualSize;
  vmmMaxChunks = virtualSize / chunkSize;

  MORI_APP_TRACE("VMM Heap Init: vSize={} chunkSize={} maxChunks={} world={}",
                 vmmVirtualSize, vmmChunkSize, vmmMaxChunks, worldSize);

  // Step 2: Calculate total virtual address space size
  size_t totalVirtualSize = CalculateTotalVirtualSize(virtualSize, worldSize);

  // Step 3: Reserve virtual address space
  if (!ReserveVirtualAddressSpace(totalVirtualSize, chunkSize)) {
    return false;
  }

  // Step 4: Setup peer base pointers
  SetupPeerBasePointers(worldSize, myPe);

  // Step 5: Initialize chunk tracking
  InitializeChunkTracking();

  // Step 6: Create VMM heap object
  vmmHeapObj = CreateVMMHeapObject(virtualSize, worldSize, myPe);

  vmmInitialized = true;
  return true;
}

/*                                    VMM Heap Finalization                                      */
/* ---------------------------------------------------------------------------------------------- */

void SymmMemManager::FinalizeVMMHeap() {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    return;
  }

  int rank = bootNet.GetLocalRank();

  // Deregister per-chunk RDMA registrations and clean up resources
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (rdmaDeviceContext) {
    for (size_t i = 0; i < vmmMaxChunks; ++i) {
      if (vmmChunks[i].isAllocated && vmmChunks[i].rdmaRegistered) {
        void* chunkPtr =
            static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + i * vmmChunkSize);
        rdmaDeviceContext->DeregisterRdmaMemoryRegion(chunkPtr);
        MORI_APP_TRACE("FinalizeVMMHeap: Deregistered RDMA for chunk {} at {:p}", i, chunkPtr);
      }
    }
  }

  // Step 1: First unmap all peer virtual address spaces (imported chunks)
  // This must be done before unmapping local chunks and before releasing imported handles
  int worldSize = bootNet.GetWorldSize();
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe == rank) continue;  // Skip self
    
    // Only process P2P accessible PEs
    if (context.GetTransportType(pe) == TransportType::P2P && vmmPeerBasePtrs[pe] != nullptr) {
      for (size_t i = 0; i < vmmMaxChunks; ++i) {
        if (vmmChunks[i].mappedPeers.count(pe) > 0) {
          void* peerChunkPtr =
              static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) + i * vmmChunkSize);
          
          hipError_t result = hipMemUnmap(peerChunkPtr, vmmChunkSize);
          if (result != hipSuccess) {
            MORI_APP_WARN("FinalizeVMMHeap: Failed to unmap peer chunk {} from PE {}, err={}",
                          i, pe, result);
          } else {
            MORI_APP_TRACE("FinalizeVMMHeap: Unmapped chunk {} from PE {} at {:p}",
                           i, pe, peerChunkPtr);
          }
        }
      }
    }
  }

  // Step 2: Free all allocated chunks in local PE's virtual address space
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    if (vmmChunks[i].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + i * vmmChunkSize);
      
      // Close shareable file descriptor to prevent FD leak (shared by P2P and RDMA)
      if (vmmChunks[i].shareableHandle != -1) {
        close(vmmChunks[i].shareableHandle);
        MORI_APP_TRACE("FinalizeVMMHeap: Closed FD {} for chunk {} (P2P & RDMA)", 
                       vmmChunks[i].shareableHandle, i);
        vmmChunks[i].shareableHandle = -1;
      }
      
      // Release all imported handles from P2P peers
      for (auto& pair : vmmChunks[i].importedHandles) {
        HIP_RUNTIME_CHECK(hipMemRelease(pair.second));
        MORI_APP_TRACE("FinalizeVMMHeap: Released imported handle from PE {} for chunk {}",
                       pair.first, i);
      }
      vmmChunks[i].importedHandles.clear();
      
      // All chunks use granularity size (vmmChunkSize)
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunkSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[i].handle));
      vmmChunks[i].isAllocated = false;
    }
  }

  // Step 3: Synchronize GPU to ensure all operations are complete
  hipError_t syncResult = hipDeviceSynchronize();
  if (syncResult != hipSuccess) {
    MORI_APP_WARN("FinalizeVMMHeap: hipDeviceSynchronize failed: {}", syncResult);
  }

  // Step 4: Free virtual address space (entire multi-PE space)
  if (vmmVirtualBasePtr) {
    size_t totalVirtualSize = vmmPerPeerSize * worldSize;
    MORI_APP_TRACE("FinalizeVMMHeap: Freeing virtual address space at {:p}, size={} bytes",
                   vmmVirtualBasePtr, totalVirtualSize);
    HIP_RUNTIME_CHECK(hipMemAddressFree(vmmVirtualBasePtr, totalVirtualSize));
    vmmVirtualBasePtr = nullptr;
  }

  // Step 5: Clean up VMM heap object
  if (vmmHeapObj.IsValid()) {
    // Free CPU-side memory first
    if (vmmHeapObj.cpu) {
      free(vmmHeapObj.cpu->peerPtrs);
      free(vmmHeapObj.cpu->vmmRkeyInfo);
      free(vmmHeapObj.cpu->vmmLkeyInfo);
      free(vmmHeapObj.cpu->ipcMemHandles);
      free(vmmHeapObj.cpu);
      vmmHeapObj.cpu = nullptr;
    }
    
    // Free GPU-side memory with synchronization
    if (vmmHeapObj.gpu) {
      hipError_t err;
      if ((err = hipFree(vmmHeapObj.gpu->peerPtrs)) != hipSuccess) {
        MORI_APP_WARN("FinalizeVMMHeap: Failed to free GPU peerPtrs: {}", err);
      }
      if ((err = hipFree(vmmHeapObj.gpu->vmmRkeyInfo)) != hipSuccess) {
        MORI_APP_WARN("FinalizeVMMHeap: Failed to free GPU vmmRkeyInfo: {}", err);
      }
      if ((err = hipFree(vmmHeapObj.gpu->vmmLkeyInfo)) != hipSuccess) {
        MORI_APP_WARN("FinalizeVMMHeap: Failed to free GPU vmmLkeyInfo: {}", err);
      }
      if ((err = hipFree(vmmHeapObj.gpu)) != hipSuccess) {
        MORI_APP_WARN("FinalizeVMMHeap: Failed to free GPU states: {}", err);
      }
      vmmHeapObj.gpu = nullptr;
    }
    
    vmmHeapObj = SymmMemObjPtr{nullptr, nullptr};
  }

  // Clean up VA Manager
  if (heapVAManager) {
    heapVAManager->Reset();
    heapVAManager.reset();
    MORI_APP_TRACE("VA Manager cleaned up for rank {}", rank);
  }

  vmmChunks.clear();
  vmmPeerBasePtrs.clear();
  vmmMinChunkSize = 0;
  vmmPerPeerSize = 0;
  vmmInitialized = false;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Chunk Allocation Helpers                               */
/* ---------------------------------------------------------------------------------------------- */

// Step 1: Allocate virtual address from VA manager
uintptr_t SymmMemManager::AllocateVirtualAddress(size_t size) {
  uintptr_t allocAddr = heapVAManager->Allocate(size, 256);
  
  if (allocAddr == 0) {
    MORI_APP_ERROR("VA allocation failed for size {} bytes", size);
    
    // Log VA manager stats for debugging
    size_t totalBlocks, freeBlocks, allocatedBlocks, totalFreeSpace, largestFreeBlock;
    heapVAManager->GetStats(totalBlocks, freeBlocks, allocatedBlocks, totalFreeSpace,
                            largestFreeBlock);
    MORI_APP_ERROR("VA stats: total={} free={} alloc={} freeSpace={} largest={}", totalBlocks,
                   freeBlocks, allocatedBlocks, totalFreeSpace, largestFreeBlock);
  }
  
  return allocAddr;
}

// Step 2: Verify VA allocation consistency across all PEs
bool SymmMemManager::VerifyVAConsistency(uintptr_t allocAddr, size_t size, 
                                         size_t offset, int rank, int worldSize) {
  struct VAInfo {
    size_t offset;
    size_t size;
  };
  VAInfo myVAInfo = {offset, size};
  std::vector<VAInfo> allVAInfo(worldSize);
  
  bootNet.Allgather(&myVAInfo, allVAInfo.data(), sizeof(VAInfo));
  
  bool vaConsistent = true;
  for (int pe = 0; pe < worldSize; ++pe) {
    if (allVAInfo[pe].offset != offset || allVAInfo[pe].size != size) {
      MORI_APP_ERROR(
          "VMMAlloc: rank={} symmetric memory violated! Self: offset=0x{:x} size={}, "
          "PE {}: offset=0x{:x} size={}",
          rank, offset, size, pe, allVAInfo[pe].offset, allVAInfo[pe].size);
      vaConsistent = false;
    }
  }
  
  if (!vaConsistent) {
    MORI_APP_ERROR("VMMAlloc: rank={} aborting due to inconsistent offset/size", rank);
    heapVAManager->Free(allocAddr);
    return false;
  }
  
  MORI_APP_TRACE("VMMAlloc: rank={} verified all {} PEs have matching offset=0x{:x} size={}", 
                 rank, worldSize, offset, size);
  return true;
}

// Helper: Cleanup already allocated chunks on error
void SymmMemManager::CleanupAllocatedChunks(size_t startChunk, size_t numToClean) {
  for (size_t j = 0; j < numToClean; ++j) {
    size_t cleanupIdx = startChunk + j;
    if (vmmChunks[cleanupIdx].refCount > 1) {
      // Reused chunk, just decrement refCount
      vmmChunks[cleanupIdx].refCount--;
    } else if (vmmChunks[cleanupIdx].refCount == 1) {
      // Newly created chunk, fully release it
      void* cleanupPtr = static_cast<void*>(
          static_cast<char*>(vmmPeerBasePtrs[bootNet.GetLocalRank()]) + cleanupIdx * vmmChunkSize);
      HIP_RUNTIME_CHECK(hipMemUnmap(cleanupPtr, vmmChunkSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
      vmmChunks[cleanupIdx].isAllocated = false;
      vmmChunks[cleanupIdx].refCount = 0;
    }
  }
}

// Step 3: Configure HIP allocation properties based on heap type
hipMemAllocationProp SymmMemManager::ConfigureAllocationProp(HeapType heapType, int deviceId) {
  hipMemAllocationProp allocProp = {};
#if HIP_VERSION >= 70000000  // ROCm 7.0+
  allocProp.type = (heapType == HeapType::Normal) ? hipMemAllocationTypePinned 
                                                   : hipMemAllocationTypeUncached;
#else
  // ROCm < 7.0: hipMemAllocationTypeUncached not supported, fallback to Pinned
  allocProp.type = hipMemAllocationTypePinned;
  if (heapType == HeapType::Uncached) {
    MORI_APP_WARN("Uncached heap type requested but ROCm version < 7.0 does not support "
                  "hipMemAllocationTypeUncached, falling back to Pinned memory");
  }
#endif
  allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = deviceId;
  return allocProp;
}

// Step 4: Allocate single physical chunk
bool SymmMemManager::AllocateSingleChunk(VMMChunkInfo& chunk, size_t chunkIdx, void* chunkPtr,
                                         size_t chunkSize, const hipMemAllocationProp& allocProp,
                                         int rank, int currentDev) {
  hipError_t result;
  
  // Create physical memory handle
  result = hipMemCreate(&chunk.handle, chunkSize, &allocProp, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN("VMMAlloc failed: hipMemCreate chunk={} size={} dev={} err={}",
                  chunkIdx, chunkSize, currentDev, result);
    return false;
  }

  // Map physical memory to local virtual address
  result = hipMemMap(chunkPtr, chunkSize, 0, chunk.handle, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN("VMMAlloc failed: hipMemMap chunk={} addr={:p} size={} err={}",
                  chunkIdx, chunkPtr, chunkSize, result);
    HIP_RUNTIME_CHECK(hipMemRelease(chunk.handle));
    return false;
  }

  // Set access permissions for local device
  hipMemAccessDesc accessDesc;
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = currentDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  result = hipMemSetAccess(chunkPtr, chunkSize, &accessDesc, 1);
  if (result != hipSuccess) {
    MORI_APP_WARN("VMMAlloc failed: hipMemSetAccess chunk={} addr={:p} err={}",
                  chunkIdx, chunkPtr, result);
    HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, chunkSize));
    HIP_RUNTIME_CHECK(hipMemRelease(chunk.handle));
    return false;
  }

  // Export shareable handle for cross-process sharing (P2P and RDMA)
  result = hipMemExportToShareableHandle((void*)&chunk.shareableHandle, chunk.handle,
                                         hipMemHandleTypePosixFileDescriptor, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN("VMMAlloc: hipMemExport failed chunk={} err={}, P2P and RDMA may not work",
                  chunkIdx, result);
    chunk.shareableHandle = -1;
  }

  MORI_APP_TRACE("VMMAlloc: rank={} created chunk={} size={} fd={} (shared for P2P & RDMA)",
                 rank, chunkIdx, chunkSize, chunk.shareableHandle);

  chunk.isAllocated = true;
  chunk.refCount = 1;
  chunk.size = chunkSize;
  return true;
}

// Step 5: Import and map peer chunk for P2P access
void SymmMemManager::ImportPeerChunk(VMMChunkInfo& chunk, size_t chunkIdx, void* peerChunkPtr,
                                     size_t chunkSize, int handleValue, int pe, int rank,
                                     int currentDev) {
  hipError_t result;
  
  // Import the shareable handle from the target PE
  hipMemGenericAllocationHandle_t importedHandle;
  result = hipMemImportFromShareableHandleCompat(&importedHandle, handleValue,
                                                  hipMemHandleTypePosixFileDescriptor);
  if (result != hipSuccess) {
    MORI_APP_WARN("Failed to import shareable handle from PE {}, chunk {}, hipError: {}",
                  pe, chunkIdx, result);
    return;
  }

  // Map to peer's virtual address space
  result = hipMemMap(peerChunkPtr, chunkSize, 0, importedHandle, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN("Failed hipMemMap imported PE={} chunk={} err={}", pe, chunkIdx, result);
    HIP_RUNTIME_CHECK(hipMemRelease(importedHandle));
    return;
  }

  // Set access permissions for this peer virtual mapping
  hipMemAccessDesc accessDesc;
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = currentDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  result = hipMemSetAccess(peerChunkPtr, chunkSize, &accessDesc, 1);
  if (result != hipSuccess) {
    MORI_APP_WARN("Failed hipMemSetAccess PE={} chunk={} err={}", pe, chunkIdx, result);
  }

  // Mark this chunk as mapped from this peer and save the imported handle
  chunk.mappedPeers.insert(pe);
  chunk.importedHandles[pe] = importedHandle;

  MORI_APP_TRACE("Mapped chunk={} from PE={} to {:p}", chunkIdx, pe, peerChunkPtr);
}

// Step 6: Register P2P peer memory mapping (exchange FDs and map peer chunks)
bool SymmMemManager::RegisterP2PPeerMemory(const std::vector<int>& localShareableHandles,
                                           size_t startChunk, size_t chunksNeeded,
                                           int rank, int worldSize, int currentDev) {
  MORI_APP_TRACE("VMMAlloc: rank={} registering P2P peer memory", rank);
  
  // Find all P2P peers
  std::vector<int> p2pPeers;
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe != rank && context.GetTransportType(pe) == TransportType::P2P) {
      p2pPeers.push_back(pe);
    }
  }

  if (p2pPeers.empty()) {
    MORI_APP_TRACE("VMMAlloc: rank={} no P2P peers, skip P2P registration", rank);
    return true;
  }

  MORI_APP_TRACE("VMMAlloc: rank={} found {} P2P peers for memory mapping", rank, p2pPeers.size());

  // Build global-to-peer rank mapping
  std::vector<int> globalToPeerRank(worldSize, -1);
  std::vector<int> sortedP2pPeers = p2pPeers;
  sortedP2pPeers.push_back(rank);
  std::sort(sortedP2pPeers.begin(), sortedP2pPeers.end());

  int peerRank = 0;
  for (int globalRank : sortedP2pPeers) {
    globalToPeerRank[globalRank] = peerRank++;
  }

  int myPeerRank = globalToPeerRank[rank];
  int p2pWorldSize = sortedP2pPeers.size();

  MORI_APP_TRACE("VMMAlloc: rank={} peerRank={}/{}", rank, myPeerRank, p2pWorldSize);

  // Initialize local bootstrap network for P2P group
  application::LocalBootstrapNetwork localBootstrap(myPeerRank, p2pWorldSize);
  localBootstrap.Initialize();

  // Verify chunk allocation consistency across P2P peers
  struct ChunkInfo {
    size_t startChunk;
    size_t chunksNeeded;
  };
  ChunkInfo myChunkInfo = {startChunk, chunksNeeded};
  std::vector<ChunkInfo> allChunkInfo(worldSize);
  
  bootNet.Allgather(&myChunkInfo, allChunkInfo.data(), sizeof(ChunkInfo));
  
  bool chunkConsistent = true;
  for (int i = 0; i < p2pWorldSize; ++i) {
    int globalRank = sortedP2pPeers[i];
    if (allChunkInfo[globalRank].startChunk != startChunk || 
        allChunkInfo[globalRank].chunksNeeded != chunksNeeded) {
      MORI_APP_ERROR(
          "VMMAlloc: rank={} chunk mismatch! Self=[{},+{}), peer_idx={} global_rank={} has=[{},+{})",
          rank, startChunk, chunksNeeded, i, globalRank,
          allChunkInfo[globalRank].startChunk, allChunkInfo[globalRank].chunksNeeded);
      chunkConsistent = false;
    }
  }
  
  if (!chunkConsistent) {
    MORI_APP_ERROR("VMMAlloc: rank={} aborting due to inconsistent chunk allocation", rank);
    localBootstrap.Finalize();
    return false;
  }
  
  MORI_APP_TRACE("VMMAlloc: rank={} verified all {} P2P peers have matching chunks=[{},+{})",
                 rank, p2pWorldSize - 1, startChunk, chunksNeeded);

  // Prepare local FDs for exchange
  std::vector<int> localFdsForExchange;
  for (size_t i = 0; i < chunksNeeded; ++i) {
    localFdsForExchange.push_back(static_cast<int>(localShareableHandles[i]));
  }

  // Exchange file descriptors via LocalBootstrap
  std::vector<std::vector<int>> p2pFds;
  bool exchangeSuccess = localBootstrap.ExchangeFileDescriptors(localFdsForExchange, p2pFds);

  if (!exchangeSuccess) {
    MORI_APP_ERROR("VMMAlloc: rank={} FD exchange failed! P2P requires same physical machine", rank);
    localBootstrap.Finalize();
    return false;
  }

  MORI_APP_TRACE("VMMAlloc: rank={} exchanged FDs with {} peers", rank, p2pPeers.size());

  // Convert peer-rank-indexed FDs to global-rank-indexed FDs
  std::vector<std::vector<int>> allFds(worldSize);
  for (int globalRank = 0; globalRank < worldSize; ++globalRank) {
    int pRank = globalToPeerRank[globalRank];
    if (pRank >= 0 && pRank < (int)p2pFds.size()) {
      allFds[globalRank] = p2pFds[pRank];
    }
  }

  // Import and map peer chunks
  for (int pe : p2pPeers) {
    MORI_APP_TRACE("VMMAlloc: rank={} importing from peer={}", rank, pe);

    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;

      // Skip if this peer chunk has already been mapped
      if (chunkIdx < vmmMaxChunks && vmmChunks[chunkIdx].mappedPeers.count(pe) > 0) {
        MORI_APP_TRACE("VMMAlloc: rank={} chunk={} already mapped from PE={}, skip",
                       rank, chunkIdx, pe);
        continue;
      }

      // Get the imported FD from exchange result
      int handleValue = -1;
      if (pe < (int)allFds.size() && i < allFds[pe].size()) {
        handleValue = allFds[pe][i];
      }

      if (handleValue == -1) {
        MORI_APP_WARN("RANK {} skipping invalid shareable handle from PE {}, chunk {}",
                      rank, pe, i);
        continue;
      }

      // Calculate target address in peer's virtual space
      void* peerChunkPtr = static_cast<void*>(
          static_cast<char*>(vmmPeerBasePtrs[pe]) + chunkIdx * vmmChunkSize);

      // Import and map peer chunk
      ImportPeerChunk(vmmChunks[chunkIdx], chunkIdx, peerChunkPtr, vmmChunkSize,
                      handleValue, pe, rank, currentDev);
    }
  }

  // Clean up LocalBootstrapNetwork
  localBootstrap.Finalize();
  MORI_APP_TRACE("VMMAlloc: rank={} LocalBootstrap finalized", rank);
  MORI_APP_TRACE("VMMAlloc: rank={} P2P peer memory registration done, chunks={}", rank, chunksNeeded);
  
  return true;
}

// Step 7: Register RDMA memory regions for all chunks
void SymmMemManager::RegisterRdmaChunks(size_t startChunk, size_t chunksNeeded,
                                        int rank, int worldSize) {
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  if (!rdmaDeviceContext) {
    return;
  }

  MORI_APP_TRACE("VMMAlloc: rank={} RDMA register {} chunks", rank, chunksNeeded);

  // Collect local chunk RDMA keys
  std::vector<uint32_t> localChunkRkeys(chunksNeeded);

  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;

    // Skip if this chunk already has RDMA registration (for reused chunks)
    if (vmmChunks[chunkIdx].rdmaRegistered) {
      localChunkRkeys[i] = vmmChunks[chunkIdx].peerRkeys[rank];
      MORI_APP_TRACE("VMMAlloc: rank={} chunk={} RDMA reuse lkey={}",
                     rank, chunkIdx, vmmChunks[chunkIdx].lkey);
      continue;
    }

    void* chunkPtr = static_cast<void*>(
        static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);

    // Register this chunk for RDMA access using dmabuf
    int dmabufFd = vmmChunks[chunkIdx].shareableHandle;
    if (dmabufFd < 0) {
      MORI_APP_ERROR("VMMAlloc: rank={} chunk={} fd not exported, cannot register RDMA",
                     rank, chunkIdx);
      continue;
    }

    application::RdmaMemoryRegion mr =
        rdmaDeviceContext->RegisterRdmaMemoryRegionDmabuf(chunkPtr, vmmChunkSize, dmabufFd);

    vmmChunks[chunkIdx].lkey = mr.lkey;
    vmmChunks[chunkIdx].peerRkeys[rank] = mr.rkey;
    vmmChunks[chunkIdx].rdmaRegistered = true;
    localChunkRkeys[i] = mr.rkey;

    // Update vmmHeapObj's VMMChunkKey arrays
    vmmHeapObj.cpu->vmmLkeyInfo[chunkIdx].key = mr.lkey;
    vmmHeapObj.cpu->vmmRkeyInfo[chunkIdx * worldSize + rank].key = mr.rkey;

    MORI_APP_TRACE("VMMAlloc: rank={} RDMA chunk={} addr={:p} fd={} lkey={} rkey={}",
                   rank, chunkIdx, chunkPtr, dmabufFd, mr.lkey, mr.rkey);
  }

  // Exchange rkeys via bootstrap network
  std::vector<uint32_t> allChunkRkeysFlat(worldSize * chunksNeeded, 0);
  for (size_t i = 0; i < chunksNeeded; ++i) {
    allChunkRkeysFlat[rank * chunksNeeded + i] = localChunkRkeys[i];
  }

  bootNet.Allgather(localChunkRkeys.data(), allChunkRkeysFlat.data(),
                    sizeof(uint32_t) * chunksNeeded);

  MORI_APP_TRACE("VMMAlloc: rank={} RDMA rkeys exchanged", rank);

  // Store remote rkeys for each chunk
  for (int pe = 0; pe < worldSize; ++pe) {
    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;
      uint32_t rkeyValue = allChunkRkeysFlat[pe * chunksNeeded + i];
      vmmChunks[chunkIdx].peerRkeys[pe] = rkeyValue;
      vmmHeapObj.cpu->vmmRkeyInfo[chunkIdx * worldSize + pe].key = rkeyValue;
    }
  }

  // Synchronize updated VMMChunkKey to GPU for these chunks
  size_t keysOffset = startChunk * worldSize * sizeof(VMMChunkKey);
  size_t keysSize = chunksNeeded * worldSize * sizeof(VMMChunkKey);
  HIP_RUNTIME_CHECK(hipMemcpy(
      reinterpret_cast<char*>(vmmHeapObj.gpu->vmmRkeyInfo) + keysOffset,
      vmmHeapObj.cpu->vmmRkeyInfo + startChunk * worldSize,
      keysSize, hipMemcpyHostToDevice));

  // Synchronize lkey info to GPU
  HIP_RUNTIME_CHECK(hipMemcpy(
      vmmHeapObj.gpu->vmmLkeyInfo + startChunk,
      vmmHeapObj.cpu->vmmLkeyInfo + startChunk,
      chunksNeeded * sizeof(VMMChunkKey), hipMemcpyHostToDevice));
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Chunk Allocation                                       */
/* ---------------------------------------------------------------------------------------------- */

SymmMemObjPtr SymmMemManager::VMMAllocChunk(size_t size, HeapType heapType) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized || !heapVAManager) {
    MORI_APP_WARN("VMMAllocChunk failed: VMM heap not initialized");
    return SymmMemObjPtr{nullptr, nullptr};
  }

  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  // Step 1: Allocate virtual address from VA manager
  uintptr_t allocAddr = AllocateVirtualAddress(size);
  if (allocAddr == 0) {
    return SymmMemObjPtr{nullptr, nullptr};
  }

  void* startPtr = reinterpret_cast<void*>(allocAddr);
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[rank]);
  size_t offset = allocAddr - baseAddr;

  // Step 2: Verify VA allocation consistency across all PEs
  if (!VerifyVAConsistency(allocAddr, size, offset, rank, worldSize)) {
    return SymmMemObjPtr{nullptr, nullptr};
  }

  // Step 3: Calculate chunk information
  size_t startChunk = offset / vmmChunkSize;
  size_t endOffset = offset + size;
  size_t endChunk = (endOffset + vmmChunkSize - 1) / vmmChunkSize;
  size_t chunksNeeded = endChunk - startChunk;

  MORI_APP_TRACE("VMMAlloc: rank={} VA={:p} size={} chunks=[{},{})", rank, startPtr, size,
                 startChunk, endChunk);

  // Step 4: Check if these chunks already have physical memory allocated
  bool needPhysicalAlloc = false;
  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;
    if (chunkIdx >= vmmMaxChunks || !vmmChunks[chunkIdx].isAllocated) {
      needPhysicalAlloc = true;
      break;
    }
  }

  // Step 5: Allocate physical memory if needed, otherwise reuse existing
  if (needPhysicalAlloc) {
    MORI_APP_TRACE("VMMAlloc: rank={} allocating {} NEW chunks", rank, chunksNeeded);

    int currentDev = 0;
    hipError_t result = hipGetDevice(&currentDev);
    if (result != hipSuccess) {
      MORI_APP_WARN("VMMAllocChunk failed: Cannot get current device, hipError: {}", result);
      heapVAManager->Free(allocAddr);
      return SymmMemObjPtr{nullptr, nullptr};
    }

    // Configure allocation properties based on heap type
    hipMemAllocationProp allocProp = ConfigureAllocationProp(heapType, currentDev);

    // Initialize local shareable handles (for P2P exchange)
    std::vector<int> localShareableHandles(chunksNeeded);
    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;
      localShareableHandles[i] = (chunkIdx < vmmMaxChunks && vmmChunks[chunkIdx].isAllocated)
                                     ? vmmChunks[chunkIdx].shareableHandle
                                     : -1;
    }

    // Allocate physical memory for each chunk
    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;

      // Reuse chunks that already have physical memory allocated
      if (chunkIdx < vmmMaxChunks && vmmChunks[chunkIdx].isAllocated) {
        vmmChunks[chunkIdx].refCount++;
        MORI_APP_TRACE("VMMAlloc: rank={} reusing chunk {} (refCount={}, fd={})", rank, chunkIdx,
                       vmmChunks[chunkIdx].refCount, vmmChunks[chunkIdx].shareableHandle);
        continue;
      }

      // Allocate new physical chunk
      void* localChunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);

      bool success = AllocateSingleChunk(vmmChunks[chunkIdx], chunkIdx, localChunkPtr,
                                         vmmChunkSize, allocProp, rank, currentDev);
      if (!success) {
        CleanupAllocatedChunks(startChunk, i);
        heapVAManager->Free(allocAddr);
        return SymmMemObjPtr{nullptr, nullptr};
      }

      localShareableHandles[i] = vmmChunks[chunkIdx].shareableHandle;
    }

    // Step 6: Register P2P peer memory mapping
    if (!RegisterP2PPeerMemory(localShareableHandles, startChunk, chunksNeeded,
                               rank, worldSize, currentDev)) {
      return SymmMemObjPtr();
    }

    // Step 7: RDMA registration for RDMA transport
    RegisterRdmaChunks(startChunk, chunksNeeded, rank, worldSize);
  } else {
    MORI_APP_TRACE("VMMAlloc: rank={} REUSE {} chunks at VA=0x{:x}", rank, chunksNeeded, allocAddr);
  }

  // Step 8: Register and return SymmMemObj
  MORI_APP_TRACE("VMMAlloc: done VA={:p} size={}", startPtr, size);
  return VMMRegisterSymmMemObj(startPtr, size, startChunk, chunksNeeded);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Chunk Deallocation                                     */
/* ---------------------------------------------------------------------------------------------- */

void SymmMemManager::VMMFreeChunk(void* localPtr) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized || !localPtr) {
    return;
  }

  int rank = bootNet.GetLocalRank();
  int worldSize = bootNet.GetWorldSize();

  // Find chunk index in local PE's virtual address space
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[rank]);
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(localPtr);

  if (ptrAddr < baseAddr || ptrAddr >= baseAddr + vmmPerPeerSize) {
    return;  // Not in local PE's VMM range
  }

  // Find allocation size by checking registered object
  auto it = memObjPool.find(localPtr);
  if (it == memObjPool.end()) {
    return;
  }

  size_t allocSize = it->second.cpu->size;
  
  // Calculate chunk range correctly
  size_t offset = ptrAddr - baseAddr;
  size_t startChunk = offset / vmmChunkSize;
  size_t endOffset = offset + allocSize;
  size_t endChunk = (endOffset + vmmChunkSize - 1) / vmmChunkSize;
  size_t chunksToFree = endChunk - startChunk;
  
  size_t chunkIdx = startChunk;
  
  MORI_APP_TRACE("VMMFreeChunk: RANK {} freeing ptr={:p} size={} chunks=[{},{})", rank, 
                 localPtr, allocSize, startChunk, endChunk);

  // Verify free consistency across all PEs (symmetric memory requirement)
  // Note: ShmemFree has already synchronized all PEs before calling this function
  struct FreeInfo {
    size_t offset;  // Offset relative to each PE's heap base (must be identical)
    size_t size;    // Allocation size (must be identical)
  };
  FreeInfo myFreeInfo = {offset, allocSize};
  std::vector<FreeInfo> allFreeInfo(worldSize);
  
  bootNet.Allgather(&myFreeInfo, allFreeInfo.data(), sizeof(FreeInfo));
  
  bool freeConsistent = true;
  for (int pe = 0; pe < worldSize; ++pe) {
    if (allFreeInfo[pe].offset != offset || allFreeInfo[pe].size != allocSize) {
      MORI_APP_ERROR(
          "VMMFree: rank={} symmetric memory violated! Self: offset=0x{:x} size={}, PE {}: offset=0x{:x} size={}",
          rank, offset, allocSize, pe, allFreeInfo[pe].offset, allFreeInfo[pe].size);
      freeConsistent = false;
    }
  }
  
  if (!freeConsistent) {
    MORI_APP_ERROR("VMMFree: rank={} detected inconsistent free, but continuing (may cause future issues)", rank);
    // Don't abort here - just log the error and continue to avoid resource leaks
  } else {
    MORI_APP_TRACE("VMMFree: rank={} verified all {} PEs freeing matching offset=0x{:x} size={}", 
                   rank, worldSize, offset, allocSize);
  }

  // No barrier needed here - ShmemFree entry barrier ensures synchronized entry
  // VA Manager is deterministic: same state + same inputs = same output
  if (heapVAManager) {
    heapVAManager->Free(ptrAddr);
    MORI_APP_TRACE("VMMFreeChunk: RANK {} freed VA at 0x{:x} of size {} bytes", rank, ptrAddr,
                   allocSize);
  }

  // Step 1: First unmap from peer virtual address spaces for P2P accessible PEs
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe == rank) continue;  // Skip self

    if (context.GetTransportType(pe) == TransportType::P2P && vmmPeerBasePtrs[pe] != nullptr) {
      for (size_t i = 0; i < chunksToFree; ++i) {
        size_t idx = chunkIdx + i;

        if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated && vmmChunks[idx].refCount == 1 &&
            vmmChunks[idx].mappedPeers.count(pe) > 0) {
          void* peerChunkPtr =
              static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) + idx * vmmChunkSize);

          // All chunks use granularity size (vmmChunkSize)
          hipError_t result = hipMemUnmap(peerChunkPtr, vmmChunkSize);
          if (result != hipSuccess) {
            MORI_APP_WARN("Failed to unmap peer memory for PE {} chunk {}, hipError: {}", pe, idx,
                          result);
          } else {
            // Release the imported handle if exists
            auto handleIt = vmmChunks[idx].importedHandles.find(pe);
            if (handleIt != vmmChunks[idx].importedHandles.end()) {
              HIP_RUNTIME_CHECK(hipMemRelease(handleIt->second));
              vmmChunks[idx].importedHandles.erase(handleIt);
              MORI_APP_TRACE("VMMFreeChunk: RANK {} released imported handle from PE {} for chunk {}",
                             rank, pe, idx);
            }
            
            // Successfully unmapped, remove from mappedPeers
            vmmChunks[idx].mappedPeers.erase(pe);
          }
        }
      }
    }
  }

  // Step 2: Free chunks from local PE's virtual address space
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  for (size_t i = 0; i < chunksToFree; ++i) {
    size_t idx = chunkIdx + i;
    if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated) {
      vmmChunks[idx].refCount--;
      
      MORI_APP_TRACE("VMMFreeChunk: RANK {} decrement chunk {} refCount to {}", rank, idx,
                     vmmChunks[idx].refCount);
      
      // Only release physical resources when refCount reaches 0
      if (vmmChunks[idx].refCount == 0) {
        void* chunkPtr =
            static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + idx * vmmChunkSize);

        // Deregister RDMA memory region if registered
        if (vmmChunks[idx].rdmaRegistered && rdmaDeviceContext) {
          rdmaDeviceContext->DeregisterRdmaMemoryRegion(chunkPtr);
          vmmChunks[idx].rdmaRegistered = false;
          vmmChunks[idx].lkey = 0;
          std::fill(vmmChunks[idx].peerRkeys.begin(), vmmChunks[idx].peerRkeys.end(), 0);
          
          // Clear vmmHeapObj's VMMChunkKey arrays for this chunk (keep next_addr, clear key)
          vmmHeapObj.cpu->vmmLkeyInfo[idx].key = 0;
          for (int pe = 0; pe < worldSize; ++pe) {
            vmmHeapObj.cpu->vmmRkeyInfo[idx * worldSize + pe].key = 0;
          }
          
          MORI_APP_TRACE("VMMFreeChunk: RANK {} deregistered RDMA for chunk {} at {:p}", rank, idx,
                         chunkPtr);
        }

        // Close shareable file descriptor to prevent FD leak (shared by P2P and RDMA)
        if (vmmChunks[idx].shareableHandle != -1) {
          close(vmmChunks[idx].shareableHandle);
          MORI_APP_TRACE("VMMFreeChunk: RANK {} closed FD {} for chunk {} (P2P & RDMA)", rank,
                         vmmChunks[idx].shareableHandle, idx);
        }

        HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunkSize));
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[idx].handle));
        vmmChunks[idx].isAllocated = false;
        vmmChunks[idx].size = 0;
        vmmChunks[idx].shareableHandle = -1;
        vmmChunks[idx].mappedPeers.clear();
        vmmChunks[idx].importedHandles.clear();
        
        MORI_APP_TRACE("VMMFreeChunk: RANK {} fully released chunk {} (physical resources freed)",
                       rank, idx);
      } else {
        MORI_APP_TRACE("VMMFreeChunk: RANK {} chunk {} still in use (refCount={}), physical resources retained",
                       rank, idx, vmmChunks[idx].refCount);
      }
    }
  }

  // Step 3: Synchronize cleared VMMChunkKey to GPU for fully freed chunks (refCount == 0)
  for (size_t i = 0; i < chunksToFree; ++i) {
    size_t idx = chunkIdx + i;
    if (idx < vmmMaxChunks && !vmmChunks[idx].isAllocated) {
      // This chunk was fully freed (refCount reached 0), sync cleared keys to GPU
      size_t keysOffset = idx * worldSize * sizeof(VMMChunkKey);
      size_t keysSize = worldSize * sizeof(VMMChunkKey);
      HIP_RUNTIME_CHECK(hipMemcpy(
          reinterpret_cast<char*>(vmmHeapObj.gpu->vmmRkeyInfo) + keysOffset,
          vmmHeapObj.cpu->vmmRkeyInfo + idx * worldSize,
          keysSize, hipMemcpyHostToDevice));
      
      HIP_RUNTIME_CHECK(hipMemcpy(
          vmmHeapObj.gpu->vmmLkeyInfo + idx,
          vmmHeapObj.cpu->vmmLkeyInfo + idx,
          sizeof(VMMChunkKey), hipMemcpyHostToDevice));
      
      MORI_APP_TRACE("VMMFreeChunk: RANK {} synced cleared keys to GPU for chunk {}", rank, idx);
    }
  }

  VMMDeregisterSymmMemObj(localPtr);
  
  // Note: ShmemFree will synchronize all PEs after this function returns
  MORI_APP_TRACE("VMMFreeChunk: rank={} free complete", rank);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM SymmMemObj Registration                                */
/* ---------------------------------------------------------------------------------------------- */

SymmMemObjPtr SymmMemManager::VMMRegisterSymmMemObj(void* localPtr, size_t size, size_t startChunk,
                                                    size_t numChunks) {
  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  SymmMemObj* cpuMemObj = new SymmMemObj();
  cpuMemObj->localPtr = localPtr;
  cpuMemObj->size = size;

  // Calculate peer pointers based on VMM per-PE virtual address spaces
  cpuMemObj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));

  uintptr_t localOffset =
      reinterpret_cast<uintptr_t>(localPtr) - reinterpret_cast<uintptr_t>(vmmVirtualBasePtr);
  // Set peer pointers to corresponding addresses in each PE's virtual address space
  for (int pe = 0; pe < worldSize; ++pe) {
    cpuMemObj->peerPtrs[pe] = vmmHeapObj.cpu->peerPtrs[pe] + localOffset;
  }
  MORI_APP_TRACE("VMMRegister: localPtr={:p} size={} offset={}", localPtr, size, localOffset);
  
  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess and shareable handles
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // For VMM allocations: directly point to vmmHeapObj's VMMChunkKey arrays (shared across all VMM objects)
  // This allows accessing keys for all chunks in the heap
  cpuMemObj->vmmLkeyInfo = vmmHeapObj.cpu->vmmLkeyInfo;
  cpuMemObj->vmmRkeyInfo = vmmHeapObj.cpu->vmmRkeyInfo;
  cpuMemObj->vmmNumChunks = vmmHeapObj.cpu->vmmNumChunks;
  cpuMemObj->worldSize = worldSize;
  
  // Keep lkey and peerRkeys as nullptr/0 for VMM allocations to distinguish from static heap
  cpuMemObj->lkey = 0;
  cpuMemObj->peerRkeys = nullptr;

  MORI_APP_TRACE("VMMRegister: startChunk={} numChunks={} chunkSize={} spans [{}, {})",
                 startChunk, numChunks, vmmChunkSize, startChunk, startChunk + numChunks);
  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  // For VMM allocations: point to vmmHeapObj's GPU VMMChunkKey arrays (not allocating new memory)
  gpuMemObj->vmmLkeyInfo = vmmHeapObj.gpu->vmmLkeyInfo;
  gpuMemObj->vmmRkeyInfo = vmmHeapObj.gpu->vmmRkeyInfo;
  gpuMemObj->peerRkeys = nullptr;  // Not used for VMM allocations

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  MORI_APP_TRACE("VMMRegister: rank={} done addr={:p} size={}", rank, localPtr, size);
  return memObjPool.at(localPtr);
}


}  // namespace application
}  // namespace mori
