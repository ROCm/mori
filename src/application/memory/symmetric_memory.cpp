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

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/local_bootstrap.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {

namespace application {

SymmMemManager::SymmMemManager(BootstrapNetwork& bootNet, Context& context)
    : bootNet(bootNet), context(context) {}

SymmMemManager::~SymmMemManager() {
  while (!memObjPool.empty()) {
    DeregisterSymmMemObj(memObjPool.begin()->first);
  }
}

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
    if (context.GetTransportType(i) != TransportType::P2P) continue;
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

SymmMemObjPtr SymmMemManager::HeapRegisterSymmMemObj(void* localPtr, size_t size,
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

  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  return memObjPool.at(localPtr);
}

void SymmMemManager::HeapDeregisterSymmMemObj(void* localPtr) {
  if (memObjPool.find(localPtr) == memObjPool.end()) return;

  // No need to deregister RDMA memory region - this is a sub-region of the heap

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

SymmMemObjPtr SymmMemManager::Get(void* localPtr) const {
  if (memObjPool.find(localPtr) == memObjPool.end()) return SymmMemObjPtr{};
  return memObjPool.at(localPtr);
}

// VMM-based symmetric memory management implementation
bool SymmMemManager::IsVMMSupported() const {
  int vmm = 0;
  int currentDev = 0;
  hipError_t result = hipGetDevice(&currentDev);
  if (result != hipSuccess) {
    return false;  // Cannot get current device
  }

  result =
      hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev);
  return (result == hipSuccess && vmm != 0);
}

bool SymmMemManager::InitializeVMMHeap(size_t virtualSize, size_t chunkSize) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (vmmInitialized) {
    return true;  // Already initialized
  }

  if (!IsVMMSupported()) {
    return false;  // VMM not supported
  }

  // Determine optimal chunk size if not provided
  if (chunkSize == 0) {
    int currentDev = 0;
    hipError_t result = hipGetDevice(&currentDev);
    if (result != hipSuccess) {
      return false;
    }

    hipMemAllocationProp allocProp = {};
    allocProp.type = hipMemAllocationTypePinned;
    allocProp.location.type = hipMemLocationTypeDevice;
    allocProp.location.id = currentDev;

    size_t granularity = 0;
    result = hipMemGetAllocationGranularity(&granularity, &allocProp,
                                            hipMemAllocationGranularityRecommended);
    if (result == hipSuccess && granularity > 0) {
      chunkSize = granularity;
      size_t minGranularity = 0;
      result = hipMemGetAllocationGranularity(&minGranularity, &allocProp,
                                              hipMemAllocationGranularityMinimum);
      if (result == hipSuccess && minGranularity > 0) {
        vmmMinChunkSize = minGranularity;
      } else {
        vmmMinChunkSize = 4 * 1024;
      }
    } else {
      // Fallback to minimal granularity if recommended fails
      result = hipMemGetAllocationGranularity(&granularity, &allocProp,
                                              hipMemAllocationGranularityMinimum);
      if (result == hipSuccess && granularity > 0) {
        chunkSize = granularity;
        vmmMinChunkSize = granularity;
      } else {
        // Final fallback to 2MB
        chunkSize = 2 * 1024 * 1024;
        vmmMinChunkSize = 4 * 1024;
      }
    }
  }

  int worldSize = bootNet.GetWorldSize();
  int myPe = bootNet.GetLocalRank();

  std::vector<TransportType> transportTypes = context.GetTransportTypes();
  // Calculate per-PE virtual address space size
  vmmPerPeerSize = virtualSize;
  size_t p2pPeCount = 0;
  for (int pe = 0; pe < worldSize; ++pe) {
    if (transportTypes[pe] == TransportType::P2P && myPe != pe) {
      p2pPeCount++;
    }
  }

  // Only allocate virtual space for P2P accessible PEs
  size_t totalVirtualSize = vmmPerPeerSize * (p2pPeCount + 1);

  vmmVirtualSize = virtualSize;  // Keep original size for local PE
  vmmChunkSize = chunkSize;
  vmmMaxChunks = virtualSize / chunkSize;

  MORI_APP_INFO(
      "VMM Heap Initialization: virtualSize = {}, chunkSize = {}, minChunkSize = {}, maxChunks = "
      "{}, "
      "worldSize = {}, p2pPeCount = {}, perPeerSize = {}, totalVirtualSize = {}",
      vmmVirtualSize, vmmChunkSize, vmmMinChunkSize, vmmMaxChunks, worldSize, p2pPeCount,
      vmmPerPeerSize, totalVirtualSize);

  // Reserve virtual address space for all PEs
  hipError_t result = hipMemAddressReserve(&vmmVirtualBasePtr, totalVirtualSize, 0, nullptr, 0);
  if (result != hipSuccess) {
    MORI_APP_WARN(
        "InitializeVMMHeap failed: hipMemAddressReserve failed for total size {}, hipError: {}",
        totalVirtualSize, result);
    return false;
  }

  // Set up peer base pointers for each PE
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

  // Initialize chunk tracking (only for local PE initially)
  vmmChunks.resize(vmmMaxChunks);

  // Initialize each chunk's peerRkeys vector
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    vmmChunks[i].peerRkeys.resize(worldSize, 0);
  }

  // Create SymmMemObjPtr for the entire VMM heap (metadata only, no RDMA registration)
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

  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess and shareable handles
  cpuHeapObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // No unified RDMA keys - each chunk will have its own per-chunk RDMA registration
  cpuHeapObj->peerRkeys =
      static_cast<uint32_t*>(calloc(worldSize * vmmMaxChunks, sizeof(uint32_t)));
  cpuHeapObj->lkey = static_cast<uint32_t*>(calloc(vmmMaxChunks, sizeof(uint32_t)));
  memset(cpuHeapObj->peerRkeys, 0, sizeof(uint32_t) * worldSize * vmmMaxChunks);
  memset(cpuHeapObj->lkey, 0, sizeof(uint32_t) * vmmMaxChunks);

  // Copy heap object to GPU memory
  SymmMemObj* gpuHeapObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj, cpuHeapObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->peerPtrs, cpuHeapObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->peerRkeys, sizeof(uint32_t) * worldSize * vmmMaxChunks));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->peerRkeys, cpuHeapObj->peerRkeys,
                              sizeof(uint32_t) * worldSize * vmmMaxChunks, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipMalloc(&gpuHeapObj->lkey, sizeof(uint32_t) * vmmMaxChunks));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuHeapObj->lkey, cpuHeapObj->lkey,
                              sizeof(uint32_t) * vmmMaxChunks, hipMemcpyHostToDevice));

  // Store the VMM heap object
  vmmHeapObj = SymmMemObjPtr{cpuHeapObj, gpuHeapObj};

  // Initialize VA Manager for tracking virtual address allocations
  // Pass granularity (chunkSize) to ensure VA blocks don't cross physical memory boundaries
  vmmVAManager = std::make_unique<VMMVAManager>(reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[myPe]),
                                                vmmPerPeerSize, chunkSize);
  MORI_APP_INFO(
      "VA Manager initialized for rank {} with base {:p}, size {} bytes, granularity {} bytes",
      myPe, vmmPeerBasePtrs[myPe], vmmPerPeerSize, chunkSize);

  vmmInitialized = true;
  return true;
}

void SymmMemManager::FinalizeVMMHeap() {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    return;
  }

  int rank = bootNet.GetLocalRank();

  // Deregister per-chunk RDMA registrations
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

  // Free all allocated chunks in local PE's virtual address space
  for (size_t i = 0; i < vmmMaxChunks; ++i) {
    if (vmmChunks[i].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + i * vmmChunkSize);
      // All chunks use granularity size (vmmChunkSize)
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunkSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[i].handle));
    }
  }

  // Free virtual address space (entire multi-PE space)
  if (vmmVirtualBasePtr) {
    size_t totalVirtualSize = vmmPerPeerSize * bootNet.GetWorldSize();
    HIP_RUNTIME_CHECK(hipMemAddressFree(vmmVirtualBasePtr, totalVirtualSize));
    vmmVirtualBasePtr = nullptr;
  }

  // Clean up VMM heap object
  if (vmmHeapObj.IsValid()) {
    free(vmmHeapObj.cpu->peerPtrs);
    free(vmmHeapObj.cpu->peerRkeys);
    free(vmmHeapObj.cpu->lkey);
    free(vmmHeapObj.cpu->ipcMemHandles);
    free(vmmHeapObj.cpu);
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu->peerPtrs));
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu->peerRkeys));
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu->lkey));
    HIP_RUNTIME_CHECK(hipFree(vmmHeapObj.gpu));
    vmmHeapObj = SymmMemObjPtr{nullptr, nullptr};
  }

  // Clean up VA Manager
  if (vmmVAManager) {
    vmmVAManager->Reset();
    vmmVAManager.reset();
    MORI_APP_INFO("VA Manager cleaned up for rank {}", rank);
  }

  vmmChunks.clear();
  vmmPeerBasePtrs.clear();
  vmmMinChunkSize = 0;
  vmmPerPeerSize = 0;
  vmmInitialized = false;
}

SymmMemObjPtr SymmMemManager::VMMAllocChunk(size_t size, uint32_t allocType) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized || !vmmVAManager) {
    MORI_APP_WARN("VMMAllocChunk failed: VMM heap not initialized");
    return SymmMemObjPtr{nullptr, nullptr};
  }

  int worldSize = bootNet.GetWorldSize();
  int rank = bootNet.GetLocalRank();

  // Step 1: Allocate virtual address from VA manager (may reuse freed VA)
  uintptr_t allocAddr = vmmVAManager->Allocate(size, 256);
  if (allocAddr == 0) {
    MORI_APP_ERROR("VMMAllocChunk failed: VA allocation failed for size {} bytes", size);

    // Log VA manager stats for debugging
    size_t totalBlocks, freeBlocks, allocatedBlocks, totalFreeSpace, largestFreeBlock;
    vmmVAManager->GetStats(totalBlocks, freeBlocks, allocatedBlocks, totalFreeSpace,
                           largestFreeBlock);
    MORI_APP_ERROR(
        "VA Manager stats: totalBlocks={}, freeBlocks={}, allocatedBlocks={}, "
        "totalFreeSpace={} bytes, largestFreeBlock={} bytes",
        totalBlocks, freeBlocks, allocatedBlocks, totalFreeSpace, largestFreeBlock);
    return SymmMemObjPtr{nullptr, nullptr};
  }

  void* startPtr = reinterpret_cast<void*>(allocAddr);

  // Calculate chunk information
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(vmmPeerBasePtrs[rank]);
  size_t offset = allocAddr - baseAddr;
  size_t startChunk = offset / vmmChunkSize;
  size_t endOffset = offset + size;
  size_t endChunk = (endOffset + vmmChunkSize - 1) / vmmChunkSize;
  size_t chunksNeeded = endChunk - startChunk;

  MORI_APP_TRACE(
      "VMMAllocChunk: RANK {} allocated VA at {:p}, offset={}, size={}, startChunk={}, "
      "endChunk={}, chunksNeeded={}",
      rank, startPtr, offset, size, startChunk, endChunk, chunksNeeded);

  // Step 2: Check if these chunks already have physical memory allocated (for reuse)
  bool needPhysicalAlloc = false;
  for (size_t i = 0; i < chunksNeeded; ++i) {
    size_t chunkIdx = startChunk + i;
    if (chunkIdx >= vmmMaxChunks || !vmmChunks[chunkIdx].isAllocated) {
      needPhysicalAlloc = true;
      break;
    }
  }

  // Step 3: Allocate physical memory only if needed
  if (needPhysicalAlloc) {
    MORI_APP_TRACE("VMMAllocChunk: RANK {} allocating NEW physical memory for {} chunks", rank,
                   chunksNeeded);

    int currentDev = 0;
    hipError_t result = hipGetDevice(&currentDev);
    if (result != hipSuccess) {
      MORI_APP_WARN("VMMAllocChunk failed: Cannot get current device, hipError: {}", result);
      vmmVAManager->Free(allocAddr);  // Free the VA on failure
      return SymmMemObjPtr{nullptr, nullptr};
    }

    hipMemAllocationProp allocProp = {};
    allocProp.type = static_cast<hipMemAllocationType>(allocType);
    allocProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    allocProp.location.type = hipMemLocationTypeDevice;
    allocProp.location.id = currentDev;

    std::vector<int> localShareableHandles(chunksNeeded);
    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;
      if (chunkIdx < vmmMaxChunks && vmmChunks[chunkIdx].isAllocated) {
        localShareableHandles[i] = vmmChunks[chunkIdx].shareableHandle;
      } else {
        localShareableHandles[i] = -1;
      }
    }

    for (size_t i = 0; i < chunksNeeded; ++i) {
      size_t chunkIdx = startChunk + i;

      // Skip chunks that already have physical memory allocated
      if (chunkIdx < vmmMaxChunks && vmmChunks[chunkIdx].isAllocated) {
        MORI_APP_TRACE("VMMAllocChunk: RANK {} skipping chunk {} (already allocated, handle={})",
                       rank, chunkIdx, vmmChunks[chunkIdx].shareableHandle);
        continue;
      }

      void* localChunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);

      result = hipMemCreate(&vmmChunks[chunkIdx].handle, vmmChunkSize, &allocProp, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "VMMAllocChunk failed: hipMemCreate failed for chunk {} with size {} bytes, allocType: "
            "{}, device: {}, hipError: {}",
            chunkIdx, vmmChunkSize, allocType, currentDev, result);
        // Cleanup already allocated chunks
        for (size_t j = 0; j < i; ++j) {
          size_t cleanupIdx = startChunk + j;
          HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
          vmmChunks[cleanupIdx].isAllocated = false;
        }
        vmmVAManager->Free(allocAddr);  // Free the VA on failure
        return SymmMemObjPtr{nullptr, nullptr};
      }

      // Export shareable handle for cross-process sharing
      result = hipMemExportToShareableHandle((void*)&vmmChunks[chunkIdx].shareableHandle,
                                             vmmChunks[chunkIdx].handle,
                                             hipMemHandleTypePosixFileDescriptor, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "VMMAllocChunk warning: hipMemExportToShareableHandle failed for chunk {}, hipError: "
            "{}. "
            "Cross-process sharing may not work.",
            chunkIdx, result);
        vmmChunks[chunkIdx].shareableHandle = -1;
      }
      localShareableHandles[i] = vmmChunks[chunkIdx].shareableHandle;
      MORI_APP_TRACE(
          "VMMAllocChunk: RANK {} Created chunk {} with granularity size {} bytes, shareable "
          "handle "
          "{}",
          rank, chunkIdx, vmmChunkSize, vmmChunks[chunkIdx].shareableHandle);

      // Map physical memory to local virtual address
      result = hipMemMap(localChunkPtr, vmmChunkSize, 0, vmmChunks[chunkIdx].handle, 0);
      if (result != hipSuccess) {
        MORI_APP_WARN(
            "VMMAllocChunk failed: hipMemMap failed for chunk {} at address {:p} with size {} "
            "bytes, "
            "hipError: {}",
            chunkIdx, localChunkPtr, vmmChunkSize, result);
        HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[chunkIdx].handle));
        // Cleanup already allocated chunks
        for (size_t j = 0; j < i; ++j) {
          size_t cleanupIdx = startChunk + j;
          void* cleanupPtr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) +
                                                cleanupIdx * vmmChunkSize);
          HIP_RUNTIME_CHECK(hipMemUnmap(cleanupPtr, vmmChunkSize));
          HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[cleanupIdx].handle));
          vmmChunks[cleanupIdx].isAllocated = false;
        }
        vmmVAManager->Free(allocAddr);  // Free the VA on failure
        return SymmMemObjPtr{nullptr, nullptr};
      }

      vmmChunks[chunkIdx].isAllocated = true;
      vmmChunks[chunkIdx].size = vmmChunkSize;  // Both virtual and physical use granularity size
    }

    MORI_APP_TRACE("RANK {} VMMAllocChunk: Starting shareable handle exchange for P2P peers", rank);
    std::vector<int> p2pPeers;
    for (int pe = 0; pe < worldSize; ++pe) {
      if (pe != rank && context.GetTransportType(pe) == TransportType::P2P) {
        p2pPeers.push_back(pe);
      }
    }

    if (p2pPeers.empty()) {
      MORI_APP_TRACE("RANK {} VMMAllocChunk: No P2P peers, skipping FD exchange", rank);
    } else {
      MORI_APP_TRACE("RANK {} VMMAllocChunk: Found {} P2P peers for FD exchange: {}", rank,
                     p2pPeers.size(), [&p2pPeers]() {
                       std::string s;
                       for (int p : p2pPeers) s += std::to_string(p) + " ";
                       return s;
                     }());

      std::vector<int> globalToPeerRank(worldSize, -1);  // -1 = not in P2P group
      int peerRank = 0;

      // Assign peer ranks to all P2P peers in ascending global rank order
      std::vector<int> sortedP2pPeers = p2pPeers;
      sortedP2pPeers.push_back(rank);
      std::sort(sortedP2pPeers.begin(), sortedP2pPeers.end());

      for (int globalRank : sortedP2pPeers) {
        globalToPeerRank[globalRank] = peerRank++;
      }

      int myPeerRank = globalToPeerRank[rank];
      int p2pWorldSize = sortedP2pPeers.size();

      MORI_APP_TRACE("RANK {} has P2P peer rank {}/{} in P2P group", rank, myPeerRank,
                     p2pWorldSize);

      application::LocalBootstrapNetwork localBootstrap(myPeerRank, p2pWorldSize);
      localBootstrap.Initialize();

      // Prepare local FDs for exchange
      std::vector<int> localFdsForExchange;
      for (size_t i = 0; i < chunksNeeded; ++i) {
        size_t chunkIdx = startChunk + i;
        int handleValue = static_cast<int>(localShareableHandles[chunkIdx]);
        localFdsForExchange.push_back(handleValue);
      }

      std::vector<std::vector<int>> p2pFds;
      bool exchangeSuccess = localBootstrap.ExchangeFileDescriptors(localFdsForExchange, p2pFds);

      if (!exchangeSuccess) {
        MORI_APP_ERROR("RANK {} VMMAllocChunk: Failed to exchange file descriptors with P2P peers",
                       rank);
        MORI_APP_ERROR("P2P FD exchange requires all P2P peers on the same physical machine!");
        localBootstrap.Finalize();
        return SymmMemObjPtr();
      }

      MORI_APP_TRACE("RANK {} VMMAllocChunk: Successfully exchanged FDs with {} P2P peers", rank,
                     p2pPeers.size());

      // Convert peer-rank-indexed FDs to global-rank-indexed FDs
      std::vector<std::vector<int>> allFds(worldSize);
      for (int globalRank = 0; globalRank < worldSize; ++globalRank) {
        int pRank = globalToPeerRank[globalRank];
        if (pRank >= 0 && pRank < (int)p2pFds.size()) {
          allFds[globalRank] = p2pFds[pRank];
        }
      }

      for (int pe : p2pPeers) {
        MORI_APP_TRACE("VMMAllocChunk: RANK {} importing and mapping from P2P peer {}", rank, pe);

        for (size_t i = 0; i < chunksNeeded; ++i) {
          // Get the imported FD from exchange result (now using global rank indexing)
          int handleValue = -1;
          if (pe < (int)allFds.size() && i < allFds[pe].size()) {
            handleValue = allFds[pe][i];
          }

          if (handleValue == -1) {
            MORI_APP_WARN("RANK {} skipping invalid shareable handle from PE {}, chunk {}", rank,
                          pe, i);
            continue;
          }

          // Calculate target address in peer's virtual space
          void* peerChunkPtr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) +
                                                  (startChunk + i) * vmmChunkSize);

          // Import the shareable handle from the target PE
          hipMemGenericAllocationHandle_t importedHandle;
          result = hipMemImportFromShareableHandle(&importedHandle, (void*)&handleValue,
                                                   hipMemHandleTypePosixFileDescriptor);
          if (result != hipSuccess) {
            MORI_APP_WARN("Failed to import shareable handle from PE {}, chunk {}, hipError: {}",
                          pe, i, result);
            continue;
          }

          // Map to peer's virtual address space (use granularity size)
          result = hipMemMap(peerChunkPtr, vmmChunkSize, 0, importedHandle, 0);
          if (result != hipSuccess) {
            MORI_APP_WARN(
                "Failed to map imported memory to PE {} virtual space, chunk {}, hipError: {}", pe,
                i, result);
            HIP_RUNTIME_CHECK(hipMemRelease(importedHandle));
            continue;
          }

          // Set access permissions for this peer virtual mapping
          hipMemAccessDesc accessDesc;
          accessDesc.location.type = hipMemLocationTypeDevice;
          accessDesc.location.id = currentDev;
          accessDesc.flags = hipMemAccessFlagsProtReadWrite;

          result = hipMemSetAccess(peerChunkPtr, vmmChunkSize, &accessDesc, 1);
          if (result != hipSuccess) {
            MORI_APP_WARN("Failed to set access for PE {} virtual space, chunk {}, hipError: {}",
                          pe, i, result);
          }

          MORI_APP_TRACE("Successfully mapped chunk {} from PE {} to address {}", i, pe,
                         peerChunkPtr);
        }
      }

      // Clean up LocalBootstrapNetwork after FD exchange is complete
      localBootstrap.Finalize();
      MORI_APP_TRACE("RANK {} LocalBootstrapNetwork finalized after FD exchange", rank);
    }

    MORI_APP_TRACE("RANK {} VMMAllocChunk: Shareable handle exchange completed", rank);

    MORI_APP_TRACE("VMMAllocChunk: RANK {} Allocated {} bytes using {} chunks starting at chunk {}",
                   rank, size, chunksNeeded, startChunk);

    // Step 4: Per-chunk RDMA registration for RDMA transport
    // Each chunk gets its own RDMA memory region with independent lkey/rkeys
    RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
    if (rdmaDeviceContext) {
      MORI_APP_TRACE("VMMAllocChunk: RANK {} starting per-chunk RDMA registration for {} chunks",
                     rank, chunksNeeded);

      // Collect local chunk RDMA keys
      std::vector<uint32_t> localChunkRkeys(chunksNeeded);

      for (size_t i = 0; i < chunksNeeded; ++i) {
        size_t chunkIdx = startChunk + i;

        // Skip if this chunk already has RDMA registration (for reused chunks)
        if (vmmChunks[chunkIdx].rdmaRegistered) {
          localChunkRkeys[i] = vmmChunks[chunkIdx].peerRkeys[rank];
          MORI_APP_TRACE(
              "VMMAllocChunk: RANK {} chunk {} already has RDMA registration, lkey={}, rkey={}",
              rank, chunkIdx, vmmChunks[chunkIdx].lkey, localChunkRkeys[i]);
          continue;
        }

        void* chunkPtr =
            static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + chunkIdx * vmmChunkSize);

        // Register this chunk for RDMA access (use granularity size)
        application::RdmaMemoryRegion mr =
            rdmaDeviceContext->RegisterRdmaMemoryRegion(chunkPtr, vmmChunkSize);

        vmmChunks[chunkIdx].lkey = mr.lkey;
        vmmChunks[chunkIdx].peerRkeys[rank] = mr.rkey;
        vmmChunks[chunkIdx].rdmaRegistered = true;
        localChunkRkeys[i] = mr.rkey;

        MORI_APP_TRACE(
            "VMMAllocChunk: RANK {} registered RDMA for chunk {} at {:p}, size={}, lkey={}, "
            "rkey={}",
            rank, chunkIdx, chunkPtr, vmmChunkSize, mr.lkey, mr.rkey);
      }

      std::vector<uint32_t> allChunkRkeysFlat(worldSize * chunksNeeded, 0);

      // Copy local rkeys to correct position
      for (size_t i = 0; i < chunksNeeded; ++i) {
        allChunkRkeysFlat[rank * chunksNeeded + i] = localChunkRkeys[i];
      }

      // Exchange rkeys via bootstrap network
      bootNet.Allgather(localChunkRkeys.data(), allChunkRkeysFlat.data(),
                        sizeof(uint32_t) * chunksNeeded);

      MORI_APP_TRACE("RANK {} VMMAllocChunk: Exchanged RDMA rkeys among all PEs", rank);

      // Store remote rkeys for each chunk
      for (int pe = 0; pe < worldSize; ++pe) {
        for (size_t i = 0; i < chunksNeeded; ++i) {
          size_t chunkIdx = startChunk + i;
          uint32_t rkeyValue = allChunkRkeysFlat[pe * chunksNeeded + i];
          vmmChunks[chunkIdx].peerRkeys[pe] = rkeyValue;

          MORI_APP_TRACE("RANK {} VMMAllocChunk: chunk[{}].peerRkeys[{}] = {}", rank, chunkIdx, pe,
                         rkeyValue);
        }
      }
    }
  } else {
    // Step 4: Reuse existing physical memory (VA was previously allocated)
    MORI_APP_TRACE(
        "VMMAllocChunk: RANK {} REUSING existing physical memory for {} chunks at virtual address "
        "0x{:x}",
        rank, chunksNeeded, allocAddr);
  }

  // Create SymmMemObj for VMM allocation
  MORI_APP_TRACE("VMMAllocChunk: Allocated memory at virtual address {:p} of size {} bytes",
                 startPtr, size);
  return VMMRegisterSymmMemObj(startPtr, size, startChunk, chunksNeeded);
}

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

  size_t chunkIdx = (ptrAddr - baseAddr) / vmmChunkSize;

  // Find allocation size by checking registered object
  auto it = memObjPool.find(localPtr);
  if (it == memObjPool.end()) {
    return;
  }

  size_t allocSize = it->second.cpu->size;
  size_t chunksToFree = (allocSize + vmmChunkSize - 1) / vmmChunkSize;

  // Free VA from VA manager
  if (vmmVAManager) {
    vmmVAManager->Free(ptrAddr);
    MORI_APP_TRACE("VMMFreeChunk: RANK {} freed VA at 0x{:x} of size {} bytes", rank, ptrAddr,
                   allocSize);
  }

  // Free chunks from local PE's virtual address space
  RdmaDeviceContext* rdmaDeviceContext = context.GetRdmaDeviceContext();
  for (size_t i = 0; i < chunksToFree; ++i) {
    size_t idx = chunkIdx + i;
    if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated) {
      void* chunkPtr =
          static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[rank]) + idx * vmmChunkSize);

      // Deregister RDMA memory region if registered
      if (vmmChunks[idx].rdmaRegistered && rdmaDeviceContext) {
        rdmaDeviceContext->DeregisterRdmaMemoryRegion(chunkPtr);
        vmmChunks[idx].rdmaRegistered = false;
        vmmChunks[idx].lkey = 0;
        std::fill(vmmChunks[idx].peerRkeys.begin(), vmmChunks[idx].peerRkeys.end(), 0);
        MORI_APP_TRACE("VMMFreeChunk: RANK {} deregistered RDMA for chunk {} at {:p}", rank, idx,
                       chunkPtr);
      }

      // All chunks use granularity size (vmmChunkSize)
      HIP_RUNTIME_CHECK(hipMemUnmap(chunkPtr, vmmChunkSize));
      HIP_RUNTIME_CHECK(hipMemRelease(vmmChunks[idx].handle));
      vmmChunks[idx].isAllocated = false;
      vmmChunks[idx].size = 0;
      vmmChunks[idx].shareableHandle = -1;
    }
  }

  // Also unmap from peer virtual address spaces for P2P accessible PEs
  for (int pe = 0; pe < worldSize; ++pe) {
    if (pe == rank) continue;  // Skip self, already unmapped above

    // Only unmap from P2P accessible PEs where we previously mapped
    if (context.GetTransportType(pe) == TransportType::P2P && vmmPeerBasePtrs[pe] != nullptr) {
      for (size_t i = 0; i < chunksToFree; ++i) {
        size_t idx = chunkIdx + i;
        if (idx < vmmMaxChunks && vmmChunks[idx].isAllocated) {
          void* peerChunkPtr =
              static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[pe]) + idx * vmmChunkSize);

          // All chunks use granularity size (vmmChunkSize)
          hipError_t result = hipMemUnmap(peerChunkPtr, vmmChunkSize);
          if (result != hipSuccess) {
            MORI_APP_WARN("Failed to unmap peer memory for PE {} chunk {}, hipError: {}", pe, idx,
                          result);
          }
        }
      }
    }
  }

  HeapDeregisterSymmMemObj(localPtr);
}

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
  MORI_APP_TRACE("VMMRegisterSymmMemObj: localOffset = {}", localOffset);
  // Set peer pointers to corresponding addresses in each PE's virtual address space
  for (int pe = 0; pe < worldSize; ++pe) {
    cpuMemObj->peerPtrs[pe] = vmmHeapObj.cpu->peerPtrs[pe] + localOffset;
    MORI_APP_TRACE("VMMRegisterSymmMemObj: peerPtrs[{}] = {:p}", pe,
                   reinterpret_cast<void*>(cpuMemObj->peerPtrs[pe]));
  }
  MORI_APP_TRACE(
      "VMMRegisterSymmMemObj: Registered memory at local address {:p} of size {} "
      "bytes",
      localPtr, size);
  // VMM doesn't need IPC handles - access is managed through hipMemSetAccess and shareable handles
  cpuMemObj->ipcMemHandles =
      static_cast<hipIpcMemHandle_t*>(calloc(worldSize, sizeof(hipIpcMemHandle_t)));

  // Use per-chunk RDMA keys
  // For allocations spanning multiple chunks, we use the first chunk's RDMA keys
  // Note: RDMA operations crossing chunk boundaries will need special handling
  cpuMemObj->peerRkeys = static_cast<uint32_t*>(calloc(worldSize, sizeof(uint32_t)));

  if (startChunk < vmmMaxChunks && vmmChunks[startChunk].rdmaRegistered) {
    // Use the first chunk's lkey
    cpuMemObj->lkey = vmmChunks[startChunk].lkey;

    // Copy all peer rkeys from the first chunk
    memcpy(cpuMemObj->peerRkeys, vmmChunks[startChunk].peerRkeys.data(),
           sizeof(uint32_t) * worldSize);

    MORI_APP_TRACE("VMMRegisterSymmMemObj: Using chunk {} RDMA keys, lkey={}, spanning {} chunks",
                   startChunk, cpuMemObj->lkey, numChunks);

    // Warning if allocation spans multiple chunks (RDMA may need multiple operations)
    if (numChunks > 1) {
      MORI_APP_WARN(
          "VMMRegisterSymmMemObj: Allocation at {:p} spans {} chunks - "
          "RDMA operations may require chunk-aware handling",
          localPtr, numChunks);
    }
  } else {
    // No RDMA registration for this chunk (shouldn't happen if RDMA is enabled)
    cpuMemObj->lkey = 0;
    memset(cpuMemObj->peerRkeys, 0, sizeof(uint32_t) * worldSize);
    MORI_APP_TRACE("VMMRegisterSymmMemObj: No RDMA registration for chunk {}", startChunk);
  }
  SymmMemObj* gpuMemObj;
  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj, sizeof(SymmMemObj)));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj, cpuMemObj, sizeof(SymmMemObj), hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerPtrs, sizeof(uintptr_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerPtrs, cpuMemObj->peerPtrs,
                              sizeof(uintptr_t) * worldSize, hipMemcpyHostToDevice));

  HIP_RUNTIME_CHECK(hipMalloc(&gpuMemObj->peerRkeys, sizeof(uint32_t) * worldSize));
  HIP_RUNTIME_CHECK(hipMemcpy(gpuMemObj->peerRkeys, cpuMemObj->peerRkeys,
                              sizeof(uint32_t) * worldSize, hipMemcpyHostToDevice));

  memObjPool.insert({localPtr, SymmMemObjPtr{cpuMemObj, gpuMemObj}});
  MORI_APP_TRACE(
      "VMMRegisterSymmMemObj: rank: {} Registered memory at local address {:p} of size {} "
      "bytes",
      rank, localPtr, size);
  return memObjPool.at(localPtr);
}

bool SymmMemManager::VMMImportPeerMemory(int peerPe, void* localBaseAddr, size_t offset,
                                         size_t size, const std::vector<int>& shareableHandles) {
  std::lock_guard<std::mutex> lock(vmmLock);

  if (!vmmInitialized) {
    MORI_APP_WARN("VMMImportPeerMemory failed: VMM heap not initialized");
    return false;
  }

  int worldSize = bootNet.GetWorldSize();
  if (peerPe >= worldSize || peerPe < 0) {
    MORI_APP_WARN("VMMImportPeerMemory failed: Invalid peerPe {}", peerPe);
    return false;
  }

  // Calculate target address in peer's dedicated virtual space
  void* targetAddr = static_cast<void*>(static_cast<char*>(vmmPeerBasePtrs[peerPe]) + offset);
  size_t chunksNeeded = (size + vmmChunkSize - 1) / vmmChunkSize;

  MORI_APP_INFO(
      "VMMImportPeerMemory: Importing {} chunks from PE {} to peer virtual space at offset {}",
      chunksNeeded, peerPe, offset);

  // Import and map each chunk to peer's virtual address space
  for (size_t i = 0; i < chunksNeeded && i < shareableHandles.size(); ++i) {
    if (shareableHandles[i] == -1) {
      MORI_APP_WARN("VMMImportPeerMemory: Invalid shareable handle for chunk {}", i);
      continue;
    }

    hipMemGenericAllocationHandle_t importedHandle;

    // Import the shareable handle
    hipError_t result = hipMemImportFromShareableHandle(
        &importedHandle, (void*)&shareableHandles[i], hipMemHandleTypePosixFileDescriptor);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory failed: hipMemImportFromShareableHandle failed for chunk {}, "
          "hipError: {}",
          i, result);

      // Cleanup already imported chunks
      for (size_t j = 0; j < i; ++j) {
        void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(chunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during import failure, hipError: {}", j,
                        unmapResult);
        }
      }
      return false;
    }

    // Map the imported handle to peer's virtual address space
    void* chunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + i * vmmChunkSize);
    size_t chunkSize = std::min(size - i * vmmChunkSize, vmmChunkSize);

    result = hipMemMap(chunkAddr, chunkSize, 0, importedHandle, 0);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory failed: hipMemMap failed for imported chunk {}, hipError: {}", i,
          result);

      // Release the imported handle and cleanup
      hipError_t releaseResult = hipMemRelease(importedHandle);
      if (releaseResult != hipSuccess) {
        MORI_APP_WARN("Failed to release imported handle, hipError: {}", releaseResult);
      }
      for (size_t j = 0; j < i; ++j) {
        void* prevChunkAddr = static_cast<void*>(static_cast<char*>(targetAddr) + j * vmmChunkSize);
        hipError_t unmapResult = hipMemUnmap(prevChunkAddr, vmmChunkSize);
        if (unmapResult != hipSuccess) {
          MORI_APP_WARN("Failed to cleanup chunk {} during map failure, hipError: {}", j,
                        unmapResult);
        }
      }
      return false;
    }

    // Set access permissions for the current device to access the imported memory
    hipMemAccessDesc accessDesc;
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;  // Current device
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;

    result = hipMemSetAccess(chunkAddr, chunkSize, &accessDesc, 1);
    if (result != hipSuccess) {
      MORI_APP_WARN(
          "VMMImportPeerMemory warning: hipMemSetAccess failed for imported chunk {}, hipError: {}",
          i, result);
      // Continue without setting access - might still work in some cases
    }
  }

  MORI_APP_INFO(
      "VMMImportPeerMemory: Successfully imported {} chunks from PE {} to peer virtual space",
      chunksNeeded, peerPe);
  return true;
}

}  // namespace application
}  // namespace mori
