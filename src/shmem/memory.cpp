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

#include "mori/application/memory/symmetric_memory.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"
#include "mori/shmem/internal.hpp"

namespace mori {
namespace shmem {

void* ShmemMalloc(size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (size == 0) {
    return nullptr;
  }

  // Use different allocation strategies based on mode
  if (states->mode == ShmemMode::StaticHeap) {
    // Static heap mode: bump allocator from pre-allocated heap
    // Align to 256 bytes for better performance
    constexpr size_t ALIGNMENT = 256;
    size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    void* ptr;
    
    {
      std::lock_guard<std::mutex> lock(states->memoryStates->heapLock);

      // Check if we have enough space
      if (states->memoryStates->staticHeapUsed + size > states->memoryStates->staticHeapSize) {
        MORI_SHMEM_ERROR("Out of symmetric heap memory! Requested: {} bytes, Available: {} bytes", size,
                         states->memoryStates->staticHeapSize - states->memoryStates->staticHeapUsed);
        return nullptr;
      }

      // Allocate from the bump pointer
      uintptr_t baseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
      ptr = reinterpret_cast<void*>(baseAddr + states->memoryStates->staticHeapUsed);
      states->memoryStates->staticHeapUsed += size;
    }

    // CRITICAL: Synchronize staticHeapUsed across all ranks
    application::BootstrapNetwork& bootNet = *states->bootStates->bootNet;
    int rank = bootNet.GetLocalRank();
    int worldSize = bootNet.GetWorldSize();
    
    size_t myHeapUsed = states->memoryStates->staticHeapUsed;
    size_t* allHeapUsed = static_cast<size_t*>(malloc(worldSize * sizeof(size_t)));
    bootNet.Allgather(&myHeapUsed, allHeapUsed, sizeof(size_t));
    
    // Verify all ranks have consistent staticHeapUsed
    for (int i = 0; i < worldSize; i++) {
      if (allHeapUsed[i] != myHeapUsed) {
        MORI_SHMEM_ERROR("Rank {} staticHeapUsed mismatch! My: {}, Rank {}: {}",
                         rank, myHeapUsed, i, allHeapUsed[i]);
        free(allHeapUsed);
        return nullptr;
      }
    }
    free(allHeapUsed);

    states->memoryStates->symmMemMgr->HeapRegisterSymmMemObj(ptr, size,
                                                             &states->memoryStates->staticHeapObj);
    
    uintptr_t baseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
    uintptr_t ptrValue = reinterpret_cast<uintptr_t>(ptr);
    MORI_SHMEM_TRACE("Allocated {} bytes at ptr={:#x} (offset={}, aligned to 256={}) (total used: {} / {})", 
                     size, ptrValue, ptrValue - baseAddr, 
                     (ptrValue % 256 == 0 ? "yes" : "no"),
                     states->memoryStates->staticHeapUsed, states->memoryStates->staticHeapSize);

    return ptr;
  } else {
    // Isolation mode: each allocation gets its own SymmMemObj
    application::SymmMemObjPtr obj = states->memoryStates->symmMemMgr->Malloc(size);
    MORI_SHMEM_TRACE("Allocated shared memory of size {} in isolation mode", size);
    if (obj.IsValid()) {
      return obj.cpu->localPtr;
    }
    return nullptr;
  }
}

void* ShmemMallocAlign(size_t alignment, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (size == 0) {
    return nullptr;
  }

  // Validate alignment: must be power of 2
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    MORI_SHMEM_ERROR("Invalid alignment: {} (must be a power of 2)", alignment);
    return nullptr;
  }

  // Align size to the requested alignment
  size = (size + alignment - 1) & ~(alignment - 1);
  return ShmemMalloc(size);
}

void* ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (size == 0) {
    return nullptr;
  }

  // Use different allocation strategies based on mode
  if (states->mode == ShmemMode::StaticHeap) {
    // In static heap mode, flags are ignored - use the same allocator as ShmemMalloc
    MORI_SHMEM_TRACE("Allocated shared memory of size {} with flags {} (flags ignored in static heap mode)", size, flags);
    return ShmemMalloc(size);
  } else {
    // Isolation mode: use ExtMallocWithFlags directly
    application::SymmMemObjPtr obj =
        states->memoryStates->symmMemMgr->ExtMallocWithFlags(size, flags);
    MORI_SHMEM_TRACE("Allocated shared memory of size {} with flags {} in isolation mode", size, flags);
    if (obj.IsValid()) {
      return obj.cpu->localPtr;
    }
    return nullptr;
  }


  // Check if using VMM heap
  if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
    // Use VMM-based allocation
    application::SymmMemObjPtr memObj = states->memoryStates->symmMemMgr->VMMAllocChunk(size);
    if (!memObj.IsValid()) {
      MORI_SHMEM_ERROR("VMM allocation failed for size {} bytes", size);
      return nullptr;
    }
    
    MORI_SHMEM_TRACE("VMM allocated {} bytes at {}", size, memObj.cpu->localPtr);
    return memObj.cpu->localPtr;
  } else {
    // Use traditional static heap bump allocator
    // Align to 256 bytes for better performance
    constexpr size_t ALIGNMENT = 256;
    size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    std::lock_guard<std::mutex> lock(states->memoryStates->heapLock);

    // Check if we have enough space
    if (states->memoryStates->staticHeapUsed + size > states->memoryStates->staticHeapSize) {
      MORI_SHMEM_ERROR("Out of symmetric heap memory! Requested: {} bytes, Available: {} bytes", size,
                       states->memoryStates->staticHeapSize - states->memoryStates->staticHeapUsed);
      return nullptr;
    }

    // Allocate from the bump pointer
    uintptr_t baseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
    void* ptr = reinterpret_cast<void*>(baseAddr + states->memoryStates->staticHeapUsed);
    states->memoryStates->staticHeapUsed += size;

    states->memoryStates->symmMemMgr->HeapRegisterSymmMemObj(ptr, size,
                                                             &states->memoryStates->staticHeapObj);
    MORI_SHMEM_TRACE("Static heap allocated {} bytes at offset {} (total used: {} / {})", size,
                     reinterpret_cast<uintptr_t>(ptr) - baseAddr,
                     states->memoryStates->staticHeapUsed, states->memoryStates->staticHeapSize);

    return ptr;
  }
}

void* ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  // For now, ignore flags and use the same allocator
  // TODO: Support different allocation flags if needed
  MORI_SHMEM_TRACE("Allocated shared memory of size {} with flags {} (flags ignored)", size, flags);
  return ShmemMalloc(size);
}

void ShmemFree(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (localPtr == nullptr) {
    return;
  }

  // Use different deallocation strategies based on mode
  if (states->mode == ShmemMode::StaticHeap) {
    // Static heap mode: just deregister from the pool (no actual free)
    // TODO: Implement a proper free list allocator for better memory reuse
    states->memoryStates->symmMemMgr->HeapDeregisterSymmMemObj(localPtr);
  } else {
    // Isolation mode: free the memory and deregister
    states->memoryStates->symmMemMgr->Free(localPtr);
  }
  // Check if using VMM heap
  if (states->memoryStates->useVMMHeap && states->memoryStates->vmmHeapInitialized) {
    // Use VMM-based deallocation (true free)
    states->memoryStates->symmMemMgr->VMMFreeChunk(localPtr);
    MORI_SHMEM_TRACE("VMM freed memory at {}", localPtr);
  } else {
    // For static heap, we can only deregister but not actually free
    // (bump allocator limitation)
    states->memoryStates->symmMemMgr->HeapDeregisterSymmMemObj(localPtr);
    MORI_SHMEM_TRACE("Static heap deregistered memory at {} (memory not reclaimed)", localPtr);
  }
}

application::SymmMemObjPtr ShmemQueryMemObjPtr(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (localPtr == nullptr) {
    return application::SymmMemObjPtr{nullptr, nullptr};
  }

  return states->memoryStates->symmMemMgr->Get(localPtr);
}

int ShmemBufferRegister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  states->memoryStates->mrMgr->RegisterBuffer(ptr, size);
  return 0;
}

int ShmemBufferDeregister(void* ptr, size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();
  states->memoryStates->mrMgr->DeregisterBuffer(ptr);
  return 0;
}

uint64_t ShmemPtrP2p(const uint64_t destPtr, const int myPe, int destPe) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  // If same PE, return the pointer directly
  if (myPe == destPe) {
    return destPtr;
  }

  if (destPe < 0 || destPe >= static_cast<int>(states->bootStates->worldSize)) {
    MORI_SHMEM_ERROR("Invalid destPe: {}", destPe);
    return 0;
  }

  application::TransportType transportType = states->rdmaStates->commContext->GetTransportType(destPe);
  if (transportType == application::TransportType::RDMA) {
    return 0;
  }

  uintptr_t localAddrInt = static_cast<uintptr_t>(destPtr);

  // Check if the pointer is within the symmetric heap
  uintptr_t heapBaseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
  uintptr_t heapEndAddr = heapBaseAddr + states->memoryStates->staticHeapSize;

  if (localAddrInt < heapBaseAddr || localAddrInt >= heapEndAddr) {
    MORI_SHMEM_ERROR("Pointer 0x{:x} is not in symmetric heap [0x{:x}, 0x{:x})", 
                     localAddrInt, heapBaseAddr, heapEndAddr);
    return 0;
  }

  // Calculate offset from heap base
  size_t offset = localAddrInt - heapBaseAddr;

  // Get the symmetric memory object for the heap
  application::SymmMemObjPtr heapObj = states->memoryStates->staticHeapObj;
  if (heapObj->Get() == nullptr) {
    MORI_SHMEM_ERROR("Failed to get heap symmetric memory object");
    return 0;
  }

  uint64_t peerBaseAddr = heapObj->peerPtrs[destPe];

  // Return the remote P2P address
  uint64_t remoteAddr = peerBaseAddr + offset;
  return remoteAddr;
}

}  // namespace shmem
}  // namespace mori
