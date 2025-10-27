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
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {


void* ShmemMalloc(size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (size == 0) {
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

}  // namespace shmem
}  // namespace mori
