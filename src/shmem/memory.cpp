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
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                    Memory Allocation Helpers                                  */
/* ---------------------------------------------------------------------------------------------- */

// Allocate memory in static heap mode (with VA manager for reuse)
static void* AllocateStaticHeap(ShmemStates* states, size_t size) {
  // Align to 256 bytes for better performance
  constexpr size_t ALIGNMENT = 256;

  // Use VA manager to allocate address (thread-safe, handles reuse)
  uintptr_t allocAddr =
      states->memoryStates->symmMemMgr->GetHeapVAManager()->Allocate(size, ALIGNMENT);

  if (allocAddr == 0) {
    MORI_SHMEM_ERROR(
        "Out of static heap memory! Requested: {} bytes. Hint: Increase via MORI_SHMEM_HEAP_SIZE "
        "(default: 2GB)",
        size);
    return nullptr;
  }

  void* ptr = reinterpret_cast<void*>(allocAddr);

  // Register the allocated region as a sub-region of the static heap
  states->memoryStates->symmMemMgr->RegisterStaticHeapSubRegion(
      ptr, size, &states->memoryStates->staticHeapObj);

  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
  MORI_SHMEM_TRACE("Allocated {} bytes at ptr={:#x} (offset={}, aligned to 256={})", size,
                   allocAddr, allocAddr - baseAddr, (allocAddr % 256 == 0 ? "yes" : "no"));

  return ptr;
}

// Allocate memory in VMM heap mode
static void* AllocateVMMHeap(ShmemStates* states, size_t size) {
  application::SymmMemObjPtr obj =
      states->memoryStates->symmMemMgr->VMMAllocChunk(size, states->memoryStates->heapType);

  if (obj.IsValid()) {
    MORI_SHMEM_TRACE("Allocated {} bytes in VMM heap mode", size);
    return obj.cpu->localPtr;
  }

  MORI_SHMEM_ERROR(
      "Failed to allocate {} bytes in VMM heap. Hint: Increase via MORI_SHMEM_HEAP_SIZE (default: "
      "8GB) or MORI_SHMEM_VMM_CHUNK_SIZE (default: 64MB)",
      size);
  return nullptr;
}

// Allocate memory in isolation mode
static void* AllocateIsolation(ShmemStates* states, size_t size) {
  application::SymmMemObjPtr obj = states->memoryStates->symmMemMgr->Malloc(size);

  if (obj.IsValid()) {
    MORI_SHMEM_TRACE("Allocated {} bytes in isolation mode", size);
    return obj.cpu->localPtr;
  }

  MORI_SHMEM_ERROR("Failed to allocate {} bytes in isolation mode", size);
  return nullptr;
}

// Allocate memory with flags in isolation mode
static void* AllocateIsolationWithFlags(ShmemStates* states, size_t size, unsigned int flags) {
  application::SymmMemObjPtr obj =
      states->memoryStates->symmMemMgr->ExtMallocWithFlags(size, flags);

  if (obj.IsValid()) {
    MORI_SHMEM_TRACE("Allocated {} bytes with flags {} in isolation mode", size, flags);
    return obj.cpu->localPtr;
  }

  MORI_SHMEM_ERROR("Failed to allocate {} bytes with flags {} in isolation mode", size, flags);
  return nullptr;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Memory Deallocation Helpers                                */
/* ---------------------------------------------------------------------------------------------- */

// Free memory in static heap mode
static void FreeStaticHeap(ShmemStates* states, void* localPtr) {
  // Deregister static heap sub-region from SymmMemObj pool
  states->memoryStates->symmMemMgr->DeregisterStaticHeapSubRegion(localPtr);

  // Free the VA address in VA manager (enables reuse)
  uintptr_t addr = reinterpret_cast<uintptr_t>(localPtr);
  if (states->memoryStates->symmMemMgr->GetHeapVAManager()->Free(addr)) {
    MORI_SHMEM_TRACE("Static heap freed memory at {} (VA reclaimed for reuse)", localPtr);
  } else {
    MORI_SHMEM_ERROR("Failed to free VA address {} in static heap", localPtr);
  }
}

// Free memory in VMM heap mode
static void FreeVMMHeap(ShmemStates* states, void* localPtr) {
  states->memoryStates->symmMemMgr->VMMFreeChunk(localPtr);
  MORI_SHMEM_TRACE("VMM heap freed memory at {}", localPtr);
}

// Free memory in isolation mode
static void FreeIsolation(ShmemStates* states, void* localPtr) {
  states->memoryStates->symmMemMgr->Free(localPtr);
  MORI_SHMEM_TRACE("Isolation mode freed memory at {}", localPtr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Public Memory Allocation APIs                              */
/* ---------------------------------------------------------------------------------------------- */

void* ShmemMalloc(size_t size) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (size == 0) {
    return nullptr;
  }

  // Dispatch to appropriate allocator based on mode
  switch (states->mode) {
    case ShmemMode::StaticHeap:
      return AllocateStaticHeap(states, size);

    case ShmemMode::VMHeap:
      return AllocateVMMHeap(states, size);

    case ShmemMode::Isolation:
      return AllocateIsolation(states, size);

    default:
      MORI_SHMEM_ERROR("Unknown ShmemMode: {}", static_cast<int>(states->mode));
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

  // Dispatch to appropriate allocator based on mode
  switch (states->mode) {
    case ShmemMode::StaticHeap:
      // Flags are ignored in static heap mode
      return AllocateStaticHeap(states, size);

    case ShmemMode::VMHeap:
      // Flags are ignored in VMM heap mode
      return AllocateVMMHeap(states, size);

    case ShmemMode::Isolation:
      // Isolation mode: flags are respected
      return AllocateIsolationWithFlags(states, size, flags);

    default:
      MORI_SHMEM_ERROR("Unknown ShmemMode: {}", static_cast<int>(states->mode));
      return nullptr;
  }
}

void ShmemFree(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (localPtr == nullptr) {
    return;
  }

  // Dispatch to appropriate deallocator based on mode
  switch (states->mode) {
    case ShmemMode::StaticHeap:
      FreeStaticHeap(states, localPtr);
      break;

    case ShmemMode::VMHeap:
      FreeVMMHeap(states, localPtr);
      break;

    case ShmemMode::Isolation:
      FreeIsolation(states, localPtr);
      break;

    default:
      MORI_SHMEM_ERROR("Unknown ShmemMode: {}", static_cast<int>(states->mode));
      break;
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Memory Query and Management APIs                           */
/* ---------------------------------------------------------------------------------------------- */

application::SymmMemObjPtr ShmemQueryMemObjPtr(void* localPtr) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  if (localPtr == nullptr) {
    return application::SymmMemObjPtr{nullptr, nullptr};
  }

  return states->memoryStates->symmMemMgr->Get(localPtr);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    Buffer Registration APIs                                   */
/* ---------------------------------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------------------------------- */
/*                                    P2P Pointer Conversion                                     */
/* ---------------------------------------------------------------------------------------------- */

uint64_t ShmemPtrP2p(const uint64_t destPtr, const int myPe, int destPe) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  // Same PE: return pointer directly
  if (myPe == destPe) {
    return destPtr;
  }

  // Validate destination PE
  if (destPe < 0 || destPe >= static_cast<int>(states->bootStates->worldSize)) {
    MORI_SHMEM_ERROR("Invalid destPe: {}", destPe);
    return 0;
  }

  // Check transport type (only P2P transport needs address translation)
  application::TransportType transportType =
      states->rdmaStates->commContext->GetTransportType(destPe);
  if (transportType == application::TransportType::RDMA) {
    return 0;  // RDMA doesn't use P2P pointers
  }

  // Validate pointer is within symmetric heap bounds
  uintptr_t localAddrInt = static_cast<uintptr_t>(destPtr);
  uintptr_t heapBaseAddr = reinterpret_cast<uintptr_t>(states->memoryStates->staticHeapBasePtr);
  uintptr_t heapEndAddr = heapBaseAddr + states->memoryStates->staticHeapSize;

  if (localAddrInt < heapBaseAddr || localAddrInt >= heapEndAddr) {
    MORI_SHMEM_ERROR("Pointer 0x{:x} is not in symmetric heap [0x{:x}, 0x{:x})", localAddrInt,
                     heapBaseAddr, heapEndAddr);
    return 0;
  }

  // Calculate offset and get peer address
  size_t offset = localAddrInt - heapBaseAddr;

  application::SymmMemObjPtr heapObj = states->memoryStates->staticHeapObj;
  if (heapObj->Get() == nullptr) {
    MORI_SHMEM_ERROR("Failed to get heap symmetric memory object");
    return 0;
  }

  uint64_t peerBaseAddr = heapObj->peerPtrs[destPe];
  uint64_t remoteAddr = peerBaseAddr + offset;

  MORI_SHMEM_TRACE("P2P pointer conversion: local=0x{:x} -> remote=0x{:x} (PE {} -> {})", destPtr,
                   remoteAddr, myPe, destPe);

  return remoteAddr;
}

}  // namespace shmem
}  // namespace mori
