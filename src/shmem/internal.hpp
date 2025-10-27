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
#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"

namespace mori {
namespace shmem {

// Configuration for static symmetric heap
constexpr size_t DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE = 2ULL * 1024 * 1024 * 1024;  // 2GB default
constexpr size_t DEFAULT_VMM_SYMMETRIC_HEAP_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8GB default

struct BootStates {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
};

using RdmaEndpointList = std::vector<application::RdmaEndpoint>;
using RdmaEndpointHandleList = std::vector<application::RdmaEndpointHandle>;

struct RdmaStates {
  application::Context* commContext{nullptr};
};

struct MemoryStates {
  application::SymmMemManager* symmMemMgr{nullptr};
  application::RdmaMemoryRegionManager* mrMgr{nullptr};

  std::mutex heapLock;                       // Lock for thread-safe allocation
  // Static heap
  void* staticHeapBasePtr{nullptr};          // Base address of the static symmetric heap
  size_t staticHeapSize{0};                  // Total size of the static heap
  size_t staticHeapUsed{0};                  // Currently used bytes
  application::SymmMemObjPtr staticHeapObj;  // SymmMemObj for the entire heap
  
  // VMM-based dynamic heap
  bool useVMMHeap{false};                    // Whether to use VMM-based heap
  bool vmmHeapInitialized{false};            // VMM heap initialization status
  void* vmmHeapBaseAddr{nullptr};              // Base address of VMM heap
  size_t vmmHeapVirtualSize{0};              // Total virtual address space size
  size_t vmmHeapChunkSize{0};                // Size of each chunk
  application::SymmMemObjPtr vmmHeapObj;  // SymmMemObj for the entire heap
};

enum ShmemStatesStatus {
  New = 0,
  Initialized = 1,
  Finalized = 2,
};

struct ShmemStates {
  ShmemStatesStatus status{ShmemStatesStatus::New};
  BootStates* bootStates{nullptr};
  RdmaStates* rdmaStates{nullptr};
  MemoryStates* memoryStates{nullptr};

  // This is a temporary API for debugging only
  void CheckStatusValid() {
    if (status == ShmemStatesStatus::New) {
      std::cout
          << "Shmem state is not initialized, initialize it by calling ShmemMpiInitialize first."
          << std::endl;
      assert(false);
    }
    if (status == ShmemStatesStatus::Finalized) {
      std::cout << "Shmem state has been finalized." << std::endl;
      assert(false);
    }
  }
};

struct GpuStates {
  int rank{-1};
  int worldSize{-1};
  int numQpPerPe{4};  // Default to 4 QPs per peer, consistent with Context default
  application::TransportType* transportTypes{nullptr};
  application::RdmaEndpoint* rdmaEndpoints{nullptr};
  uint32_t* endpointLock{nullptr};

  // Heap information (supports both static and VMM modes)
  bool useVMMHeap{false};                     // Whether using VMM-based heap
  uintptr_t heapBaseAddr{0};                  // Base address of symmetric heap
  uintptr_t heapEndAddr{0};                   // End address of symmetric heap (base + size)
  application::SymmMemObj* heapObj{nullptr};  // Pointer to the heap's SymmMemObj on device
};

extern __constant__ GpuStates globalGpuStates;

static __device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

/* ---------------------------------------------------------------------------------------------- */
/*                                Address to Remote Address Translation                           */
/* ---------------------------------------------------------------------------------------------- */
struct RemoteAddrInfo {
  uintptr_t raddr;  // Remote address
  uintptr_t rkey;   // Remote key for RDMA
  bool valid;

  __device__ RemoteAddrInfo() : raddr(0), rkey(0), valid(false) {}
  __device__ RemoteAddrInfo(uintptr_t r, uintptr_t k)
      : raddr(r), rkey(k), valid(true) {}
};

inline __device__ RemoteAddrInfo ShmemAddrToRemoteAddr(const void* localAddr, int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  uintptr_t localAddrInt = reinterpret_cast<uintptr_t>(localAddr);

  if (globalGpuStates->useVMMHeap) {
    // VMM mode: All PEs see the same virtual address
    // This greatly simplifies address translation
    uintptr_t raddr = localAddrInt;  // Same virtual address for all PEs
    
    // For RDMA, we still need the correct rkey
    // In VMM mode, each allocation has its own SymmMemObj with proper rkeys
    // For now, we'll use a simplified approach (rkey would be obtained from the allocation's SymmMemObj)
    uintptr_t rkey = 0;  // TODO: Implement proper rkey lookup for VMM allocations
    
    return RemoteAddrInfo(raddr, rkey);
  } else {
    // Static heap mode - traditional offset-based approach
    if (localAddrInt < globalGpuStates->heapBaseAddr ||
        localAddrInt >= globalGpuStates->heapEndAddr) {
      assert(false && "dest addr not in symmetric heap");
      return RemoteAddrInfo();
    }

    // Calculate offset within the symmetric heap
    size_t offset = localAddrInt - globalGpuStates->heapBaseAddr;

    // Get remote address using the heap's SymmMemObj
    application::SymmMemObj* heapObj = globalGpuStates->heapObj;
    uintptr_t raddr = heapObj->peerPtrs[pe] + offset;
    uintptr_t rkey = heapObj->peerRkeys[pe];

    return RemoteAddrInfo(raddr, rkey);
  }
}

class ShmemStatesSingleton {
 public:
  ShmemStatesSingleton(const ShmemStatesSingleton& obj) = delete;

  static ShmemStates* GetInstance() {
    static ShmemStates states;
    return &states;
  }
};

}  // namespace shmem
}  // namespace mori
