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

// Shmem operation mode
enum class ShmemMode {
  Isolation,   // Original mode: each allocation gets its own SymmMemObj
  StaticHeap,   // single static heap with unified memory space
  VMHeap   // TODO: implement virtual memory heap
};

// Configuration for static symmetric heap
constexpr size_t DEFAULT_SYMMETRIC_HEAP_SIZE = 2ULL * 1024 * 1024 * 1024;  // 2GB default

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

  // Static heap mode fields (only used when mode == StaticHeap)
  void* staticHeapBasePtr{nullptr};          // Base address of the static symmetric heap
  size_t staticHeapSize{0};                  // Total size of the static heap
  size_t staticHeapUsed{0};                  // Currently used bytes
  application::SymmMemObjPtr staticHeapObj;  // SymmMemObj for the entire heap
  std::mutex heapLock;                       // Lock for thread-safe allocation
};

enum ShmemStatesStatus {
  New = 0,
  Initialized = 1,
  Finalized = 2,
};

struct ShmemStates {
  ShmemStatesStatus status{ShmemStatesStatus::New};
  ShmemMode mode{ShmemMode::StaticHeap};  // Default to static heap mode
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

  uintptr_t heapBaseAddr{0};                  // Base address of symmetric heap
  uintptr_t heapEndAddr{0};                   // End address of symmetric heap (base + size)
  application::SymmMemObj* heapObj{nullptr};  // Pointer to the heap's SymmMemObj on device
};

// Changed from __constant__ to __device__ to allow hipMemcpyToSymbol updates (like rocshmem)
extern __device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;

static __device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

/* ---------------------------------------------------------------------------------------------- */
/*                                Address to Remote Address Translation                           */
/* ---------------------------------------------------------------------------------------------- */
struct RemoteAddrInfo {
  uintptr_t raddr;  // Remote address
  uintptr_t rkey;   // Remote key for RDMA
  bool valid;

  __device__ RemoteAddrInfo() : raddr(0), rkey(0), valid(false) {}
  __device__ RemoteAddrInfo(uintptr_t r, uintptr_t k) : raddr(r), rkey(k), valid(true) {}
};

inline __device__ RemoteAddrInfo ShmemAddrToRemoteAddr(const void* localAddr, int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  uintptr_t localAddrInt = reinterpret_cast<uintptr_t>(localAddr);

  if (globalGpuStates->heapObj == nullptr) {
    assert(false);
    return RemoteAddrInfo();
  }

  // Check if address is in symmetric heap
  if (localAddrInt < globalGpuStates->heapBaseAddr ||
      localAddrInt >= globalGpuStates->heapEndAddr) {
    assert(false);
    return RemoteAddrInfo();
  }

  // Calculate offset within the symmetric heap
  size_t offset = reinterpret_cast<const char*>(localAddr) -
                  reinterpret_cast<const char*>(globalGpuStates->heapBaseAddr);

  // Get remote address using the heap's SymmMemObj
  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  uintptr_t raddr = heapObj->peerPtrs[pe] + offset;
  uintptr_t rkey = heapObj->peerRkeys[pe];

  return RemoteAddrInfo(raddr, rkey);
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
