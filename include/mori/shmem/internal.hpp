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

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/hip_compat.hpp"

namespace mori {
namespace shmem {

// Shmem operation mode
enum class ShmemMode {
  Isolation,   // Original mode: each allocation gets its own SymmMemObj
  StaticHeap,  // single static heap with unified memory space
  VMHeap       // TODO: implement virtual memory heap
};

constexpr size_t DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE = 4ULL * 1024 * 1024 * 1024;  // 4GB default
constexpr size_t DEFAULT_VMM_SYMMETRIC_HEAP_SIZE = 16ULL * 1024 * 1024 * 1024;    // 16GB default
constexpr size_t DEFAULT_VMM_MIN_CHUNK_SIZE = 64ULL * 1024 * 1024;                // 64MB default

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
  std::mutex heapLock;  // Lock for thread-safe allocation
  application::HeapType heapType{
      application::HeapType::Uncached};  // Type of heap memory (default: uncached)

  // Static heap
  void* staticHeapBasePtr{nullptr};          // Base address of the static symmetric heap
  size_t staticHeapSize{0};                  // Total size of the static heap
  size_t staticHeapUsed{0};                  // Currently used bytes
  application::SymmMemObjPtr staticHeapObj;  // SymmMemObj for the entire heap

  // VMM-based dynamic heap
  bool useVMMHeap{false};                 // Whether to use VMM-based heap
  bool vmmHeapInitialized{false};         // VMM heap initialization status
  void* vmmHeapBaseAddr{nullptr};         // Base address of VMM heap
  size_t vmmHeapVirtualSize{0};           // Total virtual address space size
  size_t vmmHeapChunkSize{0};             // Size of each chunk
  application::SymmMemObjPtr vmmHeapObj;  // SymmMemObj for the entire heap
};

// GpuStates must be declared before ModuleStates and ShmemStates which embed it.
struct GpuStates {
  int rank{-1};
  int worldSize{-1};
  int numQpPerPe{4};  // Default to 4 QPs per peer, consistent with Context default
  application::TransportType* transportTypes{nullptr};
  application::RdmaEndpoint* rdmaEndpoints{nullptr};
  uint32_t* endpointLock{nullptr};

  // Heap information (supports both static and VMM modes)
  bool useVMMHeap{false};                     // Whether using VMM-based heap
  uint8_t vmmChunkSizeShift{0};               // log2(chunkSize) for VMM heap, 0 for static heap
  uintptr_t heapBaseAddr{0};                  // Base address of symmetric heap
  uintptr_t heapEndAddr{0};                   // End address of symmetric heap (base + size)
  application::SymmMemObj* heapObj{nullptr};  // Pointer to the heap's SymmMemObj on device
  uint64_t* internalSyncPtr{nullptr};         // Pointer to the internal synchronization object
};

enum ShmemStatesStatus {
  New = 0,
  Initialized = 1,
  Finalized = 2,
};

// Per-GPU JIT module state (HIP module handle + device symbol pointers)
struct ModuleStates {
  hipModule_t module{nullptr};
  GpuStates* gpuStatesPtr{nullptr};  // device-side globalGpuStates address in JIT module
  hipFunction_t barrierFunc{nullptr};
};

struct ShmemStates {
  ShmemStatesStatus status{ShmemStatesStatus::New};
  ShmemMode mode{ShmemMode::StaticHeap};  // Default to static heap mode
  BootStates* bootStates{nullptr};
  RdmaStates* rdmaStates{nullptr};
  MemoryStates* memoryStates{nullptr};
  ModuleStates moduleStates;  // JIT module state for this GPU
  GpuStates gpuStates;        // host-side copy of device GpuStates for this GPU

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

// Internal functions shared between init.cpp and runtime.cpp
void CopyGpuStatesToDevice(ShmemStates* states);
void FinalizeRuntime(ShmemStates* states);

// Max GPUs per node (fixed array avoids deque resize/realloc issues)
static constexpr int kMaxGpusPerNode = 8;

class ShmemStatesSingleton {
 public:
  ShmemStatesSingleton(const ShmemStatesSingleton& obj) = delete;

  static ShmemStates* GetInstance();

 private:
#ifdef MORI_MULTITHREAD_SUPPORT
  // One ShmemStates slot per GPU, indexed by hipGetDevice().
  // std::array gives stable addresses (no realloc unlike deque/vector).
  std::array<ShmemStates, kMaxGpusPerNode> states_{};
  std::mutex mutex_;
  ShmemStatesSingleton() = default;
#endif
};

}  // namespace shmem
}  // namespace mori
