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

// Device-safe: no STL/ibverbs, safe for HIP/CUDA device compilation.
#include <cassert>

#include "mori/application/application_device_types.hpp"
#include "mori/core/utils/utils.hpp"
#include "mori/hip_compat.hpp"
#include "mori/utils/limits.hpp"

// Host-only (STL/ibverbs); guarded out of device (.hip) compilation.
#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"
#endif

namespace mori {
namespace shmem {

// Host-only shmem state structures

#if !defined(__HIPCC__) && !defined(__CUDACC__)

enum class ShmemMode {
  Isolation,   // per-allocation SymmMemObj
  StaticHeap,  // single unified heap
  VMHeap       // virtual memory heap (unimplemented)
};

constexpr size_t DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE = 4ULL * 1024 * 1024 * 1024;
constexpr size_t DEFAULT_VMM_SYMMETRIC_HEAP_SIZE = 16ULL * 1024 * 1024 * 1024;
constexpr size_t DEFAULT_VMM_MIN_CHUNK_SIZE = 64ULL * 1024 * 1024;

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

  // StaticHeap-mode fields
  std::mutex heapLock;
  application::HeapType heapType{
      application::HeapType::Uncached};

  void* staticHeapBasePtr{nullptr};
  size_t staticHeapSize{0};
  size_t staticHeapUsed{0};
  application::SymmMemObjPtr staticHeapObj;

  bool useVMMHeap{false};
  bool vmmHeapInitialized{false};
  void* vmmHeapBaseAddr{nullptr};
  size_t vmmHeapVirtualSize{0};
  size_t vmmHeapChunkSize{0};
  application::SymmMemObjPtr vmmHeapObj;
};

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)

// Device-safe GPU-side structures

// Device POD projection of application::RdmaEndpoint (core's WQ/CQ/Ibuf handles).
using ShmemRdmaEndpoint = core::RdmaEndpointDevice;

// GpuStates must be declared before ModuleStates and ShmemStates which embed it.
struct GpuStates {
  int rank{-1};
  int worldSize{-1};
  int numQpPerPe{4};  // must match Context default
  application::TransportType* transportTypes{nullptr};
  ShmemRdmaEndpoint* rdmaEndpoints{nullptr};
  uint32_t* endpointLock{nullptr};

  bool useVMMHeap{false};
  uint8_t vmmChunkSizeShift{0};               // log2(chunkSize); 0 for static heap
  uintptr_t heapBaseAddr{0};
  uintptr_t heapEndAddr{0};                   // base + size
  application::SymmMemObj* heapObj{nullptr};
  uint64_t* internalSyncPtr{nullptr};
};

// __device__ (not __constant__) so hipMemcpyToSymbol can update it.
// Default visibility so JIT EP (MORI_DEFINE_GPU_STATES) matches this declaration.
extern __device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;

static __device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

// Address → remote address translation
struct RemoteAddrInfo {
  uintptr_t raddr;
  uintptr_t rkey;
  bool valid;

  __device__ RemoteAddrInfo() : raddr(0), rkey(0), valid(false) {}
  __device__ RemoteAddrInfo(uintptr_t r, uintptr_t k) : raddr(r), rkey(k), valid(true) {}
};

// Host-only internal functions

#if !defined(__HIPCC__) && !defined(__CUDACC__)

enum ShmemStatesStatus {
  New = 0,
  Initialized = 1,
  // Reserved: Finalize currently resets the slot to New so the GPU can be
  // re-initialized (SPMT multi-cycle test suites).
  Finalized = 2,
};

// Per-GPU JIT module state (HIP module handle + device symbol pointers).
struct ModuleStates {
  hipModule_t module{nullptr};
  GpuStates* gpuStatesPtr{nullptr};  // device globalGpuStates address in JIT module
  hipFunction_t barrierFunc{nullptr};
  // nullptr if the loaded module predates it => launcher falls back to funnel barrier.
  hipFunction_t dissemBarrierFunc{nullptr};
  hipFunction_t hierBarrierFunc{nullptr};
};

struct ShmemStates {
  ShmemStatesStatus status{ShmemStatesStatus::New};
  ShmemMode mode{ShmemMode::StaticHeap};
  BootStates* bootStates{nullptr};
  RdmaStates* rdmaStates{nullptr};
  MemoryStates* memoryStates{nullptr};
  ModuleStates moduleStates;
  GpuStates gpuStates;        // host-side copy of device GpuStates

  // Asserts the slot is initialized and not finalized (usable).
  void CheckStatusValid() {
    if (status == ShmemStatesStatus::New) {
      std::cout << "Shmem state is not initialized, call ShmemInit*/shmem_init_attr first."
                << std::endl;
      assert(false);
    }
    if (status == ShmemStatesStatus::Finalized) {
      std::cout << "Shmem state has been finalized." << std::endl;
      assert(false);
    }
  }
};

// Internal functions shared between init.cpp and runtime.cpp
void CopyGpuStatesToDevice(ShmemStates* states);
void FinalizeRuntime(ShmemStates* states);

class ShmemStatesSingleton {
 public:
  ShmemStatesSingleton(const ShmemStatesSingleton& obj) = delete;

  static ShmemStates* GetInstance();

#ifdef MORI_MULTITHREAD_SUPPORT
  // SPMT rank → HIP device id map (populated at ShmemInit): lets FFI/XLA worker
  // threads hipSetDevice to a rank's device before touching MORI state.
  // GetDeviceByRank returns -1 if unmapped (caller falls back to hipGetDevice).
  static void RegisterRankDevice(int rank, int deviceId);
  static int GetDeviceByRank(int rank);
#endif

 private:
#ifdef MORI_MULTITHREAD_SUPPORT
  // One slot per GPU, indexed by hipGetDevice. No lock: SPMT contract is one
  // thread per GPU, so each slot is accessed serially by its owner.
  std::array<ShmemStates, mori::kMaxGpusPerNode> states_{};
  ShmemStatesSingleton() = default;
#endif
};

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)

}  // namespace shmem
}  // namespace mori
