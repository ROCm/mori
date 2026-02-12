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

#include <hip/hip_runtime_api.h>
#include <linux/types.h>
#include <stdint.h>

#include <set>
#include <unordered_map>
#include <vector>

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/application/context/context.hpp"
#include "mori/application/memory/va_manager.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/transport/transport.hpp"

namespace mori {
namespace application {

// Heap type for memory allocation
enum class HeapType {
  Normal,   // Normal cached memory
  Uncached  // Uncached memory
};

struct VMMChunkKey {
  uint32_t key;         // RDMA lkey or rkey
  uintptr_t next_addr;  // Address of next chunk boundary (for calculating chunk_size)

  VMMChunkKey() : key(0), next_addr(0) {}
  VMMChunkKey(uint32_t k, uintptr_t addr) : key(k), next_addr(addr) {}
};

struct SymmMemObj {
  void* localPtr{nullptr};
  uintptr_t* peerPtrs{nullptr};
  size_t size{0};
  // For Rdma
  uint32_t lkey{0};
  uint32_t* peerRkeys{nullptr};

  // For VMM allocations: chunk key information (nvshmem-style)
  // vmmLkeyInfo[i] contains lkey and next_addr for chunk i
  // vmmRkeyInfo[i * worldSize + pe] contains rkey and next_addr for chunk i, PE pe
  VMMChunkKey* vmmLkeyInfo{nullptr};
  VMMChunkKey* vmmRkeyInfo{nullptr};
  size_t vmmNumChunks{0};  // Total number of chunks in VMM heap
  int worldSize{0};
  // For IPC
  hipIpcMemHandle_t* ipcMemHandles{nullptr};  // should only placed on cpu

  // For Sdma
  anvil::SdmaQueueDeviceHandle** deviceHandles_d = nullptr;  // should only placed on GPU
  HSAuint64* signalPtrs = nullptr;                           // should only placed on GPU
  uint32_t sdmaNumQueue = 8;                                 // number of sdma queue
  HSAuint64* expectSignalsPtr = nullptr;                     // should only placed on GPU

  __device__ __host__ RdmaMemoryRegion GetRdmaMemoryRegion(int pe) const {
    RdmaMemoryRegion mr;
    mr.addr = peerPtrs[pe];
    mr.lkey = lkey;
    mr.rkey = peerRkeys[pe];
    mr.length = size;
    return mr;
  }

  // Get pointers
  inline __device__ __host__ void* Get() const { return localPtr; }
  inline __device__ __host__ void* Get(int pe) const {
    return reinterpret_cast<void*>(peerPtrs[pe]);
  }

  template <typename T>
  inline __device__ __host__ T GetAs() const {
    return reinterpret_cast<T>(localPtr);
  }
  template <typename T>
  inline __device__ __host__ T GetAs(int pe) const {
    return reinterpret_cast<T>(peerPtrs[pe]);
  }
};

struct SymmMemObjPtr {
  SymmMemObj* cpu{nullptr};
  SymmMemObj* gpu{nullptr};

  bool IsValid() { return (cpu != nullptr) && (gpu != nullptr); }

  __host__ SymmMemObj* operator->() { return cpu; }
  __device__ SymmMemObj* operator->() { return gpu; }
  __host__ const SymmMemObj* operator->() const { return cpu; }
  __device__ const SymmMemObj* operator->() const { return gpu; }
};

class SymmMemManager {
 public:
  // Constructor and Destructor
  SymmMemManager(BootstrapNetwork& bootNet, Context& context);
  ~SymmMemManager();

  // Basic Memory Allocation (Isolation Mode)
  SymmMemObjPtr HostMalloc(size_t size, size_t alignment = sysconf(_SC_PAGE_SIZE));
  void HostFree(void* localPtr);
  SymmMemObjPtr Malloc(size_t size);
  SymmMemObjPtr ExtMallocWithFlags(size_t size, unsigned int flags);
  void Free(void* localPtr);
  SymmMemObjPtr RegisterSymmMemObj(void* localPtr, size_t size, bool heap_begin = false);
  void DeregisterSymmMemObj(void* localPtr);

  // Static Heap Operations
  SymmMemObjPtr RegisterStaticHeapSubRegion(void* localPtr, size_t size, SymmMemObjPtr* heapObj);
  void DeregisterStaticHeapSubRegion(void* localPtr);

  // VMM Heap Operations
  bool InitializeVMMHeap(size_t virtualSize, size_t chunkSize = 0,
                         HeapType heapType = HeapType::Uncached);
  void FinalizeVMMHeap();
  bool IsVMMSupported() const;
  SymmMemObjPtr VMMAllocChunk(size_t size, HeapType heapType = HeapType::Uncached);
  void VMMFreeChunk(void* localPtr);
  SymmMemObjPtr VMMRegisterSymmMemObj(void* localPtr, size_t size, size_t startChunk,
                                      size_t numChunks);
  void VMMDeregisterSymmMemObj(void* localPtr);
  SymmMemObjPtr GetVMMHeapObj() const { return vmmHeapObj; }
  size_t GetVMMChunkSize() const { return vmmChunkSize; }

  // Common Utilities
  SymmMemObjPtr Get(void* localPtr) const;
  HeapVAManager* GetHeapVAManager() const { return heapVAManager.get(); }
  void InitHeapVAManager(uintptr_t baseAddr, size_t size, size_t granularity = 0) {
    heapVAManager = std::make_unique<HeapVAManager>(baseAddr, size, granularity);
  }

 private:
  // Member Variables
  BootstrapNetwork& bootNet;
  Context& context;
  std::unordered_map<void*, SymmMemObjPtr> memObjPool;

  // VMM Heap State
  struct VMMChunkInfo {
    hipMemGenericAllocationHandle_t handle;
    int shareableHandle;  // File descriptor for POSIX systems (shared by P2P and RDMA/dmabuf)
    size_t size;          // Chunk size (always equals granularity/vmmChunkSize)
    bool isAllocated;
    int refCount;  // Reference count: how many allocations are using this chunk

    // RDMA registration info (per-chunk, for RDMA transport)
    uint32_t lkey;                    // Local key for RDMA access
    std::vector<uint32_t> peerRkeys;  // Remote keys from all PEs
    bool rdmaRegistered;              // Whether this chunk is RDMA registered

    // P2P mapping tracking: which peers have already mapped this chunk
    std::set<int> mappedPeers;  // Set of peer ranks that have imported and mapped this chunk

    // P2P imported handles tracking: handles imported from other PEs need to be released
    std::map<int, hipMemGenericAllocationHandle_t> importedHandles;  // pe -> imported handle

    VMMChunkInfo()
        : handle(0),
          shareableHandle(-1),
          size(0),
          isAllocated(false),
          refCount(0),
          lkey(0),
          rdmaRegistered(false) {}
  };

  bool vmmInitialized{false};
  void* vmmVirtualBasePtr{nullptr};
  size_t vmmVirtualSize{0};
  size_t vmmChunkSize{0};
  size_t vmmMinChunkSize{0};
  size_t vmmMaxChunks{0};
  std::vector<VMMChunkInfo> vmmChunks;
  std::mutex vmmLock;
  bool vmmRdmaRegistered{false};
  SymmMemObjPtr vmmHeapObj{nullptr, nullptr};
  std::vector<void*> vmmPeerBasePtrs;
  size_t vmmPerPeerSize{0};

  // VA Manager (used by both VMM and Static heap)
  std::unique_ptr<HeapVAManager> heapVAManager;

  // Helper Functions - InitializeVMMHeap
  size_t DetermineVMMChunkSize(size_t userChunkSize, HeapType heapType);
  size_t CalculateTotalVirtualSize(size_t perPeerSize, int worldSize);
  bool ReserveVirtualAddressSpace(size_t totalSize, size_t chunkSize);
  void SetupPeerBasePointers(int worldSize, int myPe);
  void InitializeChunkTracking();
  SymmMemObjPtr CreateVMMHeapObject(size_t virtualSize, int worldSize, int myPe);

  // Helper Functions - VMMAllocChunk
  uintptr_t AllocateVirtualAddress(size_t size);
  bool VerifyVAConsistency(uintptr_t allocAddr, size_t size, size_t offset, int rank,
                           int worldSize);
  void CleanupAllocatedChunks(size_t startChunk, size_t numToClean);
  hipMemAllocationProp ConfigureAllocationProp(HeapType heapType, int deviceId);
  bool AllocateSingleChunk(VMMChunkInfo& chunk, size_t chunkIdx, void* chunkPtr, size_t chunkSize,
                           const hipMemAllocationProp& allocProp, int rank, int currentDev);
  void ImportPeerChunk(VMMChunkInfo& chunk, size_t chunkIdx, void* peerChunkPtr, size_t chunkSize,
                       int handleValue, int pe, int rank, int currentDev);
  bool RegisterP2PPeerMemory(const std::vector<int>& localShareableHandles, size_t startChunk,
                             size_t chunksNeeded, int rank, int worldSize, int currentDev);
  void RegisterRdmaChunks(size_t startChunk, size_t chunksNeeded, int rank, int worldSize);
};

}  // namespace application

namespace shmem {
// Note: temporary API for testing
application::SymmMemObjPtr ShmemQueryMemObjPtr(void*);
}  // namespace shmem

}  // namespace mori
