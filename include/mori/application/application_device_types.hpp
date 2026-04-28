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
//
// Device-safe subset of mori::application types.
// Contains only POD structs, enums, and types with __device__ __host__ methods.
// No C++ STL (std::vector/map/etc.), no ibverbs, no host-only class definitions.
// Safe to include from both host and device (HIP/CUDA) compilation units.
//
// Host code should continue to include "mori/application/application.hpp" for the full API
// (SymmMemManager, RdmaDevice, Context, etc.).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mori/application/transport/sdma/anvil_device.hpp"
#include "mori/core/transport/rdma/core_device_types.hpp"
#include "mori/hip_compat.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                       Transport Types                                          */
/* ---------------------------------------------------------------------------------------------- */

enum TransportType { RDMA = 0, P2P = 1, SDMA = 2 };

/* ---------------------------------------------------------------------------------------------- */
/*                                      RDMA Types (device-safe)                                  */
/* ---------------------------------------------------------------------------------------------- */

enum class RdmaDeviceVendorId : uint32_t {
  Unknown = 0,
  Mellanox = 0x02c9,
  Broadcom = 0x14E4,
  Pensando = 0x1dd8,
};

struct RdmaMemoryRegion {
  uintptr_t addr{0};
  uint32_t lkey{0};
  uint32_t rkey{0};
  size_t length{0};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                    Symmetric Memory Types */
/* ---------------------------------------------------------------------------------------------- */

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
  uintptr_t* p2pPeerPtrs{nullptr};
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
  uint32_t sdmaNumQueue = 2;                                 // number of sdma queue
  HSAuint64* expectSignalsPtr = nullptr;                     // should only placed on GPU
  // Remote signal: peerSignalPtrs[pe] points to PE pe's signalPtrs mapped into local address space.
  // SdmaPutThread writes ATOMIC to peerSignalPtrs[remotePe] + myPe*sdmaNumQueue + qId,
  // so the remote PE can directly read its own signalPtrs to detect completion.
  HSAuint64** peerSignalPtrs = nullptr;  // should only placed on GPU

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
    return reinterpret_cast<void*>(p2pPeerPtrs[pe]);
  }

  template <typename T>
  inline __device__ __host__ T GetAs() const {
    return reinterpret_cast<T>(localPtr);
  }
  template <typename T>
  inline __device__ __host__ T GetAs(int pe) const {
    return reinterpret_cast<T>(p2pPeerPtrs[pe]);
  }
};

struct SymmMemObjPtr {
  SymmMemObj* cpu{nullptr};
  SymmMemObj* gpu{nullptr};

  bool IsValid() { return (cpu != nullptr) && (gpu != nullptr); }

#if defined(__HIPCC__) || defined(__CUDACC__)
  __host__ SymmMemObj* operator->() { return cpu; }
  __device__ SymmMemObj* operator->() { return gpu; }
  __host__ const SymmMemObj* operator->() const { return cpu; }
  __device__ const SymmMemObj* operator->() const { return gpu; }
#else
  SymmMemObj* operator->() { return cpu; }
  const SymmMemObj* operator->() const { return cpu; }
#endif
};

}  // namespace application
}  // namespace mori
