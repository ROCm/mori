#pragma once

#include <hip/hip_runtime_api.h>
#include <linux/types.h>
#include <stdint.h>

#include <unordered_map>
#include <vector>

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/application/context/context.hpp"
#include "mori/application/transport/transport.hpp"

namespace mori {
namespace application {

struct SymmMemObj {
  void* localPtr{nullptr};
  uintptr_t* peerPtrs{nullptr};
  size_t size{0};
  // For Rdma
  uint32_t lkey{0};
  uint32_t* peerRkeys{nullptr};
  // For IPC
  hipIpcMemHandle_t* ipcMemHandles{nullptr};  // should only placed on cpu

  __device__ __host__ MemoryRegion GetMemoryRegion(int pe) const {
    MemoryRegion mr;
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
  SymmMemManager(BootstrapNetwork& bootNet, Context& context);
  ~SymmMemManager();

  SymmMemObjPtr HostMalloc(size_t size, size_t alignment = sysconf(_SC_PAGE_SIZE));
  void HostFree(void* localPtr);

  SymmMemObjPtr Malloc(size_t size);
  // See hipExtMallocWithFlags for flags settings
  SymmMemObjPtr ExtMallocWihFlags(size_t size, unsigned int flags);
  void Free(void* localPtr);

  SymmMemObjPtr RegisterSymmMemObj(void* localPtr, size_t size);
  void DeRegisterSymmMemObj(void* localPtr);

  SymmMemObjPtr Get(void* localPtr) const;

 private:
  BootstrapNetwork& bootNet;
  Context& context;
  std::unordered_map<void*, SymmMemObjPtr> memObjPool;
};

}  // namespace application
}  // namespace mori