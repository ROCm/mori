#pragma once

#include <linux/types.h>
#include <stdint.h>

#include <unordered_map>
#include <vector>

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

struct SymmMemObj {
  void* localPtr{nullptr};
  uintptr_t* peerPtrs{nullptr};
  size_t size{0};
  uint32_t lkey{0};
  uint32_t* peerRkeys{nullptr};

  __device__ __host__ MemoryRegion GetMemoryRegion(int pe) const {
    MemoryRegion mr;
    mr.addr = peerPtrs[pe];
    mr.lkey = lkey;
    mr.rkey = peerRkeys[pe];
    mr.length = size;
    return mr;
  }

  // Get pointers
  __device__ __host__ void* Get() const { return localPtr; }
  __device__ __host__ void* Get(int pe) const { return reinterpret_cast<void*>(peerPtrs[pe]); }

  template <typename T>
  __device__ __host__ T GetAs() const {
    return reinterpret_cast<T>(localPtr);
  }
  template <typename T>
  __device__ __host__ T GetAs(int pe) const {
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
  SymmMemManager(BootstrapNetwork& bootNet, RdmaDeviceContext& context);
  ~SymmMemManager();

  SymmMemObjPtr HostMalloc(size_t size, size_t alignment = sysconf(_SC_PAGE_SIZE));
  void HostFree(void* localPtr);

  SymmMemObjPtr Malloc(size_t size);
  void Free(void* localPtr);

  SymmMemObjPtr RegisterSymmMemObj(void* localPtr, size_t size);
  void DeRegisterSymmMemObj(void* localPtr);

  SymmMemObjPtr Get(void* localPtr) const;

 private:
  BootstrapNetwork& bootNet;
  RdmaDeviceContext& context;
  std::unordered_map<void*, SymmMemObjPtr> memObjPool;
};

}  // namespace application
}  // namespace mori