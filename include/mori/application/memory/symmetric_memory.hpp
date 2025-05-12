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

  __device__ MemoryRegion GetMemoryRegion(int pe) {
    MemoryRegion mr;
    mr.addr = peerPtrs[pe];
    mr.lkey = lkey;
    mr.rkey = peerRkeys[pe];
    mr.length = size;
    return mr;
  }
};

struct SymmMemObjPtr {
  SymmMemObj* cpu{nullptr};
  SymmMemObj* gpu{nullptr};

  bool IsValid() { return (cpu != nullptr) && (gpu != nullptr); }
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