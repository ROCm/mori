#pragma once

#include <stdint.h>

#include <unordered_map>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/core/core.hpp"

namespace mori {
namespace application {

class MemoryRegionManager {
 public:
  MemoryRegionManager(RdmaDeviceContext& context);
  ~MemoryRegionManager();

  MemoryRegion RegisterBuffer(void* ptr, size_t size);
  void DeRegisterBuffer(void* ptr);

  MemoryRegion Get(void* ptr) const;

 private:
  RdmaDeviceContext& context;
  std::unordered_map<void*, MemoryRegion> mrPool;
};

}  // namespace application
}  // namespace mori