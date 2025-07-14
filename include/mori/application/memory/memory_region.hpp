#pragma once

#include <stdint.h>

#include <unordered_map>

#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

class RdmaMemoryRegionManager {
 public:
  RdmaMemoryRegionManager(RdmaDeviceContext& context);
  ~RdmaMemoryRegionManager();

  RdmaMemoryRegion RegisterBuffer(void* ptr, size_t size);
  void DeRegisterBuffer(void* ptr);

  RdmaMemoryRegion Get(void* ptr) const;

 private:
  RdmaDeviceContext& context;
  std::unordered_map<void*, RdmaMemoryRegion> mrPool;
};

}  // namespace application
}  // namespace mori