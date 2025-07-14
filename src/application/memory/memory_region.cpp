#include "mori/application/memory/memory_region.hpp"

namespace mori {
namespace application {

RdmaMemoryRegionManager::RdmaMemoryRegionManager(RdmaDeviceContext& context) : context(context) {}

RdmaMemoryRegionManager::~RdmaMemoryRegionManager() {}

application::RdmaMemoryRegion RdmaMemoryRegionManager::RegisterBuffer(void* ptr, size_t size) {
  application::RdmaMemoryRegion mr = context.RegisterRdmaMemoryRegion(ptr, size);
  mrPool.insert({ptr, mr});
  return mr;
}

void RdmaMemoryRegionManager::DeRegisterBuffer(void* ptr) {
  if (mrPool.find(ptr) == mrPool.end()) return;
  context.DeRegisterRdmaMemoryRegion(ptr);
  mrPool.erase(ptr);
}

application::RdmaMemoryRegion RdmaMemoryRegionManager::Get(void* ptr) const {
  if (mrPool.find(ptr) == mrPool.end()) return {};
  return mrPool.at(ptr);
}

}  // namespace application
}  // namespace mori