#include "mori/application/memory/memory_region.hpp"

namespace mori {
namespace application {

MemoryRegionManager::MemoryRegionManager(RdmaDeviceContext& context) : context(context) {}

MemoryRegionManager::~MemoryRegionManager() {}

application::MemoryRegion MemoryRegionManager::RegisterBuffer(void* ptr, size_t size) {
  application::MemoryRegion mr = context.RegisterMemoryRegion(ptr, size);
  mrPool.insert({ptr, mr});
  return mr;
}

void MemoryRegionManager::DeRegisterBuffer(void* ptr) {
  if (mrPool.find(ptr) == mrPool.end()) return;
  context.DeRegisterMemoryRegion(ptr);
  mrPool.erase(ptr);
}

application::MemoryRegion MemoryRegionManager::Get(void* ptr) const {
  if (mrPool.find(ptr) == mrPool.end()) return {};
  return mrPool.at(ptr);
}

}  // namespace application
}  // namespace mori