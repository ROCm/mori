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

void RdmaMemoryRegionManager::DeregisterBuffer(void* ptr) {
  if (mrPool.find(ptr) == mrPool.end()) return;
  context.DeregisterRdmaMemoryRegion(ptr);
  mrPool.erase(ptr);
}

application::RdmaMemoryRegion RdmaMemoryRegionManager::Get(void* ptr) const {
  if (mrPool.find(ptr) == mrPool.end()) return {};
  return mrPool.at(ptr);
}

}  // namespace application
}  // namespace mori
