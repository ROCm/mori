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
#include "mori/application/topology/gpu.hpp"

#include <cstdio>
#include <cstring>
#include <string>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemGpu                                         */
/* ---------------------------------------------------------------------------------------------- */
TopoSystemGpu::TopoSystemGpu() { Load(); }

TopoSystemGpu::~TopoSystemGpu() {}

void TopoSystemGpu::Load() {
  // GPU topology is sourced purely from the HIP runtime (hipDeviceGetPCIBusId).
  //
  // We intentionally do NOT use rocm-smi (rsmi_*) here: its rsmi_init() races
  // across processes (multiple ranks contending on the per-device shared-memory
  // mutexes under /dev/shm/rocm_smi_*), which returned RSMI_STATUS_INIT_ERROR
  // and killed ranks. The only extra information rocm-smi provided over HIP was
  // the GPU<->GPU P2P link graph (type/hops/weight), which is currently unused
  // anywhere in the codebase. HIP gives us each GPU's PCI bus id, which is all
  // the NIC-matching logic (TopoSystem::CollectAndSortCandidates) needs.
  int hipDevCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&hipDevCount));
  for (int i = 0; i < hipDevCount; ++i) {
    char buf[16] = {};
    HIP_RUNTIME_CHECK(hipDeviceGetPCIBusId(buf, sizeof(buf), i));
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpu->busId = PciBusId(std::string(buf));
    gpus.emplace_back(gpu);
  }
}

std::vector<TopoNodeGpu*> TopoSystemGpu::GetGpus() const {
  std::vector<TopoNodeGpu*> v(gpus.size());
  for (int i = 0; i < gpus.size(); i++) v[i] = gpus[i].get();
  return v;
}

TopoNodeGpu* TopoSystemGpu::GetGpuByLogicalId(int id) const {
  char buf[16] = {};
  HIP_RUNTIME_CHECK(hipDeviceGetPCIBusId(buf, sizeof(buf), id));
  PciBusId target{std::string(buf)};
  for (auto& gpuPtr : gpus) {
    TopoNodeGpu* gpu = gpuPtr.get();
    if (gpu->busId == target) return gpu;
  }
  return nullptr;
}

}  // namespace application
}  // namespace mori
