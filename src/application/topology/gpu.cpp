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
#include <cstdlib>
#include <mutex>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {
namespace detail {

void ShutdownRsmiAtExit() {
  rsmi_status_t status = rsmi_shut_down();
  if (status != RSMI_STATUS_SUCCESS) {
    const char* msg = nullptr;
    rsmi_status_string(status, &msg);
    if (msg == nullptr) msg = "unknown rocm smi error";
    fprintf(stderr, "[%s:%d] warning: rocm smi shutdown failed at exit with %s\n", __FILE__,
            __LINE__, msg);
  }
}

void EnsureRsmiInitialized() {
  static std::once_flag initOnce;
  std::call_once(initOnce, []() {
    ROCM_SMI_CHECK(rsmi_init(0));
    std::atexit(ShutdownRsmiAtExit);
  });
}

}  // namespace detail

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemGpu                                         */
/* ---------------------------------------------------------------------------------------------- */
TopoSystemGpu::TopoSystemGpu() { Load(); }

TopoSystemGpu::~TopoSystemGpu() {}

PciBusId RsmiBusId2PciBusId(uint64_t rsmiBusId) {
  uint16_t domain = (rsmiBusId >> 32);
  uint8_t bus = (rsmiBusId >> 8);
  uint8_t dev = (rsmiBusId >> 3) & 0x1f;
  uint8_t func = rsmiBusId & 0x7;
  return PciBusId(domain, bus, dev, func);
}

void TopoSystemGpu::Load() {
  uint32_t numGpus;

  detail::EnsureRsmiInitialized();
  ROCM_SMI_CHECK(rsmi_num_monitor_devices(&numGpus));

  for (uint32_t i = 0; i < numGpus; ++i) {
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpus.emplace_back(gpu);
    uint64_t rsmiBusId = 0;
    ROCM_SMI_CHECK(rsmi_dev_pci_id_get(i, &rsmiBusId));
    gpu->busId = RsmiBusId2PciBusId(rsmiBusId);
    // ROCM_SMI_CHECK(rsmi_topo_numa_affinity_get(reinterpret_cast<uint32_t>(i), &gpu->numaNode));
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    for (uint32_t j = i; j < numGpus; ++j) {
      if (i == j) continue;
      bool accessible = false;
      ROCM_SMI_CHECK(rsmi_is_P2P_accessible(i, j, &accessible));
      if (!accessible) continue;

      TopoNodeGpuP2pLink* p2p = new TopoNodeGpuP2pLink();
      ROCM_SMI_CHECK(rsmi_topo_get_link_type(i, j, &p2p->hops, &p2p->type));
      ROCM_SMI_CHECK(rsmi_topo_get_link_weight(i, j, &p2p->weight));
      p2p->gpu1 = gpus[i].get();
      p2p->gpu2 = gpus[j].get();
      p2ps.emplace_back(p2p);

      gpus[i]->p2ps.push_back(p2p);
      gpus[j]->p2ps.push_back(p2p);
    }
  }

  // RSMI owns process-global state. Keep it initialized after topology discovery instead of
  // shutting it down per TopoSystemGpu instance; shutdown is registered once at process exit.
}

std::vector<TopoNodeGpu*> TopoSystemGpu::GetGpus() const {
  std::vector<TopoNodeGpu*> v(gpus.size());
  for (int i = 0; i < gpus.size(); i++) v[i] = gpus[i].get();
  return v;
}

TopoNodeGpu* TopoSystemGpu::GetGpuByLogicalId(int id) const {
  std::string str;
  str.resize(13);
  HIP_RUNTIME_CHECK(hipDeviceGetPCIBusId(str.data(), str.size(), id));
  PciBusId target{str};
  for (auto& gpuPtr : gpus) {
    TopoNodeGpu* gpu = gpuPtr.get();
    if (gpu->busId == target) return gpu;
  }
  return nullptr;
}

}  // namespace application
}  // namespace mori
