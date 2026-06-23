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

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

// rocm-smi is loaded via a private dlopen(RTLD_LOCAL) instead of being linked.
//
// On the ROCm 7.14 docker we observed that the rsmi_* symbols resolve to two
// different shared objects in the same process: some calls (e.g. rsmi_init,
// rsmi_is_P2P_accessible) bind to libamd_smi.so (amdsmi), while the rest bind to
// librocm_smi64. The two libraries keep separate singletons, so init lands on
// one and num_monitor_devices on the other -> RSMI_STATUS_INIT_ERROR. This does
// not happen on ROCm 7.2, so we pin the so ourselves: load librocm_smi64 with
// RTLD_LOCAL and resolve every rsmi_* symbol from that one handle. Keeping it out
// of the global scope means all calls hit the same instance.
//
// Doing it lazily (after torch has initialized HIP) also lets its NEEDED
// libamdhip64 resolve to the already-loaded copy instead of pulling in a second
// HIP runtime, which an LD_PRELOAD of librocm_smi64 would do (causing
// hipErrorInvalidImage).
namespace {

void* OpenRocmSmi() {
  const char* candidates[] = {std::getenv("MORI_ROCM_SMI_PATH"), "librocm_smi64.so.1",
                              "librocm_smi64.so", "/opt/rocm/lib/librocm_smi64.so.1"};
  for (const char* path : candidates) {
    if (path && path[0]) {
      if (void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL)) return h;
    }
  }
  fprintf(stderr, "[ROCm-SMI] dlopen(librocm_smi64) failed: %s\n", dlerror());
  exit(-1);
}

template <typename Fn>
Fn Sym(void* handle, const char* name) {
  void* sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "[ROCm-SMI] missing symbol %s\n", name);
    exit(-1);
  }
  return reinterpret_cast<Fn>(sym);
}

// Declare a local function pointer `name` resolved from `lib`, reusing the exact
// signature declared in rocm_smi.h so callers below read like normal rsmi calls.
#define RSMI_FN(lib, name) auto name = Sym<decltype(&::name)>(lib, #name)

}  // namespace

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
  void* lib = OpenRocmSmi();
  RSMI_FN(lib, rsmi_init);
  RSMI_FN(lib, rsmi_num_monitor_devices);
  RSMI_FN(lib, rsmi_dev_pci_id_get);
  RSMI_FN(lib, rsmi_is_P2P_accessible);
  RSMI_FN(lib, rsmi_topo_get_link_type);
  RSMI_FN(lib, rsmi_topo_get_link_weight);
  RSMI_FN(lib, rsmi_shut_down);
  RSMI_FN(lib, rsmi_status_string);

  uint32_t numGpus = 0;
  ROCM_SMI_CHECK(rsmi_init(0));
  ROCM_SMI_CHECK(rsmi_num_monitor_devices(&numGpus));

  if (numGpus == 0) {
    fprintf(stderr, "[ROCm-SMI] rsmi_num_monitor_devices reported 0 GPUs\n");
    exit(-1);
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpus.emplace_back(gpu);
    uint64_t rsmiBusId = 0;
    ROCM_SMI_CHECK(rsmi_dev_pci_id_get(i, &rsmiBusId));
    gpu->busId = RsmiBusId2PciBusId(rsmiBusId);
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

  ROCM_SMI_CHECK(rsmi_shut_down());
  dlclose(lib);
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
