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
#include <vector>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

// SMI library is loaded via dlopen(RTLD_LOCAL) instead of being linked.
// On ROCm 10.1+ (TheRock#6852 removed rocm_smi_lib), we use amd_smi.
// On ROCm 10.0 and earlier, we use rocm_smi (librocm_smi64).
namespace {

#if MORI_USE_AMDSMI
// ---- amd_smi path (ROCm 10.1+) ----

void* OpenAmdSmi() {
  const char* candidates[] = {
    std::getenv("MORI_AMD_SMI_PATH"),
    "libamd_smi.so",
    "/opt/rocm/lib/libamd_smi.so",
    std::getenv("MORI_ROCM_SMI_PATH"),
    "librocm_smi64.so.1",
    "librocm_smi64.so",
    "/opt/rocm/lib/librocm_smi64.so.1"
  };
  for (const char* path : candidates) {
    if (path && path[0]) {
      if (void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL)) return h;
    }
  }
  fprintf(stderr, "[AMD-SMI] dlopen(libamd_smi) failed: %s\n", dlerror());
  exit(-1);
}

template <typename Fn>
Fn Sym(void* handle, const char* name) {
  void* sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "[AMD-SMI] missing symbol %s\n", name);
    exit(-1);
  }
  return reinterpret_cast<Fn>(sym);
}

#define AMDSMI_FN(lib, name) auto name = Sym<decltype(&::name)>(lib, #name)

#else
// ---- rocm_smi path (ROCm <= 10.0) ----

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

#define RSMI_FN(lib, name) auto name = Sym<decltype(&::name)>(lib, #name)

#endif

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemGpu                                         */
/* ---------------------------------------------------------------------------------------------- */
TopoSystemGpu::TopoSystemGpu() { Load(); }

TopoSystemGpu::~TopoSystemGpu() {}

#if MORI_USE_AMDSMI
// ---- amd_smi Load() (ROCm 10.1+) ----
// Uses socket/processor handle model instead of flat device indices.

PciBusId AmdSmiBdf2PciBusId(amdsmi_bdf_t bdf) {
  uint16_t domain = bdf.domain_number;
  uint8_t bus = bdf.bus_number;
  uint8_t dev = bdf.device_number;
  uint8_t func = bdf.function_number;
  return PciBusId(domain, bus, dev, func);
}

void TopoSystemGpu::Load() {
  void* lib = OpenAmdSmi();
  AMDSMI_FN(lib, amdsmi_init);
  AMDSMI_FN(lib, amdsmi_shut_down);
  AMDSMI_FN(lib, amdsmi_get_socket_handles);
  AMDSMI_FN(lib, amdsmi_get_processor_handles);
  AMDSMI_FN(lib, amdsmi_get_gpu_device_bdf);
  AMDSMI_FN(lib, amdsmi_topo_get_p2p_status);
  AMDSMI_FN(lib, amdsmi_topo_get_link_type);
  AMDSMI_FN(lib, amdsmi_topo_get_link_weight);
  AMDSMI_FN(lib, amdsmi_status_code_to_string);

  ROCM_SMI_CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

  uint32_t socketCount = 0;
  ROCM_SMI_CHECK(amdsmi_get_socket_handles(&socketCount, nullptr));
  std::vector<amdsmi_socket_handle> sockets(socketCount);
  ROCM_SMI_CHECK(amdsmi_get_socket_handles(&socketCount, sockets.data()));

  uint32_t numGpus = 0;
  ROCM_SMI_CHECK(amdsmi_get_processor_handles(sockets[0], &numGpus, nullptr));
  std::vector<amdsmi_processor_handle> handles(numGpus);
  ROCM_SMI_CHECK(amdsmi_get_processor_handles(sockets[0], &numGpus, handles.data()));

  if (numGpus == 0) {
    fprintf(stderr, "[AMD-SMI] amdsmi_get_processor_handles reported 0 GPUs\n");
    exit(-1);
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpus.emplace_back(gpu);
    amdsmi_bdf_t bdf = {};
    ROCM_SMI_CHECK(amdsmi_get_gpu_device_bdf(handles[i], &bdf));
    gpu->busId = AmdSmiBdf2PciBusId(bdf);
  }

  for (uint32_t i = 0; i < numGpus; ++i) {
    for (uint32_t j = i; j < numGpus; ++j) {
      if (i == j) continue;
      amdsmi_link_type_t link_type = {};
      amdsmi_p2p_capability_t cap = {};
      ROCM_SMI_CHECK(amdsmi_topo_get_p2p_status(handles[i], handles[j], &link_type, &cap));
      if (!cap.is_iolink_coherent && !cap.is_iolink_atomics_32bit && !cap.is_iolink_atomics_64bit) continue;

      TopoNodeGpuP2pLink* p2p = new TopoNodeGpuP2pLink();
      ROCM_SMI_CHECK(amdsmi_topo_get_link_type(handles[i], handles[j], &p2p->hops, &p2p->type));
      ROCM_SMI_CHECK(amdsmi_topo_get_link_weight(handles[i], handles[j], &p2p->weight));
      p2p->gpu1 = gpus[i].get();
      p2p->gpu2 = gpus[j].get();
      p2ps.emplace_back(p2p);

      gpus[i]->p2ps.push_back(p2p);
      gpus[j]->p2ps.push_back(p2p);
    }
  }

  ROCM_SMI_CHECK(amdsmi_shut_down());
  dlclose(lib);
}

#else
// ---- rocm_smi Load() (ROCm <= 10.0) ----
// Uses flat device index model (rsmi_num_monitor_devices, rsmi_dev_pci_id_get).

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

#endif

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
