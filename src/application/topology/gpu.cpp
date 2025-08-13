#include "mori/application/topology/gpu.hpp"

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

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

  ROCM_SMI_CHECK(rsmi_init(0));
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

  ROCM_SMI_CHECK(rsmi_shut_down());
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