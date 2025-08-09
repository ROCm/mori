#include "mori/application/topology/gpu.hpp"

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemGpu                                         */
/* ---------------------------------------------------------------------------------------------- */
TopoSystemGpu::TopoSystemGpu() { Load(); }

TopoSystemGpu::~TopoSystemGpu() {}

void TopoSystemGpu::Load() {
  uint32_t numGpus;
  PciBusId busId;

  ROCM_SMI_CHECK(rsmi_init(0));
  ROCM_SMI_CHECK(rsmi_num_monitor_devices(&numGpus));

  for (uint32_t i = 0; i < numGpus; ++i) {
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpus.emplace_back(gpu);
    ROCM_SMI_CHECK(rsmi_dev_pci_id_get(i, &gpu->busId));
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

}  // namespace application
}  // namespace mori