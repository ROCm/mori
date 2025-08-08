#include "mori/application/topology/system.hpp"

#include <libgen.h>
#include <stdio.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mori/application/utils/check.hpp"
extern "C" {
#include <pci/pci.h>
}

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                            TopoNode                                            */
/* ---------------------------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoSystem                                           */
/* ---------------------------------------------------------------------------------------------- */
TopoSystem::TopoSystem() {}

TopoSystem::~TopoSystem() {}

std::string bdfToString(uint64_t bdf) {
  uint8_t bus = (bdf >> 8) & 0xFF;
  uint8_t device = (bdf >> 3) & 0x1F;
  uint8_t function = bdf & 0x07;
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(bus) << ":"
      << std::setw(2) << static_cast<int>(device) << "." << static_cast<int>(function);
  printf("-%d %s-\n", function, oss.str().c_str());

  return oss.str();
}

void TopoSystem::InitTopoGpuPlane() {
  uint32_t numGpus;
  PciBusId busId;

  ROCM_SMI_CHECK(rsmi_init(0));
  ROCM_SMI_CHECK(rsmi_num_monitor_devices(&numGpus));

  for (uint32_t i = 0; i < numGpus; ++i) {
    TopoNodeGpu* gpu = new TopoNodeGpu();
    gpus.emplace_back(gpu);
    ROCM_SMI_CHECK(rsmi_dev_pci_id_get(i, &gpu->busId));
    ROCM_SMI_CHECK(rsmi_topo_numa_affinity_get(reinterpret_cast<uint32_t>(i), &gpu->numaNode));
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

PciBusId GetPciDevBusId(struct pci_dev* d) {
  return ((uint64_t)d->domain << 32) | ((uint64_t)d->bus << 16) | ((uint64_t)d->dev << 8) |
         ((uint64_t)d->func << 0);
}

bool IsUnderRootComplex(struct pci_dev* dev) {
  char devpath[128];
  snprintf(devpath, sizeof(devpath), "/sys/bus/pci/devices/%04x:%02x:%02x.%d", dev->domain,
           dev->bus, dev->dev, dev->func);

  char link[256], parent[256];
  ssize_t len = readlink(devpath, link, sizeof(link) - 1);
  link[len] = '\0';

  strcpy(parent, dirname(link));
  const char* last = basename(parent);
  return strncmp(last, "pci", 3) == 0;
}

void TopoSystem::InitTopoPciPlane() {
  struct pci_access* pacc = pci_alloc();
  pci_init(pacc);
  pci_scan_bus(pacc);

  std::unordered_map<PciBusId, pci_dev*> dsp2dev;
  std::unordered_map<PciBusId, pci_dev*> bus2dev;
  std::unordered_set<uint32_t> domains;

  // Collect all pcie nodes
  for (struct pci_dev* dev = pacc->devices; dev; dev = dev->next) {
    pci_fill_info(dev, PCI_FILL_CLASS | PCI_FILL_NUMA_NODE);
    uint16_t cls = dev->device_class;
    uint8_t baseCls = (cls >> 8) & 0xFF;
    uint8_t subCls = cls & 0xFF;
    if ((cls != PCI_CLASS_NETWORK_ETHERNET) && (cls != PCI_CLASS_DISPLAY_3D) &&
        (cls != PCI_CLASS_BRIDGE_HOST) && (cls != PCI_CLASS_BRIDGE_PCI) &&
        (cls != PCI_CLASS_SERIAL_INFINIBAND))
      continue;

    uint8_t headerType = pci_read_byte(dev, PCI_HEADER_TYPE) & 0x7f;
    if ((headerType != PCI_HEADER_TYPE_NORMAL) && (headerType != PCI_HEADER_TYPE_BRIDGE)) continue;

    domains.insert(dev->domain);

    TopoNodePci* node = new TopoNodePci();
    node->busId = GetPciDevBusId(dev);
    node->numaNode = dev->numa_node;
    pcis.emplace(node->busId, node);

    if (headerType == PCI_HEADER_TYPE_BRIDGE) {
      uint8_t secondary = pci_read_byte(dev, PCI_SECONDARY_BUS);
      uint64_t globalSecondary = ((uint64_t)dev->domain << 32) | ((uint64_t)secondary << 16);
      assert(dsp2dev.find(globalSecondary) == dsp2dev.end());
      dsp2dev.insert({globalSecondary, dev});
      assert(dev->bus == pci_read_byte(dev, PCI_PRIMARY_BUS));
    }

    bus2dev.insert({node->busId, dev});
  }

  // Build virtual root complex node becuase libpci does not include rc as pci_dev
  for (auto& dom : domains) {
    TopoNodePci* rc = new TopoNodePci();
    rc->busId = static_cast<uint64_t>(dom) << 32;
    rc->numaNode = -1;
    pcis.emplace(rc->busId, rc);
    rcs.push_back(rc);
  }

  // Connect upstream port and downstream port
  for (auto& it : pcis) {
    PciBusId busId = it.first;
    TopoNodePci* node = it.second.get();
    if (busId == 0) continue;

    PciBusId dspBusId = busId & 0xFFFFFFFFFFFF0000ULL;
    PciBusId parentBusId = 0;
    if (dsp2dev.find(dspBusId) == dsp2dev.end()) {
      assert(IsUnderRootComplex(bus2dev[busId]));
    } else {
      pci_dev* dev = dsp2dev[dspBusId];
      parentBusId = GetPciDevBusId(dev);
    }

    assert(pcis.find(parentBusId) != pcis.end());
    TopoNodePci* parent = pcis[parentBusId].get();

    node->usp = parent;
    parent->dsps.push_back(node);
  }

  // TODO: correctness check, remove later
  for (auto& it : pcis) {
    TopoNodePci* node = it.second.get();
    assert(node->usp || !node->dsps.empty());
  }

  pci_cleanup(pacc);
}

void TopoSystem::Load() {
  InitTopoGpuPlane();
  InitTopoPciPlane();
}

}  // namespace io
}  // namespace mori