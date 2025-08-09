#include "mori/application/topology/pci.hpp"

#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>

extern "C" {
#include <pci/pci.h>
}

namespace mori {
namespace application {

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

TopoSystemPci::TopoSystemPci() { Load(); }
TopoSystemPci::~TopoSystemPci() {}

void TopoSystemPci::Load() {
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

}  // namespace application
}  // namespace mori