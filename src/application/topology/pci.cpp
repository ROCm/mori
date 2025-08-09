#include "mori/application/topology/pci.hpp"

#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

extern "C" {
#include <pci/pci.h>
}

namespace mori {
namespace application {

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
/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoNodePci                                          */
/* ---------------------------------------------------------------------------------------------- */
TopoNodePci* TopoNodePci::CreateVirtualRoot() {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::VirtualRoot;
  return n;
}

TopoNodePci* TopoNodePci::CreateRootComplex(PciBusId bus, NumaNodeId numa, TopoNodePci* vr) {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::RootComplex;
  n->busId = bus;
  n->numaNode = numa;
  assert(vr->type == TopoNodePciType::VirtualRoot);
  n->usp = vr;
  return n;
}

TopoNodePci* TopoNodePci::CreateBridge(PciBusId bus, NumaNodeId numa) {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::Bridge;
  n->busId = bus;
  n->numaNode = numa;
  return n;
}

TopoNodePci* TopoNodePci::CreateGpu(PciBusId bus, NumaNodeId numa) {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::Gpu;
  n->busId = bus;
  n->numaNode = numa;
  return n;
}

TopoNodePci* TopoNodePci::CreateNet(PciBusId bus, NumaNodeId numa) {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::Net;
  n->busId = bus;
  n->numaNode = numa;
  return n;
}

void TopoNodePci::SetUpstreamPort(TopoNodePci* n) {
  if (type == TopoNodePciType::VirtualRoot) {
    assert(false && "virtual root cannot have usp");
  }

  else if (type == TopoNodePciType::RootComplex) {
    assert((n->Type() == TopoNodePciType::VirtualRoot) &&
           "root port can only connect to virtual port as its usp");
    usp = n;
  }

  else {
    usp = n;
  }
}

void TopoNodePci::AddDownstreamPort(TopoNodePci* n) {
  if ((type == TopoNodePciType::Gpu) || (type == TopoNodePciType::Net)) {
    assert(false && "pci endpoints(gpu/net) cannot have dsp");
  }
  for (auto* dsp : dsps) {
    if (dsp == n) return;
  }
  dsps.push_back(n);
}

void TopoNodePci::RemoveDownstreamPort(PciBusId bus) {
  for (int i = 0; i < dsps.size(); i++) {
    if (dsps[i]->BusId() == bus) {
      dsps.erase(dsps.begin() + i);
      return;
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemPci                                         */
/* ---------------------------------------------------------------------------------------------- */
TopoSystemPci::TopoSystemPci() {
  Load();
  Validate();
}

TopoSystemPci::~TopoSystemPci() {}

TopoNodePci* CreateTopoNodePciFrom(pci_dev* dev) {
  uint16_t cls = dev->device_class;
  PciBusId bus = PciBusId(dev->domain, dev->bus, dev->dev, dev->func);
  NumaNodeId numa = dev->numa_node;
  if ((cls == PCI_CLASS_BRIDGE_HOST) || (cls == PCI_CLASS_BRIDGE_PCI)) {
    return TopoNodePci::CreateBridge(bus, numa);
  } else if ((cls == PCI_CLASS_NETWORK_ETHERNET) || (cls == PCI_CLASS_SERIAL_INFINIBAND)) {
    return TopoNodePci::CreateNet(bus, numa);
  } else if (cls == PCI_CLASS_DISPLAY_3D) {
    return TopoNodePci::CreateGpu(bus, numa);
  }
  return nullptr;
}

void TopoSystemPci::Load() {
  struct pci_access* pacc = pci_alloc();
  pci_init(pacc);
  pci_scan_bus(pacc);

  std::unordered_map<uint64_t, pci_dev*> dsp2dev;
  std::unordered_map<uint64_t, pci_dev*> bus2dev;
  std::unordered_set<uint32_t> domains;

  root = TopoNodePci::CreateVirtualRoot();
  pcis.emplace(root->BusId().packed, root);

  // Collect all pcie nodes
  for (struct pci_dev* dev = pacc->devices; dev; dev = dev->next) {
    pci_fill_info(dev, PCI_FILL_CLASS | PCI_FILL_NUMA_NODE);
    uint8_t headerType = pci_read_byte(dev, PCI_HEADER_TYPE) & 0x7f;
    if ((headerType != PCI_HEADER_TYPE_NORMAL) && (headerType != PCI_HEADER_TYPE_BRIDGE)) continue;

    TopoNodePci* node = CreateTopoNodePciFrom(dev);
    if (node == nullptr) continue;

    domains.insert(dev->domain);
    pcis.emplace(node->BusId().packed, node);

    if (headerType == PCI_HEADER_TYPE_BRIDGE) {
      uint8_t secondary = pci_read_byte(dev, PCI_SECONDARY_BUS);
      uint64_t globalSecondary = PciBusId(dev->domain, secondary, 0, 0).packed;
      assert(dsp2dev.find(globalSecondary) == dsp2dev.end());
      dsp2dev.insert({globalSecondary, dev});
      assert(dev->bus == pci_read_byte(dev, PCI_PRIMARY_BUS));
    }

    bus2dev.insert({node->BusId().packed, dev});
  }

  // Create root port
  for (auto& dom : domains) {
    printf("domain %d\n", dom);
    TopoNodePci* n = TopoNodePci::CreateRootComplex(PciBusId(dom, 0, 0, 0), -1, root);
    pcis.emplace(n->BusId().packed, n);
  }

  // Connect upstream port and downstream port
  for (auto& it : pcis) {
    PciBusId busId = it.first;
    TopoNodePci* node = it.second.get();
    if (busId.packed == 0) continue;

    uint64_t parentDsp = PciBusId(busId.Domain(), busId.Bus(), 0, 0).packed;
    uint64_t parentBus = 0;
    if (dsp2dev.find(parentDsp) == dsp2dev.end()) {
      assert(IsUnderRootComplex(bus2dev[busId.packed]));
      parentBus = PciBusId(busId.Domain(), 0, 0, 0).packed;
    } else {
      pci_dev* dev = dsp2dev[parentDsp];
      parentBus = PciBusId(dev->domain, dev->bus, dev->dev, dev->func).packed;
    }

    assert(pcis.find(parentBus) != pcis.end());
    TopoNodePci* parent = pcis[parentBus].get();

    node->SetUpstreamPort(parent);
    parent->AddDownstreamPort(node);
  }

  pci_cleanup(pacc);
}

void TopoSystemPci::Validate() {
  std::unordered_set<uint64_t> seen;

  // Make sure every node can be reached from root and no cycles
  std::queue<TopoNodePci*> nodes;
  nodes.push(root);

  while (!nodes.empty()) {
    size_t nnodes = nodes.size();

    while (nnodes > 0) {
      TopoNodePci* cur = nodes.front();
      nodes.pop();

      assert(seen.find(cur->BusId().packed) == seen.end());
      seen.insert(cur->BusId().packed);

      for (auto* child : cur->DownstreamPort()) {
        nodes.push(child);
      }

      nnodes--;
    }
  }

  assert(seen.size() == pcis.size());
  printf("total nodes %d\n", seen.size());
}

}  // namespace application
}  // namespace mori