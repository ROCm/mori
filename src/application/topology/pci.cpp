#include "mori/application/topology/pci.hpp"

#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <iomanip>
#include <queue>
#include <regex>
#include <sstream>
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

bool IsBdfString(const std::string& s) {
  static const std::regex bdfRegex(R"(^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$)");
  return std::regex_match(s.c_str(), bdfRegex);
}
std::vector<uint64_t> ParseBdfFromString(const std::string& str) {
  if (!IsBdfString(str)) return {};

  uint64_t domain = 0, bus = 0, dev = 0, func = 0;
  char dot;

  std::stringstream ss(str);
  ss >> std::hex >> domain;
  ss.ignore(1, ':');
  ss >> std::hex >> bus;
  ss.ignore(1, ':');
  ss >> std::hex >> dev;
  ss >> dot;
  ss >> std::hex >> func;

  return {domain, bus, dev, func};
}

std::string BdfToString(uint64_t domain, uint64_t bus, uint64_t dev, uint64_t func) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(4) << domain << ":" << std::setw(2) << bus << ":"
     << std::setw(2) << dev << "." << std::dec << func;
  return ss.str();
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

TopoNodePci* TopoNodePci::CreateOthers(PciBusId bus, NumaNodeId numa) {
  TopoNodePci* n = new TopoNodePci();
  n->type = TopoNodePciType::Others;
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
/*                                           TopoPathPci                                          */
/* ---------------------------------------------------------------------------------------------- */
TopoPathPci::TopoPathPci(TopoNodePci* h, TopoNodePci* t) : head(h), tail(t) {
  assert(head && tail);
  BuildPath();
  Validate();
}

void TopoPathPci::BuildPath() {
  std::vector<TopoNodePci*> hAnces;
  std::vector<TopoNodePci*> tAnces;
  std::unordered_map<TopoNodePci*, int> tAnces2Idx;

  for (TopoNodePci* n = head; n != nullptr; n = n->UpstreamPort()) hAnces.push_back(n);

  int i = 0;
  for (TopoNodePci* n = tail; n != nullptr; n = n->UpstreamPort(), i++) {
    tAnces.push_back(n);
    tAnces2Idx.insert({n, i});
  }

  for (auto* n : hAnces) {
    if (tAnces2Idx.find(n) == tAnces2Idx.end()) {
      nodes.push_back(n);
    } else {
      int idx = tAnces2Idx[n];
      for (int i = idx; i >= 0; i--) nodes.push_back(tAnces[i]);
      break;
    }
  }
}

void TopoPathPci::Validate() {
  assert(!nodes.empty());
  assert(nodes[0] == head);
  assert(nodes[nodes.size() - 1] == tail);
}

size_t TopoPathPci::Hops() const {
  if ((head == nullptr) || (tail == nullptr)) return 0;

  size_t distance = nodes.size();
  if (distance <= 2) return 0;
  return distance - 2;
}

bool TopoPathPci::CrossRootComplex() const {
  for (auto* n : nodes) {
    if (n->Type() == TopoNodePciType::RootComplex) return true;
  }
  return false;
}

bool TopoPathPci::CrossMultipleNuma() const {
  // No root complex, check numa directly
  if ((head->Type() != TopoNodePciType::RootComplex) &&
      (tail->Type() != TopoNodePciType::RootComplex)) {
    return (head->NumaNode() != tail->NumaNode());
  }
  // If one is root complex, check
  bool crossMultiRc = false;
  for (auto* n : nodes) {
    if (n->Type() == TopoNodePciType::VirtualRoot) {
      crossMultiRc = true;
      break;
    }
  }
  return crossMultiRc;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          TopoSystemPci                                         */
/* ---------------------------------------------------------------------------------------------- */

TopoSystemPci::TopoSystemPci() : pathCache() {
  Load();
  Validate();
}

TopoSystemPci::~TopoSystemPci() {}

TopoNodePci* CreateTopoNodePciFrom(pci_dev* dev) {
  uint16_t cls = dev->device_class;
  uint16_t baseCls = (cls >> 8);
  PciBusId bus = PciBusId(dev->domain, dev->bus, dev->dev, dev->func);
  NumaNodeId numa = dev->numa_node;
  if ((cls == PCI_CLASS_BRIDGE_HOST) || (cls == PCI_CLASS_BRIDGE_PCI)) {
    return TopoNodePci::CreateBridge(bus, numa);
  } else if (baseCls == PCI_BASE_CLASS_NETWORK) {
    return TopoNodePci::CreateNet(bus, numa);
  } else if (cls == 0x1200) {
    return TopoNodePci::CreateGpu(bus, numa);
  }
  //  else {
  //   return TopoNodePci::CreateOthers(bus, numa);
  // }

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
    TopoNodePci* n = TopoNodePci::CreateRootComplex(PciBusId(dom, 0, 0, 0), -1, root);
    printf("domain bus %s\n", n->BusId().String().c_str());
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
      if (busId.Bus() == 0) {
        printf("bus %s \n", busId.String().c_str());
      }
      assert(IsUnderRootComplex(bus2dev[busId.packed]));
      parentBus = PciBusId(busId.Domain(), 0, 0, 0).packed;
    } else {
      if (busId.packed == 0) {
        printf("root port\n");
      }
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
}

TopoPathPci* TopoSystemPci::Path(PciBusId head, PciBusId tail) {
  if (pcis.find(head.packed) == pcis.end()) return nullptr;
  if (pcis.find(tail.packed) == pcis.end()) return nullptr;

  PathKey key{head.packed, tail.packed};
  if (pathCache.find(key) == pathCache.end()) {
    TopoPathPci* p = new TopoPathPci(pcis[head.packed].get(), pcis[tail.packed].get());
    pathCache.emplace(key, p);
  }
  return pathCache[key].get();
}

TopoNodePci* TopoSystemPci::Node(PciBusId busId) {
  if (pcis.find(busId.packed) == pcis.end()) return nullptr;
  return pcis[busId.packed].get();
}

}  // namespace application
}  // namespace mori