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
#include "mori/application/topology/pci.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mori {
namespace application {

namespace {

constexpr const char* kSysfsPciDevices = "/sys/bus/pci/devices";

// PCI configuration header offsets (PCI Local Bus Specification).
constexpr off_t kCfgHeaderType = 0x0e;
constexpr off_t kCfgPrimaryBus = 0x18;
constexpr off_t kCfgSecondaryBus = 0x19;
constexpr off_t kCfgSubordinateBus = 0x1a;

constexpr uint8_t kHeaderTypeNormal = 0x00;
constexpr uint8_t kHeaderTypeBridge = 0x01;

constexpr uint16_t kClassBridgePci = 0x0604;
constexpr uint8_t kBaseClassNetwork = 0x02;

// One PCI function as described by sysfs. Unprivileged readers of sysfs
// "config" only see the first 64 bytes of configuration space, which covers
// every field read here.
struct SysfsPciDev {
  uint16_t domain{0};
  uint8_t bus{0};
  uint8_t dev{0};
  uint8_t func{0};
  uint16_t deviceClass{0};  // class code without the programming interface byte
  NumaNodeId numaNode{-1};
  uint8_t headerType{0};
  uint8_t primaryBus{0};
  uint8_t secondaryBus{0};
  uint8_t subordinateBus{0};

  PciBusId BusId() const { return PciBusId(domain, bus, dev, func); }
};

bool ReadSysfsLine(const std::string& path, std::string& out) {
  std::ifstream f(path);
  return static_cast<bool>(std::getline(f, out));
}

bool ReadSysfsPciDev(const std::string& name, SysfsPciDev& out) {
  std::vector<uint64_t> bdf = ParseBdfFromString(name);
  if (bdf.size() != 4) return false;
  out.domain = bdf[0];
  out.bus = bdf[1];
  out.dev = bdf[2];
  out.func = bdf[3];

  const std::string dir = std::string(kSysfsPciDevices) + "/" + name;

  std::string line;
  if (!ReadSysfsLine(dir + "/class", line)) return false;
  out.deviceClass = static_cast<uint16_t>(strtoul(line.c_str(), nullptr, 16) >> 8);

  out.numaNode = -1;
  if (ReadSysfsLine(dir + "/numa_node", line)) out.numaNode = strtol(line.c_str(), nullptr, 10);

  int fd = open((dir + "/config").c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  uint8_t cfg[kCfgSubordinateBus + 1];
  ssize_t n = pread(fd, cfg, sizeof(cfg), 0);
  close(fd);
  if (n != static_cast<ssize_t>(sizeof(cfg))) return false;

  out.headerType = cfg[kCfgHeaderType] & 0x7f;
  out.primaryBus = cfg[kCfgPrimaryBus];
  out.secondaryBus = cfg[kCfgSecondaryBus];
  out.subordinateBus = cfg[kCfgSubordinateBus];
  return true;
}

std::vector<SysfsPciDev> ScanSysfsPci() {
  std::vector<SysfsPciDev> devs;
  DIR* d = opendir(kSysfsPciDevices);
  if (d == nullptr) return devs;
  while (struct dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    SysfsPciDev dev;
    if (ReadSysfsPciDev(e->d_name, dev)) devs.push_back(dev);
  }
  closedir(d);
  std::sort(devs.begin(), devs.end(), [](const SysfsPciDev& a, const SysfsPciDev& b) {
    return a.BusId().packed < b.BusId().packed;
  });
  return devs;
}

bool IsUnderRootComplex(const SysfsPciDev& dev) {
  const std::string devpath = std::string(kSysfsPciDevices) + "/" + dev.BusId().String();

  char link[256], parent[256];
  ssize_t len = readlink(devpath.c_str(), link, sizeof(link) - 1);
  if (len < 0) return false;
  link[len] = '\0';

  strcpy(parent, dirname(link));
  const char* last = basename(parent);
  return strncmp(last, "pci", 3) == 0;
}

}  // namespace

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
  n->busId = PciBusId(std::numeric_limits<uint64_t>::max());
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

static TopoNodePci* CreateTopoNodePciFrom(const SysfsPciDev& dev) {
  uint16_t cls = dev.deviceClass;
  uint16_t baseCls = (cls >> 8);
  PciBusId bus = dev.BusId();
  NumaNodeId numa = dev.numaNode;
  if (cls == kClassBridgePci) {
    return TopoNodePci::CreateBridge(bus, numa);
  } else if (baseCls == kBaseClassNetwork) {
    return TopoNodePci::CreateNet(bus, numa);
  } else if (cls == 0x1200 || cls == 0x0300 || cls == 0x0302) {
    // 0x1200: Processing Accelerator (data-center GPUs)
    // 0x0300: VGA compatible controller (RDNA 4 consumer/workstation GPUs, e.g. R9700)
    // 0x0302: 3D controller (some workstation GPUs)
    return TopoNodePci::CreateGpu(bus, numa);
  }

  return nullptr;
}

void TopoSystemPci::Load() {
  const std::vector<SysfsPciDev> devs = ScanSysfsPci();

  std::unordered_map<uint64_t, const SysfsPciDev*> dsp2dev;
  std::unordered_map<uint64_t, const SysfsPciDev*> bus2dev;
  std::unordered_set<uint32_t> domains;

  root = TopoNodePci::CreateVirtualRoot();
  pcis.emplace(root->BusId().packed, root);

  // Collect all pcie nodes
  for (const SysfsPciDev& dev : devs) {
    if ((dev.headerType != kHeaderTypeNormal) && (dev.headerType != kHeaderTypeBridge)) continue;

    TopoNodePci* node = CreateTopoNodePciFrom(dev);
    if (node == nullptr) continue;

    domains.insert(dev.domain);
    pcis.emplace(node->BusId().packed, node);

    if (dev.headerType == kHeaderTypeBridge) {
      // Use a wider counter than the bus numbers themselves: a bridge may report
      // a subordinate bus of 0xff (e.g. empty downstream ports on some NICs), and
      // a uint8_t loop variable would wrap from 0xff back to 0 and spin forever.
      uint32_t secondary = dev.secondaryBus;
      uint32_t subordinate = dev.subordinateBus;
      for (uint32_t i = secondary; i <= subordinate; i++) {
        uint64_t dspBusId = PciBusId(dev.domain, i, 0, 0).packed;
        if (dsp2dev.find(dspBusId) != dsp2dev.end()) {
          const SysfsPciDev* lastDev = dsp2dev[dspBusId];
          if (dev.primaryBus < lastDev->primaryBus) continue;
        }
        dsp2dev[dspBusId] = &dev;
        assert(dev.bus == dev.primaryBus);
      }
    }

    bus2dev.insert({node->BusId().packed, &dev});
  }

  // Create root port
  for (auto& dom : domains) {
    TopoNodePci* n = nullptr;
    PciBusId busId(dom, 0, 0, 0);
    // No root port or device found, create a virtual one
    if (pcis.find(busId.packed) == pcis.end()) {
      n = TopoNodePci::CreateRootComplex(busId, -1, root);
      pcis.emplace(n->BusId().packed, n);
    }
    n = pcis[busId.packed].get();
    root->AddDownstreamPort(n);
    n->SetUpstreamPort(root);
  }

  // Connect upstream port and downstream port
  for (auto& it : pcis) {
    PciBusId busId = it.first;
    TopoNodePci* node = it.second.get();

    // These nodes should already be connected
    if ((node->Type() == TopoNodePciType::RootComplex) ||
        node->Type() == TopoNodePciType::VirtualRoot)
      continue;

    uint64_t parentDsp = PciBusId(busId.Domain(), busId.Bus(), 0, 0).packed;
    uint64_t parentBus = 0;
    if (dsp2dev.find(parentDsp) == dsp2dev.end()) {
      assert(IsUnderRootComplex(*bus2dev[busId.packed]));
      parentBus = PciBusId(busId.Domain(), 0, 0, 0).packed;
    } else {
      parentBus = dsp2dev[parentDsp]->BusId().packed;
    }

    // Prevent loopback to self, this could happen in SR-IOV mode where all devices directly connect
    // to root complex
    if (parentBus == node->BusId().packed) continue;

    assert(pcis.find(parentBus) != pcis.end());
    TopoNodePci* parent = pcis[parentBus].get();

    node->SetUpstreamPort(parent);
    parent->AddDownstreamPort(node);
  }
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
