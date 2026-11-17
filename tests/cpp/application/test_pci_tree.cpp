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

// TopoSystemPci built from synthetic sysfs functions, so the tree rules run without the host's PCI
// bus. The layouts follow an MI355X tray: root port -> switch -> GPU, and a Pensando NIC below a
// second switch port.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "mori/application/topology/pci.hpp"

using mori::application::PciBusId;
using mori::application::SysfsPciDev;
using mori::application::TopoNodePci;
using mori::application::TopoNodePciType;
using mori::application::TopoSystemPci;

namespace {

constexpr uint16_t kClassHostBridge = 0x0600;
constexpr uint16_t kClassBridge = 0x0604;
constexpr uint16_t kClassEthernet = 0x0200;
constexpr uint16_t kClassAccel = 0x1200;
constexpr uint8_t kHeaderDead = 0x7f;  // config space reads all ones

SysfsPciDev Fn(uint16_t dom, uint8_t bus, uint8_t dev, uint8_t fn, uint16_t cls,
               bool underRootComplex = false) {
  SysfsPciDev d;
  d.domain = dom;
  d.bus = bus;
  d.dev = dev;
  d.func = fn;
  d.deviceClass = cls;
  d.numaNode = 0;
  d.underRootComplex = underRootComplex;
  return d;
}

SysfsPciDev Bridge(uint16_t dom, uint8_t bus, uint8_t dev, uint8_t fn, uint8_t pri, uint8_t sec,
                   uint8_t sub, bool underRootComplex = false) {
  SysfsPciDev d = Fn(dom, bus, dev, fn, kClassBridge, underRootComplex);
  d.headerType = 0x01;
  d.primaryBus = pri;
  d.secondaryBus = sec;
  d.subordinateBus = sub;
  return d;
}

PciBusId Bdf(uint16_t dom, uint8_t bus, uint8_t dev, uint8_t fn) {
  return PciBusId(dom, bus, dev, fn);
}

// Root port 00:01.1 -> switch 01:00.0 -> port 02:00.0 -> 03:00.0 -> 04:00.0 -> GPU 05:00.0, and
// port 02:01.0 -> NIC switch 06:00.0 -> 07:00.0 (accel functions 08:00.x) / 07:01.0 (NIC 09:00.0).
std::vector<SysfsPciDev> Tray(uint8_t nicSwitchPrimary) {
  return {
      Fn(0, 0x00, 0x00, 0, kClassHostBridge, true),
      Bridge(0, 0x00, 0x01, 1, 0x00, 0x01, 0x0c, true),
      Bridge(0, 0x01, 0x00, 0, 0x01, 0x02, 0x0c),
      Bridge(0, 0x02, 0x00, 0, 0x02, 0x03, 0x05),
      Bridge(0, 0x02, 0x01, 0, 0x02, 0x06, 0x09),
      Bridge(0, 0x03, 0x00, 0, 0x03, 0x04, 0x05),
      Bridge(0, 0x04, 0x00, 0, 0x04, 0x05, 0x05),
      Fn(0, 0x05, 0x00, 0, kClassAccel),
      Bridge(0, 0x06, 0x00, 0, nicSwitchPrimary, 0x07, 0x09),
      Bridge(0, 0x07, 0x00, 0, 0x07, 0x08, 0x08),
      Bridge(0, 0x07, 0x01, 0, 0x07, 0x09, 0x09),
      Fn(0, 0x08, 0x00, 0, kClassAccel),
      Fn(0, 0x08, 0x00, 1, kClassAccel),
      Fn(0, 0x09, 0x00, 0, kClassEthernet),
  };
}

TopoNodePci* Parent(TopoSystemPci& sys, PciBusId id) {
  TopoNodePci* n = sys.Node(id);
  return n ? n->UpstreamPort() : nullptr;
}

// Every node has one parent, is listed once among its children, and the root reaches each node
// exactly once.
void ExpectTree(TopoSystemPci& sys, const std::vector<SysfsPciDev>& devs) {
  std::unordered_set<TopoNodePci*> nodes;
  TopoNodePci* root = nullptr;
  for (const SysfsPciDev& d : devs) {
    TopoNodePci* n = sys.Node(d.BusId());
    if (n == nullptr) continue;
    nodes.insert(n);
    nodes.insert(sys.Node(Bdf(d.domain, 0, 0, 0)));

    TopoNodePci* top = n;
    for (size_t hops = 0; top->UpstreamPort() != nullptr; hops++) {
      ASSERT_LT(hops, devs.size() + 2) << d.BusId().String() << " is on a cycle";
      TopoNodePci* up = top->UpstreamPort();
      const auto& kids = up->DownstreamPort();
      EXPECT_EQ(std::count(kids.begin(), kids.end(), top), 1) << top->BusId().String();
      top = up;
    }
    ASSERT_EQ(top->Type(), TopoNodePciType::VirtualRoot) << d.BusId().String();
    if (root == nullptr) root = top;
    ASSERT_EQ(top, root);
  }
  ASSERT_NE(root, nullptr);
  nodes.insert(root);

  std::unordered_set<TopoNodePci*> seen;
  std::vector<TopoNodePci*> stack{root};
  while (!stack.empty()) {
    TopoNodePci* cur = stack.back();
    stack.pop_back();
    ASSERT_TRUE(seen.insert(cur).second) << cur->BusId().String() << " is reached twice";
    for (TopoNodePci* kid : cur->DownstreamPort()) stack.push_back(kid);
  }
  EXPECT_EQ(seen, nodes);
}

}  // namespace

TEST(PciTree, NestedRangesPickTheDeepestBridgeInAnyOrder) {
  // 06:00.0 leaves its primary bus register at 0; its own bus still places it.
  std::vector<SysfsPciDev> devs = Tray(0x00);
  for (bool reversed : {false, true}) {
    SCOPED_TRACE(reversed ? "deepest first" : "sysfs order");
    if (reversed) std::reverse(devs.begin(), devs.end());
    TopoSystemPci sys(devs);
    ExpectTree(sys, devs);

    EXPECT_EQ(Parent(sys, Bdf(0, 0x05, 0, 0)), sys.Node(Bdf(0, 0x04, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x03, 0, 0)), sys.Node(Bdf(0, 0x02, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x02, 1, 0)), sys.Node(Bdf(0, 0x01, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x06, 0, 0)), sys.Node(Bdf(0, 0x02, 1, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x07, 0, 0)), sys.Node(Bdf(0, 0x06, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x07, 1, 0)), sys.Node(Bdf(0, 0x06, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x08, 0, 1)), sys.Node(Bdf(0, 0x07, 0, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x09, 0, 0)), sys.Node(Bdf(0, 0x07, 1, 0)));
    EXPECT_EQ(Parent(sys, Bdf(0, 0x00, 1, 1))->Type(), TopoNodePciType::RootComplex);

    auto* path = sys.Path(Bdf(0, 0x05, 0, 0), Bdf(0, 0x09, 0, 0));
    ASSERT_NE(path, nullptr);
    EXPECT_EQ(path->Hops(), 7u);  // 04, 03, 02:00, 01, 02:01, 06, 07:01
    EXPECT_FALSE(path->CrossRootComplex());
  }
}

TEST(PciTree, UnprogrammedBridgeForwardsNoBuses) {
  // The NIC switch port reads 00/00/00 and its children are dead, as on a Pensando card that did
  // not come up.
  std::vector<SysfsPciDev> devs = Tray(0x06);
  devs.erase(
      std::remove_if(devs.begin(), devs.end(), [](const SysfsPciDev& d) { return d.bus >= 0x06; }),
      devs.end());
  devs.push_back(Bridge(0, 0x06, 0x00, 0, 0x00, 0x00, 0x00));
  SysfsPciDev dead = Bridge(0, 0x07, 0x00, 0, 0xff, 0xff, 0xff);
  dead.headerType = kHeaderDead;
  devs.push_back(dead);
  devs.push_back(Bridge(0, 0x00, 0x07, 1, 0x00, 0x0d, 0x0d, true));

  TopoSystemPci sys(devs);
  ExpectTree(sys, devs);

  TopoNodePci* port = sys.Node(Bdf(0, 0x06, 0, 0));
  ASSERT_NE(port, nullptr);
  EXPECT_TRUE(port->DownstreamPort().empty());
  EXPECT_EQ(port->UpstreamPort(), sys.Node(Bdf(0, 0x02, 1, 0)));
  EXPECT_EQ(sys.Node(Bdf(0, 0x07, 0, 0)), nullptr);
  // Bus 0 stays with the domain root.
  EXPECT_EQ(Parent(sys, Bdf(0, 0x00, 1, 1))->Type(), TopoNodePciType::RootComplex);
  EXPECT_EQ(Parent(sys, Bdf(0, 0x00, 7, 1))->Type(), TopoNodePciType::RootComplex);
}

TEST(PciTree, DeviceNoBridgeForwardsToGoesUnderItsDomainRoot) {
  // The NIC's own root port is missing from the scan.
  std::vector<SysfsPciDev> devs = Tray(0x06);
  devs.push_back(Fn(0, 0xc1, 0x00, 0, kClassEthernet));

  TopoSystemPci sys(devs);
  ExpectTree(sys, devs);

  TopoNodePci* domainRoot = sys.Node(Bdf(0, 0, 0, 0));
  ASSERT_NE(domainRoot, nullptr);
  EXPECT_EQ(Parent(sys, Bdf(0, 0xc1, 0, 0)), domainRoot);
  auto* path = sys.Path(Bdf(0, 0x05, 0, 0), Bdf(0, 0xc1, 0, 0));
  ASSERT_NE(path, nullptr);
  EXPECT_TRUE(path->CrossRootComplex());
}

TEST(PciTree, EndpointAtDomainRootTakesNoChildren) {
  // A passed-through two-port NIC on its own domain, with a GPU beside it.
  std::vector<SysfsPciDev> devs = Tray(0x06);
  devs.push_back(Fn(1, 0x00, 0x00, 0, kClassEthernet));
  devs.push_back(Fn(1, 0x00, 0x00, 1, kClassEthernet));
  devs.push_back(Fn(1, 0x00, 0x01, 0, kClassAccel));

  TopoSystemPci sys(devs);
  ExpectTree(sys, devs);

  TopoNodePci* nic0 = sys.Node(Bdf(1, 0, 0, 0));
  ASSERT_NE(nic0, nullptr);
  EXPECT_EQ(nic0->Type(), TopoNodePciType::Net);
  EXPECT_TRUE(nic0->DownstreamPort().empty());
  EXPECT_EQ(nic0->UpstreamPort()->Type(), TopoNodePciType::VirtualRoot);
  EXPECT_EQ(Parent(sys, Bdf(1, 0, 0, 1)), nic0->UpstreamPort());
  EXPECT_EQ(Parent(sys, Bdf(1, 0, 1, 0)), nic0->UpstreamPort());
  EXPECT_NE(sys.Path(Bdf(1, 0, 1, 0), Bdf(1, 0, 0, 1)), nullptr);
}
