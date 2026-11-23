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

// TopoSystemPci over a PCI tree built on disk, so the shapes a real host only
// shows by accident (a bridge left in reset) can be pinned down. No hardware.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mori/application/topology/pci.hpp"

namespace {

using mori::application::PciBusId;
using mori::application::TopoNodePci;
using mori::application::TopoNodePciType;
using mori::application::TopoSystemPci;

constexpr uint32_t kClassBridge = 0x060400;
constexpr uint32_t kClassNic = 0x020000;
constexpr uint32_t kClassGpu = 0x120000;

// The bus registers of a bridge: where it sits, and the buses behind it.
struct BusNumbers {
  uint8_t primary = 0;
  uint8_t secondary = 0;
  uint8_t subordinate = 0;
};

// The part of sysfs TopoSystemPci reads: <root>/bus/pci/devices/<bdf> linking
// into <root>/devices/<pci root>/<bdf>/.../<bdf>, each with class, numa_node
// and the first 64 bytes of configuration space.
class FakeSysfs {
 public:
  FakeSysfs() {
    root_ = std::filesystem::temp_directory_path() /
            ("mori_topo_pci_" + std::to_string(::getpid()) + "_" + std::to_string(counter_++));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_ / "bus/pci/devices");
  }
  ~FakeSysfs() { std::filesystem::remove_all(root_); }

  // `path` is the device's place in the hierarchy, e.g.
  // "pci0000:10/0000:10:01.1/0000:11:00.0".
  void AddBridge(const std::string& path, BusNumbers buses) {
    Add(path, kClassBridge, 0x01, buses);
  }
  void AddDevice(const std::string& path, uint32_t device_class) {
    Add(path, device_class, 0x00, BusNumbers{});
  }

  std::string Devices() const { return (root_ / "bus/pci/devices").string(); }

 private:
  void Add(const std::string& path, uint32_t device_class, uint8_t header_type, BusNumbers buses) {
    const std::filesystem::path dir = root_ / "devices" / path;
    std::filesystem::create_directories(dir);
    char text[16];
    snprintf(text, sizeof(text), "0x%06x\n", device_class);
    std::ofstream(dir / "class") << text;
    std::ofstream(dir / "numa_node") << "0\n";
    std::vector<char> config(64, 0);
    config[0x0e] = static_cast<char>(header_type);
    config[0x18] = static_cast<char>(buses.primary);
    config[0x19] = static_cast<char>(buses.secondary);
    config[0x1a] = static_cast<char>(buses.subordinate);
    std::ofstream(dir / "config", std::ios::binary).write(config.data(), config.size());
    std::filesystem::create_symlink(std::filesystem::path("../../../devices") / path,
                                    root_ / "bus/pci/devices" / dir.filename());
  }

  static inline int counter_ = 0;
  std::filesystem::path root_;
};

// One root port on bus 0x10 leading to a switch: a GPU behind one downstream
// port, and behind another a second switch (0000:16:00.0) with a NIC below it.
// A NIC on bus 0 hangs off the root complex directly.
void AddSwitchTree(FakeSysfs* sysfs, BusNumbers inner_switch) {
  const std::string root_port = "pci0000:10/0000:10:01.1";
  const std::string upstream = root_port + "/0000:11:00.0";
  sysfs->AddDevice("pci0000:00/0000:00:01.0", kClassNic);
  sysfs->AddBridge(root_port, BusNumbers{0x10, 0x11, 0x1a});
  sysfs->AddBridge(upstream, BusNumbers{0x11, 0x12, 0x1a});
  sysfs->AddBridge(upstream + "/0000:12:00.0", BusNumbers{0x12, 0x13, 0x13});
  sysfs->AddDevice(upstream + "/0000:12:00.0/0000:13:00.0", kClassGpu);
  sysfs->AddBridge(upstream + "/0000:12:01.0", BusNumbers{0x12, 0x16, 0x19});
  sysfs->AddBridge(upstream + "/0000:12:01.0/0000:16:00.0", inner_switch);
  sysfs->AddDevice(upstream + "/0000:12:01.0/0000:16:00.0/0000:17:00.0", kClassNic);
}

TopoNodePci* Upstream(TopoSystemPci* pci, const std::string& bdf) {
  TopoNodePci* node = pci->Node(PciBusId(bdf));
  return node == nullptr ? nullptr : node->UpstreamPort();
}

TEST(TopoSystemPci, BridgesNestByTheirBusRanges) {
  FakeSysfs sysfs;
  AddSwitchTree(&sysfs, BusNumbers{0x16, 0x17, 0x17});
  TopoSystemPci pci(sysfs.Devices());

  EXPECT_EQ(Upstream(&pci, "0000:13:00.0"), pci.Node(PciBusId("0000:12:00.0")));
  EXPECT_EQ(Upstream(&pci, "0000:16:00.0"), pci.Node(PciBusId("0000:12:01.0")));
  EXPECT_EQ(Upstream(&pci, "0000:17:00.0"), pci.Node(PciBusId("0000:16:00.0")));
  ASSERT_NE(Upstream(&pci, "0000:00:01.0"), nullptr);
  EXPECT_EQ(Upstream(&pci, "0000:00:01.0")->Type(), TopoNodePciType::RootComplex);
  EXPECT_NE(pci.Path(PciBusId("0000:13:00.0"), PciBusId("0000:17:00.0")), nullptr);
}

// A switch left in reset reads its primary, secondary and subordinate bus as
// 0. Taken at face value that makes it the bridge to bus 0, which is the root
// bus: everything there would hang below it, and what is really behind it
// would have no parent.
TEST(TopoSystemPci, ABridgeWithoutBusNumbersClaimsNoBus) {
  FakeSysfs sysfs;
  AddSwitchTree(&sysfs, BusNumbers{0, 0, 0});
  TopoSystemPci pci(sysfs.Devices());

  // Bus 0 is still the root complex's.
  ASSERT_NE(Upstream(&pci, "0000:00:01.0"), nullptr);
  EXPECT_EQ(Upstream(&pci, "0000:00:01.0")->Type(), TopoNodePciType::RootComplex);
  // The bridge stays where its parent's bus range puts it.
  EXPECT_EQ(Upstream(&pci, "0000:16:00.0"), pci.Node(PciBusId("0000:12:01.0")));
  // The device behind it is placed by where sysfs enumerated it.
  EXPECT_EQ(Upstream(&pci, "0000:17:00.0"), pci.Node(PciBusId("0000:16:00.0")));
  // The rest of the tree is untouched.
  EXPECT_EQ(Upstream(&pci, "0000:13:00.0"), pci.Node(PciBusId("0000:12:00.0")));
  EXPECT_NE(pci.Path(PciBusId("0000:13:00.0"), PciBusId("0000:17:00.0")), nullptr);
}

}  // namespace
