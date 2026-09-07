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
#include "mori/application/topology/net.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

TopoSystemNet::TopoSystemNet() { Load(); }

TopoSystemNet::~TopoSystemNet() {}

PciBusId ParseBusIdFromSysfs(std::filesystem::path path) {
  // Regex to match PCI BDF like 0000:8c:00.0
  std::regex bdf_pattern(R"(^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$)");

  for (auto it = path; !it.empty(); it = it.parent_path()) {
    auto comp = it.filename().string();
    if (IsBdfString(comp)) return PciBusId(comp);
  }

  return PciBusId(0);
}

// Parse a CXI link-speed token (e.g. "ck400G", "BS200G", "cd50G") into Gbps.
// The encoding is a 2-char media prefix + decimal rate + 'G'; only the digits
// carry the rate, so extract the numeric run. Returns 0 if none is found.
static double ParseCxiLinkGbps(const std::string& token) {
  std::string digits;
  for (char c : token) {
    if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
  }
  if (digits.empty()) return 0.0;
  return static_cast<double>(std::stol(digits));
}

void TopoSystemNet::Load() {
  // Verbs (ibverbs) NICs. Absent on CXI/Slingshot-only hosts, so tolerate an
  // empty list or enumeration failure and fall back to the CXI scan below.
  try {
    application::RdmaContext rdma(application::RdmaBackendType::IBVerbs);
    auto devices = rdma.GetRdmaDeviceList();

    for (auto& dev : devices) {
      // TODO: finish nic plane
      TopoNodeNic* nic = new TopoNodeNic();
      auto rPath = std::filesystem::canonical(dev->GetIbvDevice()->ibdev_path);
      nic->name = dev->Name();
      nic->busId = ParseBusIdFromSysfs(rPath);
      nic->totalGbps = dev->TotalActiveGbps();

      nics.emplace_back(nic);
    }
  } catch (const std::exception& e) {
    MORI_APP_WARN("TopoSystemNet: ibverbs enumeration failed ({}); continuing with CXI scan",
                  e.what());
  }

  // Only probe CXI when ibverbs discovered no NICs (verbs-less Slingshot hosts).
  if (nics.empty()) LoadCxiNics();
}

void TopoSystemNet::LoadCxiNics() {
  // HPE Slingshot (Cassini) NICs are not exposed through ibverbs; the kernel
  // driver publishes them under /sys/class/cxi/cxiN, each with a `device`
  // symlink to its PCI node. libfabric's CXI provider names its domains the
  // same way (cxi0, cxi1, ...), so the sysfs entry name doubles as the NIC name.
  const std::filesystem::path cxiRoot{"/sys/class/cxi"};
  std::error_code ec;
  if (!std::filesystem::exists(cxiRoot, ec)) return;

  // Fallback when the link-speed attribute is missing/unreadable. Used only as a
  // relative ranking weight; when NICs share it, selection falls through to NUMA
  // then PCIe hops.
  constexpr double kCxiDefaultGbps = 200.0;

  for (const auto& entry : std::filesystem::directory_iterator(cxiRoot, ec)) {
    const std::filesystem::path devLink = entry.path() / "device";
    std::filesystem::path devPath = std::filesystem::canonical(devLink, ec);
    if (ec) {
      ec.clear();
      continue;
    }

    PciBusId busId = ParseBusIdFromSysfs(devPath);
    if (busId.packed == 0) continue;

    // e.g. /sys/class/cxi/cxi0/device/port/0/link/speed -> "ck400G".
    double gbps = kCxiDefaultGbps;
    std::ifstream speedFile(entry.path() / "device" / "port" / "0" / "link" / "speed");
    std::string speedToken;
    if (speedFile && (speedFile >> speedToken)) {
      double parsed = ParseCxiLinkGbps(speedToken);
      if (parsed > 0.0) gbps = parsed;
    }

    TopoNodeNic* nic = new TopoNodeNic();
    nic->name = entry.path().filename().string();
    nic->busId = busId;
    nic->totalGbps = gbps;
    nics.emplace_back(nic);
  }
}

std::vector<TopoNodeNic*> TopoSystemNet::GetNics() const {
  std::vector<TopoNodeNic*> v(nics.size());
  for (int i = 0; i < nics.size(); i++) v[i] = nics[i].get();
  return v;
}

}  // namespace application
}  // namespace mori
