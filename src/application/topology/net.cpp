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

#include <filesystem>
#include <regex>
#include <string>
#include <utility>

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

void TopoSystemNet::Load() {
  // Verbs (ibverbs) NICs. Non-verbs fabrics (e.g. HPE Slingshot/CXI) are
  // discovered by their transport backend and injected via AddNic().
  try {
    application::RdmaContext rdma(application::RdmaBackendType::IBVerbs);
    auto devices = rdma.GetRdmaDeviceList();

    for (auto& dev : devices) {
      // TODO: finish nic plane
      auto nic = std::make_unique<TopoNodeNic>();
      auto rPath = std::filesystem::canonical(dev->GetIbvDevice()->ibdev_path);
      nic->name = dev->Name();
      nic->busId = ParseBusIdFromSysfs(rPath);
      nic->totalGbps = dev->TotalActiveGbps();

      nics.emplace_back(std::move(nic));
    }
  } catch (const std::exception& e) {
    MORI_APP_WARN("TopoSystemNet: ibverbs enumeration failed ({}); continuing without verbs NICs",
                  e.what());
  }
}

void TopoSystemNet::AddNic(std::string name, PciBusId busId, double totalGbps) {
  auto nic = std::make_unique<TopoNodeNic>();
  nic->name = std::move(name);
  nic->busId = busId;
  nic->totalGbps = totalGbps;
  nics.emplace_back(std::move(nic));
}

std::vector<TopoNodeNic*> TopoSystemNet::GetNics() const {
  std::vector<TopoNodeNic*> v(nics.size());
  for (int i = 0; i < nics.size(); i++) v[i] = nics[i].get();
  return v;
}

}  // namespace application
}  // namespace mori
