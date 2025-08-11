#include "mori/application/topology/net.hpp"

#include <filesystem>
#include <regex>

#include "mori/application/transport/rdma/rdma.hpp"

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
  application::RdmaContext rdma(application::RdmaBackendType::IBVerbs);
  auto devices = rdma.GetRdmaDeviceList();

  for (auto& dev : devices) {
    // TODO: finish nic plane
    TopoNodeNic* nic = new TopoNodeNic();
    auto rPath = std::filesystem::canonical(dev->GetIbvDevice()->ibdev_path);
    nic->busId = ParseBusIdFromSysfs(rPath);
    nic->totalGbps = dev->TotalActiveGbps();

    nics.emplace_back(nic);
    printf("nic %s gbps %f path %s bdf %s\n", dev->Name().c_str(), nic->totalGbps, rPath.c_str(),
           nic->busId.String().c_str());
  }
}

std::vector<TopoNodeNic*> TopoSystemNet::GetNICs() const {
  std::vector<TopoNodeNic*> v(nics.size());
  for (int i = 0; i < nics.size(); i++) v[i] = nics[i].get();
  return v;
}

}  // namespace application
}  // namespace mori