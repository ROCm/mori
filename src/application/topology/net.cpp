#include "mori/application/topology/net.hpp"

#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

TopoSystemNet::TopoSystemNet() { Load(); }

TopoSystemNet::~TopoSystemNet() {}

void TopoSystemNet::Load() {
  application::RdmaContext rdma(application::RdmaBackendType::IBVerbs);
  auto devices = rdma.GetRdmaDeviceList();

  for (auto& dev : devices) {
    // TODO: finish nic plane
    TopoNodeNic* nic = new TopoNodeNic();
    nic->totalGbps = dev->TotalActiveGbps();

    printf("nic %s gbps %f path %s\n", dev->Name().c_str(), nic->totalGbps,
           dev->GetIbvDevice()->ibdev_path);
  }
}

}  // namespace application
}  // namespace mori