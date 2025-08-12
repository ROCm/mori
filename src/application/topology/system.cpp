#include "mori/application/topology/system.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoSystem                                           */
/* ---------------------------------------------------------------------------------------------- */
TopoSystem::TopoSystem() { Load(); }

TopoSystem::~TopoSystem() {}

void TopoSystem::Load() {
  gpu.reset(new TopoSystemGpu());
  pci.reset(new TopoSystemPci());
  net.reset(new TopoSystemNet());
}

std::string TopoSystem::MatchGpuAndNic(int id) const {
  auto nics = net->GetNics();
  TopoNodeGpu* d = gpu->GetGpuByLogicalId(id);
  int minHops = std::numeric_limits<int>::max();
  std::string best;

  for (auto* nic : nics) {
    TopoPathPci* path = pci->Path(d->busId, nic->busId);
    if (!path) continue;
    if (path->Hops() < minHops) {
      minHops = path->Hops();
      best = nic->name;
    }
  }

  return best;
}

std::vector<std::string> TopoSystem::MatchAllGpusAndNics() const {
  int count;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&count));

  auto nics = net->GetNics();

  std::vector<std::string> matches(count);
  for (int i = 0; i < count; i++) {
    matches[i] = MatchGpuAndNic(i);
  }

  return matches;
}

}  // namespace application
}  // namespace mori