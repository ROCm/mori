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

  using CandType = std::pair<TopoPathPci*, TopoNodeNic*>;
  std::vector<CandType> candidates;
  for (auto* nic : nics) {
    TopoPathPci* path = pci->Path(d->busId, nic->busId);
    if (!path) continue;
    candidates.push_back({path, nic});
  }

  std::sort(candidates.begin(), candidates.end(), [](CandType a, CandType b) {
    if (a.second->totalGbps == b.second->totalGbps) {
      return a.first->Hops() <= b.first->Hops();
    }
    return a.second->totalGbps > b.second->totalGbps;
  });

  if (candidates.empty()) return "";
  return candidates[0].second->name;
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