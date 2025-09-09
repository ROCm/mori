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

struct Candidate {
  TopoPathPci* path{nullptr};
  TopoNodePci* node{nullptr};
  TopoNodeNic* nic{nullptr};
};

std::vector<Candidate> CollectAndSortCandidates(TopoSystem* sys, int id) {
  assert(sys != nullptr);

  TopoSystemGpu* gpu = sys->GetTopoSystemGpu();
  TopoSystemPci* pci = sys->GetTopoSystemPci();
  TopoSystemNet* net = sys->GetTopoSystemNet();

  TopoNodeGpu* dev = gpu->GetGpuByLogicalId(id);
  NumaNodeId gpuNumaNodeId = pci->Node(dev->busId)->NumaNode();

  // Collect nic candidates
  auto nics = net->GetNics();
  std::vector<Candidate> candidates;
  for (auto* nic : nics) {
    TopoPathPci* path = pci->Path(dev->busId, nic->busId);
    TopoNodePci* nicPci = pci->Node(nic->busId);
    if (!path) continue;
    candidates.push_back({path, nicPci, nic});
  }

  // Sort by 1) speed 2) numa 3) hops 4) name
  std::sort(candidates.begin(), candidates.end(), [&gpuNumaNodeId](Candidate a, Candidate b) -> bool {
    bool tie = (a.nic->totalGbps == b.nic->totalGbps);
    if (!tie) return a.nic->totalGbps > b.nic->totalGbps;

    if ((a.node->NumaNode() == gpuNumaNodeId) && (b.node->NumaNode() != gpuNumaNodeId)) return true;
    if ((a.node->NumaNode() != gpuNumaNodeId) && (b.node->NumaNode() == gpuNumaNodeId))
      return false;

    tie = (a.path->Hops() == b.path->Hops());
    if (!tie) return a.path->Hops() <= b.path->Hops();

    return a.nic->name <= b.nic->name;
  });

  return candidates;
}

std::string TopoSystem::MatchGpuAndNic(int id) {
  std::vector<std::string> matches = MatchAllGpusAndNics();
  assert(id < matches.size());
  return matches[id];
}

std::vector<std::string> TopoSystem::MatchAllGpusAndNics() {
  int count;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&count));

  std::vector<std::string> matches;
  std::unordered_set<std::string> matched;

  for (int i = 0; i < count; i++) {
    std::vector<Candidate> candidates = CollectAndSortCandidates(this, i);
    if (candidates.empty()) {
      matches.push_back("");
      continue;
    }

    bool found = false;
    for (auto& cand : candidates) {
      std::string name = cand.nic->name;
      if (matched.find(name) == matched.end()) {
        matches.push_back(name);
        matched.insert(name);
        found = true;
        break;
      }
    }

    if (!found) matches.push_back(candidates[i % candidates.size()].nic->name);
  }

  return matches;
}

}  // namespace application
}  // namespace mori
