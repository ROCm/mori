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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/mori_log.hpp"

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
  if (dev == nullptr) {
    MORI_APP_WARN("Topology: GPU logical id {} not found; no NIC candidates", id);
    return {};
  }
  TopoNodePci* devPci = pci->Node(dev->busId);
  if (devPci == nullptr) {
    MORI_APP_WARN("Topology: GPU {} bdf {} not in PCI tree; no NIC candidates", id,
                  dev->busId.String());
    return {};
  }
  NumaNodeId gpuNumaNodeId = devPci->NumaNode();

  // Collect nic candidates
  auto nics = net->GetNics();
  std::vector<Candidate> candidates;
  for (auto* nic : nics) {
    TopoNodePci* nicPci = pci->Node(nic->busId);
    if (!nicPci) continue;  // NIC BDF not present in the scanned PCI tree
    TopoPathPci* path = pci->Path(dev->busId, nic->busId);
    if (!path) continue;
    candidates.push_back({path, nicPci, nic});
  }

  // Sort by 1) speed 2) numa 3) hops 4) name
  std::sort(candidates.begin(), candidates.end(),
            [&gpuNumaNodeId](Candidate a, Candidate b) -> bool {
              bool tie = (a.nic->totalGbps == b.nic->totalGbps);
              if (!tie) return a.nic->totalGbps > b.nic->totalGbps;

              if ((a.node->NumaNode() == gpuNumaNodeId) && (b.node->NumaNode() != gpuNumaNodeId))
                return true;
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

std::vector<std::string> TopoSystem::MatchGpuAndNics(int id, int k) {
  if (k <= 0) return {};

  std::vector<Candidate> candidates = CollectAndSortCandidates(this, id);
  std::vector<std::string> matches;
  matches.reserve(std::min<int>(k, candidates.size()));

  std::unordered_set<std::string> seen;
  for (const auto& candidate : candidates) {
    if (candidate.nic == nullptr) continue;
    const std::string& name = candidate.nic->name;
    if (!seen.insert(name).second) continue;
    matches.push_back(name);
    if (static_cast<int>(matches.size()) >= k) break;
  }

  return matches;
}

std::vector<std::string> TopoSystem::MatchCpuNics(int numaNode, int k) {
  if (k <= 0) return {};

  TopoSystemPci* pci = GetTopoSystemPci();
  TopoSystemNet* net = GetTopoSystemNet();

  auto nics = net->GetNics();
  std::vector<TopoNodeNic*> candidates;
  candidates.reserve(nics.size());
  for (auto* nic : nics) {
    if (nic == nullptr) continue;
    candidates.push_back(nic);
  }

  std::sort(candidates.begin(), candidates.end(), [&](TopoNodeNic* a, TopoNodeNic* b) -> bool {
    if (numaNode >= 0) {
      TopoNodePci* aNode = pci->Node(a->busId);
      TopoNodePci* bNode = pci->Node(b->busId);
      const bool aLocal = (aNode != nullptr) && (aNode->NumaNode() == numaNode);
      const bool bLocal = (bNode != nullptr) && (bNode->NumaNode() == numaNode);
      if (aLocal != bLocal) return aLocal;
    }

    bool tie = (a->totalGbps == b->totalGbps);
    if (!tie) return a->totalGbps > b->totalGbps;

    return a->name <= b->name;
  });

  std::vector<std::string> matches;
  matches.reserve(std::min<int>(k, candidates.size()));
  std::unordered_set<std::string> seen;
  for (const auto* nic : candidates) {
    if (nic == nullptr) continue;
    if (!seen.insert(nic->name).second) continue;
    matches.push_back(nic->name);
    if (static_cast<int>(matches.size()) >= k) break;
  }

  return matches;
}

std::vector<std::string> TopoSystem::MatchAllGpusAndNics() {
  int count;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&count));

  std::vector<std::vector<Candidate>> perGpu(count);
  std::unordered_set<std::string> nicNames;
  for (int i = 0; i < count; i++) {
    perGpu[i] = CollectAndSortCandidates(this, i);
    for (auto& cand : perGpu[i]) nicNames.insert(cand.nic->name);
  }

  std::vector<std::string> matches(count);
  if (nicNames.empty()) return matches;

  // Each NIC serves at most ceil(GPUs / NICs) GPUs. Pairs are assigned closest-first across all
  // GPUs, so a NIC goes to the GPUs under its own PCIe switch before any farther GPU can take it.
  const size_t cap = (count + nicNames.size() - 1) / nicNames.size();

  struct Pair {
    int gpu;
    size_t rank;
    const Candidate* cand;
  };
  std::vector<Pair> pairs;
  for (int i = 0; i < count; i++) {
    for (size_t r = 0; r < perGpu[i].size(); r++) pairs.push_back({i, r, &perGpu[i][r]});
  }
  std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
    if (a.cand->nic->totalGbps != b.cand->nic->totalGbps)
      return a.cand->nic->totalGbps > b.cand->nic->totalGbps;
    if (a.cand->path->Hops() != b.cand->path->Hops())
      return a.cand->path->Hops() < b.cand->path->Hops();
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.gpu < b.gpu;
  });

  std::unordered_map<std::string, size_t> load;
  std::vector<bool> assigned(count, false);
  for (const auto& p : pairs) {
    if (assigned[p.gpu]) continue;
    const std::string& name = p.cand->nic->name;
    if (load[name] >= cap) continue;
    matches[p.gpu] = name;
    load[name]++;
    assigned[p.gpu] = true;
  }

  for (int i = 0; i < count; i++) {
    if (!assigned[i] && !perGpu[i].empty()) matches[i] = perGpu[i][0].nic->name;
  }

  return matches;
}

}  // namespace application
}  // namespace mori
