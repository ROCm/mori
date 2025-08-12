#include <dlfcn.h>

#include <cassert>

#include "mori/application/topology/topology.hpp"

int TestTopoNodeGpu() {
  mori::application::TopoSystem sys{};
  auto* gpuSys = sys.GetTopoSystemGpu();
  auto* netSys = sys.GetTopoSystemNet();
  auto* pciSys = sys.GetTopoSystemPci();

  auto gpus = gpuSys->GetGpus();
  auto nics = netSys->GetNics();

  for (auto* gpu : gpus) {
    assert(pciSys->Node(gpu->busId));
    for (auto* nic : nics) {
      assert(pciSys->Node(nic->busId));
      auto* path = pciSys->Path(gpu->busId, nic->busId);
      if (!path) {
        printf("gpu %s nic %s no direct link\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str());
      } else {
        printf("gpu %s nic %s %s hops %d\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str(), nic->name.c_str(), path->Hops());
      }
    }
  }

  std::vector<std::string> matches = sys.MatchAllGpusAndNics();
  for (int i = 0; i < matches.size(); i++) {
    printf("gpu %d matches %s\n", i, matches[i].c_str());
  }

  return 0;
}

int main() { return TestTopoNodeGpu(); }