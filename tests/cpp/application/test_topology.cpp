#include <dlfcn.h>

#include <cassert>

#include "mori/application/topology/topology.hpp"

int TestTopoNodeGpu() {
  mori::application::TopoSystem sys{};
  auto* gpuSys = sys.GetTopoSystemGpu();
  auto* netSys = sys.GetTopoSystemNet();
  auto* pciSys = sys.GetTopoSystemPci();

  auto gpus = gpuSys->GetGpus();
  auto nics = netSys->GetNICs();

  for (auto* gpu : gpus) {
    assert(pciSys->Node(gpu->busId));
    for (auto* nic : nics) {
      assert(pciSys->Node(nic->busId));
      auto* path = pciSys->Path(gpu->busId, nic->busId);
      if (!path) {
        printf("gpu %s nic %s no direct link\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str());
      } else {
        printf("gpu %s nic %s hops %d\n", gpu->busId.String().c_str(), nic->busId.String().c_str(),
               path->Hops());
      }
    }
  }

  // for (auto* nic : net->GetNICs()) {
  //   printf("bdf %s rate %f\n", nic->busId.String().c_str(), nic->totalGbps);
  // }

  return 0;
}

int main() { return TestTopoNodeGpu(); }