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
      auto* nicPci = pciSys->Node(nic->busId);
      if (!path) {
        printf("gpu %s nic %s no direct link\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str());
      } else {
        printf("gpu %s nic %s %s hops %d speed %f numa %d\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str(), nic->name.c_str(), path->Hops(), nic->totalGbps,
               nicPci->NumaNode());
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
