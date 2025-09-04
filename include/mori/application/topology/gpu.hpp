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
#pragma once

#include <memory>
#include <vector>

#include "mori/application/topology/node.hpp"
#include "mori/application/topology/pci.hpp"
#include "rocm_smi/rocm_smi.h"

namespace mori {
namespace application {
/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoNodeGpu                                          */
/* ---------------------------------------------------------------------------------------------- */
class TopoNodeGpu;

class TopoNodeGpuP2pLink : public TopoNode {
 public:
  TopoNodeGpuP2pLink() = default;
  ~TopoNodeGpuP2pLink() = default;

 public:
  RSMI_IO_LINK_TYPE type;
  uint64_t hops{0};
  uint64_t weight{0};

  TopoNodeGpu* gpu1{nullptr};
  TopoNodeGpu* gpu2{nullptr};
};

class TopoNodeGpu : public TopoNode {
 public:
  TopoNodeGpu() = default;
  ~TopoNodeGpu() = default;

 public:
  PciBusId busId{0};
  std::vector<TopoNodeGpuP2pLink*> p2ps;
};

class TopoSystemGpu {
 public:
  TopoSystemGpu();
  ~TopoSystemGpu();

  int NumGpus() const { return gpus.size(); }
  std::vector<TopoNodeGpu*> GetGpus() const;
  TopoNodeGpu* GetGpuByLogicalId(int) const;

 private:
  void Load();

 private:
  std::vector<std::unique_ptr<TopoNodeGpu>> gpus;
  std::vector<std::unique_ptr<TopoNodeGpuP2pLink>> p2ps;
};

}  // namespace application
}  // namespace mori
