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

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoNodeNic                                          */
/* ---------------------------------------------------------------------------------------------- */
class TopoNodeNic : public TopoNode {
 public:
  TopoNodeNic() = default;
  ~TopoNodeNic() = default;

 public:
  std::string name{};
  PciBusId busId{0};
  double totalGbps{0};
};

class TopoSystemNet {
 public:
  TopoSystemNet();
  ~TopoSystemNet();

  std::vector<TopoNodeNic*> GetNics() const;

 private:
  void Load();

 private:
  std::vector<std::unique_ptr<TopoNodeNic>> nics;
};

}  // namespace application
}  // namespace mori
