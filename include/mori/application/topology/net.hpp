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
  PciBusId busId{0};
  double totalGbps{0};
};

class TopoSystemNet {
 public:
  TopoSystemNet();
  ~TopoSystemNet();

 private:
  void Load();

 private:
  std::vector<std::unique_ptr<TopoNodeNic>> nics;
};

}  // namespace application
}  // namespace mori