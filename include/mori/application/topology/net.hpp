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