#pragma once
#include <stdint.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "mori/application/topology/node.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoNodePci                                          */
/* ---------------------------------------------------------------------------------------------- */
class TopoNodePci : public TopoNode {
 public:
  TopoNodePci() = default;
  virtual ~TopoNodePci() = default;

 public:
  PciBusId busId{0};
  NumaNodeId numaNode{0};
  TopoNodePci* usp{nullptr};
  std::vector<TopoNodePci*> dsps;
};

class TopoSystemPci {
 public:
  TopoSystemPci();
  ~TopoSystemPci();

 private:
  void Load();

 private:
  std::unordered_map<PciBusId, std::unique_ptr<TopoNodePci>> pcis;
  std::vector<TopoNodePci*> rcs;
};

}  // namespace application
}  // namespace mori