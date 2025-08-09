#pragma once

namespace mori {
namespace application {

using PciBusId = uint64_t;
using NumaNodeId = int32_t;
/* ---------------------------------------------------------------------------------------------- */
/*                                            TopoNode                                            */
/* ---------------------------------------------------------------------------------------------- */
class TopoNode {
 public:
  TopoNode() = default;
  virtual ~TopoNode() = default;
};
}  // namespace application
}  // namespace mori