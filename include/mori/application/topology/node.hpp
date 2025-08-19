#pragma once

namespace mori {
namespace application {

using NumaNodeId = int32_t;

class TopoNode {
 public:
  TopoNode() = default;
  virtual ~TopoNode() = default;
};
}  // namespace application
}  // namespace mori