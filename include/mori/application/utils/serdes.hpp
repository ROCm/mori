#pragma once

#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

class RdmaEndpointHandlePacker {
 public:
  RdmaEndpointHandlePacker();
  ~RdmaEndpointHandlePacker();
  size_t PackedSizeCompact() const;
  void PackCompact(const RdmaEndpointHandle&, void* packed);
  void UnpackCompact(RdmaEndpointHandle&, void* packed);
};

}  // namespace application
}  // namespace mori