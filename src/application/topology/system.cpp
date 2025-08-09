#include "mori/application/topology/system.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoSystem                                           */
/* ---------------------------------------------------------------------------------------------- */
TopoSystem::TopoSystem() { Load(); }

TopoSystem::~TopoSystem() {}

void TopoSystem::Load() {
  gpu.reset(new TopoSystemGpu());
  pci.reset(new TopoSystemPci());
  net.reset(new TopoSystemNet());
}

}  // namespace application
}  // namespace mori