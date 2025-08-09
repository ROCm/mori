#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "mori/application/topology/gpu.hpp"
#include "mori/application/topology/net.hpp"
#include "mori/application/topology/pci.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoSystem                                           */
/* ---------------------------------------------------------------------------------------------- */
class TopoSystem {
 public:
  TopoSystem();
  ~TopoSystem();

 private:
  void Load();

 public:
  std::unique_ptr<TopoSystemGpu> gpu;
  std::unique_ptr<TopoSystemPci> pci;
  std::unique_ptr<TopoSystemNet> net;
};

}  // namespace application
}  // namespace mori