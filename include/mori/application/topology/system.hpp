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

  TopoSystemGpu* GetTopoSystemGpu() { return gpu.get(); }
  TopoSystemPci* GetTopoSystemPci() { return pci.get(); }
  TopoSystemNet* GetTopoSystemNet() { return net.get(); }

 private:
  void Load();

 private:
  std::unique_ptr<TopoSystemGpu> gpu;
  std::unique_ptr<TopoSystemPci> pci;
  std::unique_ptr<TopoSystemNet> net;
};

}  // namespace application
}  // namespace mori