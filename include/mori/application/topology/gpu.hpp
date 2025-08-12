#pragma once

#include <memory>
#include <vector>

#include "mori/application/topology/node.hpp"
#include "mori/application/topology/pci.hpp"
#include "rocm_smi/rocm_smi.h"

namespace mori {
namespace application {
/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoNodeGpu                                          */
/* ---------------------------------------------------------------------------------------------- */
class TopoNodeGpu;

class TopoNodeGpuP2pLink : public TopoNode {
 public:
  TopoNodeGpuP2pLink() = default;
  ~TopoNodeGpuP2pLink() = default;

 public:
  RSMI_IO_LINK_TYPE type;
  uint64_t hops{0};
  uint64_t weight{0};

  TopoNodeGpu* gpu1{nullptr};
  TopoNodeGpu* gpu2{nullptr};
};

class TopoNodeGpu : public TopoNode {
 public:
  TopoNodeGpu() = default;
  ~TopoNodeGpu() = default;

 public:
  PciBusId busId{0};
  std::vector<TopoNodeGpuP2pLink*> p2ps;
};

class TopoSystemGpu {
 public:
  TopoSystemGpu();
  ~TopoSystemGpu();

  std::vector<TopoNodeGpu*> GetGpus() const;
  TopoNodeGpu* GetGpuByLogicalId(int) const;

 private:
  void Load();

 private:
  std::vector<std::unique_ptr<TopoNodeGpu>> gpus;
  std::vector<std::unique_ptr<TopoNodeGpuP2pLink>> p2ps;
};

}  // namespace application
}  // namespace mori