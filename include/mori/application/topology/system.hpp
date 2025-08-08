#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace mori {
namespace io {

using PciBusId = uint64_t;
using NumaNodeId = int32_t;
/* ---------------------------------------------------------------------------------------------- */
/*                                            TopoNode                                            */
/* ---------------------------------------------------------------------------------------------- */
class TopoNode {
 public:
  TopoNode() = default;
  virtual ~TopoNode() = default;

 public:
};

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

class TopoNodeGpu : public TopoNodePci {
 public:
  TopoNodeGpu() = default;
  ~TopoNodeGpu() = default;

 public:
  int gpuId{-1};
  std::vector<TopoNodeGpuP2pLink*> p2ps;
};

class TopoSystem {
 public:
  TopoSystem();
  ~TopoSystem();

  void Load();

 private:
  void InitTopoGpuPlane();
  void InitTopoPciPlane();
  void InitTopoCpuPlane();

 public:
  // Gpu plane
  std::vector<std::unique_ptr<TopoNodeGpu>> gpus;
  std::vector<std::unique_ptr<TopoNodeGpuP2pLink>> p2ps;
  // PCI plane
  std::unordered_map<PciBusId, std::unique_ptr<TopoNodePci>> pcis;
  std::vector<TopoNodePci*> rcs;
};

}  // namespace io
}  // namespace mori