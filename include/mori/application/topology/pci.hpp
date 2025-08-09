#pragma once
#include <stdint.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "mori/application/topology/node.hpp"

namespace mori {
namespace application {

struct PciBusId {
  PciBusId(uint64_t v) : packed(v) {}
  PciBusId(uint16_t domain, uint8_t bus, uint8_t dev, uint8_t func) {
    packed = (static_cast<uint64_t>(domain) << 16) | (static_cast<uint64_t>(bus) << 8) |
             (static_cast<uint64_t>(dev) << 3) | (static_cast<uint64_t>(func));
  }
  ~PciBusId() = default;

  uint16_t Domain() const { return (packed >> 16); }
  uint8_t Bus() const { return (packed >> 8); }
  uint8_t Dev() const { return (packed >> 3); }
  uint8_t Func() const { return (packed & 0x7); }

  bool operator==(const PciBusId& rhs) { return packed == rhs.packed; }
  explicit operator uint64_t() const { return packed; }

  uint64_t packed{0};
};

PciBusId PackPciBusId(uint16_t domain, uint8_t bus, uint8_t);

enum class TopoNodePciType {
  VirtualRoot = 0,  // A vritual root to traverse pci tree
  RootComplex = 1,
  Bridge = 2,
  Gpu = 3,
  Net = 4,
  Unknown = 9,
};

class TopoNodePci : public TopoNode {
 public:
  TopoNodePci() = default;
  ~TopoNodePci() = default;

  static TopoNodePci* CreateVirtualRoot();
  static TopoNodePci* CreateRootComplex(PciBusId, NumaNodeId, TopoNodePci* vr);
  static TopoNodePci* CreateBridge(PciBusId, NumaNodeId);
  static TopoNodePci* CreateGpu(PciBusId, NumaNodeId);
  static TopoNodePci* CreateNet(PciBusId, NumaNodeId);

  TopoNodePciType Type() const { return type; }
  PciBusId BusId() const { return busId; }
  NumaNodeId NumaNode() const { return numaNode; }
  TopoNodePci* UpstreamPort() const { return usp; }
  const std::vector<TopoNodePci*>& DownstreamPort() const { return dsps; }

  void SetUpstreamPort(TopoNodePci*);
  void AddDownstreamPort(TopoNodePci*);
  void RemoveDownstreamPort(PciBusId);

 private:
  TopoNodePciType type{TopoNodePciType::Unknown};
  PciBusId busId{0};
  NumaNodeId numaNode{-1};
  TopoNodePci* usp{nullptr};
  std::vector<TopoNodePci*> dsps;
};

class TopoSystemPci {
 public:
  TopoSystemPci();
  ~TopoSystemPci();

 private:
  void Load();
  void Validate();

 private:
  std::unordered_map<uint64_t, std::unique_ptr<TopoNodePci>> pcis;
  TopoNodePci* root{nullptr};
};

}  // namespace application
}  // namespace mori