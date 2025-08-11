#pragma once
#include <stdint.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "mori/application/topology/node.hpp"

namespace std {
template <>
struct hash<std::pair<uint64_t, uint64_t>> {
  size_t operator()(const std::pair<uint64_t, uint64_t>& p) const noexcept {
    size_t h1 = std::hash<uint64_t>{}(p.first);
    size_t h2 = std::hash<uint64_t>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};
}  // namespace std

namespace mori {
namespace application {

// Valid bdf string: xxxx:xx:xx.x
bool IsBdfString(const std::string& s);
// Return empty vector when bdf is invalid
std::vector<uint64_t> ParseBdfFromString(const std::string&);
std::string BdfToString(uint64_t domain, uint64_t bus, uint64_t dev, uint64_t func);

struct PciBusId {
  PciBusId(uint64_t v) : packed(v) {}
  PciBusId(uint16_t domain, uint8_t bus, uint8_t dev, uint8_t func)
      : packed(Pack(domain, bus, dev, func)) {}
  PciBusId(const std::string& str) {
    std::vector<uint64_t> dbdf = ParseBdfFromString(str);
    if (dbdf.size() != 4) {
      packed = 0;
      return;
    }
    packed = Pack(dbdf[0], dbdf[1], dbdf[2], dbdf[3]);
  }

  ~PciBusId() = default;

  static uint64_t Pack(uint16_t domain, uint8_t bus, uint8_t dev, uint8_t func) {
    return (static_cast<uint64_t>(domain) << 16) | (static_cast<uint64_t>(bus) << 8) |
           (static_cast<uint64_t>(dev) << 3) | (static_cast<uint64_t>(func));
  }

  uint16_t Domain() const { return (packed >> 16); }
  uint8_t Bus() const { return (packed >> 8); }
  uint8_t Dev() const { return ((packed >> 3) & 0x1f); }
  uint8_t Func() const { return (packed & 0x7); }
  std::string String() const { return BdfToString(Domain(), Bus(), Dev(), Func()); }

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
  Others = 8,
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
  static TopoNodePci* CreateOthers(PciBusId, NumaNodeId);

  TopoNodePciType Type() const { return type; }
  PciBusId BusId() const { return busId; }
  NumaNodeId NumaNode() const { return numaNode; }
  TopoNodePci* UpstreamPort() const { return usp; }
  const std::vector<TopoNodePci*>& DownstreamPort() const { return dsps; }

  void SetUpstreamPort(TopoNodePci*);
  void AddDownstreamPort(TopoNodePci*);
  void RemoveDownstreamPort(PciBusId);

 private:
  TopoNodePciType type{TopoNodePciType::Others};
  PciBusId busId{0};
  NumaNodeId numaNode{-1};
  TopoNodePci* usp{nullptr};
  std::vector<TopoNodePci*> dsps;
};

class TopoPathPci {
 public:
  TopoPathPci(TopoNodePci* head, TopoNodePci* tail);
  ~TopoPathPci() = default;

  // Returns how many pci hops the path go through
  size_t Hops() const;
  // Whether the path crosses root complex
  bool CrossRootComplex() const;
  // Whether the path crosses multiple numa nodes
  bool CrossMultipleNuma() const;
  const std::vector<TopoNodePci*>& Nodes() const { return nodes; }

 private:
  void BuildPath();
  void Validate();

 private:
  TopoNodePci* head{nullptr};
  TopoNodePci* tail{nullptr};
  std::vector<TopoNodePci*> nodes;
};

class TopoSystemPci {
 public:
  TopoSystemPci();
  ~TopoSystemPci();

  TopoPathPci* Path(PciBusId, PciBusId);
  TopoNodePci* Node(PciBusId);

 private:
  void Load();
  void Validate();

 private:
  std::unordered_map<uint64_t, std::unique_ptr<TopoNodePci>> pcis;
  TopoNodePci* root{nullptr};

  using PathKey = std::pair<uint64_t, uint64_t>;
  using PathCache = std::unordered_map<PathKey, std::unique_ptr<TopoPathPci>>;

  PathCache pathCache;
};

}  // namespace application
}  // namespace mori
