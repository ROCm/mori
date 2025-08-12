#pragma once

#include <string>
#include <vector>

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/application/topology/topology.hpp"
#include "mori/application/transport/transport.hpp"

namespace mori {
namespace application {

class Context {
 public:
  Context(BootstrapNetwork& bootNet);
  ~Context();

  int LocalRank() const { return bootNet.GetLocalRank(); }
  int WorldSize() const { return bootNet.GetWorldSize(); }
  int LocalRankInNode() const { return rankInNode; }
  std::string HostName() const;

  TransportType GetTransportType(int destRank) const { return transportTypes[destRank]; }
  std::vector<TransportType> GetTransportTypes() const { return transportTypes; }

  RdmaContext* GetRdmaContext() const { return rdmaContext.get(); }
  RdmaDeviceContext* GetRdmaDeviceContext() const { return rdmaDeviceContext.get(); }
  bool RdmaTransportEnabled() const { return GetRdmaDeviceContext() != nullptr; }

  const std::vector<RdmaEndpoint>& GetRdmaEndpoints() const { return rdmaEps; }

 private:
  void CollectHostNames();
  void IntializePossibleTransports();

 private:
  BootstrapNetwork& bootNet;
  int rankInNode{-1};
  std::vector<std::string> hostnames;
  std::vector<TransportType> transportTypes;

  std::unique_ptr<RdmaContext> rdmaContext{nullptr};
  std::unique_ptr<RdmaDeviceContext> rdmaDeviceContext{nullptr};

  std::vector<RdmaEndpoint> rdmaEps;

  std::unique_ptr<TopoSystem> topo{nullptr};
};

}  // namespace application
}  // namespace mori