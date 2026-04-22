// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
  std::string HostName() const { return myHostname; }

  TransportType GetTransportType(int destRank) const { return transportTypes[destRank]; }
  const std::vector<TransportType>& GetTransportTypes() const { return transportTypes; }
  int GetNumQpPerPe() const { return numQpPerPe; }

  RdmaContext* GetRdmaContext() const { return rdmaContext.get(); }
  RdmaDeviceContext* GetRdmaDeviceContext() const { return rdmaDeviceContext.get(); }
  bool RdmaTransportEnabled() const { return GetRdmaDeviceContext() != nullptr; }

  // Check if P2P connection is possible with a peer (same node)
  bool CanUseP2P(int destRank) const;
  // Check if peer is in the same OS process (enables direct pointer access, skip IPC handle)
  bool SameProcessP2P(int destRank) const;

  const std::vector<RdmaEndpoint>& GetRdmaEndpoints() const { return rdmaEps; }

 private:
  void CollectHostNames();
  void InitializePossibleTransports();

  struct PeerInfo {
    bool sameHost{false};     // on the same node (same hostname+IP)
    bool sameProcess{false};  // in the same OS process (same pid + same host)
  };

 private:
  BootstrapNetwork& bootNet;
  int rankInNode{-1};
  int numQpPerPe{4};
  std::string myHostname;
  std::vector<PeerInfo> peerInfos;
  std::vector<TransportType> transportTypes;

  std::unique_ptr<RdmaContext> rdmaContext{nullptr};
  std::unique_ptr<RdmaDeviceContext> rdmaDeviceContext{nullptr};

  std::vector<RdmaEndpoint> rdmaEps;

  std::unique_ptr<TopoSystem> topo{nullptr};
};

}  // namespace application
}  // namespace mori
