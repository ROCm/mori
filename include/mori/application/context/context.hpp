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

#include <memory>
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
  const std::string& HostName() const { return myHostname; }

  TransportType GetTransportType(int destRank) const { return transportTypes[destRank]; }
  const std::vector<TransportType>& GetTransportTypes() const { return transportTypes; }
  int GetNumQpPerPe() const { return numQpPerPe; }

  RdmaContext* GetRdmaContext() const { return rdmaContext.get(); }
  RdmaDeviceContext* GetRdmaDeviceContext() const { return rdmaDeviceContext.get(); }
  bool RdmaTransportEnabled() const { return GetRdmaDeviceContext() != nullptr; }
  // DUAL-RAIL (idle-NIC fan-out): the second RDMA device context (an otherwise-
  // idle NIC) on which the upper half of each peer's QPs are created. nullptr
  // unless MORI_HIER_DUAL_RAIL is enabled.
  RdmaDeviceContext* GetRdmaDeviceContext2() const { return rdmaDeviceContext2.get(); }
  bool DualRailEnabled() const { return rdmaDeviceContext2.get() != nullptr; }
  // Local QP index (within a peer's [0,numQpPerPe) block) at/after which QPs live
  // on the second rail. Equals numQpPerPe when dual-rail is off (no rail-2 QP).
  int GetRail2QpStart() const { return rail2QpStart; }

  // Check if P2P connection is possible with a peer (same node)
  bool CanUseP2P(int destRank) const;
  // Check if peer is in the same OS process (enables direct pointer access, skip IPC handle)
  bool SameProcessP2P(int destRank) const;

  // Cached env-var snapshot taken at construction time. All later code MUST
  // consult these (not getenv) so that env-var changes after Context init
  // cannot create an inconsistent state -- e.g. transport selected one way
  // at init but SymmMemManager::Malloc later switching to uncached
  // hipExtMallocWithFlags allocations because someone set MORI_ENABLE_SDMA
  // in a test function after the workers had already been spawned.
  bool IsSdmaEnabled() const { return sdmaEnabled; }
  bool IsP2PDisabled() const { return p2pDisabled; }

  const std::vector<RdmaEndpoint>& GetRdmaEndpoints() const { return rdmaEps; }

 private:
  void CollectHostNames();
  void InitializePossibleTransports();

  struct PeerInfo {
    // True if peer is on this rank's physical node. Keyed on node identity, not
    // raw hostname, so it holds even when all machines share one hostname.
    bool sameHost{false};
    bool sameProcess{false};  // in the same OS process (same pid + same host)
  };

 private:
  BootstrapNetwork& bootNet;
  int rankInNode{-1};
  int numQpPerPe{4};
  // Snapshotted at construction; see IsSdmaEnabled() / IsP2PDisabled() above.
  bool sdmaEnabled{false};
  bool p2pDisabled{false};
  std::string myHostname;
  std::vector<PeerInfo> peerInfos;
  std::vector<TransportType> transportTypes;

  std::unique_ptr<RdmaContext> rdmaContext{nullptr};
  std::unique_ptr<RdmaDeviceContext> rdmaDeviceContext{nullptr};
  // DUAL-RAIL second device context (idle NIC). nullptr unless MORI_HIER_DUAL_RAIL.
  std::unique_ptr<RdmaDeviceContext> rdmaDeviceContext2{nullptr};
  int rail2QpStart{-1};

  std::vector<RdmaEndpoint> rdmaEps;

  std::unique_ptr<TopoSystem> topo{nullptr};
};

}  // namespace application
}  // namespace mori
