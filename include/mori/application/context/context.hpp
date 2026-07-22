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

// PeerCapabilities — transports physically available to reach a peer.
// Capability only, no policy: transportTypes[] below applies the default policy
// to pick one; peerCaps[] lets CCO apply its own (e.g. NIC QPs to intra-node
// peers).
struct PeerCapabilities {
  // canP2P/canSDMA are conservative (default sameHost, no hardware probe).
  // canRDMA is set whenever a NIC was selected, even same-host (NIC loopback).
  bool sameHost{false};     // peer is on the same physical node
  bool sameProcess{false};  // peer is in the same OS process (loopback IPC ok)
  bool canP2P{false};       // intra-node GPU peer access *likely* reachable
  bool canSDMA{false};      // intra-node SDMA *likely* reachable
  bool canRDMA{false};      // NIC reachable for this Context (host or cross)
};

class Context {
 public:
  Context(BootstrapNetwork& bootNet);
  ~Context();

  int LocalRank() const { return bootNet.GetLocalRank(); }
  int WorldSize() const { return bootNet.GetWorldSize(); }
  int LocalRankInNode() const { return rankInNode; }
  const std::string& HostName() const { return myHostname; }

  // Single-value selection via default policy (P2P > SDMA > RDMA). Stable for
  // SHMEM's device-side DISPATCH_TRANSPORT_TYPE macro.
  TransportType GetTransportType(int destRank) const { return transportTypes[destRank]; }
  const std::vector<TransportType>& GetTransportTypes() const { return transportTypes; }
  int GetNumQpPerPe() const { return numQpPerPe; }

  // All transports physically available to the peer, no policy baked in.
  const PeerCapabilities& GetPeerCapabilities(int destRank) const { return peerCaps[destRank]; }
  const std::vector<PeerCapabilities>& GetAllPeerCapabilities() const { return peerCaps; }

  RdmaContext* GetRdmaContext() const { return rdmaContext.get(); }
  RdmaDeviceContext* GetRdmaDeviceContext() const { return rdmaDeviceContext.get(); }
  bool RdmaTransportEnabled() const { return GetRdmaDeviceContext() != nullptr; }

  bool CanUseP2P(int destRank) const;
  // Same OS process: enables direct pointer access, skips IPC handle.
  bool SameProcessP2P(int destRank) const;

  // Env-var snapshot taken at construction; all later code MUST consult these
  // (not getenv) so post-init env changes can't create an inconsistent state.
  bool IsSdmaEnabled() const { return sdmaEnabled; }
  bool IsP2PDisabled() const { return p2pDisabled; }

  // Initial RDMA endpoint set; empty until BuildInitialEndpoints(). Consumed by
  // SHMEM; CCO builds its own via CreateAdditionalEndpoints.
  const std::vector<RdmaEndpoint>& GetRdmaEndpoints() const { return rdmaEps; }

  // Build+connect the initial worldSize×numQpPerPe endpoint set. Idempotent.
  // Collective: all ranks must call together (one AllToAll + per-peer RTR/RTS).
  // Side effect: applies the default policy to populate transportTypes[] and
  // lazily inits SDMA queues for any SDMA-resolved peer.
  void BuildInitialEndpoints();

  // Idempotent setup of anvil SDMA queues for all canSDMA peers (on-demand path
  // for CCO). No-op if already done or no peer has canSDMA.
  void EnsureSdmaTransport();

  // New independent QP set. peerMask[i] selects which peers get numQpPerPe real
  // QPs; others get empty stubs so the vector is always worldSize×numQpPerPe.
  // Self and peers with canRDMA==false are silently skipped even when masked.
  std::vector<RdmaEndpoint> CreateAdditionalEndpoints(int numQpPerPe,
                                                      const std::vector<bool>& peerMask);

  // Exchange endpoint handles via AllToAll then move each masked QP through
  // INIT → RTR → RTS. Must use the same peerMask as CreateAdditionalEndpoints.
  void ConnectAdditionalEndpoints(std::vector<RdmaEndpoint>& endpoints, int numQpPerPe,
                                  const std::vector<bool>& peerMask);

 private:
  void CollectHostNames();
  void InitializeTopologyAndTransports();  // lightweight: topology + NIC + transport type decision
                                           // + SDMA queues
  void BuildAndConnectInitialEndpoints();  // heavyweight: build initial QP set + AllToAll + connect

  // Derive a single TransportType from caps: P2P > SDMA > RDMA; self always P2P.
  // Aborts if none available (legacy SHMEM-init behavior).
  TransportType DefaultPolicyResolve(const PeerCapabilities& cap, bool isSelf) const;

  struct PeerInfo {
    // True if peer is on this rank's physical node; keyed on node identity, not
    // raw hostname, so it holds when machines share a hostname.
    bool sameHost{false};
    bool sameProcess{false};  // in the same OS process (same pid + same host)
  };

 private:
  BootstrapNetwork& bootNet;
  int rankInNode{-1};
  int numQpPerPe{4};
  bool sdmaEnabled{false};
  bool p2pDisabled{false};
  std::string myHostname;
  std::vector<PeerInfo> peerInfos;
  std::vector<PeerCapabilities> peerCaps;
  std::vector<TransportType> transportTypes;  // derived via DefaultPolicyResolve

  std::unique_ptr<RdmaContext> rdmaContext{nullptr};
  std::unique_ptr<RdmaDeviceContext> rdmaDeviceContext{nullptr};

  std::vector<RdmaEndpoint> rdmaEps;
  bool initialEndpointsBuilt{false};
  bool sdmaSetupDone{false};

  std::unique_ptr<TopoSystem> topo{nullptr};

  int savedPortId{-1};
  RdmaEndpointConfig savedEpConfig;
};

}  // namespace application
}  // namespace mori
