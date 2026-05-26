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

/* ---------------------------------------------------------------------------
 *  PeerCapabilities — per-peer transport capability discovery
 *
 *  Describes WHICH transports are physically available to reach a given peer,
 *  taking into account hardware topology + env-var snapshots. Decoupled from
 *  the policy "which one do we actually use" (`transportTypes` below applies
 *  a default policy to derive a single TransportType from these caps).
 *
 *  - `transportTypes[i]` (legacy single-value field) is the historical
 *    "Context picked one for you" interface, kept for SHMEM compatibility.
 *  - `peerCaps[i]` is the new capability set, intended for CCO and other
 *    consumers that want to make their own policy decisions (e.g. CCO's
 *    gdaConnectionType chooses whether intra-node peers also get NIC QPs).
 * --------------------------------------------------------------------------- */
struct PeerCapabilities {
  // Objective hardware/topology facts only. NO policy / env-var influence.
  // Env vars MORI_DISABLE_P2P / MORI_ENABLE_SDMA flip *policy* (which
  // transport gets picked); they do not change capability — the hardware can
  // still do P2P even if env var disables it. Use the policy layer
  // (DefaultPolicyResolve or CCO's resolver) to combine cap + env intent.
  //
  // CURRENT LIMITATIONS:
  //  * canP2P/canSDMA are conservative — both default to `sameHost`. mori
  //    has no real hardware probe for either (we assume HIP peer access is
  //    always enabled, and anvil has no IsSupported() API). True capability
  //    is determined when anvil queues / hipDeviceEnablePeerAccess actually
  //    get invoked, which happens later in EnsureSdmaTransport() / window
  //    registration. A future probe-based check would refine these bits
  //    without changing the public API.
  //  * canRDMA is true whenever a NIC was selected for this Context, even
  //    for same-host peers. This lets future FULL-style policies allocate
  //    NIC QPs to intra-node peers (NIC loopback for uniform addressing).
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

  // Single-value transport selection driven by Context's default policy
  // (intra-node P2P > SDMA > cross-node RDMA). Kept stable so SHMEM's
  // device-side DISPATCH_TRANSPORT_TYPE macro continues to work unchanged.
  TransportType GetTransportType(int destRank) const { return transportTypes[destRank]; }
  const std::vector<TransportType>& GetTransportTypes() const { return transportTypes; }
  int GetNumQpPerPe() const { return numQpPerPe; }

  // Capability-level query: reveals all transports physically available to
  // reach the peer, without baking any policy choice. Use this when you need
  // to apply a custom policy (e.g. CCO's gdaConnectionType FULL forces NIC
  // QPs to intra-node peers even though canP2P is also true).
  const PeerCapabilities& GetPeerCapabilities(int destRank) const {
    return peerCaps[destRank];
  }
  const std::vector<PeerCapabilities>& GetAllPeerCapabilities() const {
    return peerCaps;
  }

  RdmaContext* GetRdmaContext() const { return rdmaContext.get(); }
  RdmaDeviceContext* GetRdmaDeviceContext() const { return rdmaDeviceContext.get(); }
  bool RdmaTransportEnabled() const { return GetRdmaDeviceContext() != nullptr; }

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

  // Returns the initial RDMA endpoint set. Empty until BuildInitialEndpoints()
  // has been called. SHMEM consumes this set; CCO does not (it creates its own
  // per-DevComm sets via CreateAdditionalEndpoints).
  const std::vector<RdmaEndpoint>& GetRdmaEndpoints() const { return rdmaEps; }

  // Build and connect the initial RDMA endpoint set sized worldSize×numQpPerPe.
  // Idempotent: safe to call multiple times, second+ calls are no-ops.
  //
  // This is a collective operation: all ranks must call it together. It runs
  // one AllToAll to exchange QP handles plus per-peer RTR/RTS transitions, so
  // it's heavy. Modules that don't need the initial set (e.g. CCO) can skip
  // this entirely and only use CreateAdditionalEndpoints later.
  //
  // Side effects: also applies Context's default policy to populate the
  // transportTypes[] vector (consumed by GetTransportType[s]) and lazily
  // initializes the SDMA queues if any peer was resolved to SDMA.
  void BuildInitialEndpoints();

  // Idempotent setup of anvil SDMA queues for all canSDMA peers. Useful when
  // SHMEM-style "BuildInitialEndpoints does it for me" is not in play — e.g.
  // CCO chooses SDMA per-DevComm and needs the queues materialized on demand.
  // No-op if already set up, or if no peer has canSDMA capability.
  void EnsureSdmaTransport();

  // Create a new independent set of QP endpoints for RDMA peers (does NOT connect).
  std::vector<RdmaEndpoint> CreateAdditionalEndpoints(int numQpPerPe);

  // Exchange new endpoint handles via AllToAll, then connect RDMA QPs. Collective.
  void ConnectAdditionalEndpoints(std::vector<RdmaEndpoint>& endpoints, int numQpPerPe);

 private:
  void CollectHostNames();
  void InitializeTopologyAndTransports();   // lightweight: topology + NIC + transport type decision + SDMA queues
  void BuildAndConnectInitialEndpoints();   // heavyweight: build initial QP set + AllToAll + connect

  // Apply Context's built-in policy to derive a single TransportType from a
  // PeerCapabilities entry. Preference: P2P > SDMA > RDMA. Self always P2P.
  // Aborts (assert) if no transport is available, matching legacy behavior
  // expected by SHMEM init.
  TransportType DefaultPolicyResolve(const PeerCapabilities& cap,
                                     bool isSelf) const;

  struct PeerInfo {
    bool sameHost{false};     // on the same node (same hostname+IP)
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
  std::vector<PeerCapabilities> peerCaps;        // raw capability discovery
  std::vector<TransportType> transportTypes;     // derived via DefaultPolicyResolve

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
