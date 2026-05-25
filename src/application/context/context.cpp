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
#include "mori/application/context/context.hpp"

#include <arpa/inet.h>
#include <hip/hip_runtime_api.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mori/application/transport/sdma/anvil.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

Context::Context(BootstrapNetwork& bootNet) : bootNet(bootNet) {
  // Snapshot env vars once at construction. Every subsequent decision (transport
  // selection, hipMalloc vs hipExtMallocWithFlags(uncached), etc.) must read
  // from this cached state, not getenv. Otherwise late env mutations -- e.g.
  // a test setting MORI_ENABLE_SDMA after worker init -- can produce a state
  // where the transport layer chose P2P but per-allocation paths flip to
  // uncached SDMA buffers, leading to cache/IPC inconsistency hangs.
  sdmaEnabled = env::IsEnvVarEnabled("MORI_ENABLE_SDMA");
  p2pDisabled = env::IsEnvVarEnabled("MORI_DISABLE_P2P");
  CollectHostNames();
  // Lightweight: topology, NIC selection, transport type decision, SDMA queues.
  // No QP creation, no AllToAll. Modules that need the initial RDMA endpoint
  // set must explicitly call BuildInitialEndpoints() afterwards.
  InitializeTopologyAndTransports();
}

void Context::BuildInitialEndpoints() {
  if (initialEndpointsBuilt) return;
  BuildAndConnectInitialEndpoints();
  initialEndpointsBuilt = true;
}

Context::~Context() {}

std::string GetLocalIP() {
  struct ifaddrs *ifaddr, *ifa;
  char host[NI_MAXHOST];
  std::string localIP = "127.0.0.1";

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return localIP;
  }

  for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) continue;

    if (ifa->ifa_addr->sa_family == AF_INET) {
      int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, NULL, 0,
                          NI_NUMERICHOST);
      if (s != 0) {
        continue;
      }

      if (strcmp(host, "127.0.0.1") == 0) {
        continue;
      }

      localIP = host;
      break;
    }
  }

  freeifaddrs(ifaddr);
  return localIP;
}

bool Context::CanUseP2P(int destRank) const {
  if (destRank == LocalRank()) {
    return false;  // Cannot use P2P with self
  }
  return peerInfos[destRank].sameHost;
}

bool Context::SameProcessP2P(int destRank) const {
  if (destRank == LocalRank()) {
    return false;
  }
  return peerInfos[destRank].sameProcess;
}

void Context::CollectHostNames() {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  // Keep node identity stable across ranks on the same machine.
  // Using hostname+IP can split local ranks when different NICs are selected.

  // Pack pid + hostname into a fixed-size buffer for Allgather.
  // Using a fixed layout avoids string parsing ambiguity.
  constexpr int kPidSize = sizeof(pid_t);
  constexpr int kStrMax = HOST_NAME_MAX + 1;  // +1 for '\0'
  constexpr int kRecordSize = kPidSize + kStrMax;

  pid_t myPid = getpid();
  char localBuffer[kRecordSize];
  memcpy(localBuffer, &myPid, kPidSize);
  snprintf(localBuffer + kPidSize, kStrMax, "%s", hostname);

  std::vector<char> global(kRecordSize * WorldSize());
  bootNet.Allgather(localBuffer, global.data(), kRecordSize);

  myHostname = std::string(localBuffer + kPidSize);
  peerInfos.resize(WorldSize());
  for (int i = 0; i < WorldSize(); i++) {
    const char* rec = global.data() + i * kRecordSize;
    pid_t peerPid;
    memcpy(&peerPid, rec, kPidSize);
    std::string peerHost(rec + kPidSize);
    peerInfos[i].sameHost = (peerHost == myHostname);
    peerInfos[i].sameProcess = peerInfos[i].sameHost && (peerPid == myPid);
    if (LocalRank() == 0) {
      MORI_APP_TRACE("rank {} hostname={} pid={} sameHost={} sameProcess={}", i, peerHost, peerPid,
                     peerInfos[i].sameHost, peerInfos[i].sameProcess);
    }
  }
}

// MORI_ENABLE_SDMA / MORI_DISABLE_P2P are now read exactly once in the
// Context constructor and cached as members; consult Context::IsSdmaEnabled()
// / Context::IsP2PDisabled() instead of getenv anywhere outside the
// constructor.

void Context::InitializeTopologyAndTransports() {
  // Find my rank in node
  for (int i = 0; i <= LocalRank(); i++) {
    if (peerInfos[i].sameHost) rankInNode++;
  }
  assert(rankInNode < 8);

  // Init rdma context
  rdmaContext.reset(new RdmaContext(RdmaBackendType::DirectVerbs));
  const RdmaDeviceList& devices = rdmaContext->GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);

  if (rankInNode == 0) {
    std::string rdma_devices;
    if (activeDevicePortList.empty()) {
      rdma_devices = "None";
    } else {
      for (size_t i = 0; i < activeDevicePortList.size(); ++i) {
        if (i > 0) rdma_devices += ", ";
        rdma_devices += activeDevicePortList[i].first->Name();
      }
    }
    MORI_APP_INFO("rank {} RDMA devices: {}", LocalRank(), rdma_devices);
  }

  // Match gpu and nic
  bool disableTopo = env::IsEnvVarEnabled("MORI_DISABLE_TOPO");
  int portId = -1;
  int devicePortId = -1;
  RdmaDevice* device = nullptr;

  if (disableTopo) {
    std::cout << "MORI Topology detection is disabled, use static matching" << std::endl;
    if (!activeDevicePortList.empty()) {
      devicePortId = (rankInNode % activeDevicePortList.size());
      device = activeDevicePortList[devicePortId].first;
      portId = activeDevicePortList[devicePortId].second;
      rdmaDeviceContext.reset(device->CreateRdmaDeviceContext());
    }
  } else {
    int deviceId = -1;
    HIP_RUNTIME_CHECK(hipGetDevice(&deviceId));
    topo.reset(new TopoSystem());
    std::string nicName = topo->MatchGpuAndNic(deviceId);
    MORI_APP_TRACE("rank {} rankInNode {} matched nic {} for gpu {}", LocalRank(), rankInNode,
                   nicName, deviceId);
    for (int i = 0; i < activeDevicePortList.size(); i++) {
      auto& dp = activeDevicePortList[i];
      if (dp.first->Name() != nicName) continue;
      device = dp.first;
      portId = activeDevicePortList[i].second;
      rdmaDeviceContext.reset(device->CreateRdmaDeviceContext());
      devicePortId = i;
      break;
    }
  }

  if (device == nullptr) {
    MORI_APP_INFO("rank {} rankInNode {} select no device", LocalRank(), rankInNode);
  } else {
    MORI_APP_INFO("rank {} rankInNode {} select device [{}] {}", LocalRank(), rankInNode,
                  devicePortId, device->Name());
  }

  int numQpPerPe = 4;
  const char* envNumQp = std::getenv("MORI_NUM_QP_PER_PE");
  if (envNumQp != nullptr) {
    numQpPerPe = std::max(1, std::atoi(envNumQp));  // ensure at least 1 QP
  }
  this->numQpPerPe = numQpPerPe;
  // Initialize transport
  int peerRankInNode = -1;
  if (!IsP2PDisabled() && IsSdmaEnabled()) anvil::anvil.init();

  int sdmaNumChannels = anvil::GetSdmaNumChannels();
  MORI_APP_INFO("SDMA num channels per GPU pair: {}", sdmaNumChannels);

  // Per-peer loop, split into two phases:
  //
  //   (A) Capability discovery: fill peerCaps[i] purely from environment +
  //       NIC presence. No "we picked X" decision. This is what CCO consults
  //       to apply its own policy (gdaConnectionType FULL/CROSSNODE/RAIL).
  //
  //   (B) Default-policy resolve: call DefaultPolicyResolve(cap) to derive a
  //       single transportTypes[i]. This is what SHMEM's GpuStates copy step
  //       reads, so its DISPATCH_TRANSPORT_TYPE macro keeps working.
  //
  // Side effects (anvil queue setup, savedEpConfig lazy init) happen in (A)
  // because they depend on the discovered capability, not on the policy.
  peerCaps.resize(WorldSize());
  for (int i = 0; i < WorldSize(); i++) {
    PeerCapabilities& cap = peerCaps[i];
    cap.sameHost = peerInfos[i].sameHost;
    cap.sameProcess = peerInfos[i].sameProcess;

    // ── (A.1) intra-node capabilities ──
    if (!IsP2PDisabled() && cap.sameHost) {
      peerRankInNode++;
      // TODO: should use TopoSystemGpu to determine if peer access is enabled,
      // but that requires exchanging gpu bdf id, hence for simplicity we
      // assume peer access is enabled.
      const bool canAccessPeer = true;

      if (i == LocalRank() || canAccessPeer) {
        if (IsSdmaEnabled()) {
          // Same-host with SDMA enabled: also wire up the anvil queues now,
          // because SHMEM's default policy resolver picks SDMA over P2P and
          // expects the queues already exist.
          cap.canSDMA = true;
          if (i != LocalRank()) {
            anvil::EnablePeerAccess(LocalRank() % 8, i % 8);
          }
          anvil::anvil.connect(LocalRank() % 8, i % 8, sdmaNumChannels);
        } else {
          cap.canP2P = true;
        }
      }
    } else if (IsP2PDisabled() && i == LocalRank()) {
      // Self-loop always reachable via "P2P" semantics even with P2P disabled.
      cap.canP2P = true;
    }

    // ── (A.2) cross-node capability ──
    //
    // Cross-node peers require an RDMA NIC. If we have no NIC and we have a
    // cross-node peer, we still mark canRDMA=false and let the policy layer
    // (DefaultPolicyResolve or CCO's resolver) decide whether to assert or
    // gracefully report "no transport".
    const bool isCrossNode = !cap.sameHost;
    const bool needsRdma = isCrossNode || (IsP2PDisabled() && i != LocalRank());
    if (needsRdma && rdmaDeviceContext.get() != nullptr) {
      cap.canRDMA = true;
      // Lazy-initialize the EP config the first time we encounter a peer
      // that would need an RDMA endpoint.
      if (savedPortId < 0) {
        savedPortId = portId;
        savedEpConfig.portId = portId;
        savedEpConfig.gidIdx = -1;
        const char* envGidIdx = std::getenv("MORI_IB_GID_INDEX");
        if (envGidIdx != nullptr) savedEpConfig.gidIdx = std::atoi(envGidIdx);
        savedEpConfig.maxMsgsNum = 4096;
        uint32_t vid = rdmaDeviceContext->GetRdmaDevice()->GetDeviceAttr()->orig_attr.vendor_id;
        savedEpConfig.maxCqeNum =
            (vid == static_cast<uint32_t>(RdmaDeviceVendorId::Broadcom)) ? 1 : 4096;
        savedEpConfig.alignment = 4096;
        savedEpConfig.onGpu = true;
      }
    }

    // ── (B) default policy resolve ──
    //
    // Equivalent to the old per-peer if/else cascade. SHMEM and any caller
    // of GetTransportType() / GetTransportTypes() sees the same behavior as
    // before this refactor.
    transportTypes.push_back(DefaultPolicyResolve(cap, /*isSelf=*/i == LocalRank()));
  }
}

/* ------------------------------------------------------------------------ */
/*                          Default policy resolver                         */
/* ------------------------------------------------------------------------ */

TransportType Context::DefaultPolicyResolve(const PeerCapabilities& cap,
                                            bool isSelf) const {
  // Self always uses the cheapest path (P2P semantics for local read/write).
  // Among real peers: P2P first if available, then SDMA, then RDMA.
  if (isSelf) {
    // If SDMA is enabled, we have set canSDMA=true for self too (matches the
    // pre-refactor behavior). Otherwise default to P2P.
    return cap.canSDMA ? TransportType::SDMA : TransportType::P2P;
  }
  if (cap.canSDMA) return TransportType::SDMA;
  if (cap.canP2P) return TransportType::P2P;
  if (cap.canRDMA) return TransportType::RDMA;
  // No transport available — historical behavior was to assert here. Keep it
  // so SHMEM's init still fails fast on misconfigured deployments.
  assert(false && "no transport available for peer");
  return TransportType::RDMA;  // unreachable
}

/* ------------------------------------------------------------------------ */
/*               Phase B: build + connect initial RDMA endpoint set         */
/* ------------------------------------------------------------------------ */

void Context::BuildAndConnectInitialEndpoints() {
  // Build the worldSize × numQpPerPe rdmaEps vector. Non-RDMA peer slots are
  // populated with empty stubs to keep the indexing uniform.
  rdmaEps.reserve(static_cast<size_t>(WorldSize()) * numQpPerPe);
  for (int i = 0; i < WorldSize(); i++) {
    if (transportTypes[i] == TransportType::RDMA) {
      for (int qp = 0; qp < numQpPerPe; qp++) {
        RdmaEndpoint ep = rdmaDeviceContext->CreateRdmaEndpoint(savedEpConfig);
        rdmaEps.push_back(ep);
      }
    } else {
      for (int qp = 0; qp < numQpPerPe; qp++) {
        rdmaEps.push_back({});
      }
    }
  }

  // Exchange endpoint handles via AllToAll (worldSize × numQpPerPe handles).
  int totalEps = WorldSize() * numQpPerPe;
  std::vector<RdmaEndpointHandle> localToPeerEpHandles(totalEps);
  std::vector<RdmaEndpointHandle> peerToLocalEpHandles(totalEps);
  for (int i = 0; i < rdmaEps.size(); i++) {
    localToPeerEpHandles[i] = rdmaEps[i].handle;
  }
  bootNet.AllToAll(localToPeerEpHandles.data(), peerToLocalEpHandles.data(),
                   sizeof(RdmaEndpointHandle) * numQpPerPe);

  // Connect each RDMA peer's QPs (INIT -> RTR -> RTS).
  for (int peer = 0; peer < WorldSize(); peer++) {
    if (transportTypes[peer] != TransportType::RDMA) {
      continue;
    }
    for (int qp = 0; qp < numQpPerPe; qp++) {
      int epIndex = peer * numQpPerPe + qp;
      rdmaDeviceContext->ConnectEndpoint(localToPeerEpHandles[epIndex],
                                         peerToLocalEpHandles[epIndex], qp);
    }
  }
}

std::vector<RdmaEndpoint> Context::CreateAdditionalEndpoints(int qpPerPe) {
  std::vector<RdmaEndpoint> eps;
  eps.reserve(WorldSize() * qpPerPe);

  for (int i = 0; i < WorldSize(); i++) {
    if (transportTypes[i] != TransportType::RDMA || !rdmaDeviceContext) {
      for (int qp = 0; qp < qpPerPe; qp++) {
        eps.push_back({});
      }
      continue;
    }
    for (int qp = 0; qp < qpPerPe; qp++) {
      RdmaEndpoint ep = rdmaDeviceContext->CreateRdmaEndpoint(savedEpConfig);
      eps.push_back(ep);
    }
  }
  return eps;
}

void Context::ConnectAdditionalEndpoints(std::vector<RdmaEndpoint>& endpoints, int qpPerPe) {
  int totalEps = WorldSize() * qpPerPe;
  std::vector<RdmaEndpointHandle> localHandles(totalEps);
  std::vector<RdmaEndpointHandle> peerHandles(totalEps);

  for (int i = 0; i < totalEps; i++) {
    localHandles[i] = endpoints[i].handle;
  }

  bootNet.AllToAll(localHandles.data(), peerHandles.data(), sizeof(RdmaEndpointHandle) * qpPerPe);

  for (int peer = 0; peer < WorldSize(); peer++) {
    if (transportTypes[peer] != TransportType::RDMA) continue;
    for (int qp = 0; qp < qpPerPe; qp++) {
      int idx = peer * qpPerPe + qp;
      rdmaDeviceContext->ConnectEndpoint(localHandles[idx], peerHandles[idx], qp);
    }
  }
}

}  // namespace application
}  // namespace mori
