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
#include "mori/utils/host_utils.hpp"
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
  InitializePossibleTransports();
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
  myHostname = std::string(hostname);

  // Key co-location on node id, not hostname: identical hostnames would mark
  // cross-node ranks as co-located, over-counting rankInNode (trips assert below).
  std::string nodeId = ResolveNodeId(myHostname);

  // Allgather a fixed-layout {pid, nodeId} record; fixed size avoids parsing.
  constexpr int kPidSize = sizeof(pid_t);
  constexpr int kStrMax = 256;  // node id: boot_id, hostname, or override
  constexpr int kRecordSize = kPidSize + kStrMax;

  pid_t myPid = getpid();
  char localBuffer[kRecordSize] = {};
  memcpy(localBuffer, &myPid, kPidSize);
  snprintf(localBuffer + kPidSize, kStrMax, "%s", nodeId.c_str());

  std::vector<char> global(kRecordSize * WorldSize());
  bootNet.Allgather(localBuffer, global.data(), kRecordSize);

  std::string myNodeId(localBuffer + kPidSize);
  peerInfos.resize(WorldSize());
  for (int i = 0; i < WorldSize(); i++) {
    const char* rec = global.data() + i * kRecordSize;
    pid_t peerPid;
    memcpy(&peerPid, rec, kPidSize);
    std::string peerNodeId(rec + kPidSize);
    peerInfos[i].sameHost = (peerNodeId == myNodeId);
    peerInfos[i].sameProcess = peerInfos[i].sameHost && (peerPid == myPid);
    if (LocalRank() == 0) {
      MORI_APP_TRACE("rank {} nodeId={} pid={} sameHost={} sameProcess={}", i, peerNodeId, peerPid,
                     peerInfos[i].sameHost, peerInfos[i].sameProcess);
    }
  }
}

// MORI_ENABLE_SDMA / MORI_DISABLE_P2P are now read exactly once in the
// Context constructor and cached as members; consult Context::IsSdmaEnabled()
// / Context::IsP2PDisabled() instead of getenv anywhere outside the
// constructor.

void Context::InitializePossibleTransports() {
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

  // DUAL-RAIL (idle-NIC fan-out): each node exposes more active RDMA NICs than the
  // GPUs used (e.g. 8 ionic ports, 4 GPU/node) -- the extra NICs sit idle. When
  // MORI_HIER_DUAL_RAIL is set we bind a SECOND device (the idle partner of the
  // matched NIC) and create the upper half of each peer's QPs on it, so a single
  // rank's inter-node writes fan across TWO NICs. The second NIC is chosen as the
  // matched index offset by half the active-port list (0->4,1->5,... with 8 ports),
  // which lands on the idle partners and stays symmetric across nodes (both nodes
  // run identical code, so rank g uses the same pair on each). rail2QpStart splits
  // a peer's [0,numQpPerPe) block: [0,start) on rail 1, [start,numQpPerPe) on rail 2.
  int portId2 = -1;
  RdmaDevice* device2 = nullptr;
  rail2QpStart = numQpPerPe;  // default: no QP on rail 2 (single-rail)
  bool dualRail = false;
  {
    const char* eDR = std::getenv("MORI_HIER_DUAL_RAIL");
    dualRail = (eDR != nullptr && eDR[0] != '\0' && eDR[0] != '0');
  }
  if (dualRail && device != nullptr && activeDevicePortList.size() >= 2 && numQpPerPe >= 2) {
    int n = static_cast<int>(activeDevicePortList.size());
    int idx2 = (devicePortId + n / 2) % n;
    if (idx2 == devicePortId) idx2 = (devicePortId + 1) % n;
    device2 = activeDevicePortList[idx2].first;
    portId2 = activeDevicePortList[idx2].second;
    rdmaDeviceContext2.reset(device2->CreateRdmaDeviceContext());
    rail2QpStart = numQpPerPe / 2;
    MORI_APP_INFO("rank {} DUAL-RAIL rail2 device [{}] {} port {} (rail2QpStart {})", LocalRank(),
                  idx2, device2->Name(), portId2, rail2QpStart);
  }
  // Initialize transport
  int peerRankInNode = -1;
  // HIP-visible device id of THIS rank within its node (0-based) = number of
  // same-host peers ordered before us. We must use the within-node index, NOT
  // (global rank % 8): with ranks-per-node != 8 or a sliced HIP_VISIBLE_DEVICES
  // (e.g. a 2-node, 4-GPU/node run) the global ranks 4..7 on node 1 map to local
  // HIP devices 0..3 -- (rank % 8) would pass 4..7 to HIP and fault (only 4
  // devices visible). peerRankInNode (below) is the same 0-based index per peer.
  int localDevId = 0;
  for (int j = 0; j < LocalRank(); j++)
    if (peerInfos[j].sameHost) localDevId++;
  if (!IsP2PDisabled() && IsSdmaEnabled()) anvil::anvil.init();

  int sdmaNumChannels = anvil::GetSdmaNumChannels();
  MORI_APP_INFO("SDMA num channels per GPU pair: {}", sdmaNumChannels);

  for (int i = 0; i < WorldSize(); i++) {
    // Check P2P availability
    if (!IsP2PDisabled()) {
      if (peerInfos[i].sameHost) {
        peerRankInNode++;

        // TODO: should use TopoSystemGpu to determine if peer access is enabled, but that requires
        // exchanging gpu bdf id, hence for simplicity we assume peer access is enabled
        bool canAccessPeer = true;

        if ((i == LocalRank()) || canAccessPeer) {
          if (IsSdmaEnabled()) {
            if (i != LocalRank()) {
              transportTypes.push_back(TransportType::SDMA);
              anvil::EnablePeerAccess(localDevId, peerRankInNode);
              // Better performance if allocating all 8 queues
              anvil::anvil.connect(localDevId, peerRankInNode, sdmaNumChannels);
            } else {
              transportTypes.push_back(TransportType::SDMA);
              anvil::anvil.connect(localDevId, peerRankInNode, sdmaNumChannels);
            }
          } else {
            transportTypes.push_back(TransportType::P2P);
          }
          for (int qp = 0; qp < numQpPerPe; qp++) {
            rdmaEps.push_back({});
          }
          continue;
        }
      }
    } else {
      if (i == LocalRank()) {
        transportTypes.push_back(TransportType::P2P);
        for (int qp = 0; qp < numQpPerPe; qp++) {
          rdmaEps.push_back({});
        }
        continue;
      }
    }

    if (rdmaDeviceContext.get() == nullptr) assert(false && "no rdma device found");
    // Create multiple QPs for this peer
    application::RdmaEndpointConfig config;
    config.portId = portId;
    config.gidIdx = -1;
    const char* envGidIdx = std::getenv("MORI_IB_GID_INDEX");
    if (envGidIdx != nullptr) {
      config.gidIdx = std::atoi(envGidIdx);
    }
    config.maxMsgsNum = 4096;
    uint32_t vid = rdmaDeviceContext->GetRdmaDevice()->GetDeviceAttr()->orig_attr.vendor_id;
    config.maxCqeNum = (vid == static_cast<uint32_t>(RdmaDeviceVendorId::Broadcom)) ? 1 : 4096;
    config.alignment = 4096;
    config.onGpu = true;
    // Phase-6 WRITE_WITH_IMM: the receiver polls recvCqHandle for RDMA_WRITE_WITH_IMM
    // completions with its OWN consumer index (starting at 0). Without a dedicated recv
    // CQ, ionic_recv_cq_buf mirrors the send CQ (ionic.cpp), so recvCqHandle.consIdx=0
    // aliases send CQEs already consumed via cqHandle.consIdx -> the recv poll reads
    // stale/foreign CQEs and spins forever (both ring peers deadlock). Give WRITE_IMM
    // its own recv CQ so consIdx=0 is valid and only recv CQEs land there.
    {
      const char* eImm = std::getenv("MORI_HIER_RING_WRITE_IMM");
      if (eImm != nullptr && eImm[0] != '\0' && eImm[0] != '0') config.dedicatedRecvCq = true;
    }
    for (int qp = 0; qp < numQpPerPe; qp++) {
      // DUAL-RAIL: QPs [rail2QpStart, numQpPerPe) are created on the second device
      // (idle NIC) with its own port; the rest on the primary. Same flat rdmaEps
      // layout (numQpPerPe consecutive per peer) so init.cpp copies each QP's own
      // qpn/wq/cq handle unchanged -- only the underlying NIC differs.
      if (rdmaDeviceContext2 && qp >= rail2QpStart) {
        RdmaEndpointConfig config2 = config;
        config2.portId = portId2;
        RdmaEndpoint ep = rdmaDeviceContext2->CreateRdmaEndpoint(config2);
        rdmaEps.push_back(ep);
      } else {
        RdmaEndpoint ep = rdmaDeviceContext->CreateRdmaEndpoint(config);
        rdmaEps.push_back(ep);
      }
    }
    transportTypes.push_back(TransportType::RDMA);
  }

  // All2All rdma eps
  // Exchange endpoint handles (now with multiple QPs per peer)
  int totalEps = WorldSize() * numQpPerPe;
  std::vector<RdmaEndpointHandle> localToPeerEpHandles(totalEps);
  std::vector<RdmaEndpointHandle> peerToLocalEpHandles(totalEps);

  // Fill local endpoint handles
  for (int i = 0; i < rdmaEps.size(); i++) {
    localToPeerEpHandles[i] = rdmaEps[i].handle;
  }

  bootNet.AllToAll(localToPeerEpHandles.data(), peerToLocalEpHandles.data(),
                   sizeof(RdmaEndpointHandle) * numQpPerPe);

  // Connect RDMA endpoints
  for (int peer = 0; peer < WorldSize(); peer++) {
    if (transportTypes[peer] != TransportType::RDMA) {
      continue;
    }
    for (int qp = 0; qp < numQpPerPe; qp++) {
      int epIndex = peer * numQpPerPe + qp;
      // DUAL-RAIL: rail-2 QPs were created on rdmaDeviceContext2, whose qpPool
      // (keyed by local qpn) holds them; connect them there. ionic's ConnectEndpoint
      // resolves the QP from local.qpn, so the per-context call is all that matters.
      if (rdmaDeviceContext2 && qp >= rail2QpStart) {
        rdmaDeviceContext2->ConnectEndpoint(localToPeerEpHandles[epIndex],
                                            peerToLocalEpHandles[epIndex], qp);
      } else {
        rdmaDeviceContext->ConnectEndpoint(localToPeerEpHandles[epIndex],
                                           peerToLocalEpHandles[epIndex], qp);
      }
    }
  }
}

}  // namespace application
}  // namespace mori
