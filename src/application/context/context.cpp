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
#include <hip/hip_runtime.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

Context::Context(BootstrapNetwork& bootNet) : bootNet(bootNet) {
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

std::string Context::HostName() const { return hostnames[LocalRank()]; }

void Context::CollectHostNames() {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  std::string localIP = GetLocalIP();
  std::string hostIdentifier = std::string(hostname) + ":" + localIP;

  constexpr int IDENTIFIER_MAX = HOST_NAME_MAX + INET_ADDRSTRLEN;
  std::vector<char> globalIdentifiers(IDENTIFIER_MAX * WorldSize());
  // Create a non-const buffer for Allgather
  char localBuffer[IDENTIFIER_MAX];
  strncpy(localBuffer, hostIdentifier.c_str(), IDENTIFIER_MAX - 1);
  localBuffer[IDENTIFIER_MAX - 1] = '\0';
  bootNet.Allgather(localBuffer, globalIdentifiers.data(), IDENTIFIER_MAX);

  for (int i = 0; i < WorldSize(); i++) {
    hostnames.push_back(&globalIdentifiers.data()[i * IDENTIFIER_MAX]);
  }

  if (LocalRank() == 0) {
    MORI_APP_TRACE("Collected hostnames:");
    for (int i = 0; i < hostnames.size(); i++) {
      MORI_APP_TRACE("  rank {}: {}", i, hostnames[i]);
    }
  }
}

bool IsP2PDisabled() {
  const char* varName = "MORI_DISABLE_P2P";
  return getenv(varName) != nullptr;
}

void Context::InitializePossibleTransports() {
  // Find my rank in node
  for (int i = 0; i <= LocalRank(); i++) {
    if (HostName() == hostnames[i]) rankInNode++;
  }
  assert(rankInNode < 8);

  // Init rdma context
  rdmaContext.reset(new RdmaContext(RdmaBackendType::DirectVerbs));
  const RdmaDeviceList& devices = rdmaContext->GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);

  if (rankInNode == 0) {
    std::cout << "rank " << LocalRank() << " RDMA devices: ";
    if (activeDevicePortList.empty()) {
      std::cout << "None" << std::endl;
    } else {
      for (size_t i = 0; i < activeDevicePortList.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << activeDevicePortList[i].first->Name();
      }
      std::cout << std::endl;
    }
  }

  // Match gpu and nic
  const char* disableTopo = std::getenv("MORI_DISABLE_TOPO");
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
    std::cout << "rank " << LocalRank() << " rankInNode " << rankInNode << " select no device"
              << std::endl;
  } else {
    std::cout << "rank " << LocalRank() << " rankInNode " << rankInNode << " select device "
              << "[" << devicePortId << "] " << device->Name() << std::endl;
  }

  int numQpPerPe = 4;
  const char* envNumQp = std::getenv("MORI_NUM_QP_PER_PE");
  if (envNumQp != nullptr) {
    numQpPerPe = std::max(1, std::atoi(envNumQp));  // ensure at least 1 QP
  }
  this->numQpPerPe = numQpPerPe;
  // Initialize transport
  int peerRankInNode = -1;
  for (int i = 0; i < WorldSize(); i++) {
    // Check P2P availability
    if (!IsP2PDisabled()) {
      if (HostName() == hostnames[i]) {
        peerRankInNode++;

        // TODO: should use TopoSystemGpu to determine if peer access is enabled, but that requires
        // exchanging gpu bdf id, hence for simplicity we assume peer access is enabled
        bool canAccessPeer = true;

        if ((i == LocalRank()) || canAccessPeer) {
          transportTypes.push_back(TransportType::P2P);
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
#ifdef ENABLE_BNXT
    config.maxCqeNum = 1;
#else
    config.maxCqeNum = 4096;
#endif
    config.alignment = 4096;
    config.onGpu = true;
    for (int qp = 0; qp < numQpPerPe; qp++) {
      RdmaEndpoint ep = rdmaDeviceContext->CreateRdmaEndpoint(config);
      rdmaEps.push_back(ep);
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
      rdmaDeviceContext->ConnectEndpoint(localToPeerEpHandles[epIndex],
                                         peerToLocalEpHandles[epIndex], qp);
    }
  }
}

}  // namespace application
}  // namespace mori
