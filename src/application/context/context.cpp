#include "mori/application/context/context.hpp"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <vector>

#include "mori/application/utils/hip_check.hpp"

namespace mori {
namespace application {

Context::Context(BootstrapNetwork& bootNet) : bootNet(bootNet) {
  CollectHostNames();
  IntializePossibleTransports();
}

Context::~Context() {}

std::string Context::HostName() const { return hostnames[LocalRank()]; }

void Context::CollectHostNames() {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  // char globalHostNames[HOST_NAME_MAX * WorldSize()];
  std::vector<char> globalHostNames(HOST_NAME_MAX * WorldSize());
  bootNet.Allgather(hostname, globalHostNames.data(), HOST_NAME_MAX);

  for (int i = 0; i < WorldSize(); i++) {
    hostnames.push_back(&globalHostNames.data()[i * HOST_NAME_MAX]);
  }
}

bool IsP2PDisabled() {
  const char* varName = "MORI_DISABLE_P2P";
  return getenv(varName) != nullptr;
}

void Context::IntializePossibleTransports() {
  // Find my rank in node
  for (int i = 0; i <= LocalRank(); i++) {
    if (HostName() == hostnames[i]) rankInNode++;
  }
  int gpuCount;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&gpuCount));
  assert(rankInNode < gpuCount);
  HIP_RUNTIME_CHECK(hipSetDevice(rankInNode));

  // Init rdma context
  rdmaContext.reset(new RdmaContext());
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

  int portId;
  RdmaDevice* device = nullptr;
  if (!activeDevicePortList.empty()) {
    int devicePortId = (rankInNode % activeDevicePortList.size());
    RdmaDevice* device = activeDevicePortList[devicePortId].first;
    std::cout << "rank " << LocalRank() << " rankInNode " << rankInNode << " select device "
              << "[" << devicePortId << "] " << device->Name() << std::endl;
    portId = activeDevicePortList[devicePortId].second;
    rdmaDeviceContext.reset(device->CreateRdmaDeviceContext());
  }

  // Intialize transport
  int peerRankInNode = -1;
  for (int i = 0; i < WorldSize(); i++) {
    // Check P2P availability
    if (!IsP2PDisabled()) {
      if (HostName() == hostnames[i]) {
        peerRankInNode++;

        int canAccessPeer;
        HIP_RUNTIME_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, rankInNode, peerRankInNode));

        if (canAccessPeer) {
          HIP_RUNTIME_CHECK(hipDeviceEnablePeerAccess(peerRankInNode, 0));
        }

        if ((i == LocalRank()) || canAccessPeer) {
          transportTypes.push_back(TransportType::P2P);
          rdmaEps.push_back({});
          continue;
        }
      }
    } else {
      if (i == LocalRank()) {
        transportTypes.push_back(TransportType::P2P);
        rdmaEps.push_back({});
        continue;
      }
    }

    if (rdmaDeviceContext.get() == nullptr) assert(false && "no rdma device found");

    application::RdmaEndpointConfig config;
    config.portId = portId;
    config.gidIdx = 3;
    config.maxMsgsNum = 4096;
    config.maxCqeNum = 4096;
    config.alignment = 4096;
    config.onGpu = true;
    RdmaEndpoint ep = rdmaDeviceContext->CreateRdmaEndpoint(config);
    rdmaEps.push_back(ep);
    transportTypes.push_back(TransportType::RDMA);
  }

  // All2All rdma eps
  // Exchange endpoint handles
  std::vector<RdmaEndpointHandle> localToPeerEpHandles(WorldSize());
  std::vector<RdmaEndpointHandle> peerToLocalEpHandles(WorldSize());
  for (int i = 0; i < WorldSize(); i++) localToPeerEpHandles[i] = rdmaEps[i].handle;
  bootNet.AllToAll(localToPeerEpHandles.data(), peerToLocalEpHandles.data(),
                   sizeof(RdmaEndpointHandle));

  // Connect RDMA endpoints
  for (int i = 0; i < WorldSize(); i++) {
    if (transportTypes[i] != TransportType::RDMA) continue;
    rdmaDeviceContext->ConnectEndpoint(localToPeerEpHandles[i], peerToLocalEpHandles[i]);
  }
}

}  // namespace application
}  // namespace mori