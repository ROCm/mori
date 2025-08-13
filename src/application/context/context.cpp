#include "mori/application/context/context.hpp"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <vector>

#include "mori/application/utils/check.hpp"

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
  int deviceId = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&deviceId));
  topo.reset(new TopoSystem());
  std::string nicName = topo->MatchGpuAndNic(deviceId);

  int portId;
  RdmaDevice* device = nullptr;
  for (int i = 0; i < activeDevicePortList.size(); i++) {
    auto& dp = activeDevicePortList[i];
    if (dp.first->Name() != nicName) continue;
    device = dp.first;
    portId = activeDevicePortList[i].second;
    rdmaDeviceContext.reset(device->CreateRdmaDeviceContext());

    std::cout << "rank " << LocalRank() << " rankInNode " << rankInNode << " select device "
              << "[" << i << "] " << device->Name() << std::endl;
    break;
  }

  if (device == nullptr) {
    std::cout << "rank " << LocalRank() << " rankInNode " << rankInNode << " select no device"
              << std::endl;
  }

  // Intialize transport
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