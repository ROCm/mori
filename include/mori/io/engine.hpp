#pragma once

#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

struct IOEngineConfig {
  std::string host;
  uint16_t port;
  int gpuId;
  BackendTypeVec backends;
};

using RdmaEpPair = std::pair<application::RdmaEndpointHandle, application::RdmaEndpointHandle>;
using RdmaEpPairVec = std::vector<RdmaEpPair>;

class IOEngine {
 public:
  IOEngine(EngineKey, IOEngineConfig);
  ~IOEngine();

  EngineDesc GetEngineDesc();
  void RegisterRemoteEngine(EngineDesc);
  void DeRegisterRemoteEngine(EngineDesc);

  //   MemoryDesc RegisterMemory(void* data, size_t length, MemLoc loc);
  //   void DeRegisterMemory(MemoryDesc);

  //   Status Write(MemoryDesc local, MemoryDesc remote, EngineDesc agent);

  // Data plane methods
 private:
  application::RdmaEndpointHandle CreateRdmaEndpoint();

 private:
  // Control plane methods
  RdmaEpPair BuildRdmaConnection(const application::TCPEndpointHandle&, bool isInitiator);

  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);
  void ControlPlaneLoop();
  void StartControlPlane();
  void ShutdownControlPlane();

  void InitDataPlane();

 public:
  // Config and descriptors
  IOEngineConfig config;
  EngineDesc desc;

 private:
  // Meta data store
  std::unordered_map<EngineKey, EngineDesc> engineKV;
  std::unordered_map<EngineKey, RdmaEpPairVec> rdmaEpKV;
  // std::unordered_map<MemoryDescId, MemoryDesc> memKV;

 private:
  // Control plane related members
  std::unique_ptr<application::TCPContext> tcpContext;
  std::unordered_map<int, application::TCPEndpointHandle> tcpEpKV;
  int epollFd{-1};
  std::thread ctrlPlaneThd;
  std::atomic<bool> running{false};

  // Data plane related members
  application::ActiveDevicePort devicePort;
  std::unique_ptr<application::RdmaContext> rdmaContext;
  application::RdmaDeviceContext* rdmaDeviceContext;
};

}  // namespace io
}  // namespace mori