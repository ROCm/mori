#pragma once

#include <atomic>
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

using RdmaEpPair = std::pair<application::RdmaEndpoint, application::RdmaEndpointHandle>;
using RdmaEpPairVec = std::vector<RdmaEpPair>;
using MemoryBackendDescsPool = std::unordered_map<MemoryUniqueId, MemoryBackendDescs>;

class IOEngine {
 public:
  IOEngine(EngineKey, IOEngineConfig);
  ~IOEngine();

  EngineDesc GetEngineDesc();
  void RegisterRemoteEngine(EngineDesc);
  void DeRegisterRemoteEngine(EngineDesc);

  MemoryDesc RegisterMemory(void* data, size_t length, int device, MemoryLocationType loc);
  void DeRegisterMemory(const MemoryDesc& desc);

  void Read(MemoryDesc local, size_t localOffset, MemoryDesc remote, size_t remoteOffset,
            size_t size, TransferStatus& status);

 private:
  // Data plane methods
  application::RdmaEndpoint CreateRdmaEndpoint();

 private:
  // Control plane methods
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

  // memory meta data
  std::atomic<uint32_t> nextMemUid;
  std::unordered_map<MemoryUniqueId, MemoryDesc> memPool;

 private:
  // Control plane related members
  std::unique_ptr<application::TCPContext> tcpContext;
  std::unordered_map<int, application::TCPEndpointHandle> tcpEpKV;
  int epollFd{-1};
  std::thread ctrlPlaneThd;
  std::atomic<bool> running{false};

  // Data plane related members
  application::ActiveDevicePort devicePort;
  application::RdmaDeviceContext* rdmaDeviceContext;
  std::unique_ptr<application::RdmaContext> rdmaContext;

  std::thread rdmaPollThd;
};

}  // namespace io
}  // namespace mori