#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

using RdmaEpPair = std::pair<application::RdmaEndpoint, application::RdmaEndpointHandle>;
using RdmaEpPairVec = std::vector<RdmaEpPair>;

class RdmaBackend : public Backend {
 public:
  RdmaBackend(EngineKey, IOEngineConfig);
  ~RdmaBackend();

  EngineDesc GetEngineDesc();
  void RegisterRemoteEngine(EngineDesc);
  void DeregisterRemoteEngine(EngineDesc);

  void RegisterMemory(MemoryDesc& desc);
  void DeregisterMemory(MemoryDesc& desc);

  TransferUniqueId AllocateTransferUniqueId();
  void Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc, size_t remoteOffset,
            size_t size, TransferStatus* status, TransferUniqueId id);
  void Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest, size_t remoteOffset,
             size_t size, TransferStatus* status, TransferUniqueId id);

  // Take the transfer status of an inbound op
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

 private:
  // Control plane methods
  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);
  void ControlPlaneLoop();
  void StartControlPlane();
  void ShutdownControlPlane();

  // Data plane methods
  application::RdmaEndpoint CreateRdmaEndpoint();
  void RdmaPollLoop();
  void StartDataPlane();
  void ShutdownDataPlane();
  application::RdmaEndpointConfig GetRdmaEndpointConfig();

  void RdmaNotifyTransfer(const application::RdmaEndpoint& ep, TransferStatus* status,
                          TransferUniqueId id);

 public:
  // Config and descriptors
  IOEngineConfig config;
  EngineDesc desc;

 private:
  // TODO: add a read-write for per-engine / per device members

  // Meta data store
  std::unordered_map<EngineKey, EngineDesc> engineKV;
  std::unordered_map<EngineKey, RdmaEpPairVec> rdmaEpKV;
  std::unordered_map<uint32_t, std::pair<EngineKey, RdmaEpPair>> qpn2EngineKV;

  // transfer meta data
  uint32_t rdmaTrsfUidNum{1024};
  TransferUniqueId* rdmaTrsfUidBuf{nullptr};
  application::RdmaMemoryRegion rdmaTrsfUidMr;

  struct TransferUidNotifMap {
    mutable std::mutex mu;  // TODO：use read-write lock
    std::unordered_set<TransferUniqueId> map;
  };
  std::unordered_map<EngineKey, std::unique_ptr<TransferUidNotifMap>> trsfUidNotifMaps;

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
  int rdmaCompChEpollFd{-1};
  std::thread rdmaPollThd;
};

}  // namespace io
}  // namespace mori