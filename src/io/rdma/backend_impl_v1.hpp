#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "mori/application/topology/topology.hpp"
#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                     Common Data Structures                                     */
/* ---------------------------------------------------------------------------------------------- */
struct TopoKey {
  int deviceId;
  MemoryLocationType loc;

  bool operator==(const TopoKey& rhs) const noexcept {
    return (deviceId == rhs.deviceId) && (loc == rhs.loc);
  }

  MSGPACK_DEFINE(deviceId, loc);
};

struct TopoKeyPair {
  TopoKey local;
  TopoKey remote;

  bool operator==(const TopoKeyPair& rhs) const noexcept {
    return (local == rhs.local) && (remote == rhs.remote);
  }

  MSGPACK_DEFINE(local, remote);
};

struct MemoryKey {
  int devId;
  MemoryUniqueId id;

  bool operator==(const MemoryKey& rhs) const noexcept {
    return (id == rhs.id) && (devId == rhs.devId);
  }
};

struct EpPair {
  int weight;
  int ldevId;
  int rdevId;
  EngineKey remoteEngineKey;
  application::RdmaEndpoint local;
  application::RdmaEndpointHandle remote;
};

}  // namespace io
}  // namespace mori

namespace std {
template <>
struct hash<mori::io::TopoKey> {
  std::size_t operator()(const mori::io::TopoKey& k) const noexcept {
    std::size_t h1 = std::hash<uint32_t>{}(k.deviceId);
    std::size_t h2 = std::hash<uint32_t>{}(static_cast<uint32_t>(k.loc));
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

template <>
struct hash<mori::io::TopoKeyPair> {
  std::size_t operator()(const mori::io::TopoKeyPair& kp) const noexcept {
    std::size_t h1 = std::hash<mori::io::TopoKey>{}(kp.local);
    std::size_t h2 = std::hash<mori::io::TopoKey>{}(kp.remote);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

template <>
struct hash<mori::io::MemoryKey> {
  std::size_t operator()(const mori::io::MemoryKey& k) const noexcept {
    std::size_t h1 = std::hash<mori::io::MemoryUniqueId>{}(k.id);
    std::size_t h2 = std::hash<int>{}(k.devId);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace std

namespace mori {
namespace io {

using EpPairVec = std::vector<EpPair>;
using RouteTable = std::unordered_map<TopoKeyPair, EpPairVec>;
using MemoryTable = std::unordered_map<MemoryKey, application::RdmaMemoryRegion>;

struct RemoteEngineMeta {
  EngineKey key;
  RouteTable rTable;
  MemoryTable mTable;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */
class RdmaManager {
 public:
  RdmaManager(application::RdmaContext* ctx);
  ~RdmaManager() = default;

  application::RdmaEndpointConfig GetRdmaEndpointConfig(int portId);

  // Topology APIs
  std::vector<std::pair<int, int>> Search(TopoKey);

  // Local memory management APIs
  std::optional<application::RdmaMemoryRegion> GetLocalMemory(int ldevId, MemoryUniqueId);
  application::RdmaMemoryRegion RegisterLocalMemory(int ldevId, const MemoryDesc& desc);
  void DeregisterLocalMemory(int ldevId, const MemoryDesc& desc);

  // Remote memory management APIs
  std::optional<application::RdmaMemoryRegion> GetRemoteMemory(EngineKey, int remRdmaDevId,
                                                               MemoryUniqueId);
  void RegisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId,
                            application::RdmaMemoryRegion);
  void DeregisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId);

  // Endpoint management APIs
  int CountEndpoint(EngineKey, TopoKeyPair);
  EpPairVec GetAllEndpoint(EngineKey, TopoKeyPair);
  application::RdmaEndpoint CreateEndpoint(int devId);
  void ConnectEndpoint(EngineKey remoteKey, int ldevId, application::RdmaEndpoint local, int rdevId,
                       application::RdmaEndpointHandle remote, TopoKeyPair key, int weight);
  std::optional<EpPair> GetEpPairByQpn(uint32_t qpn);

  application::RdmaDeviceContext* GetRdmaDeviceContext(int devId);

 private:
  application::RdmaDeviceContext* GetOrCreateDeviceContext(int devId);

 private:
  mutable std::mutex mu;

  application::RdmaContext* ctx;
  application::ActiveDevicePortList availDevices;
  std::vector<application::RdmaDeviceContext*> deviceCtxs;

  MemoryTable mTable;
  std::unordered_map<EngineKey, RemoteEngineMeta> remotes;
  std::unordered_map<uint32_t, EpPair> epsMap;

  std::unique_ptr<application::TopoSystem> topo{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
class NotifManager {
 public:
  NotifManager(RdmaManager*);
  ~NotifManager();

  void RegisterEndpointByQpn(uint32_t qpn);
  // void DeregisterEndpoint(EpPair*);

  void RegisterDevice(int devId);

  void MainLoop();
  void Start();
  void Shutdown();

 private:
  mutable std::mutex mu;

  int epfd{-1};
  std::atomic<bool> running{false};
  std::thread thd;
  RdmaManager* rdma;

 private:
  struct DeviceNotifContext {
    ibv_srq* srq;
    application::RdmaMemoryRegion mr;
  };

  uint32_t maxNotifNum{8192};
  std::unordered_map<int, DeviceNotifContext> notifCtx;
  std::unordered_map<EngineKey, std::unordered_set<TransferUniqueId>> notifPool;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       Control Plane Serer                                      */
/* ---------------------------------------------------------------------------------------------- */
class ControlPlaneServer {
 public:
  ControlPlaneServer(std::string host, int port, RdmaManager*, NotifManager*);
  ~ControlPlaneServer();

  // Remote engine meta management
  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  // Endpoint management
  void BuildRdmaConn(EngineKey, TopoKeyPair);

  // MemoryRegion management
  void RegisterMemory(const MemoryDesc&);
  void DeregisterMemory(const MemoryDesc&);
  application::RdmaMemoryRegion AskRemoteMemoryRegion(EngineKey, int rdevId, MemoryUniqueId);

  // Server management
  void MainLoop();
  void Start();
  void Shutdown();

 private:
  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);

 private:
  mutable std::mutex mu;

  int epfd{-1};
  std::atomic<bool> running{false};
  std::unique_ptr<application::TCPContext> ctx{nullptr};
  std::unordered_map<int, application::TCPEndpointHandle> eps;
  std::thread thd;

  RdmaManager* rdma{nullptr};
  NotifManager* notif{nullptr};
  std::unordered_map<EngineKey, EngineDesc> engines;
  std::unordered_map<MemoryUniqueId, MemoryDesc> mems;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       RdmaBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
class RdmaBackendSession : public BackendSession {
 public:
  RdmaBackendSession() = default;
  RdmaBackendSession(const application::RdmaMemoryRegion& local,
                     const application::RdmaMemoryRegion& remote, const EpPair& eps);
  ~RdmaBackendSession() = default;

  void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
            TransferUniqueId id);
  void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
             TransferUniqueId id);

  void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets, const SizeVec& sizes,
                 TransferStatus* status, TransferUniqueId id);

  bool Alive() const;

 private:
  application::RdmaMemoryRegion local{};
  application::RdmaMemoryRegion remote{};
  EpPair eps{};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

class RdmaBackend : public Backend {
 public:
  RdmaBackend(EngineKey, IOEngineConfig);
  ~RdmaBackend();

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);
  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);
  void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
            size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id);
  void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
             size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id);
  void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                 const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets, const SizeVec& sizes,
                 TransferStatus* status, TransferUniqueId id);

  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote);
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

 private:
  void CreateSession(const MemoryDesc& local, const MemoryDesc& remote, RdmaBackendSession& sess);

 private:
  std::unique_ptr<RdmaManager> rdma;
  std::unique_ptr<NotifManager> notif;
  std::unique_ptr<ControlPlaneServer> server;
  std::vector<std::unique_ptr<RdmaBackendSession>> sessions;
};

}  // namespace io
}  // namespace mori