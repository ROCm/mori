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

#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "mori/application/topology/topology.hpp"
#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "src/io/rdma/common.hpp"
#include "src/io/rdma/executor.hpp"

namespace mori {
namespace io {
/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */
class RdmaManager {
 public:
  RdmaManager(const RdmaBackendConfig cfg, application::RdmaContext* ctx);
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
  EpHandleVec GetAllEndpointHandles(EngineKey, TopoKeyPair);
  application::RdmaEndpoint CreateEndpoint(int devId);
  void ConnectEndpoint(EngineKey remoteKey, int ldevId, application::RdmaEndpoint local, int rdevId,
                       application::RdmaEndpointHandle remote, TopoKeyPair key, int weight);
  std::optional<EpPairPtr> GetEpPairPtrByQpn(uint32_t qpn);

  std::optional<EpPairPtr> CommitPreparedRebuild(uint32_t old_qpn,
                                                 const application::RdmaEndpoint& new_local,
                                                 const application::RdmaEndpointHandle& remoteNew);

  // Deferred reclamation support -------------------------------------------------------------
  struct RetiredEndpoint {
    application::RdmaEndpoint ep;  // owns ibvHandle objects to destroy later
    EpPairPtr epPair;              // keep shared_ptr so use_count reflects external refs
    std::chrono::steady_clock::time_point retire_time;  // for diagnostics / optional timeout
  };

  // Periodically invoked (e.g. from NotifManager loop) to destroy endpoints whose EpPair
  // is no longer referenced elsewhere (use_count == 1 meaning only retired list holds it).
  void ReclaimRetiredEndpoints();

  application::RdmaDeviceContext* GetRdmaDeviceContext(int devId);

  int GetActiveDevicePort(int devId);

  // Endpoint enumeration
  using EnumerateEpCallbackFunc = std::function<void(int qpn, const EpPair& ep)>;
  void EnumerateEndpoints(const EnumerateEpCallbackFunc&);

 private:
  application::RdmaDeviceContext* GetOrCreateDeviceContext(int devId);

  RdmaBackendConfig config;
  mutable std::shared_mutex mu;

  application::RdmaContext* ctx;
  application::ActiveDevicePortList availDevices;
  std::vector<application::RdmaDeviceContext*> deviceCtxs;

  MemoryTable mTable;
  std::unordered_map<EngineKey, RemoteEngineMeta> remotes;
  std::unordered_map<uint32_t, EpPairPtr> epsMap;

  // Retired endpoints awaiting safe destruction. Guarded by mu.
  std::vector<RetiredEndpoint> retiredEps;

  std::unique_ptr<application::TopoSystem> topo{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
class NotifManager {
 public:
  NotifManager(RdmaManager*, const RdmaBackendConfig&, Backend* ownerBackend);
  ~NotifManager();

  void RegisterEndpointByQpn(uint32_t qpn);
  void UnregisterEndpointByQpn(uint32_t qpn);
  // void DeregisterEndpoint(EpPair*);

  void RegisterDevice(int devId);

  bool PopInboundTransferStatus(const EngineKey&, TransferUniqueId, TransferStatus*);

  void MainLoop();
  void Start();
  void Shutdown();

 private:
  void ProcessOneCqe(int qpn, const EpPair& ep);

 private:
  RdmaBackendConfig config;
  mutable std::mutex mu;

  int epfd{-1};
  std::atomic<bool> running{false};
  std::thread thd;
  RdmaManager* rdma;
  Backend* owner{nullptr};

  // Notification context
 private:
  struct DeviceNotifContext {
    ibv_srq* srq;
    application::RdmaMemoryRegion mr;
  };

  uint32_t maxNotifNum{8192};
  std::unordered_map<int, DeviceNotifContext> notifCtx;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, int>> notifPool;

  std::unordered_map<TransferStatus*, int> localNotif;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       Control Plane Serer                                      */
/* ---------------------------------------------------------------------------------------------- */
class ControlPlaneServer {
 public:
  ControlPlaneServer(const std::string& key, const std::string& host, int port,
                     Backend* ownerBackend, RdmaManager*, NotifManager*);
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

  // Placeholder: coordinated rebuild initiation (synchronous minimal version)
  // Returns true if handshake succeeded; false means fallback to local-only.
  bool SendRebuildRequest(uint32_t old_qpn, EpPairPtr epPtr, const RdmaBackendConfig& cfg);

  // Server management
  void MainLoop();
  void Start();
  void Shutdown();

 private:
  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);

  Backend* owner{nullptr};
  EngineKey myEngKey;

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
  RdmaBackendSession(const RdmaBackendConfig& config, const application::RdmaMemoryRegion& local,
                     const application::RdmaMemoryRegion& remote, const EpHandleVec& handles,
                     Executor* executor);
  ~RdmaBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

  bool Alive() const;

 private:
  RdmaBackendConfig config{};
  application::RdmaMemoryRegion local{};
  application::RdmaMemoryRegion remote{};
  EpHandleVec handles{};  // vector of EpHandle shared_ptr
  Executor* executor{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

class RdmaBackend : public Backend {
 public:
  RdmaBackend(EngineKey, const IOEngineConfig&, const RdmaBackendConfig&);
  ~RdmaBackend();

  friend class NotifManager;
  friend class ControlPlaneServer;

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);
  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);
  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead);
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);
  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote);
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

 private:
  void CreateSession(const MemoryDesc& local, const MemoryDesc& remote, RdmaBackendSession& sess);
  RdmaBackendSession* GetOrCreateSessionCached(const MemoryDesc& local, const MemoryDesc& remote);
  void InvalidateSessionsForMemory(MemoryUniqueId id);

  EngineKey myEngKey;
  RdmaBackendConfig config;
  std::unique_ptr<RdmaManager> rdma{nullptr};
  std::unique_ptr<NotifManager> notif{nullptr};
  std::unique_ptr<ControlPlaneServer> server{nullptr};
  std::unique_ptr<Executor> executor{nullptr};
  // session cache
  std::unordered_map<SessionCacheKey, std::unique_ptr<RdmaBackendSession>, SessionCacheKeyHash>
      sessionCache;
  std::mutex sessionCacheMu;
};

}  // namespace io
}  // namespace mori
