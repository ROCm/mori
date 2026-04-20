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
#include <cstdint>
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
  ~RdmaManager();

  application::RdmaEndpointConfig GetRdmaEndpointConfig(int devId);

  // Topology APIs
  std::vector<std::pair<int, int>> Search(TopoKey);

  // Chunked local memory registration
  std::shared_ptr<const RdmaLocalMemoryRegistration> GetOrMaterializeLocalRegistration(
      const MemoryDesc& desc);
  void DeregisterLocalRegistration(const MemoryDesc& desc);

  // Remote layout management
  std::shared_ptr<const RdmaRemoteMemoryLayout> GetRemoteLayout(EngineKey, MemoryUniqueId);
  void RegisterRemoteLayout(EngineKey, MemoryUniqueId,
                            std::shared_ptr<const RdmaRemoteMemoryLayout>);
  void DeregisterRemoteLayout(EngineKey, MemoryUniqueId);

  // Endpoint management APIs
  int CountEndpoint(EngineKey, TopoKeyPair);
  EpPairVec GetAllEndpoint(EngineKey, TopoKeyPair);
  application::RdmaEndpoint CreateEndpoint(int devId);
  EndpointId ConnectEndpoint(EngineKey remoteKey, int ldevId, application::RdmaEndpoint local,
                             int rdevId, application::RdmaEndpointHandle remote, TopoKeyPair key,
                             int weight);
  std::shared_ptr<EndpointRuntime> GetEndpointRuntime(EndpointId id);
  std::vector<std::shared_ptr<EndpointRuntime>> SnapshotEndpointRuntimes();

  application::RdmaDeviceContext* GetRdmaDeviceContext(int devId);
  size_t GetEffectiveChunkSize(application::RdmaDeviceContext* devCtx);

 private:
  application::RdmaDeviceContext* GetOrCreateDeviceContext(int devId);
  int PickRdmaDeviceForMemory(const MemoryDesc& desc);

 private:
  RdmaBackendConfig config;
  mutable std::shared_mutex mu;

  application::RdmaContext* ctx;
  application::ActiveDevicePortList availDevices;
  std::vector<application::RdmaDeviceContext*> deviceCtxs;

  // Per-memory device affinity: pinned on first materialization
  std::unordered_map<MemoryUniqueId, int> memoryDeviceAffinity_;
  // Local registration cache: MemoryUniqueId -> immutable registration
  std::unordered_map<MemoryUniqueId, std::shared_ptr<const RdmaLocalMemoryRegistration>>
      localRegistrations_;
  // Per-memory single-flight mutex for materialization
  std::unordered_map<MemoryUniqueId, std::shared_ptr<std::mutex>> materializationMutexes_;

  std::unordered_map<EngineKey, RemoteEngineMeta> remotes;
  std::atomic<EndpointId> nextEndpointId_{1};
  std::unordered_map<EndpointId, std::shared_ptr<EndpointRuntime>> endpointsById_;

  std::unique_ptr<application::TopoSystem> topo{nullptr};
  std::atomic<uint32_t> roundRobinCounter{0};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
class NotifManager {
 public:
  NotifManager(RdmaManager*, const RdmaBackendConfig&);
  ~NotifManager();

  void RegisterEndpoint(const std::shared_ptr<EndpointRuntime>& rt);

  void RegisterDevice(int devId);

  bool PopInboundTransferStatus(const EngineKey&, TransferUniqueId, TransferStatus*);

  void MainLoop();
  void Start();
  void Shutdown();

 private:
  struct FlushDrainStats {
    uint64_t count{0};
    uint32_t firstQpNum{0};

    void Record(uint32_t qpNum) {
      if (count == 0) firstQpNum = qpNum;
      count++;
    }

    bool Empty() const { return count == 0; }
  };

  struct FlushRoundStats {
    uint64_t total{0};
    uint32_t endpointCount{0};
    EndpointId sampleEndpointId{0};
    uint32_t sampleQpNum{0};

    void Merge(EndpointId eid, const FlushDrainStats& drain) {
      if (drain.Empty()) return;
      if (total == 0) {
        sampleEndpointId = eid;
        sampleQpNum = drain.firstQpNum;
      }
      total += drain.count;
      endpointCount++;
    }

    bool Empty() const { return total == 0; }
  };

  FlushDrainStats ProcessOneCqe(const std::shared_ptr<EndpointRuntime>& rt);
  void EmitFlushSummaryIfNeeded(const FlushRoundStats& roundStats);

 private:
  RdmaBackendConfig config;
  mutable std::mutex mu;

  int epfd{-1};
  std::atomic<bool> running{false};
  std::thread thd;
  RdmaManager* rdma;

  // Notification context
 private:
  struct QpNotifContext {
    application::RdmaMemoryRegion mr;
    void* buf;
  };

  std::unordered_map<EndpointId, std::shared_ptr<EndpointRuntime>> registeredRuntimes_;
  std::unordered_map<EndpointId, QpNotifContext> notifCtxById_;
  std::unordered_map<EngineKey, std::unordered_map<TransferUniqueId, int>> notifPool;

  std::unordered_map<TransferStatus*, int> localNotif;

  // Accessed only by the single NotifManager poll loop thread to rate-limit
  // repeated summaries for the same consecutive flush episode.
  uint64_t flushSummaryStreak_{0};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       Control Plane Serer                                      */
/* ---------------------------------------------------------------------------------------------- */
class ControlPlaneServer {
 public:
  ControlPlaneServer(const std::string& key, const std::string& host, int port, RdmaManager*,
                     NotifManager*);
  ~ControlPlaneServer();

  std::optional<uint16_t> GetListenPort() const {
    if (!ctx) return std::nullopt;
    return static_cast<uint16_t>(ctx->GetPort());
  }

  // Remote engine meta management
  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  // Endpoint management
  void BuildRdmaConn(EngineKey, TopoKeyPair, int ldevId = -1, int rdevId = -1);

  // MemoryRegion management
  void RegisterMemory(MemoryDesc&);
  void DeregisterMemory(const MemoryDesc&);
  std::shared_ptr<const RdmaRemoteMemoryLayout> AskRemoteMemoryLayout(EngineKey, MemoryUniqueId,
                                                                      size_t expectedSize);

  // Server management
  void MainLoop();
  void Start();
  void Shutdown();

 private:
  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);

 private:
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
struct RdmaResolvedLocalMemory {
  bool singleMrFastPath{false};
  application::RdmaMemoryRegion singleMr{};
  std::shared_ptr<const RdmaLocalMemoryRegistration> registration;
};

struct RdmaResolvedRemoteMemory {
  bool singleMrFastPath{false};
  application::RdmaMemoryRegion singleMr{};
  std::shared_ptr<const RdmaRemoteMemoryLayout> layout;
};

class RdmaBackendSession : public BackendSession {
 public:
  RdmaBackendSession() = default;
  RdmaBackendSession(const RdmaBackendConfig& config, RdmaResolvedLocalMemory local,
                     RdmaResolvedRemoteMemory remote, const EpPairVec& eps, Executor* executor);
  ~RdmaBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

  bool Alive() const;

 private:
  bool UseFastPath(const SizeVec& sizes) const;

  RdmaBackendConfig config{};
  RdmaResolvedLocalMemory local{};
  RdmaResolvedRemoteMemory remote{};
  EpPairVec eps{};
  Executor* executor{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

class RdmaBackend : public Backend {
 public:
  RdmaBackend(EngineKey, const IOEngineConfig&, const RdmaBackendConfig&);
  ~RdmaBackend();

  std::optional<uint16_t> GetListenPort() const {
    if (!server) return std::nullopt;
    return server->GetListenPort();
  }

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);
  void RegisterMemory(MemoryDesc& desc);
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
  std::shared_ptr<RdmaBackendSession> CreateSessionImpl(const MemoryDesc& local,
                                                        const MemoryDesc& remote);
  // Session cache helpers
  struct SessionCacheKey {
    EngineKey remoteEngineKey;
    MemoryUniqueId localMemId;
    MemoryUniqueId remoteMemId;
    bool operator==(const SessionCacheKey& o) const {
      return remoteEngineKey == o.remoteEngineKey && localMemId == o.localMemId &&
             remoteMemId == o.remoteMemId;
    }
  };
  struct SessionCacheKeyHash {
    std::size_t operator()(const SessionCacheKey& k) const noexcept {
      auto hash_combine = [](std::size_t& seed, std::size_t v) {
        seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      std::size_t seed = 0;
      hash_combine(seed, std::hash<std::string>()(k.remoteEngineKey));
      hash_combine(seed, std::hash<uint64_t>()(k.localMemId));
      hash_combine(seed, std::hash<uint64_t>()(k.remoteMemId));
      return seed;
    }
  };
  std::shared_ptr<RdmaBackendSession> GetOrCreateSessionCached(const MemoryDesc& local,
                                                               const MemoryDesc& remote);
  void InvalidateSessionsForMemory(MemoryUniqueId id);

 private:
  EngineKey myEngKey;
  RdmaBackendConfig config;
  std::unique_ptr<RdmaManager> rdma{nullptr};
  std::unique_ptr<NotifManager> notif{nullptr};
  std::unique_ptr<ControlPlaneServer> server{nullptr};
  std::unique_ptr<Executor> executor{nullptr};
  std::unordered_map<SessionCacheKey, std::shared_ptr<RdmaBackendSession>, SessionCacheKeyHash>
      sessionCache;
  std::mutex sessionCacheMu;
};

}  // namespace io
}  // namespace mori
