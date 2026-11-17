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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
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

class RdmaAsyncEventMonitor;

namespace internal {

// Placeholder written into engine_desc.port when an RDMA backend request is
// rerouted to XGMI-only mode because this host has no active RDMA device.
inline constexpr uint16_t kXgmiOnlyFallbackPlaceholderPort = 1;

}  // namespace internal

void ValidateRdmaTransferConfig(const RdmaBackendConfig& config);
// Applies the MORI_IO_* env overrides to beConfig, then validates the result.
RdmaBackendConfig ResolveRdmaBackendConfig(const RdmaBackendConfig& beConfig);
bool UsesInlineOnly(const RdmaBackendConfig& config);
int ResolveRequestedNics(const RdmaBackendConfig& config, const TopoKey& local,
                         const TopoKey& remote);
std::vector<int> BuildDesiredQpCounts(int totalQp, int numRanks);
EpPairVec InterleaveEndpointsByLocalDevice(const EpPairVec& eps,
                                           const std::vector<int>& localDevOrder,
                                           const std::vector<int>& wantPerRank);

/* ---------------------------------------------------------------------------------------------- */
/*                                       Endpoint resources                                       */
/* ---------------------------------------------------------------------------------------------- */
// The verbs objects behind one connected endpoint, shared by its EpPair copies.
struct EndpointResources {
  int devId{-1};
  application::RdmaEndpoint ep{};
  // NotifManager's receive buffer for notifications on this QP, and its MR.
  void* notifBuf{nullptr};
  application::RdmaMemoryRegion notifMr{};
};

// Holds endpoints whose last EpPair copy is gone until RdmaManager destroys them. A
// copy can go on any thread and under any lock, so letting go of one only queues it.
class ReleasedEndpoints {
 public:
  void Push(EndpointResources* res);
  std::vector<std::unique_ptr<EndpointResources>> TakeAll();
  // The device contexts are being destroyed, and their QPs with them: later pushes are
  // dropped without touching verbs.
  void Close();

 private:
  std::mutex mu_;
  bool closed_{false};
  std::vector<std::unique_ptr<EndpointResources>> released_;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */
class RdmaManager {
 public:
  using SnapshotVector = std::vector<std::shared_ptr<EndpointRuntime>>;

  RdmaManager(const RdmaBackendConfig cfg, application::RdmaContext* ctx);
  ~RdmaManager();

  application::RdmaEndpointConfig GetRdmaEndpointConfig(int devId);

  // Topology APIs
  std::vector<std::pair<int, int>> Search(TopoKey, int requestedNics = -1);

  // Local memory management APIs
  std::optional<application::RdmaMemoryRegion> GetLocalMemory(int ldevId, MemoryUniqueId);
  application::RdmaMemoryRegion RegisterLocalMemory(int ldevId, const MemoryDesc& desc);
  void DeregisterLocalMemory(int ldevId, const MemoryDesc& desc);
  void DeregisterLocalMemory(const MemoryDesc& desc);

  // Remote memory management APIs
  std::optional<application::RdmaMemoryRegion> GetRemoteMemory(EngineKey, int remRdmaDevId,
                                                               MemoryUniqueId);
  void RegisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId,
                            application::RdmaMemoryRegion);
  void DeregisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId);

  // Endpoint management APIs
  int CountEndpoint(EngineKey, TopoKeyPair);
  EpPairVec GetAllEndpoint(EngineKey, TopoKeyPair);
  // GetAllEndpoint without the endpoints whose QP has been reported failed: an RC QP
  // in the error state never carries traffic again, so it must not count or be reused.
  EpPairVec GetLiveEndpoints(EngineKey, TopoKeyPair);
  application::RdmaEndpoint CreateEndpoint(int devId);
  bool DestroyEndpointNoThrow(int devId, const application::RdmaEndpoint& ep) noexcept;
  EndpointId ConnectEndpoint(EngineKey remoteKey, int ldevId, application::RdmaEndpoint local,
                             int rdevId, application::RdmaEndpointHandle remote, TopoKeyPair key,
                             int weight);
  std::shared_ptr<EndpointRuntime> GetEndpointRuntime(EndpointId id);
  void SnapshotEndpointRuntimes(SnapshotVector& result);
  // Removes every endpoint to `remote` from the route table and forgets the remote
  // engine's memory regions, so no new session can reach them. Returns them, along with
  // those an earlier call could not cut; they are still polled.
  SnapshotVector DetachRemoteEngine(const EngineKey& remote);
  // Cuts every endpoint's QP, then fails what is still in flight on the ones it cut with
  // `message`. All of them reject new submissions, and Alive() reports them dead. A QP
  // that could not be cut can still complete work, so its transfers are left to their
  // completions. Returns the endpoints it cut.
  SnapshotVector RetireEndpoints(const SnapshotVector& runtimes, const std::string& message);
  // Stops polling endpoints, once nothing posted on them can complete any more.
  void RemoveEndpoints(const SnapshotVector& runtimes);
  // Destroys the endpoints no EpPair copy refers to any more.
  void DestroyReleasedEndpoints();
  // QPs connected and not destroyed yet, including detached ones a session still holds.
  size_t NumConnectedEndpoints() const {
    return connectedEndpoints_.load(std::memory_order_relaxed);
  }
  application::RdmaMemoryRegion RegisterNotifBuffer(int devId, void* buf, size_t size);
  // Monotonic generation counter bumped whenever the endpoint set changes.
  // Lets hot pollers skip the locked snapshot rebuild when nothing changed.
  uint64_t EndpointsEpoch() const noexcept {
    return endpointsEpoch_.load(std::memory_order_acquire);
  }

  application::RdmaDeviceContext* GetRdmaDeviceContext(int devId);
  size_t NumAvailDevices() const { return availDevices.size(); }
  bool HasIonicDevice() const;

  // Fails every in-flight transfer on the endpoints implicated by a fatal verbs
  // async event, and marks them degraded so later submissions fail fast instead
  // of queueing onto a dead QP. Returns the number of transfers this call moved
  // to a failed state.
  int FailInFlightTransfers(const QpErrorEvent& event);

  /* -------------------------------- Peer failure -------------------------------- */
  // Takes the oldest recorded failure; polled by an application thread.
  bool PopPeerFailure(PeerFailureEvent* out);
  // Drops the queued failures of `remote`. Returns how many.
  size_t DropPeerFailures(const EngineKey& remote);
  // False once a failure covering this QP is observed; a slow QP stays alive.
  bool IsQpAlive(int devId, uint32_t qpNum) const;
  bool NeedsQpReplacement(int devId, uint32_t qpNum) const;
  // Records IBV_WC_RETRY_EXC_ERR: the only place peer death is observable.
  void RecordPeerUnreachable(const EpPair& ep, const char* statusText);
  // Records any other error completion: the QP is in ERR, but no event is raised.
  void RecordQpFailed(const EpPair& ep);
  // Name of an available device; availDevices is fixed, so this takes no lock.
  std::string DeviceName(int devId) const;
  // Session admission only: never reject part-way through posting a transfer.
  bool RejectIfPortDown(const EpPairVec& eps, TransferStatus* status);

 private:
  application::RdmaDeviceContext* GetOrCreateDeviceContext(int devId);

 private:
  RdmaBackendConfig config;
  mutable std::shared_mutex mu;

  application::RdmaContext* ctx;
  application::ActiveDevicePortList availDevices;
  std::vector<application::RdmaDeviceContext*> deviceCtxs;

  MemoryTable mTable;
  std::unordered_map<EngineKey, RemoteEngineMeta> remotes;
  std::atomic<EndpointId> nextEndpointId_{1};
  std::unordered_map<EndpointId, std::shared_ptr<EndpointRuntime>> endpointsById_;
  std::atomic<uint64_t> endpointsEpoch_{0};
  std::shared_ptr<ReleasedEndpoints> released_{std::make_shared<ReleasedEndpoints>()};
  std::atomic<size_t> connectedEndpoints_{0};

  std::unique_ptr<application::TopoSystem> topo{nullptr};
  std::atomic<uint32_t> roundRobinCounter{0};

  // Self-synchronizing, so not guarded by `mu`; its writer threads join first.
  PeerFailureTracker peerFailures_;
  bool portDownFailFast_{true};
  std::unique_ptr<std::atomic<bool>[]> portDown_;
  std::unique_ptr<std::atomic<uint64_t>[]> portRejected_;

  // Declared last so it is the first member destroyed, and explicitly reset at
  // the top of ~RdmaManager: the monitor thread calls into peerFailures_.
  std::unique_ptr<RdmaAsyncEventMonitor> asyncEventMonitor_;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                      Notification Manager                                      */
/* ---------------------------------------------------------------------------------------------- */
class NotifManager {
 public:
  NotifManager(RdmaManager*, const RdmaBackendConfig&);
  ~NotifManager();

  void RegisterEndpoint(const std::shared_ptr<EndpointRuntime>& rt);
  // Stops watching endpoints detached from RdmaManager, and returns once the poll loop
  // holds none of them any more.
  void UnregisterEndpoints(const RdmaManager::SnapshotVector& runtimes);
  // Drops the notifications received from `remote`.
  void ForgetRemoteEngine(const EngineKey& remote);

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
  // Poll loop iterations so far. An iteration drops every runtime the previous one held,
  // so once it has moved on by two, none detached before can still be held there.
  std::atomic<uint64_t> pollRounds_{0};

  // Notification context
 private:
  std::unordered_map<EndpointId, std::shared_ptr<EndpointRuntime>> registeredRuntimes_;
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
  ControlPlaneServer(const std::string& key, const std::string& host, int port,
                     const RdmaBackendConfig& config, RdmaManager*, NotifManager*);
  ~ControlPlaneServer();

  std::optional<uint16_t> GetListenPort() const {
    if (!ctx) return std::nullopt;
    return static_cast<uint16_t>(ctx->GetPort());
  }

  // Remote engine meta management
  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);
  std::optional<int> TryGetRemoteEnginePort(const EngineKey&) const;
  // Refuses the connections a remote engine asks for until AllowIncoming, and waits out
  // the one being accepted, so none is published while that engine is being torn down.
  // Not to be called on the server thread.
  void BlockIncoming(const EngineKey&);
  void AllowIncoming(const EngineKey&);
#ifdef MORI_IO_FAULT_INJECTION
  // Tests only: runs on the server thread between connecting the endpoint a remote engine
  // asked for and publishing it.
  static std::function<void(const EngineKey&)> testIncomingConnected;
#endif

  // Endpoint management
  void BuildRdmaConn(EngineKey, TopoKeyPair, int nicRank);

  // MemoryRegion management
  void RegisterMemory(MemoryDesc&);
  void DeregisterMemory(const MemoryDesc&);
  application::RdmaMemoryRegion AskRemoteMemoryRegion(EngineKey, int rdevId, MemoryUniqueId);

  // Server management
  void MainLoop();
  void Start();
  void Shutdown();

 private:
  // Counts a connection a remote engine asked for while its endpoint is created and
  // published; refuses it while that engine is blocked.
  class IncomingConn;

  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);
  void DropConnection(int fd) noexcept;

 private:
  EngineKey myEngKey;
  RdmaBackendConfig config{};

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

  std::mutex incomingMu_;
  std::condition_variable incomingDone_;
  std::unordered_set<EngineKey> incomingBlocked_;
  std::unordered_map<EngineKey, int> incomingInFlight_;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                       RdmaBackendSession                                       */
/* ---------------------------------------------------------------------------------------------- */
class RdmaBackendSession : public BackendSession {
 public:
  RdmaBackendSession() = default;
  RdmaBackendSession(const RdmaBackendConfig& config,
                     std::vector<application::RdmaMemoryRegion> localMrPerEp,
                     std::vector<application::RdmaMemoryRegion> remoteMrPerEp, const EpPairVec& eps,
                     Executor* executor, RdmaManager* rdma,
                     MemoryLocationType localLoc = MemoryLocationType::CPU);
  ~RdmaBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

  bool Alive() const;
  bool NeedsQpReplacement() const;

 private:
  RdmaBackendConfig config{};
  std::vector<application::RdmaMemoryRegion> localMrPerEp{};
  std::vector<application::RdmaMemoryRegion> remoteMrPerEp{};
  EpPairVec eps{};
  Executor* executor{nullptr};
  // Source of QP liveness for Alive(). Null on a default-constructed session,
  // which reports alive because nothing is known to have failed.
  RdmaManager* rdma_{nullptr};
  MemoryLocationType localLoc_{MemoryLocationType::CPU};
  std::shared_ptr<std::atomic<bool>> warnedChunkedWorkerFallback_{
      std::make_shared<std::atomic<bool>>(false)};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaBackend                                          */
/* ---------------------------------------------------------------------------------------------- */

class RdmaBackend : public Backend {
 public:
  RdmaBackend(EngineKey, const IOEngineConfig&, const RdmaBackendConfig&);
  ~RdmaBackend();

  static bool HasActiveDevices();

  std::optional<uint16_t> GetListenPort() const {
    if (!server) return std::nullopt;
    return server->GetListenPort();
  }

  void RegisterRemoteEngine(const EngineDesc&) override;
  void DeregisterRemoteEngine(const EngineDesc&) override;
  void RegisterMemory(MemoryDesc& desc) override;
  void DeregisterMemory(const MemoryDesc& desc) override;
  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead) override;
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead) override;
  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) override;
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                TransferStatus* status) override;
  bool PopPeerFailure(PeerFailureEvent* out) override;
  bool CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const override;
  size_t NumConnectedEndpoints() const { return rdma->NumConnectedEndpoints(); }

 private:
  // With paceQpReplacement, building QPs in place of failed ones is refused with
  // QpReplacementBackoff while PaceQpReplacement says to wait.
  void CreateSession(const MemoryDesc& local, const MemoryDesc& remote, RdmaBackendSession& sess,
                     bool paceQpReplacement = false);
  // Session cache helpers
  struct SessionCacheKey {
    EngineKey remoteEngineKey;  // use remote memory's engine key
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
        // 64-bit variant of boost::hash_combine / splitmix64 inspired
        seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      std::size_t seed = 0;
      hash_combine(seed, std::hash<std::string>()(k.remoteEngineKey));
      hash_combine(seed, std::hash<uint64_t>()(k.localMemId));
      hash_combine(seed, std::hash<uint64_t>()(k.remoteMemId));
      return seed;
    }
  };
  struct ConnBuildKey {
    EngineKey remoteEngineKey;
    TopoKeyPair topo;
    bool operator==(const ConnBuildKey& o) const {
      return remoteEngineKey == o.remoteEngineKey && topo == o.topo;
    }
  };
  struct ConnBuildKeyHash {
    std::size_t operator()(const ConnBuildKey& k) const noexcept {
      std::size_t topoHash = std::hash<TopoKeyPair>{}(k.topo);
      std::size_t engineHash = std::hash<std::string>{}(k.remoteEngineKey);
      return topoHash ^ (engineHash + 0x9e3779b97f4a7c15ULL + (topoHash << 6) + (topoHash >> 2));
    }
  };
  struct QpReplacementPacing {
    std::chrono::steady_clock::time_point lastAt{};
    std::chrono::milliseconds interval{0};
  };
  // Records a replacement of failed QPs to `key` and returns how long the next one waits;
  // with `enforce`, throws QpReplacementBackoff instead while the last one says to wait.
  std::chrono::milliseconds PaceQpReplacement(const ConnBuildKey& key, bool enforce);
  std::shared_ptr<RdmaBackendSession> GetOrCreateSessionCached(const MemoryDesc& local,
                                                               const MemoryDesc& remote);
  void InvalidateSessionsForMemory(MemoryUniqueId id);
  std::shared_ptr<std::mutex> GetConnBuildLock(const EngineKey& remoteEngineKey,
                                               const TopoKeyPair& topo);

 private:
  EngineKey myEngKey;
  RdmaBackendConfig config;
  std::unique_ptr<RdmaManager> rdma{nullptr};
  std::unique_ptr<NotifManager> notif{nullptr};
  std::unique_ptr<ControlPlaneServer> server{nullptr};
  std::unique_ptr<Executor> executor{nullptr};
  // session cache; a caller keeps its session alive while it uses it, even once the
  // session is replaced or dropped from the cache.
  std::unordered_map<SessionCacheKey, std::shared_ptr<RdmaBackendSession>, SessionCacheKeyHash>
      sessionCache;
  // After an attempt to replace a failed session failed (the peer may be gone, or the
  // replacement backed off), when to try again. Guarded by sessionCacheMu.
  std::unordered_map<ConnBuildKey, std::chrono::steady_clock::time_point, ConnBuildKeyHash>
      replaceRetryAt;
  std::mutex sessionCacheMu;
  // Serializes registering and deregistering one remote engine key. Per key, because a
  // deregistration can wait out a connection build to a dead peer.
  std::shared_ptr<std::mutex> GetPeerLock(const EngineKey& remoteEngineKey);
  std::mutex peerLocksMu_;
  std::unordered_map<EngineKey, std::shared_ptr<std::mutex>> peerLocks_;
  std::mutex connBuildMapMu_;
  std::unordered_map<ConnBuildKey, std::shared_ptr<std::mutex>, ConnBuildKeyHash> connBuildMu_;
  // Failed QPs stay allocated until their remote engine is deregistered, so QPs that fail
  // again soon after being replaced, as on a path that stays down, are replaced at a
  // growing interval. Guarded by connBuildMapMu_.
  std::unordered_map<ConnBuildKey, QpReplacementPacing, ConnBuildKeyHash> qpReplacementPacing_;
};

}  // namespace io
}  // namespace mori
