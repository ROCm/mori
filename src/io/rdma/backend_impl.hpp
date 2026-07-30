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
#include <deque>
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

namespace internal {

// Placeholder written into engine_desc.port when an RDMA backend request is
// rerouted to XGMI-only mode because this host has no active RDMA device.
inline constexpr uint16_t kXgmiOnlyFallbackPlaceholderPort = 1;

}  // namespace internal

void ValidateRdmaTransferConfig(const RdmaBackendConfig& config);
bool UsesInlineOnly(const RdmaBackendConfig& config);
int ResolveRequestedNics(const RdmaBackendConfig& config, const TopoKey& local,
                         const TopoKey& remote);
std::vector<int> BuildDesiredQpCounts(int totalQp, int numRanks);
EpPairVec InterleaveEndpointsByLocalDevice(const EpPairVec& eps,
                                           const std::vector<int>& localDevOrder,
                                           const std::vector<int>& wantPerRank);

/* ---------------------------------------------------------------------------------------------- */
/*                                           RdmaManager                                          */
/* ---------------------------------------------------------------------------------------------- */
class RdmaManager {
 public:
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
  // The in-flight gate for one local memory id, created on first use. A
  // session captures this at CreateSession time and every post takes a token
  // from it, so `QuiesceLocalMemory` below can wait the NIC out before
  // ibv_dereg_mr destroys the lkey those posts carry. See MemoryInflightGate.
  std::shared_ptr<MemoryInflightGate> GetOrCreateLocalMemoryGate(MemoryUniqueId);
  // Close the gate to new posts and wait for outstanding ones to complete.
  // Returns true if it drained. KEEPS the retired gate in the map as a
  // tombstone (creating one if the id had none) so that a session built during
  // the dereg is REFUSED rather than handed a fresh live gate -- see the
  // definition for the exact interleaving that erasing here left open
  // (REVIEW_M #73-1/#74-1/#75-1). `ClearLocalMemoryGate` lifts the tombstone.
  bool QuiesceLocalMemory(MemoryUniqueId, int timeoutMs);
  // Lift the tombstone: the id has been re-registered, so posting against it is
  // safe again. Returns true if a RETIRED gate was cleared (i.e. the tombstone
  // was doing work), false if there was nothing or it was still live.
  bool ClearLocalMemoryGate(MemoryUniqueId);
  // Publish the retention to the dereg census, so the tombstone is a NUMBER a
  // test can watch rise and fall rather than a reassurance in a comment. Reads
  // the incrementally-maintained `retiredGateCount_`; it does NOT scan
  // memGates_ (REVIEW_M #76-1 -- the scan made every dereg O(N)). Takes `mu`;
  // do not call with it held.
  void PublishGateTombstoneCount();
  // Evict retired gates in excess of a bound, oldest first, and return how many
  // went (REVIEW_M #76-1). The tombstone exists to refuse a post built from a
  // session that was already in flight when the dereg started; once that post
  // has been refused, or the drain window has long closed, the entry is dead
  // weight. `IOEngine::RegisterMemory` mints a FRESH id per call, so in
  // production no tombstone is ever LIFTED and the map would otherwise grow one
  // entry per deregistered id for the life of the process -- a curve across a
  // flip stress, which is what turns 44-45 removed from two other registries.
  // Bounded rather than time-based: a bound is what a test can assert FLAT.
  std::size_t ReapMemoryGates();

  // Remote memory management APIs
  std::optional<application::RdmaMemoryRegion> GetRemoteMemory(EngineKey, int remRdmaDevId,
                                                               MemoryUniqueId);
  void RegisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId,
                            application::RdmaMemoryRegion);
  void DeregisterRemoteMemory(EngineKey, int remRdmaDevId, MemoryUniqueId);
  // Drops EVERY cached remote MR for one engine. Needed on a PD role flip: the
  // peer tears down and re-registers its buffers, so every rkey we cached for
  // it is dead. Returns how many entries were dropped (0 == we had nothing
  // cached for that engine), so the caller can log it.
  std::size_t InvalidateRemoteMemoryForEngine(const EngineKey&);

  // Endpoint management APIs
  int CountEndpoint(EngineKey, TopoKeyPair);
  EpPairVec GetAllEndpoint(EngineKey, TopoKeyPair);
  application::RdmaEndpoint CreateEndpoint(int devId);
  EndpointId ConnectEndpoint(EngineKey remoteKey, int ldevId, application::RdmaEndpoint local,
                             int rdevId, application::RdmaEndpointHandle remote, TopoKeyPair key,
                             int weight);
  std::shared_ptr<EndpointRuntime> GetEndpointRuntime(EndpointId id);
  std::vector<std::shared_ptr<EndpointRuntime>> SnapshotEndpointRuntimes();
  // Remove one endpoint from the ROUTE TABLE so CreateSession's
  // `CountEndpoint < qpPerTransfer` test builds a replacement instead of
  // handing out the retired one. Called when a QP takes a non-flush completion
  // error, i.e. when it is in the RC ERROR state and unusable forever.
  //
  // Deliberately does NOT erase `endpointsById_`: the CQ poll loop holds a
  // shared_ptr to that runtime and is mid-drain of the flush cascade when this
  // runs, and the flushed WRs still need their ledger records released so the
  // transfers they belong to report failure rather than hang. The runtime is
  // reaped with the backend; retiring it from routing is what stops NEW work
  // landing on it. Returns how many route-table entries were removed (0 if it
  // was already retired, so the caller's log is not duplicated per CQE).
  std::size_t RetireEndpoint(EndpointId id);
  // Reap retired endpoints out of `endpointsById_`, and hence out of the CQ
  // poll set. RetireEndpoint deliberately does NOT do this (see above): the
  // poll loop is mid-drain of the flush cascade when it runs, and those
  // flushed WRs still need their ledger records released so their transfers
  // report failure instead of hanging. So the erase is DEFERRED to here and
  // gated on the ledger being empty, which is the point at which the NIC has
  // nothing left referencing the QP.
  //
  // Without this, `endpointsById_` was insert-only for the life of the
  // process and every retirement permanently added one `ibv_poll_cq` to
  // NotifManager::MainLoop's per-round walk (backend_impl.cpp, the POLLING
  // branch snapshots the whole map every round, and sglang runs
  // PollCqMode::POLLING -- conn.py:370). REVIEW_M #72-2: after 24 flip cycles
  // T37 measured 768 distinct QPs, i.e. 768 dead CQs polled per round. That
  // makes it a LATENCY regression on the flip stress, not merely a leak, so
  // it threatens ACCEPTANCE *Performance* and not just *Robustness*.
  //
  // Safe to call from the poll thread itself: the caller iterates a SNAPSHOT
  // of shared_ptrs, so erasing from the map does not invalidate anything it
  // holds, and the runtime stays alive until that snapshot is dropped.
  // Returns the ids that were reaped, NOT just a count: NotifManager holds a
  // second shared_ptr to each runtime in `registeredRuntimes_` (plus, when
  // notification is on, a pinned buffer and an MR in `notifCtxById_`), and the
  // caller must reap those under NotifManager's own lock. Erasing only here
  // leaves the QP resident for the life of the process and keeps
  // `numRegisteredRuntimes` -- the counter that is live at sglang's
  // `enableNotification=false` -- growing per dead QP. REVIEW_M #74-2.
  std::vector<EndpointId> ReapRetiredEndpoints();
  // Route-table endpoints for one engine+topo that are NOT QP-fatal.
  int CountUsableEndpoint(EngineKey, TopoKeyPair);

  application::RdmaDeviceContext* GetRdmaDeviceContext(int devId);

  // --- Per-remote-engine retention counters (read-only diagnostics) ---------
  // A PD flip gives the flipping side a brand-new engine key, so the SURVIVING
  // peer accumulates one dead-key entry per flip in each of the maps below and
  // never drops any of them (see the reachability note at
  // RdmaBackend::DeregisterRemoteEngine). These make that accumulation
  // OBSERVABLE from a test, which is the only way to tell a real leak from a
  // plausible source reading -- the same reason ShmemGetHeapStats exists.
  // Rank-local, read-only, taken under the same lock as the mutators, so they
  // are a consistent snapshot rather than a torn read.
  std::size_t GetNumRemoteEngines() const;
  std::size_t GetNumEndpointRuntimes() const;
  // Endpoints retained for ONE engine key, summed over its topo pairs. 0 both
  // when the key is unknown and when it is known but holds no endpoints.
  std::size_t GetNumEndpointsForEngine(const EngineKey&) const;

 private:
  application::RdmaDeviceContext* GetOrCreateDeviceContext(int devId);

 private:
  RdmaBackendConfig config;
  mutable std::shared_mutex mu;

  application::RdmaContext* ctx;
  application::ActiveDevicePortList availDevices;
  std::vector<application::RdmaDeviceContext*> deviceCtxs;

  MemoryTable mTable;
  // Keyed by memory id, NOT MemoryKey{devId,id}: one dereg call tears down
  // every per-device MR for the id (see DeregisterLocalMemory(desc)), so the
  // barrier has to cover the id as a whole or a multi-NIC session would drain
  // one device and dereg the other out from under itself.
  std::unordered_map<MemoryUniqueId, std::shared_ptr<MemoryInflightGate>> memGates_;
  // REVIEW_M #76-1. The tombstone's retention, maintained INCREMENTALLY under
  // `mu` at the two places a gate can start or stop retiring, rather than
  // recounted by walking memGates_. The walk made every dereg O(N) in the
  // number of retired ids, and N is unbounded on sglang's path -- an O(flips^2)
  // cost on the *Performance* clause, not merely a leak. An entry can only
  // become retiring in QuiesceLocalMemory and can only stop by being erased in
  // ClearLocalMemoryGate/ReapMemoryGate, so an exact count needs no scan.
  std::size_t retiredGateCount_{0};
  // Retirement order, so ReapMemoryGates can evict OLDEST first: the youngest
  // tombstone is the one a racing post is most likely to still hit. Holds ids,
  // not iterators/pointers, so an entry lifted by ClearLocalMemoryGate in the
  // meantime is simply not found at reap time and skipped.
  std::deque<MemoryUniqueId> retiredOrder_;
  // How many tombstones to retain, read from MORI_IO_MAX_MEM_GATE_TOMBSTONES
  // ONCE PER MANAGER at construction. Was a function-local `static const`, i.e.
  // latched process-wide on first use; T43 measured that failing
  // `rdma_gate_tombstones_are_bounded` (`settled=24 bound=4`) because six
  // earlier cases had already cached the default. Per-instance is also what the
  // knob's name promises when a process hosts more than one engine.
  const std::size_t maxMemGateTombstones_;
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

  // The NotifManager half of the reap. Takes the ids returned by
  // RdmaManager::ReapRetiredEndpoints -- i.e. QPs that are qpFatal AND whose
  // ledger has drained -- and drops them from `registeredRuntimes_`, removes
  // their comp-channel from epoll in EVENT mode, and frees + deregisters their
  // notification context. Returns how many were found and reaped.
  std::size_t ReapEndpoints(const std::vector<EndpointId>& ids);

  // Read-only diagnostics, mirroring RdmaManager's. `notifCtxById_` is the one
  // that costs real resources: RegisterEndpoint posix_memaligns
  // notifPerQp*sizeof(NotifMessage) AND registers an RDMA MR per QP, and there
  // is no per-endpoint removal anywhere -- only Shutdown() frees them. So this
  // counter IS the pinned-host-memory + MR leak, in units of QPs.
  std::size_t GetNumRegisteredRuntimes() const;
  std::size_t GetNumNotifContexts() const;
  // Bytes of pinned host memory held by those notification contexts.
  std::size_t GetNotifBufferBytes() const;

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
  // Read-only diagnostic: how many remote engine descriptors are retained.
  std::size_t GetNumRemoteEngines() const;

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
  void AcceptRemoteEngineConn();
  void HandleControlPlaneProtocol(int fd);
  // Close and forget one control-plane fd. Called both on peer-close and when
  // HandleControlPlaneProtocol throws, so a bad message costs the connection
  // rather than std::terminate-ing the engine from MainLoop's thread.
  void DropControlPlaneConn(int fd);

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
                     Executor* executor,
                     std::shared_ptr<MemoryInflightGate> localGate = nullptr);
  ~RdmaBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);

  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

  bool Alive() const;

 private:
  RdmaBackendConfig config{};
  std::vector<application::RdmaMemoryRegion> localMrPerEp{};
  std::vector<application::RdmaMemoryRegion> remoteMrPerEp{};
  EpPairVec eps{};
  Executor* executor{nullptr};
  // Gate for the LOCAL memory id this session was built against. `localMrPerEp`
  // is a set of POD copies with no back-reference to the ibv_mr, so this is the
  // only thing tying the session's posts to the lifetime of the registration.
  // Null only for a default-constructed session (the CreateSession(local,
  // remote) overload fills it in before use).
  std::shared_ptr<MemoryInflightGate> localGate{};
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

  // One snapshot of everything this backend retains per remote engine. Exists
  // so a multi-cycle flip stress can assert on NUMBERS instead of on a reading
  // of the source: every field below is expected to stay FLAT across repeated
  // peer churn and today several of them grow without bound.
  struct RemoteRetentionStats {
    std::size_t numRemoteEngines{0};    // ControlPlaneServer::engines
    std::size_t numRemoteMetas{0};      // RdmaManager::remotes (rTable+mTable)
    std::size_t numEndpointRuntimes{0}; // RdmaManager::endpointsById_
    std::size_t numSessions{0};         // RdmaBackend::sessionCache
    std::size_t numNotifContexts{0};    // NotifManager::notifCtxById_ (QPs)
    std::size_t notifBufferBytes{0};    // pinned host memory behind those QPs
    // NotifManager::registeredRuntimes_. This is the ONE counter above that is
    // live in sglang's configuration, and it was missing, which made the whole
    // struct misleading there: sglang constructs RdmaBackendConfig with
    // enableNotification=FALSE (conn.py:376-382), and under that
    // NotifManager::RegisterEndpoint early-returns right after
    // `registeredRuntimes_[rt->id] = rt` and never touches notifCtxById_. So
    // numNotifContexts and notifBufferBytes are permanently 0 as E runs it,
    // and the per-flip retention shows up HERE instead. (REVIEW_M #69-2/#70-2.)
    std::size_t numRegisteredRuntimes{0};
  };
  RemoteRetentionStats GetRemoteRetentionStats() const;

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
  bool CanHandle(const MemoryDesc& local, const MemoryDesc& remote) const override;

 private:
  void CreateSession(const MemoryDesc& local, const MemoryDesc& remote, RdmaBackendSession& sess);
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
  // Returns a SHARED owner, by value, and not a raw pointer into the cache.
  // The two Invalidate* below erase entries while a transfer thread is between
  // its lookup and its `sess->ReadWrite`; with `unique_ptr` + `.get()` that
  // erase destroyed the object under the transfer. Handing out a shared_ptr
  // makes the cache one owner among several: an erase unpublishes the session
  // but the in-flight transfer's copy keeps it alive until it returns.
  std::shared_ptr<RdmaBackendSession> GetOrCreateSessionCached(const MemoryDesc& local,
                                                               const MemoryDesc& remote);
  // Same, but reports failure through `status` and returns nullptr instead of
  // throwing on the caller's thread. See the definition for why that matters.
  std::shared_ptr<RdmaBackendSession> GetOrCreateSessionCachedNoThrow(const MemoryDesc& local,
                                                                      const MemoryDesc& remote,
                                                                      TransferStatus* status);
  void InvalidateSessionsForMemory(MemoryUniqueId id);
  // Engine-scoped counterpart of the above, for a peer that flipped role. The
  // sessionCache is keyed {engineKey, localId, remoteId} and holds MRs/endpoint
  // sets captured at CreateSession time, so without this a post-flip transfer
  // to the same {local,remote} id pair reuses the peer's PRE-flip rkeys and
  // never calls CreateSession at all.
  void InvalidateSessionsForEngine(const EngineKey& ekey);
  std::shared_ptr<std::mutex> GetConnBuildLock(const EngineKey& remoteEngineKey,
                                               const TopoKeyPair& topo);

 private:
  EngineKey myEngKey;
  RdmaBackendConfig config;
  // How long DeregisterMemory waits for the NIC before it deregisters anyway,
  // read from MORI_IO_DEREG_QUIESCE_MS ONCE PER BACKEND at construction. Was a
  // function-local `static const`, i.e. latched process-wide on first use --
  // the same defect b2453e44 removed from maxMemGateTombstones_, and the reason
  // no test could ever drive the timed-out-quiesce path.
  const int deregQuiesceTimeoutMs_;
  std::unique_ptr<RdmaManager> rdma{nullptr};
  std::unique_ptr<NotifManager> notif{nullptr};
  std::unique_ptr<ControlPlaneServer> server{nullptr};
  std::unique_ptr<Executor> executor{nullptr};
  // session cache
  std::unordered_map<SessionCacheKey, std::shared_ptr<RdmaBackendSession>, SessionCacheKeyHash>
      sessionCache;
  // mutable: GetRemoteRetentionStats() is a const read-only accessor and must
  // still take this lock to get a consistent snapshot rather than a torn read.
  mutable std::mutex sessionCacheMu;
  std::mutex connBuildMapMu_;
  std::unordered_map<ConnBuildKey, std::shared_ptr<std::mutex>, ConnBuildKeyHash> connBuildMu_;
};

}  // namespace io
}  // namespace mori
