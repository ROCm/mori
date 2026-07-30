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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/msgpack_adaptor.hpp"
#include "src/io/call_diagnostics_internal.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                     Common Data Structures                                     */
/* ---------------------------------------------------------------------------------------------- */
struct TopoKey {
  int deviceId;
  MemoryLocationType loc;
  int numaNode{-1};

  bool operator==(const TopoKey& rhs) const noexcept {
    return (deviceId == rhs.deviceId) && (loc == rhs.loc) && (numaNode == rhs.numaNode);
  }

  MSGPACK_DEFINE(deviceId, loc, numaNode);
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

}  // namespace io
}  // namespace mori

namespace std {
template <>
struct hash<mori::io::TopoKey> {
  std::size_t operator()(const mori::io::TopoKey& k) const noexcept {
    std::size_t h1 = std::hash<uint32_t>{}(k.deviceId);
    std::size_t h2 = std::hash<uint32_t>{}(static_cast<uint32_t>(k.loc));
    std::size_t h3 = std::hash<int>{}(k.numaNode);
    std::size_t seed = h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    return seed ^ (h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2));
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

struct NotifMessage {
  TransferUniqueId id{0};
  int qpIndex{-1};
  int totalNum{-1};
};

// wr_id namespace:
//   Zone A: notification RECV indices [0, notifPerQp)
//   Zone B: ledger record IDs [notifPerQp, 2^63)
//   Zone C: notification SEND IDs tagged with bit 63
static constexpr uint64_t kNotifSendWrIdTag = uint64_t{1} << 63;

inline bool IsNotifSendWrId(uint64_t wr_id) { return (wr_id & kNotifSendWrIdTag) != 0; }

inline TransferUniqueId ExtractTransferIdFromWrId(uint64_t wr_id) {
  return static_cast<TransferUniqueId>(wr_id & ~kNotifSendWrIdTag);
}

uint64_t MakeNotifSendWrId(TransferUniqueId id);

std::vector<std::pair<uint64_t, uint32_t>> PlanChunks(uint32_t total, size_t chunkBytes,
                                                      int maxChunks);

// Keeps `ibv_dereg_mr` from running while work requests that carry the MR's
// lkey are still outstanding on a send queue.
//
// `8f2d80b2` made the C++ *session object* outlive a concurrent
// `DeregisterMemory`, but the session holds its MRs BY VALUE as POD
// {addr,lkey,rkey,length}, and `RdmaBackend::DeregisterMemory` calls
// `ibv_dereg_mr` (rdma.cpp:381) before it invalidates the cache. So the
// transfer the shared_ptr just kept alive goes on to post a WR with a lkey the
// driver has already destroyed. Best case a CQE error; worst case the flip's
// immediate re-registration recycles that key and the DMA lands in the NEW KV
// buffer -- silent corruption, which is exactly the failure a gsm8k A/B shows
// as "within noise" one run and garbage the next.
//
// Ordering alone does NOT close that: moving the invalidate before the dereg
// still leaves a poster that read the MR a microsecond earlier, and it does
// nothing for a WR already sitting in the SQ. What is needed is a barrier, so
// this is a refcount whose release is driven by the CQE, not by the post
// returning: the token below is parked in `CqCallbackMeta`, the ledger holds
// that meta until the completion arrives, so the count reaches zero only once
// the NIC is done reading the region.
//
// SCOPE, stated so it is not over-read: this is ONE-SIDED. It protects the
// local lkey of the process that deregisters. A peer that deregisters while we
// have a read outstanding against its rkey is a different failure (CQE
// status 10) and is handled by the QP retirement in `d862b1c5`, not here.
class MemoryInflightGate : public std::enable_shared_from_this<MemoryInflightGate> {
 public:
  // Returns a token that must be kept alive until the NIC is done with the
  // region, or nullptr if the memory is already retiring -- in which case the
  // caller must NOT post.
  static std::shared_ptr<void> Acquire(const std::shared_ptr<MemoryInflightGate>& gate);

  // Closes the gate to new posts and blocks until every outstanding token is
  // released. Returns true if it drained, false if `timeoutMs` expired first
  // (the caller then has to decide; a flip cannot block forever).
  bool Quiesce(int timeoutMs);

  bool Retiring() const { return retiring_.load(std::memory_order_acquire); }
  int Inflight() const { return inflight_.load(std::memory_order_acquire); }

  // How many posts this gate was holding at the moment Quiesce closed it, i.e.
  // how many work requests would have had their lkey destroyed under them by
  // the old code. Zero means the barrier was a no-op for that dereg. A test
  // that does not see this go positive has not driven the race, so it must
  // report itself VACUOUS rather than green -- the same non-vacuity discipline
  // the endpoint-retirement counters exist for.
  int InflightAtQuiesce() const { return inflightAtQuiesce_.load(std::memory_order_acquire); }

 private:
  void Release();

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<int> inflight_{0};
  std::atomic<int> inflightAtQuiesce_{0};
  std::atomic<bool> retiring_{false};
};

// Process-wide census of what the barrier actually did, so a test can assert on
// NUMBERS instead of on a reading of the source. Rank-local, monotone,
// diagnostics only -- nothing branches on these.
struct DeregQuiesceCensus {
  std::size_t quiesceCalls{0};      // DeregisterMemory reached the barrier
  std::size_t quiesceWaited{0};     // ...and there was >=1 WR outstanding
  std::size_t quiesceTimedOut{0};   // ...and it gave up and deregistered anyway
  std::size_t postsRefused{0};      // a post was refused because the gate shut
  std::size_t maxInflightAtQuiesce{0};
  // REVIEW_M #72-2. Retired endpoint runtimes dropped out of the CQ poll set by
  // ReapRetiredEndpoints. Counted here rather than as a fresh instrument
  // because the reap and the barrier are the same kind of claim -- "a cleanup
  // that must be shown to have RUN, not just to have compiled" -- and every
  // test that asserts on one already reads this struct. Structurally 0 before
  // the reap existed, which is what makes an assertion on it two-sided.
  std::size_t endpointsReaped{0};
  // REVIEW_M #75-1. Retired-but-not-yet-reopened gates currently resident in
  // memGates_ -- the tombstones that make the barrier reentry-safe. Exposed,
  // and NOT monotone like the rest (it can go down when RegisterMemory lifts
  // one), because the tombstone is a deliberate RETENTION and this team does
  // not get to add one silently after spending five turns on other people's:
  // `IOEngine::RegisterMemory` mints a FRESH id per registration
  // (engine.cpp:363), so a re-registered buffer does NOT lift its
  // predecessor's tombstone, and this grows one entry per deregistered id for
  // the life of the process. An empty MemoryInflightGate is ~64 bytes plus a
  // map node, so at sglang's handful of descriptors per flip it is small --
  // but it IS a curve, it is measured here, and a test asserts on the number
  // rather than on that reassurance.
  std::size_t gateTombstones{0};
};
DeregQuiesceCensus GetDeregQuiesceCensus();
void RecordQuiesce(int inflightAtClose, bool drained);
void RecordPostRefused();
void RecordEndpointsReaped(std::size_t n);
// Absolute, not a delta: the current resident count. Set by RdmaManager under
// its own lock every time memGates_ gains or loses a retired entry.
void RecordGateTombstones(std::size_t resident);

struct CqCallbackMeta {
  CqCallbackMeta(TransferStatus* s, TransferUniqueId id_, int n)
      : status(s), id(id_), totalBatchSize(n) {}

  TransferStatus* status{nullptr};
  TransferUniqueId id{0};
  int totalBatchSize{0};
  std::atomic<uint32_t> finishedBatchSize{0};
  internal::IoCallDiagnostics diagnostics{};
  // Held for the whole lifetime of the meta, which the ledger extends until the
  // CQE lands. Destroying it is what lets a concurrent DeregisterMemory
  // proceed. Never read; its destructor is the point.
  std::shared_ptr<void> inflightToken{};
};

// SubmissionLedger: tracks per-EP WR submissions and enables precise sqDepth release.
enum class SubmissionState : uint8_t {
  Posted,    // submitted, awaiting CQE
  Orphaned,  // partial post without signaled tail; awaits recovery
};

struct SubmissionRecord {
  uint64_t recordId{0};
  int postedWr{0};
  bool hasSignaledTail{false};
  SubmissionState state{SubmissionState::Posted};
  std::shared_ptr<CqCallbackMeta> meta;
  int batchSize{0};
};

class SubmissionLedger {
 public:
  explicit SubmissionLedger(uint32_t notifPerQp) : nextId_{notifPerQp} {}

  // Allocate recordId, insert Posted record, return recordId.
  uint64_t Insert(int postedWr, bool hasSignaledTail, std::shared_ptr<CqCallbackMeta> meta,
                  int batchSize);

  // Insert an Orphaned record (partial post, no signaled tail).
  void InsertOrphaned(int postedWr, std::shared_ptr<CqCallbackMeta> meta, int batchSize);

  // CQE path: find record by recordId, release sqDepth, return CqCallbackMeta.
  // Returns nullptr if record not found.
  std::shared_ptr<CqCallbackMeta> ReleaseByCqe(uint64_t recordId, std::atomic<int>* sqDepth,
                                               int* outBatchSize);

  // Recovery path: release only Orphaned records and keep Posted records.
  int ReleaseOrphanedByRecovery(std::atomic<int>* sqDepth);

  bool HasOrphaned() const;

  // Outstanding records — Posted plus Orphaned. Zero means the NIC has nothing
  // left that references this QP, which is the precondition for reaping its
  // runtime out of the poll set.
  std::size_t NumRecords() const;

 private:
  mutable std::mutex mu_;
  uint64_t nextId_;
  std::unordered_map<uint64_t, SubmissionRecord> records_;
};

using EndpointId = uint64_t;

struct EpPair {
  // The identity of the QP this pair wraps, assigned by ConnectEndpoint and
  // copied along with the pair into the route table, the EndpointRuntime and
  // every cached session. RetireEndpoint used to match on `local.handle.qpn`
  // instead, which OVER-MATCHES: QP numbers are unique within a device
  // context, not across NICs, and a node with more than one HCA (this one has
  // bnxt + mlx5) can hand the same qpn to two live QPs on different devices.
  // The route-table walk then erased a HEALTHY endpoint on the other NIC
  // without setting its qpFatal, so it was orphaned rather than retired — a
  // silently missing route that also feeds the un-reaped-runtime leak.
  // 0 means "not from ConnectEndpoint" (default-constructed, e.g. in unit
  // tests) and must never match a retirement.
  EndpointId id{0};
  int weight;
  int ldevId;
  int rdevId;
  EngineKey remoteEngineKey;
  application::RdmaEndpoint local;
  application::RdmaEndpointHandle remote;
  // Shared across EpPair copies that refer to the same QP.
  std::shared_ptr<std::atomic<int>> sqDepth;
  int maxSqDepth{0};
  // Degraded flag — set on partial post without signaled tail.
  std::shared_ptr<std::atomic<bool>> degraded;
  std::shared_ptr<SubmissionLedger> ledger;
  // QP-fatal flag. `degraded` is RECOVERABLE (ProcessOneCqe clears it once the
  // orphaned WRs drain); this one is NOT. On a Reliable Connected QP every
  // completion error other than IBV_WC_WR_FLUSH_ERR transitions the QP to the
  // ERROR state, and mori never drives it back (there is no ibv_modify_qp
  // anywhere under src/io), so the QP is dead for the life of the process.
  // Shared across every EpPair copy of the same QP — the route table's, the
  // EndpointRuntime's and every cached session's — so one store retires it
  // everywhere at once.
  std::shared_ptr<std::atomic<bool>> qpFatal;

  bool IsQpFatal() const {
    return qpFatal && qpFatal->load(std::memory_order_relaxed);
  }
};

struct EndpointRuntime {
  EndpointRuntime() = default;
  EndpointRuntime(EndpointId id_, const EpPair& ep_) : id(id_), ep(ep_) {}

  EndpointId id{0};
  EpPair ep;
};

using EpPairVec = std::vector<EpPair>;
using RouteTable = std::unordered_map<TopoKeyPair, EpPairVec>;
using MemoryTable = std::unordered_map<MemoryKey, application::RdmaMemoryRegion>;

struct RemoteEngineMeta {
  EngineKey key;
  RouteTable rTable;
  MemoryTable mTable;
};

struct RdmaOpRet {
  StatusCode code{StatusCode::INIT};
  std::string message;

  bool Init() { return code == StatusCode::INIT; }
  bool InProgress() { return code == StatusCode::IN_PROGRESS; }
  bool Succeeded() { return code == StatusCode::SUCCESS; }
  bool Failed() { return code > StatusCode::ERR_BEGIN; }
};

RdmaOpRet RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* status, TransferUniqueId id);

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps,
                             const std::vector<application::RdmaMemoryRegion>& localMrPerEp,
                             const std::vector<application::RdmaMemoryRegion>& remoteMrPerEp,
                             const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                             const SizeVec& sizes, std::shared_ptr<CqCallbackMeta> callbackMeta,
                             TransferUniqueId id, bool isRead, int postBatchSize = -1,
                             size_t chunkBytes = 0, int maxChunks = 1,
                             bool creditByWrCount = false);

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                             const SizeVec& localOffsets,
                             const application::RdmaMemoryRegion& remote,
                             const SizeVec& remoteOffsets, const SizeVec& sizes,
                             std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                             bool isRead, int postBatchSize = -1);

inline RdmaOpRet RdmaBatchRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                               const SizeVec& localOffsets,
                               const application::RdmaMemoryRegion& remote,
                               const SizeVec& remoteOffsets, const SizeVec& sizes,
                               std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                               int postBatchSize = -1) {
  return RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, callbackMeta,
                            id, true /*isRead */, postBatchSize);
}

inline RdmaOpRet RdmaBatchWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                                const SizeVec& localOffsets,
                                const application::RdmaMemoryRegion& remote,
                                const SizeVec& remoteOffsets, const SizeVec& sizes,
                                std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                                int postBatchSize = -1) {
  return RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, callbackMeta,
                            id, false /*isRead */, postBatchSize);
}

inline RdmaOpRet RdmaReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                               size_t localOffset, const application::RdmaMemoryRegion& remote,
                               size_t remoteOffset, size_t size,
                               std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                               bool isRead) {
  return RdmaBatchReadWrite(eps, local, {localOffset}, remote, {remoteOffset}, {size}, callbackMeta,
                            id, isRead, 1);
}

inline RdmaOpRet RdmaRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                          size_t localOffset, const application::RdmaMemoryRegion& remote,
                          size_t remoteOffset, size_t size,
                          std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id) {
  return RdmaReadWrite(eps, local, localOffset, remote, remoteOffset, size, callbackMeta, id, true);
}

inline RdmaOpRet RdmaWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                           size_t localOffset, const application::RdmaMemoryRegion& remote,
                           size_t remoteOffset, size_t size,
                           std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id) {
  return RdmaReadWrite(eps, local, localOffset, remote, remoteOffset, size, callbackMeta, id,
                       false);
}
}  // namespace io
}  // namespace mori
