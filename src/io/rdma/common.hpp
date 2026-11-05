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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
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

// SQ ids carry a QP-local sequence; receive ids remain untagged indices.
inline constexpr uint64_t kSqTag = uint64_t{1} << 63;
inline constexpr uint64_t kSqSequenceMask = (uint64_t{1} << 62) - 1;
inline constexpr std::memory_order kSqAdmissionOrder = std::memory_order_seq_cst;
inline bool IsSqWrId(uint64_t id) { return (id & kSqTag) != 0; }
inline uint64_t SqSequence(uint64_t id) { return id & kSqSequenceMask; }

std::vector<std::pair<uint64_t, uint32_t>> PlanChunks(uint32_t total, size_t chunkBytes,
                                                      int maxChunks);

struct ChunkedSgeSegment {
  uint64_t remoteOffset{0};
  uint64_t localAddr{0};
  uint32_t length{0};
};

struct ChunkGeometry {
  uint64_t finalCount{0};
  uint64_t targetChunkBytes{0};
};

ChunkGeometry PlanChunkGeometry(uint64_t totalLength, size_t chunkBytes, int maxChunks,
                                uint64_t maxMessageSize);

uint64_t CountChunksForSize(uint64_t totalLength, size_t chunkBytes, int maxChunks,
                            uint64_t maxMessageSize);

void PlanSgeStreamChunks(std::vector<ChunkedSgeSegment>& plan, const std::vector<ibv_sge>& sges,
                         uint64_t totalLength, size_t chunkBytes, int maxChunks,
                         uint64_t maxMessageSize);

struct CqCallbackMeta {
  CqCallbackMeta(TransferStatus* s, TransferUniqueId id_, int n)
      : status(s), id(id_), totalBatchSize(n) {
    dataStatus.SetCode(StatusCode::IN_PROGRESS);
  }

  // Claimed with exchange(nullptr) by whoever completes the transfer first, so a
  // terminal update happens exactly once even though the CQ poller and the async
  // event monitor can both reach the same meta concurrently.
  std::atomic<TransferStatus*> status{nullptr};
  TransferUniqueId id{0};
  int totalBatchSize{0};
  std::atomic<uint32_t> finishedBatchSize{0};
  internal::IoCallDiagnostics diagnostics{};

  // Set before the data WRs are posted when a notification SEND follows them. The
  // data WRs and the notification post then finish independently, and whichever
  // finishes last reports the transfer: a notification that cannot be posted never
  // turns it terminal while its data WRs are still on the wire, and is never lost
  // behind a SUCCESS the data already reported.
  bool awaitsNotification{false};

  // Reports the data WRs' outcome through `s`, the status the caller claimed.
  void ReportData(TransferStatus* s, StatusCode code, const std::string& message) {
    if (!awaitsNotification) {
      s->Update(code, message);
      return;
    }
    dataStatus.Update(code, message);
    Finish(s, &dataOutcome, code, message);
  }

  // The data WRs' own outcome while the caller's status waits for the notification too;
  // a notification fault staged in test builds waits for it.
  TransferStatus dataStatus;

  // Reports the notification post's outcome; `s` is the posting thread's own status.
  void ReportNotification(TransferStatus* s, StatusCode code, const std::string& message) {
    Finish(s, &notificationOutcome, code, message);
  }

 private:
  struct Outcome {
    StatusCode code{StatusCode::SUCCESS};
    std::string message;
  };

  // A data failure wins over a notification failure; only both succeeding is SUCCESS.
  void Finish(TransferStatus* s, Outcome* part, StatusCode code, const std::string& message) {
    {
      std::lock_guard<std::mutex> lock(outcomeMu);
      *part = Outcome{code, message};
    }
    if (unfinishedParts.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    std::lock_guard<std::mutex> lock(outcomeMu);
    const Outcome& report = dataOutcome.code <= StatusCode::ERR_BEGIN &&
                                    notificationOutcome.code > StatusCode::ERR_BEGIN
                                ? notificationOutcome
                                : dataOutcome;
    s->Update(report.code, report.message);
  }

  std::atomic<int> unfinishedParts{2};
  std::mutex outcomeMu;
  Outcome dataOutcome;
  Outcome notificationOutcome;
};

struct SqAdmissionEvent {
  alignas(4) std::atomic<uint32_t> epoch{0};
  std::atomic<uint32_t> waiters{0};
};

class VerbsOps;

// One owner of SQ publication, physical credits, and completion evidence per QP.
class SubmissionLedger {
 public:
  enum class Kind : uint8_t { Data, DataTail, Notify, Marker };
  enum class ReserveFailure : uint8_t { None, Closed, Timeout, ExceedsCapacity };
  struct PostResult {
    int error{0};
    int posted{0};
    uint64_t dataSeq{0};
    uint64_t signaledSeq{0};
  };

  explicit SubmissionLedger(int capacity);
  bool Reserve(int count, int timeoutUs, std::string* error, ReserveFailure* failure = nullptr);
  void CancelReservation(int count);
  void NotifyAdmission();
  void Close(bool freeze = false);
  void FreezeForShutdown();
  bool Closed() const;
  bool Frozen() const;
  uint64_t Done() const;
  int Capacity() const { return capacity_; }
  int OrdinaryLimit() const { return capacity_ - 1; }
  int Used() const { return used_.load(kSqAdmissionOrder); }
  uint32_t Waiters() const { return admission_.waiters.load(kSqAdmissionOrder); }

  // The caller reserved count credits before entering this non-waiting post section.
  PostResult PostToSq(ibv_qp* qp, ibv_send_wr* first, int count, Kind kind,
                      const std::shared_ptr<CqCallbackMeta>& meta, int batchSize, bool finalBatch,
                      TransferUniqueId transferId, size_t bytes, VerbsOps& verbs);
  bool OnSqCompletion(uint64_t id, ibv_wc_status status, const std::string& message);
  Kind KindForCompletion(uint64_t id) const;
  int FailAll(StatusCode code, const std::string& message, bool freeze = false);

 private:
  static constexpr uint64_t kFrozen = uint64_t{1} << 63;
  static constexpr uint64_t kClosed = uint64_t{1} << 62;
  struct Entry {
    std::atomic<uint64_t> seq{0};
    Kind kind{Kind::Data};
    std::shared_ptr<CqCallbackMeta> meta;
    int batchSize{0};
    TransferUniqueId transferId{0};
  };
  struct Snapshot {
    bool valid{false};
    Kind kind{Kind::Data};
    std::shared_ptr<CqCallbackMeta> meta;
    int batchSize{0};
    TransferUniqueId transferId{0};
    uint64_t seq{0};
  };

  bool TryReserve(int count, int limit);
  PostResult PostToSqLocked(ibv_qp* qp, ibv_send_wr* first, int count, Kind kind,
                            const std::shared_ptr<CqCallbackMeta>& meta, int batchSize,
                            bool finalBatch, TransferUniqueId transferId, size_t bytes,
                            VerbsOps& verbs);
  Snapshot TakeEntry(uint64_t seq);
  void CompleteEntry(const Snapshot& entry, ibv_wc_status status, const std::string& message,
                     bool lost = false);

  const int capacity_;
  size_t mask_;
  std::unique_ptr<Entry[]> ring_;
  mutable std::mutex postMu_;
  uint64_t postedSeq_{0};
  uint64_t lastSignaledSeq_{0};
  std::atomic<uint64_t> state_{0};
  std::atomic<int> used_{0};
  SqAdmissionEvent admission_;
  bool markerOutstanding_{false};
  uint64_t pendingTarget_{0};
};

// A verbs async event that means outstanding work can never complete. Scope says
// which endpoints are affected: a single QP moved to Error state, or a CQ/device
// that died and took every endpoint sharing it with it.
struct QpErrorEvent {
  enum class Scope : uint8_t { kQueuePair, kCompletionQueue, kDevice };

  Scope scope{Scope::kQueuePair};
  uint32_t qpNum{0};
  // CQ handle for kCompletionQueue. Compared by value only: the verbs object may
  // already be acked and must never be dereferenced here.
  const void* cq{nullptr};
  ibv_context* context{nullptr};
  const char* eventName{""};
};

using QpErrorHandler = std::function<void(const QpErrorEvent&)>;

struct EpPair {
  int weight;
  int ldevId;
  int rdevId;
  EngineKey remoteEngineKey;
  application::RdmaEndpoint local;
  application::RdmaEndpointHandle remote;
  // Shared across every session using this physical QP.
  std::shared_ptr<SubmissionLedger> ledger;
};

// Which endpoints a fatal async event takes down with it. A QP event hits one
// endpoint; a CQ or device event hits everything sharing that CQ or context.
bool EndpointAffectedByAsyncEvent(const QpErrorEvent& event, const EpPair& ep);

using EndpointId = uint64_t;

struct EndpointRuntime {
  EndpointRuntime() = default;
  EndpointRuntime(EndpointId id_, const EpPair& ep_) : id(id_), ep(ep_) {}

  EndpointId id{0};
  EpPair ep;
  // Cached pointer to this endpoint's NotifManager::QpNotifContext, published once
  // at registration. Lets the hot CQ poller resolve the notif context without
  // locking NotifManager::mu on every poll. Type-erased (void*) to avoid a header
  // dependency on the NotifManager-private struct; nullptr until registered.
  std::atomic<void*> notifCtx{nullptr};
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

void NotifySqStateChanged(const EpPair& ep);

// `dataStatus` holds the outcome of the transfer's data WRs (CqCallbackMeta::dataStatus).
RdmaOpRet RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* dataStatus, TransferUniqueId id);

namespace detail {

// Test hook for SQ admission without proceeding to ibv_post_send on a fake QP.
bool TryReserveSqDepthForTesting(const EpPair& ep, int wrCount, std::string* errMsg = nullptr);

}  // namespace detail

struct RdmaTransferControl {
  size_t chunkBytes{0};
  int maxChunks{1};
  bool creditByWrCount{false};
  bool ownsTotalBatchSize{true};
  bool disableMerge{false};
};

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps,
                             const std::vector<application::RdmaMemoryRegion>& localMrPerEp,
                             const std::vector<application::RdmaMemoryRegion>& remoteMrPerEp,
                             const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                             const SizeVec& sizes, std::shared_ptr<CqCallbackMeta> callbackMeta,
                             TransferUniqueId id, bool isRead, int postBatchSize,
                             const RdmaTransferControl& control);

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
