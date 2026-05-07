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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
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

struct CqCallbackMeta {
  CqCallbackMeta(TransferStatus* s, TransferUniqueId id_, int n)
      : status(s), id(id_), totalBatchSize(n) {}

  TransferStatus* status{nullptr};
  TransferUniqueId id{0};
  int totalBatchSize{0};
  std::atomic<uint32_t> finishedBatchSize{0};
  internal::IoCallDiagnostics diagnostics{};
};

struct SqCqeDiagnostics {
  std::atomic<int64_t> lastPollAttemptTimeUs{0};
  std::atomic<int64_t> lastNonEmptyCqeTimeUs{0};
  std::atomic<uint64_t> recentCqeCount{0};
  std::atomic<uint64_t> recentBatchReleaseWr{0};
};

enum class SqReserveFailureKind : uint8_t {
  None = 0,
  Degraded,
  TerminalDegraded,
  ExceedsCapacity,
  Timeout,
};

enum class SqDegradeReason : uint8_t {
  None = 0,
  PartialPostOrphaned,
  FatalCqe,
  EndpointTeardown,
};

struct ReserveOptions {
  int timeoutUs{0};
};

struct ReserveResult {
  SqReserveFailureKind kind{SqReserveFailureKind::None};
  int depth{0};
  int requested{0};
  int maxDepth{0};
  int backoffCount{0};
};

class SqController {
 public:
  SqController(int maxDepth, int resumeWatermark);

  bool Reserve(int wrCount, ReserveOptions opts, ReserveResult* result);
  bool RecheckBeforePost(int reservedWrCount, ReserveResult* result);
  void Release(int wrCount);
  bool MarkDegraded(SqDegradeReason reason);
  void ReleaseDrainedOrphaned(int wrCount);
  std::shared_lock<std::shared_mutex> AcquireSubmitGuard();
  std::unique_lock<std::shared_mutex> AcquireRecoveryGuard();

  int Depth() const;
  int MaxDepth() const;
  bool IsDegraded() const;
  bool IsTerminalDegraded() const;

 private:
  void ReleaseInternal(int wrCount);
  void NotifyStateChangedLocked();

  std::atomic<int> depth_{0};
  int maxDepth_{0};
  int resumeWatermark_{0};
  std::atomic<bool> degraded_{false};
  std::atomic<bool> terminalDegraded_{false};
  std::shared_mutex submitMu_;
  std::mutex mu_;
  std::condition_variable cv_;
  uint64_t epoch_{0};
};

int GetSqResumeWatermarkWrForDepth(int maxDepth);

using EndpointId = uint64_t;

// SubmissionLedger tracks per-EP WR completions. SQ credit is released by
// SqController after callers inspect the returned SubmissionRecord.
enum class SubmissionState : uint8_t {
  Tentative,  // inserted before ibv_post_send confirms the signaled tail was posted
  Posted,     // submitted, awaiting CQE
  Orphaned,   // partial post without signaled tail; awaits recovery
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
  uint64_t InsertOrphaned(int postedWr, std::shared_ptr<CqCallbackMeta> meta, int batchSize);

  // CQE path: find record by recordId, return it, and erase it.
  bool ReleaseByCqe(uint64_t recordId, SubmissionRecord* outRecord);

  // Post path: a tentative signaled record has a posted tail and must await CQE.
  bool MarkPosted(uint64_t recordId);

  // Post failure path: erase a tentative signaled record whose tail was not posted.
  bool CancelTentative(uint64_t recordId, SubmissionRecord* outRecord);

  // Terminal degraded path: extract Orphaned records and keep Posted records.
  void ExtractOrphanedRecords(std::vector<SubmissionRecord>* outRecords);

  bool HasOrphaned() const;
  size_t RecordCount() const;

 private:
  mutable std::mutex mu_;
  uint64_t nextId_;
  std::unordered_map<uint64_t, SubmissionRecord> records_;
};

struct EpPair {
  EndpointId endpointId{0};
  int weight{0};
  int ldevId{0};
  int rdevId{0};
  EngineKey remoteEngineKey;
  application::RdmaEndpoint local;
  application::RdmaEndpointHandle remote;
  // Shared across EpPair copies that refer to the same QP. sq is the Phase 2
  // owner; sqDepth/degraded remain only for legacy/fallback EpPair instances.
  std::shared_ptr<SqController> sq;
  std::shared_ptr<std::atomic<int>> sqDepth;
  int maxSqDepth{0};
  std::shared_ptr<std::atomic<bool>> degraded;
  std::shared_ptr<SubmissionLedger> ledger;
  int qpPerTransfer{0};
  int numWorkerThreads{0};
  std::shared_ptr<SqCqeDiagnostics> sqCqeDiagnostics;
};

using EpPairVec = std::vector<EpPair>;

void RecordSqPollAttempt(const EpPair& ep);
void RecordSqPollCqes(const EpPair& ep, int cqeCount);
void RecordSqBatchReleaseWr(const EpPair& ep, int wrCount);
// The vector membership is not modified, but endpoint controller/ledger/status
// state is mutated through EpPair's shared ownership fields.
void MovePendingUnsignaledToOrphanedForEndpoint(
    const EpPairVec& eps, size_t epId, std::vector<int>& epWrsSinceSignal,
    std::vector<size_t>& epMergedSinceSignal, const std::shared_ptr<CqCallbackMeta>& callbackMeta,
    const std::string& message, const char* context,
    std::shared_lock<std::shared_mutex>* heldSubmitGuard = nullptr);

struct EndpointRuntime {
  EndpointRuntime() = default;
  EndpointRuntime(EndpointId id_, const EpPair& ep_) : id(id_), ep(ep_) {}

  EndpointId id{0};
  EpPair ep;
};

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
