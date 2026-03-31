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
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/msgpack_adaptor.hpp"

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

 private:
  mutable std::mutex mu_;
  uint64_t nextId_;
  std::unordered_map<uint64_t, SubmissionRecord> records_;
};

struct EpPair {
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
