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

#include <functional>

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

struct EpPair {
  int weight;
  int ldevId;
  int rdevId;
  EngineKey remoteEngineKey;
  application::RdmaEndpoint local;
  application::RdmaEndpointHandle remote;
  // Topology info for possible rebuild
  TopoKeyPair topoKey;
  // Rebuild / error tracking (lightweight; thresholds handled elsewhere)
  std::atomic<uint32_t> retry_exhausted_count{0};
  std::atomic<uint32_t> timeout_count{0};
  std::atomic<uint32_t> flush_err_count{0};
  std::atomic<uint32_t> rebuild_generation{0};
  std::atomic<bool> scheduled_rebuild{false};
};

// EpHandle design note:
// In the next iteration we may introduce an EpHandle abstraction wrapping a shared_ptr<EpPair>
// with an atomic swap capability so sessions can hold lightweight handles and automatically
// observe rebuilt QPs without recreating their EpPairVec. Current code mutates EpPair in place
// and updates the epsMap key; sessions that cached the pointer see the new QP fields, but any
// code storing the old QPN as key will need to refresh. EpHandle would provide:
//   struct EpHandle { std::atomic<EpPair*> cur; EpPairPtr owner; };
// Allowing transparent reload while keeping ownership semantics explicit.

using EpPairPtr = std::shared_ptr<EpPair>;

// EpHandle wraps an EpPairPtr allowing atomic pointer swap on rebuild without invalidating handles
struct EpHandle {
  explicit EpHandle(EpPairPtr ep) : owner(std::move(ep)) { cur.store(owner.get()); }
  EpHandle(const EpHandle&) = delete;
  EpHandle& operator=(const EpHandle&) = delete;
  EpHandle(EpHandle&&) = delete;
  EpHandle& operator=(EpHandle&&) = delete;
  ~EpHandle() = default;

  EpPair* get() const { return cur.load(std::memory_order_acquire); }
  EpPairPtr shared() const { return owner; }
  uint32_t qpn() const { return get()->local.handle.qpn; }
  void update(EpPairPtr n) {
    owner = std::move(n);
    cur.store(owner.get(), std::memory_order_release);
  }

 private:
  std::atomic<EpPair*> cur{nullptr};
  EpPairPtr owner;
};
using EpHandlePtr = std::shared_ptr<EpHandle>;
using EpHandleVec = std::vector<EpHandlePtr>;
// For RDMA posting functions we still reuse EpPairVec as a snapshot of current pointers.
using EpPairVec = std::vector<EpPairPtr>;
using RouteTable = std::unordered_map<TopoKeyPair, EpHandleVec>;
using MemoryTable = std::unordered_map<MemoryKey, application::RdmaMemoryRegion>;

struct RemoteEngineMeta {
  EngineKey key;
  RouteTable rTable;
  MemoryTable mTable;
};

struct NotifMessage {
  TransferUniqueId id{0};
  int qpIndex{-1};
  int totalNum{-1};
};

struct CqCallbackMeta {
  CqCallbackMeta(TransferStatus* s, TransferUniqueId id_, int n)
      : status(s), id(id_), totalBatchSize(n) {}

  TransferStatus* status{nullptr};
  TransferUniqueId id{0};
  int totalBatchSize{0};
  std::atomic<uint32_t> finishedBatchSize{0};
};

struct CqCallbackMessage {
  CqCallbackMessage(CqCallbackMeta* m, int n) : meta(m), batchSize(n) {}
  CqCallbackMeta* meta{nullptr};
  int batchSize{0};
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
                             CqCallbackMeta* callbackMeta, TransferUniqueId id, bool isRead,
                             int postBatchSize = -1);

inline RdmaOpRet RdmaBatchRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                               const SizeVec& localOffsets,
                               const application::RdmaMemoryRegion& remote,
                               const SizeVec& remoteOffsets, const SizeVec& sizes,
                               CqCallbackMeta* callbackMeta, TransferUniqueId id,
                               int postBatchSize = -1) {
  return RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, callbackMeta,
                            id, true /*isRead */, postBatchSize);
}

inline RdmaOpRet RdmaBatchWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                                const SizeVec& localOffsets,
                                const application::RdmaMemoryRegion& remote,
                                const SizeVec& remoteOffsets, const SizeVec& sizes,
                                CqCallbackMeta* callbackMeta, TransferUniqueId id,
                                int postBatchSize = -1) {
  return RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, callbackMeta,
                            id, false /*isRead */, postBatchSize);
}

inline RdmaOpRet RdmaReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                               size_t localOffset, const application::RdmaMemoryRegion& remote,
                               size_t remoteOffset, size_t size, CqCallbackMeta* callbackMeta,
                               TransferUniqueId id, bool isRead) {
  return RdmaBatchReadWrite(eps, local, {localOffset}, remote, {remoteOffset}, {size}, callbackMeta,
                            id, isRead, 1);
}

inline RdmaOpRet RdmaRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                          size_t localOffset, const application::RdmaMemoryRegion& remote,
                          size_t remoteOffset, size_t size, CqCallbackMeta* callbackMeta,
                          TransferUniqueId id) {
  return RdmaReadWrite(eps, local, localOffset, remote, remoteOffset, size, callbackMeta, id, true);
}

inline RdmaOpRet RdmaWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                           size_t localOffset, const application::RdmaMemoryRegion& remote,
                           size_t remoteOffset, size_t size, CqCallbackMeta* callbackMeta,
                           TransferUniqueId id) {
  return RdmaReadWrite(eps, local, localOffset, remote, remoteOffset, size, callbackMeta, id,
                       false);
}
}  // namespace io
}  // namespace mori
