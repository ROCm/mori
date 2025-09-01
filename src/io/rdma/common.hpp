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
};

using EpPairVec = std::vector<EpPair>;
using RouteTable = std::unordered_map<TopoKeyPair, EpPairVec>;
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

struct RdmaOpStatusHandle {
  TransferStatus* status{nullptr};
  int expectedNumCqe{0};
  std::atomic<uint32_t> curNumCqe{0};
};

void RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* status, TransferUniqueId id);
void RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                        const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                        const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
                        TransferUniqueId id, int postBatchSize, bool isRead);
inline void RdmaBatchRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                          const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                          const SizeVec& remoteOffsets, const SizeVec& sizes,
                          TransferStatus* status, TransferUniqueId id, int postBatchSize) {
  RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, status, id,
                     postBatchSize, true /*isRead */);
}

inline void RdmaBatchWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                           const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                           const SizeVec& remoteOffsets, const SizeVec& sizes,
                           TransferStatus* status, TransferUniqueId id, int postBatchSize) {
  RdmaBatchReadWrite(eps, local, localOffsets, remote, remoteOffsets, sizes, status, id,
                     postBatchSize, false /*isRead */);
}

inline void RdmaRead(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                     size_t localOffset, const application::RdmaMemoryRegion& remote,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) {
  RdmaBatchRead(eps, local, {localOffset}, remote, {remoteOffset}, {size}, status, id, 1);
}

inline void RdmaWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                      size_t localOffset, const application::RdmaMemoryRegion& remote,
                      size_t remoteOffset, size_t size, TransferStatus* status,
                      TransferUniqueId id) {
  RdmaBatchWrite(eps, local, {localOffset}, remote, {remoteOffset}, {size}, status, id, 1);
}
}  // namespace io
}  // namespace mori
