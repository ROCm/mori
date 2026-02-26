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

#include <array>
#include <functional>
#include <msgpack.hpp>
#include <mutex>

#include "mori/application/transport/p2p/p2p.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/msgpack_adaptor.hpp"

namespace mori {
namespace io {

struct BackendBitmap {
  uint32_t bits{0};

  BackendBitmap() = default;
  BackendBitmap(uint32_t backendBits) : bits(backendBits) {}
  BackendBitmap(BackendTypeVec availableBackends) {
    for (auto& be : availableBackends) SetBackend(be);
  }

  inline uint32_t GetBackendMask(BackendType type) { return 0x1 << static_cast<uint32_t>(type); }

  inline void SetBackend(BackendType type) { bits |= GetBackendMask(type); }
  inline bool IsAvailableBackend(BackendType type) { return GetBackendMask(type) & bits; }

  BackendBitmap FindCommonBackends(const BackendBitmap& rhs) {
    return BackendBitmap(bits & rhs.bits);
  }

  BackendTypeVec ToBackendTypeVec() const {
    BackendTypeVec vec;
    for (uint32_t i = 0; i < 32; i++) {
      if ((0x1 << i) & bits) vec.push_back(static_cast<BackendType>(i));
    }
    return vec;
  }

  constexpr bool operator==(const BackendBitmap& rhs) const noexcept { return bits == rhs.bits; }

  MSGPACK_DEFINE(bits);
};

using EngineKey = std::string;
using DescBlob = std::vector<std::byte>;
using BackendDescBlobMap = std::unordered_map<BackendType, DescBlob>;

struct EngineDesc {
  EngineKey key;
  std::string hostname;
  std::string host;
  int port;

  constexpr bool operator==(const EngineDesc& rhs) const noexcept {
    return (key == rhs.key) && (hostname == rhs.hostname) && (host == rhs.host) &&
           (port == rhs.port);
  }

  MSGPACK_DEFINE(key, hostname, host, port);
};

using MemoryUniqueId = uint32_t;

constexpr size_t kIpcHandleSize = 64;

struct MemoryDesc {
  EngineKey engineKey;
  MemoryUniqueId id{0};
  int deviceId{-1};
  uintptr_t data{0};
  size_t size{0};
  MemoryLocationType loc;
  std::array<char, kIpcHandleSize> ipcHandle{};

  constexpr bool operator==(const MemoryDesc& rhs) const noexcept {
    return (engineKey == rhs.engineKey) && (id == rhs.id) && (deviceId == rhs.deviceId) &&
           (data == rhs.data) && (size == rhs.size) && (loc == rhs.loc);
  }

  MSGPACK_DEFINE(engineKey, id, deviceId, data, size, loc, ipcHandle);
};

using TransferUniqueId = uint64_t;

struct TransferStatus {
 public:
  TransferStatus() = default;
  ~TransferStatus() = default;

  StatusCode Code() { return code.load(std::memory_order_acquire); }
  uint32_t CodeUint32() { return static_cast<uint32_t>(code.load(std::memory_order_acquire)); }

  std::string Message() {
    std::lock_guard<std::mutex> lock(msgMu);
    return msg;
  }

  void Update(enum StatusCode val, const std::string& message) {
    std::lock_guard<std::mutex> lock(msgMu);
    StatusCode current = code.load(std::memory_order_relaxed);
    if (current > StatusCode::ERR_BEGIN) return;

    msg = message;
    code.store(val, std::memory_order_release);
  }

  bool Init() { return Code() == StatusCode::INIT; }
  bool InProgress() { return Code() == StatusCode::IN_PROGRESS; }
  bool Succeeded() { return Code() == StatusCode::SUCCESS; }
  bool Failed() { return Code() > StatusCode::ERR_BEGIN; }

  void SetCode(enum StatusCode val) { code.store(val, std::memory_order_release); }
  void SetMessage(const std::string& val) {
    std::lock_guard<std::mutex> lock(msgMu);
    msg = val;
  }

  void Wait() {
    if (waitCallback) {
      waitCallback();
      return;
    }
    while (InProgress()) {
    }
  }

  void SetWaitCallback(std::function<void()> cb) { waitCallback = std::move(cb); }

 private:
  std::atomic<StatusCode> code{StatusCode::INIT};
  mutable std::mutex msgMu;
  std::string msg;
  std::function<void()> waitCallback;
};

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

using SizeVec = std::vector<size_t>;
using MemDescVec = std::vector<MemoryDesc>;
using BatchSizeVec = std::vector<SizeVec>;
using TransferUniqueIdVec = std::vector<TransferUniqueId>;
using TransferStatusPtrVec = std::vector<TransferStatus*>;

}  // namespace io
}  // namespace mori
