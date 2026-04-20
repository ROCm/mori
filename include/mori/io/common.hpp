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
#include <string>

#include "mori/application/transport/p2p/p2p.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/msgpack_adaptor.hpp"

namespace mori {
namespace io {
namespace internal {
struct TransferStatusAccess;
}

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
  std::string nodeId;
  std::string hostname;
  std::string host;
  int port;
  int pid{0};

  constexpr bool operator==(const EngineDesc& rhs) const noexcept {
    return (key == rhs.key) && (nodeId == rhs.nodeId) && (hostname == rhs.hostname) &&
           (host == rhs.host) && (port == rhs.port) && (pid == rhs.pid);
  }

  MSGPACK_DEFINE(key, nodeId, hostname, host, port, pid);
};

using MemoryUniqueId = uint32_t;

constexpr size_t kIpcHandleSize = 64;

struct MemoryDesc {
  EngineKey engineKey;
  MemoryUniqueId id{0};
  int deviceId{-1};
  std::string deviceBusId;
  uintptr_t data{0};
  size_t size{0};
  MemoryLocationType loc;
  std::array<char, kIpcHandleSize> ipcHandle{};

  constexpr bool operator==(const MemoryDesc& rhs) const noexcept {
    return (engineKey == rhs.engineKey) && (id == rhs.id) && (deviceId == rhs.deviceId) &&
           (deviceBusId == rhs.deviceBusId) && (data == rhs.data) && (size == rhs.size) &&
           (loc == rhs.loc);
  }

  MSGPACK_DEFINE(engineKey, id, deviceId, deviceBusId, data, size, loc, ipcHandle);
};

using TransferUniqueId = uint64_t;

struct TransferStatus {
 public:
  TransferStatus() = default;
  ~TransferStatus() = default;

  StatusCode Code() { return CodeWithProgress(); }
  uint32_t CodeUint32() { return static_cast<uint32_t>(Code()); }

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

  void SetCode(enum StatusCode val) {
    if (val <= StatusCode::ERR_BEGIN) diagnosticFlags.store(0, std::memory_order_relaxed);
    code.store(val, std::memory_order_release);
  }
  void SetMessage(const std::string& val) {
    std::lock_guard<std::mutex> lock(msgMu);
    msg = val;
  }

  void Wait() {
    PollProgress();
    if (waitCallback) {
      waitCallback();
      return;
    }
    while (InProgress()) {
    }
  }

  void SetWaitCallback(std::function<void()> cb) { waitCallback = std::move(cb); }
  void SetProgressCallback(std::function<void()> cb) { progressCallback = std::move(cb); }

 private:
  StatusCode CodeWithProgress() {
    PollProgress();
    return code.load(std::memory_order_acquire);
  }

  void PollProgress() {
    if (code.load(std::memory_order_acquire) != StatusCode::IN_PROGRESS) return;
    if (progressCallback) progressCallback();
  }

  std::atomic<StatusCode> code{StatusCode::INIT};
  // Internal-only diagnostic bits used by IO backend implementations. They are
  // intentionally not part of the public error contract.
  std::atomic<uint32_t> diagnosticFlags{0};
  mutable std::mutex msgMu;
  std::string msg;
  std::function<void()> waitCallback;
  std::function<void()> progressCallback;

  friend struct internal::TransferStatusAccess;
};

using SizeVec = std::vector<size_t>;
using MemDescVec = std::vector<MemoryDesc>;
using BatchSizeVec = std::vector<SizeVec>;
using TransferUniqueIdVec = std::vector<TransferUniqueId>;
using TransferStatusPtrVec = std::vector<TransferStatus*>;

}  // namespace io
}  // namespace mori
