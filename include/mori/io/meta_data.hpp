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

#include <msgpack.hpp>

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

struct MemoryDesc {
  EngineKey engineKey;
  MemoryUniqueId id{0};
  int deviceId{-1};
  uintptr_t data{0};
  size_t size{0};
  MemoryLocationType loc;

  constexpr bool operator==(const MemoryDesc& rhs) const noexcept {
    return (engineKey == rhs.engineKey) && (id == rhs.id) && (deviceId == rhs.deviceId) &&
           (data == rhs.data) && (size == rhs.size) && (loc == rhs.loc);
  }

  MSGPACK_DEFINE(engineKey, id, deviceId, data, size, loc);
};

using TransferUniqueId = uint64_t;

struct TransferStatus {
 public:
  TransferStatus() = default;
  ~TransferStatus() = default;

  StatusCode Code() { return code.load(std::memory_order_relaxed); }
  std::string Message() { return msg; }

  void SetCode(enum StatusCode val) { code.store(val, std::memory_order_relaxed); }
  void SetMessage(const std::string& val) { msg = val; }

 private:
  std::atomic<StatusCode> code{StatusCode::INIT};
  std::string msg;
};

using SizeVec = std::vector<size_t>;

}  // namespace io
}  // namespace mori
