#pragma once

#include <cstdint>
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
  BackendDescBlobMap backendDescs;

  constexpr bool operator==(const EngineDesc& rhs) const noexcept {
    return (key == rhs.key) && (hostname == rhs.hostname) && (host == rhs.host) &&
           (port == rhs.port) && (backendDescs == rhs.backendDescs);
  }

  MSGPACK_DEFINE(key, hostname, host, port, backendDescs);
};

using MemoryUniqueId = uint32_t;

struct MemoryDesc {
  EngineKey engineKey;
  MemoryUniqueId id{0};
  int deviceId{-1};
  void* data{nullptr};
  size_t size{0};
  MemoryLocationType loc;
  BackendDescBlobMap backendDescs;
  void set_data_ptr(uintptr_t intptr){
    this->data = reinterpret_cast<void*>(intptr);
  }

  // see PackableMemoryDesc
  // MSGPACK_DEFINE(engineKey, id, deviceId, size, loc, backendDescs);
};

// only for msg pack/unpack,not for user
struct PackableMemoryDesc{
  EngineKey engineKey;
  MemoryUniqueId id{0};
  int deviceId{-1};
  uintptr_t data{0};
  size_t size{0};
  MemoryLocationType loc;
  BackendDescBlobMap backendDescs;

  MSGPACK_DEFINE(engineKey, id, deviceId, data, size, loc, backendDescs);
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

}  // namespace io
}  // namespace mori