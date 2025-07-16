#pragma once

#include <msgpack.hpp>

#include "mori/application/transport/p2p/p2p.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/msgpack_adaptor.hpp"

namespace mori {
namespace io {

struct BackendBitmap {
  uint64_t bits{0};

  BackendBitmap() = default;
  BackendBitmap(uint64_t backendBits) : bits(backendBits) {}
  BackendBitmap(BackendTypeVec availableBackends) {
    for (auto& be : availableBackends) SetBackend(be);
  }

  inline uint64_t GetBackendMask(BackendType type) { return 0x1 << static_cast<uint32_t>(type); }

  inline void SetBackend(BackendType type) { bits |= GetBackendMask(type); }
  inline bool IsAvailableBackend(BackendType type) { return GetBackendMask(type) & bits; }

  BackendBitmap FindCommonBackends(const BackendBitmap& rhs) {
    return BackendBitmap(bits & rhs.bits);
  }

  constexpr bool operator==(const BackendBitmap& rhs) const noexcept { return bits == rhs.bits; }

  MSGPACK_DEFINE(bits);
};

using EngineKey = std::string;
using MemoryUniqueId = uint32_t;

struct EngineDesc {
  EngineKey key;
  int gpuId;
  std::string hostname;
  BackendBitmap backends;
  application::TCPContextHandle tcpHandle;

  MSGPACK_DEFINE(key, gpuId, hostname, backends, tcpHandle);
};

struct MemoryBackendDescs {
  application::RdmaMemoryRegion rdmaMr;
  MSGPACK_DEFINE(rdmaMr);
};

struct MemoryDesc {
  EngineKey engineKey;
  MemoryUniqueId id;
  int deviceId;
  void* data;
  size_t length;
  MemoryLocationType loc;
  MemoryBackendDescs backendDesc;

  MSGPACK_DEFINE(engineKey, id, deviceId, length, loc, backendDesc);
};

}  // namespace io
}  // namespace mori