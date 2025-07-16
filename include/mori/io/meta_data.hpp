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
  BackendBitmap(BackendTypeVec availableBackends) {
    for (auto& be : availableBackends) SetBackend(be);
  }

  inline uint64_t GetBackendMask(BackendType type) { return 0x1 << static_cast<uint32_t>(type); }

  inline void SetBackend(BackendType type) { bits |= GetBackendMask(type); }
  inline bool IsAvailableBackend(BackendType type) { return GetBackendMask(type) & bits; }

  constexpr bool operator==(const BackendBitmap& rhs) const noexcept { return bits == rhs.bits; }

  MSGPACK_DEFINE(bits);
};

using EngineKey = std::string;

struct EngineDesc {
  EngineKey key;
  int gpuId;
  std::string hostname;
  BackendBitmap backends;
  application::TCPContextHandle tcpHandle;

  MSGPACK_DEFINE(key, gpuId, hostname, backends, tcpHandle);
};

struct MemoryDesc {
  EngineKey engineKey;
  MemoryLocation loc;
  int gpuId;
  application::RdmaMemoryRegion rdma;
  // application::P2PMemoryRegion p2p;
  BackendBitmap backends;

  MSGPACK_DEFINE(engineKey, loc, gpuId, rdma, backends);
};

}  // namespace io
}  // namespace mori