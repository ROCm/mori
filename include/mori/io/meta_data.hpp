#pragma once

#include "mori/application/transport/p2p/p2p.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace ioengine {

enum class BackendType : uint32_t {
  Unknown = 0,
  XGMI = 1,
  RDMA = 2,
  TCP = 3,
};

using BackendTypeVec = std::vector<BackendType>;

enum class MemoryLocation : uint32_t {
  Unknown = 0,
  CPU = 1,
  GPU = 2,
};

struct BackendBitmap {
  uint64_t bits{0};

  BackendBitmap() = default;
  BackendBitmap(BackendTypeVec availableBackends) {
    for (auto& be : availableBackends) SetBackend(be);
  }

  inline uint64_t GetBackendMask(BackendType type) { return 0x1 << static_cast<uint32_t>(type); }

  inline void SetBackend(BackendType type) { bits |= GetBackendMask(type); }
  inline bool IsAvailableBackend(BackendType type) { return GetBackendMask(type) & bits; }
};

using EngineKey = std::string;

struct EngineDesc {
  EngineKey key;
  std::string hostname;
  BackendBitmap backends;
};

struct MemoryDesc {
  EngineKey engineKey;
  MemoryLocation loc;
  application::P2PMemoryRegion p2p;
  application::RdmaMemoryRegion rdma;
  BackendBitmap backends;
};

}  // namespace ioengine
}  // namespace mori