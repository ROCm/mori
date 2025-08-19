#pragma once

#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

struct IOEngineConfig {
  // Out of band TCP network configuration
  std::string host;
  uint16_t port;
};

class IOEngine {
 public:
  IOEngine(EngineKey, IOEngineConfig);
  ~IOEngine();

  void CreateBackend(BackendType, void* params);
  void RemoveBackend(BackendType);

  EngineDesc GetEngineDesc() const { return desc; }

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  MemoryDesc RegisterMemory(void* data, size_t size, int device, MemoryLocationType loc);
  void DeregisterMemory(MemoryDesc& desc);

  TransferUniqueId AllocateTransferUniqueId();
  void Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc, size_t remoteOffset,
            size_t size, TransferStatus* status, TransferUniqueId id);
  void Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest, size_t remoteOffset,
             size_t size, TransferStatus* status, TransferUniqueId id);

  // Take the transfer status of an inbound op
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

 public:
  // Config and descriptors
  IOEngineConfig config;
  EngineDesc desc;

 private:
  std::atomic<uint32_t> nextTransferUid{0};
  std::atomic<uint32_t> nextMemUid{0};
  std::unordered_map<MemoryUniqueId, MemoryDesc> memPool;
  std::unordered_map<BackendType, std::unique_ptr<Backend>> backends;
};

}  // namespace io
}  // namespace mori