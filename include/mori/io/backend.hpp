#pragma once

#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

class IOEngineConfig;

class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;

  //   virtual BackendDesc GetBackendDesc() = 0;

  virtual void RegisterRemoteEngine(EngineDesc) = 0;
  virtual void DeregisterRemoteEngine(EngineDesc) = 0;

  virtual void RegisterMemory(MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(MemoryDesc& desc) = 0;

  virtual void Read(MemoryDesc localDest, size_t localOffset, MemoryDesc remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) = 0;
  virtual void Write(MemoryDesc localSrc, size_t localOffset, MemoryDesc remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) = 0;

  // Take the transfer status of an inbound op
  virtual bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) = 0;
};

}  // namespace io
}  // namespace mori