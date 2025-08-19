#pragma once

#include "mori/io/meta_data.hpp"

namespace mori {
namespace io {

class IOEngineConfig;

class BackendSession {
 public:
  BackendSession() = default;
  virtual ~BackendSession() = default;

  virtual void Read(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) = 0;
  virtual void Write(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) = 0;

  virtual void BatchRead(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) = 0;

  virtual bool Alive() const = 0;
};

class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;

  virtual void RegisterRemoteEngine(const EngineDesc&) = 0;
  virtual void DeregisterRemoteEngine(const EngineDesc&) = 0;

  virtual void RegisterMemory(const MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(const MemoryDesc& desc) = 0;

  virtual void Read(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                    size_t remoteOffset, size_t size, TransferStatus* status,
                    TransferUniqueId id) = 0;
  virtual void Write(const MemoryDesc& localSrc, size_t localOffset, const MemoryDesc& remoteDest,
                     size_t remoteOffset, size_t size, TransferStatus* status,
                     TransferUniqueId id) = 0;

  virtual void BatchRead(const MemoryDesc& localDest, const SizeVec& localOffsets,
                         const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                         const SizeVec& sizes, TransferStatus* status, TransferUniqueId id) = 0;

  virtual BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote) = 0;

  // Take the transfer status of an inbound op
  virtual bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id,
                                        TransferStatus* status) = 0;
};

}  // namespace io
}  // namespace mori