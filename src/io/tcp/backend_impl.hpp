#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>

#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"
#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/application/utils/check.hpp"

namespace mori {
namespace io {

// Simple TCP data plane: emulate RDMA read/write by message framing and memcpy semantics.
// This backend is NOT optimized and intended as a functional placeholder.

struct TcpMessageHeader {
  uint8_t opcode; // 0 = READ_REQ, 1 = WRITE_REQ, 2 = READ_RESP, 3 = WRITE_RESP
  uint32_t id;    // transfer id
  uint64_t offset; // remote offset
  uint64_t size;   // payload size
};

class TcpBackendSession; // fwd

class TcpConnection {
 public:
  TcpConnection() = default;
  TcpConnection(application::TCPEndpointHandle h) : handle(h) {}
  ~TcpConnection() = default;
  bool Valid() const { return handle.fd >= 0; }
  application::TCPEndpointHandle handle{-1, {}};
};

class TcpBackend : public Backend {
 public:
  TcpBackend(EngineKey, const IOEngineConfig&, const TcpBackendConfig&);
  ~TcpBackend();

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);

  void ReadWrite(const MemoryDesc& localDest, size_t localOffset, const MemoryDesc& remoteSrc,
                 size_t remoteOffset, size_t size, TransferStatus* status, TransferUniqueId id,
                 bool isRead);
  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

  BackendSession* CreateSession(const MemoryDesc& local, const MemoryDesc& remote);
  bool PopInboundTransferStatus(EngineKey remote, TransferUniqueId id, TransferStatus* status);

 private:
  TcpConnection GetOrCreateConnection(const EngineDesc& rdesc);
  void ServiceLoop();
  void StartService();
  void StopService();

 private:
  EngineKey myEngKey;
  TcpBackendConfig config;
  IOEngineConfig engConfig;
  std::unique_ptr<application::TCPContext> ctx{nullptr};
  std::thread serviceThread;
  std::atomic<bool> running{false};

  // memory registered locally
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems;
  std::unordered_map<EngineKey, std::unordered_map<MemoryUniqueId, MemoryDesc>> remoteMems; // meta only
  std::unordered_map<EngineKey, EngineDesc> remotes;
  std::unordered_map<EngineKey, TcpConnection> conns;
  std::mutex mu;
};

class TcpBackendSession : public BackendSession {
 public:
  TcpBackendSession(TcpBackend* backend, const MemoryDesc& local, const MemoryDesc& remote)
      : backend(backend), local(local), remote(remote) {}
  ~TcpBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);
  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);
  bool Alive() const { return true; }

 private:
  TcpBackend* backend{nullptr};
  MemoryDesc local{};
  MemoryDesc remote{};
};

} // namespace io
} // namespace mori
