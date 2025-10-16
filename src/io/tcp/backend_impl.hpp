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

// Internal TCP backend implementation details.
#include <fcntl.h>

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"
#include "src/io/tcp/common.hpp"

namespace mori {
namespace io {

class BackendServer {
 public:
  BackendServer(const IOEngineConfig& engCfg, const TcpBackendConfig& cfg)
      : config(cfg), engConfig(engCfg) {}
  ~BackendServer() = default;

  void Start();
  void Stop();

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);

  void BatchReadWrite(const MemoryDesc& localDest, const SizeVec& localOffsets,
                      const MemoryDesc& remoteSrc, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);

 private:
  struct WorkerContext {
    application::TCPContext* listenCtx{nullptr};
    int epollFd{-1};
    size_t id{0};
  };

  // --- Core Asynchronous Engine ---
  void WorkerLoop(WorkerContext* wctx);

  // --- Event Handlers ---
  void HandleNewConnection(application::TCPContext* listener_ctx, int epoll_fd);
  void HandleReadable(Connection* conn);
  void HandleWritable(Connection* conn);

  // --- Helper Functions ---
  void SetSocketOptions(int fd);
  void SetNonBlocking(int fd);
  void RearmSocket(int epoll_fd, Connection* conn, uint32_t events);
  void CloseInbound(Connection* conn);

  void EnsureConnections(const EngineDesc& rdesc, size_t minCount);
  TcpBackendSession* GetOrCreateSessionCached(const MemoryDesc& local, const MemoryDesc& remote);

  TcpBackendConfig config;
  IOEngineConfig engConfig;
  std::vector<application::TCPContext*> listeners;
  std::vector<std::thread> workerThreads;
  std::vector<WorkerContext*> workerCtxs;  // owns WorkerContext pointers (freed in dtor)
  std::atomic<bool> running{false};
  std::atomic<size_t> nextWorker{0};  // round-robin index for assigning outbound connections

  BufferPool bufferPool;

  std::mutex inConnsMu;
  std::unordered_map<int, std::unique_ptr<Connection>> inboundConnections;
  std::unordered_map<EngineKey, std::unique_ptr<ConnectionPool>> connPools;

  std::mutex remotesMu;
  std::unordered_map<EngineKey, EngineDesc> remotes;
  std::mutex memMu;
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems;

  std::unordered_map<SessionCacheKey, std::unique_ptr<TcpBackendSession>, SessionCacheKeyHash>
      sessionCache;
  std::mutex sessionCacheMu;

  // Global monotonically increasing id source for internal sub-operations (batch).
  std::atomic<uint64_t> nextTransferId{1};  // TODO: Avoid collision

  inline TransferUniqueId NextUniqueTransferId() {
    return nextTransferId.fetch_add(1, std::memory_order_relaxed);
  }
};

class TcpBackendSession : public BackendSession {
 public:
  TcpBackendSession(BackendServer* backend, const MemoryDesc& local, const MemoryDesc& remote)
      : backend(backend), local(local), remote(remote) {}
  ~TcpBackendSession() = default;

  void ReadWrite(size_t localOffset, size_t remoteOffset, size_t size, TransferStatus* status,
                 TransferUniqueId id, bool isRead);
  void BatchReadWrite(const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                      const SizeVec& sizes, TransferStatus* status, TransferUniqueId id,
                      bool isRead);
  bool Alive() const { return true; }

 private:
  BackendServer* backend{nullptr};
  MemoryDesc local{};
  MemoryDesc remote{};
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
  EngineKey myEngKey;
  BackendServer* server;
};

}  // namespace io
}  // namespace mori
