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

  // Snapshot of current metrics (thread-safe, returns copy). Only populated when
  // config.enableMetrics == true.
  struct TcpMetricsSnapshot {
    uint64_t workerLoops{0};
    uint64_t epollWaitCalls{0};
    uint64_t epollWaitTotalNs{0};
    uint64_t headerRecvCount{0};
    uint64_t headerRecvNs{0};
    uint64_t payloadRecvCount{0};
    uint64_t payloadRecvNs{0};
    uint64_t opExecCount{0};
    uint64_t opExecNs{0};
    uint64_t gpuStagingCopies{0};
    uint64_t gpuStagingCopyNs{0};
    uint64_t headerSendCount{0};
    uint64_t headerSendNs{0};
    uint64_t payloadSendCount{0};
    uint64_t payloadSendNs{0};
    uint64_t batchScheduleCount{0};
    uint64_t batchScheduleNs{0};
  };

  TcpMetricsSnapshot GetMetricsSnapshot() const;
  void ResetMetrics();
  std::string GetMetricsJson() const;  // JSON serialization of snapshot

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
  // Unified close (inbound or outbound). Propagates failure to any pending ops/batches.
  void CloseConnection(Connection* conn, StatusCode code, const std::string& msg);

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
  // Map socket fd to its owning outbound ConnectionPool for O(1) removal on close.
  std::unordered_map<int, ConnectionPool*> fdToPool;

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

  // --- Metrics collection (optional) ---
  struct TcpMetricsInternal {
    std::atomic<uint64_t> workerLoops{0};
    std::atomic<uint64_t> epollWaitCalls{0};
    std::atomic<uint64_t> epollWaitTotalNs{0};
    std::atomic<uint64_t> headerRecvCount{0};
    std::atomic<uint64_t> headerRecvNs{0};
    std::atomic<uint64_t> payloadRecvCount{0};
    std::atomic<uint64_t> payloadRecvNs{0};
    std::atomic<uint64_t> opExecCount{0};
    std::atomic<uint64_t> opExecNs{0};
    std::atomic<uint64_t> gpuStagingCopies{0};
    std::atomic<uint64_t> gpuStagingCopyNs{0};
    std::atomic<uint64_t> headerSendCount{0};
    std::atomic<uint64_t> headerSendNs{0};
    std::atomic<uint64_t> payloadSendCount{0};
    std::atomic<uint64_t> payloadSendNs{0};
    std::atomic<uint64_t> batchScheduleCount{0};
    std::atomic<uint64_t> batchScheduleNs{0};

    void Reset() {
      workerLoops.store(0, std::memory_order_relaxed);
      epollWaitCalls.store(0, std::memory_order_relaxed);
      epollWaitTotalNs.store(0, std::memory_order_relaxed);
      headerRecvCount.store(0, std::memory_order_relaxed);
      headerRecvNs.store(0, std::memory_order_relaxed);
      payloadRecvCount.store(0, std::memory_order_relaxed);
      payloadRecvNs.store(0, std::memory_order_relaxed);
      opExecCount.store(0, std::memory_order_relaxed);
      opExecNs.store(0, std::memory_order_relaxed);
      gpuStagingCopies.store(0, std::memory_order_relaxed);
      gpuStagingCopyNs.store(0, std::memory_order_relaxed);
      headerSendCount.store(0, std::memory_order_relaxed);
      headerSendNs.store(0, std::memory_order_relaxed);
      payloadSendCount.store(0, std::memory_order_relaxed);
      payloadSendNs.store(0, std::memory_order_relaxed);
      batchScheduleCount.store(0, std::memory_order_relaxed);
      batchScheduleNs.store(0, std::memory_order_relaxed);
    }
  } metrics;
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

  // Convenience forwarding for metrics
  inline BackendServer* GetServer() { return server; }
  inline BackendServer::TcpMetricsSnapshot GetMetricsSnapshot() const {
    return server->GetMetricsSnapshot();
  }
  inline std::string GetMetricsJson() const { return server->GetMetricsJson(); }
  inline void ResetMetrics() { server->ResetMetrics(); }

 private:
  EngineKey myEngKey;
  BackendServer* server;
};

}  // namespace io
}  // namespace mori
