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
#include "src/io/tcp/common.hpp"

namespace mori {
namespace io {

// Non-blocking server-side connection state for partial I/O handling.
// Simple state machine: RECV_HDR -> (maybe RECV_PAYLOAD) -> SEND_RESP -> RECV_HDR ...
// All operations processed by service worker threads; protected by serviceStatesMu.

enum class ConnPhase { RECV_HDR, RECV_PAYLOAD, SEND_RESP };

struct ServiceConnState {
  TcpMessageHeader hdr{};  // inbound header being accumulated
  ConnPhase phase{ConnPhase::RECV_HDR};
  std::vector<char> in_payload;  // staging for WRITE request inbound data
  std::vector<char> out_buf;     // response bytes (header + optional payload)
  bool target_is_gpu{false};     // cached after header decode
  bool closed{false};
};

class TCPExecutor {
 public:
  TCPExecutor() = default;
  virtual ~TCPExecutor() = default;

  virtual void Start() = 0;
  virtual void Shutdown() = 0;
  virtual int SubmitReadWriteWork() = 0;

  virtual int SubmitServiceWork(int fd) = 0;

  virtual void RegisterRemoteEngine(const EngineDesc&) = 0;
  virtual void DeregisterRemoteEngine(const EngineDesc&) = 0;

  virtual void RegisterMemory(const MemoryDesc& desc) = 0;
  virtual void DeregisterMemory(const MemoryDesc& desc) = 0;

  virtual std::vector<TcpConnection>& EnsureConnections(const EngineDesc& rdesc,
                                                        size_t minCount) = 0;
  virtual void CloseConnections(const EngineKey& key) = 0;
};

class MultithreadTCPExecutor : public TCPExecutor {
 public:
  // ctx is owned by TcpBackend; executor only borrows pointer (lifetime > executor)
  MultithreadTCPExecutor(application::TCPContext* ctx, size_t numThreads)
      : ctx(ctx), numThreads(numThreads) {
    running.store(true);
    Start();
  }
  ~MultithreadTCPExecutor() override = default;

  void Start() override {
    for (size_t i = 0; i < numThreads; ++i) {
      readWriteWorkers.emplace_back(&MultithreadTCPExecutor::ReadWriteWorkerLoop, this);
      serviceWorkers.emplace_back(&MultithreadTCPExecutor::ServiceWorkerLoop, this);
    }
  }

  void Shutdown() override {
    workQueue.shutdown();
    serviceQueue.shutdown();
    running.store(false);
    for (auto& t : serviceWorkers) {
      if (t.joinable()) t.join();
    }
    for (auto& t : readWriteWorkers) {
      if (t.joinable()) t.join();
    }
  }

  int SubmitReadWriteWork(const ReadWriteWork& work) {
    if (!workQueue.push(work)) {
      return -1;  // queue shutdown
    }
    return 0;
  }

  int SubmitServiceWork(int fd) { return serviceQueue.push(fd) ? 0 : -1; }

  void RegisterRemoteEngine(const EngineDesc&);
  void DeregisterRemoteEngine(const EngineDesc&);

  void RegisterMemory(const MemoryDesc& desc);
  void DeregisterMemory(const MemoryDesc& desc);

  // Ensure at least minCount persistent outbound connections exist to remote engine.
  // Returns reference to internal connection vector (guarded by connsMu while mutating).
  std::vector<TcpConnection>& EnsureConnections(const EngineDesc& rdesc, size_t minCount);

  // Close and erase all persistent connections for engine key.
  void CloseConnections(const EngineKey& key);

 private:
  void ReadWriteWorkerLoop() {
    ReadWriteWork work;
    BufferPool bufferPool;
    while (running.load()) {
      if (!workQueue.pop(work)) {
        break;  // queue shutdown and empty
      }
      // Process the work item
      DoReadWrite(work, bufferPool);
    }
  }

  void ServiceWorkerLoop() {
    int fd;
    BufferPool bufferPool;
    while (running.load()) {
      if (!serviceQueue.pop(fd)) {
        break;  // queue shutdown and empty
      }
      // Process the service work item
      DoServiceWork(fd, bufferPool);
    }
  }

  void DoReadWrite(const ReadWriteWork& work, BufferPool& bufferPool);
  void DoServiceWork(int fd, BufferPool& bufferPool);

  application::TCPContext* ctx{nullptr};
  size_t numThreads{1};
  std::vector<std::thread> serviceWorkers;
  std::vector<std::thread> readWriteWorkers;
  std::atomic<bool> running{true};
  std::unordered_map<EngineKey, EngineDesc> remotes;
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems;
  std::unordered_map<EngineKey, std::unordered_map<MemoryUniqueId, MemoryDesc>>
      remoteMems;  // meta only
  SPMCQueue<ReadWriteWork> workQueue{1024};
  SPMCQueue<int> serviceQueue{1024};
  std::unordered_map<EngineKey, std::vector<TcpConnection>> conns;  // engine -> connections
  std::mutex memMu;                                                 // protects localMems
  std::mutex remotesMu;  // protects remotes & remoteMems meta
  std::mutex connsMu;    // protects conns map
  // Server-side accepted connection states (fd -> state) for non-blocking EPOLLET processing.
  std::unordered_map<int, ServiceConnState> serviceStates;
  std::mutex serviceStatesMu;
};

}  // namespace io
}  // namespace mori
