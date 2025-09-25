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

class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;

  virtual void Start() = 0;
  virtual void Shutdown() = 0;
};

class MultithreadExecutor : public Executor {
 public:
  MultithreadExecutor(size_t numThreads) : numThreads(numThreads) {
    running.store(true);
    Start();
  }
  ~MultithreadExecutor() override = default;

  void Start() override {
    for (size_t i = 0; i < numThreads; ++i) {
      readWriteWorkers.emplace_back(&MultithreadExecutor::ReadWriteWorkerLoop, this);
      serviceWorkers.emplace_back(&MultithreadExecutor::ServiceWorkerLoop, this);
    }
  }

  void Shutdown() override {
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
};

}  // namespace io
}  // namespace mori
