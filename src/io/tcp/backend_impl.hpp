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

#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

// Simple TCP data plane: emulate RDMA read/write by message framing and memcpy semantics.
// This backend is NOT optimized and intended as a functional placeholder.

struct TcpMessageHeader {
  uint8_t opcode;       // 0 = READ_REQ, 1 = WRITE_REQ, 2 = READ_RESP, 3 = WRITE_RESP
  uint8_t reserved{0};  // alignment / future flags
  uint16_t version{1};  // protocol version for evolution
  uint32_t id;          // transfer id
  uint64_t mem_id;      // target memory unique id on remote side
  uint64_t offset;      // remote offset within memory region
  uint64_t size;        // payload size
};

class TcpBackendSession;  // fwd

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

  struct BufferBlock {
    char* data{nullptr};
    size_t capacity{0};
    bool pinned{false};
  };

  class BufferPool {
   public:
    BufferPool() = default;
    void Configure(size_t maxBuffers, size_t maxBytes, bool pinned) {
      max_buffers = maxBuffers;
      max_bytes = maxBytes;
      use_pinned = pinned;
    }
    BufferBlock Acquire(size_t size) {
      std::lock_guard<std::mutex> lk(mu);
      // try find first-fit large enough
      for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->capacity >= size) {
          BufferBlock b = *it;
          current_bytes -= b.capacity;
          pool.erase(it);
          return b;
        }
      }
      // allocate new
      BufferBlock b;
      b.capacity = size;
      b.pinned = use_pinned;
      if (use_pinned) {
        if (hipHostMalloc(&b.data, size) != hipSuccess) {
          b.data = (char*)::malloc(size);
          b.pinned = false;
        }
      } else {
        b.data = (char*)::malloc(size);
      }
      return b;
    }
    void Release(BufferBlock&& b) {
      if (!b.data) return;
      std::lock_guard<std::mutex> lk(mu);
      if (pool.size() >= max_buffers || (current_bytes + b.capacity) > max_bytes) {
        Free(b);
        return;
      }
      current_bytes += b.capacity;
      pool.push_back(b);
    }
    void Free(BufferBlock& b) {
      if (!b.data) return;
      if (b.pinned)
        hipHostFree(b.data);
      else
        ::free(b.data);
      b.data = nullptr;
      b.capacity = 0;
      b.pinned = false;
    }
    ~BufferPool() {
      for (auto& b : pool) Free(b);
    }

   private:
    std::mutex mu;
    std::deque<BufferBlock> pool;
    size_t current_bytes{0};
    size_t max_buffers{0};
    size_t max_bytes{0};
    bool use_pinned{true};
  };
  BufferPool bufferPool;

  // memory registered locally
  std::unordered_map<MemoryUniqueId, MemoryDesc> localMems;
  std::unordered_map<EngineKey, std::unordered_map<MemoryUniqueId, MemoryDesc>>
      remoteMems;  // meta only
  std::unordered_map<EngineKey, EngineDesc> remotes;
  std::unordered_map<EngineKey, TcpConnection> conns;
  std::mutex memMu;      // protects localMems
  std::mutex remotesMu;  // protects remotes & remoteMems meta
  std::mutex connsMu;    // protects conns map
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

}  // namespace io
}  // namespace mori
