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
      std::lock_guard<std::mutex> lk(mu);
      max_buffers = maxBuffers;
      max_bytes = maxBytes;
      use_pinned = pinned;
      current_bytes = 0;
      total_buffers = 0;
      // pre-warm a few common sizes (4K, 64K, 1M) if budget allows
      const size_t commonSizes[] = {4 * 1024ULL, 64 * 1024ULL, 1 * 1024 * 1024ULL};
      for (size_t sz : commonSizes) {
        if (total_buffers >= max_buffers || (current_bytes + sz) > max_bytes) break;
        BufferBlock b = Allocate(sz);
        AddToBucket(std::move(b));
      }
    }
    BufferBlock Acquire(size_t size) {
      size_t bucketSize = NextPow2(std::max<size_t>(size, kMinBucket));
      std::lock_guard<std::mutex> lk(mu);
      size_t idx = BucketIndex(bucketSize);
      if (idx < buckets.size() && !buckets[idx].empty()) {
        BufferBlock b = std::move(buckets[idx].back());
        buckets[idx].pop_back();
        in_pool_buffers--;
        in_pool_bytes -= b.capacity;
        return b;
      }
      // allocate fresh (exact bucket size to reduce fragmentation)
      BufferBlock b = Allocate(bucketSize);
      return b;
    }
    void Release(BufferBlock&& b) {
      if (!b.data) return;
      size_t cap = b.capacity;
      std::lock_guard<std::mutex> lk(mu);
      if (total_buffers >= max_buffers || (current_bytes + cap) > max_bytes) {
        Free(b);
        return;
      }
      AddToBucket(std::move(b));
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
      for (auto& vec : buckets) {
        for (auto& b : vec) Free(b);
      }
    }

   private:
    static constexpr size_t kMinBucket = 256;  // smallest granularity
    static constexpr size_t kMaxBuckets = 32;  // up to 2^(31) bytes (>2GB) guard
    std::mutex mu;
    std::vector<std::vector<BufferBlock>> buckets;  // power-of-two sized bins
    size_t current_bytes{0};                        // total allocated bytes ever (live + pooled)
    size_t in_pool_bytes{0};                        // bytes currently inside pool
    size_t total_buffers{0};                        // total allocated buffers
    size_t in_pool_buffers{0};                      // buffers currently pooled
    size_t max_buffers{0};
    size_t max_bytes{0};
    bool use_pinned{true};

    inline size_t NextPow2(size_t v) {
      if (v <= 1) return 1;
      constexpr size_t kMaxPow = size_t(1) << (sizeof(size_t) * 8 - 1);
      if (v > kMaxPow) return kMaxPow;
#if defined(__GNUC__) || defined(__clang__)
      unsigned leading = __builtin_clzl(v - 1);
      unsigned bits = sizeof(size_t) * 8;
      return size_t(1) << (bits - leading);
#else
      size_t x = v - 1;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      if constexpr (sizeof(size_t) == 8) x |= x >> 32;
      return x + 1;
#endif
    }

    inline size_t BucketIndex(size_t size) {
      if (size <= 1) return 0;
#if defined(__GNUC__) || defined(__clang__)
      unsigned leading = __builtin_clzl(size);
      unsigned bits = sizeof(size_t) * 8;
      size_t idx = bits - leading - 1;
#else
      size_t s = size, idx = 0;
      while (s > 1) {
        s >>= 1;
        ++idx;
      }
#endif
      if (idx >= kMaxBuckets) idx = kMaxBuckets - 1;
      return idx;
    }

    BufferBlock Allocate(size_t size) {
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
      current_bytes += size;
      total_buffers++;
      return b;
    }
    void AddToBucket(BufferBlock&& b) {
      size_t idx = BucketIndex(b.capacity);
      if (buckets.empty()) buckets.resize(kMaxBuckets);
      buckets[idx].push_back(std::move(b));
      in_pool_buffers++;
      in_pool_bytes += buckets[idx].back().capacity;
    }
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
