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

#include <netinet/tcp.h>
#include <sys/epoll.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

#include "mori/application/utils/check.hpp"
#include "mori/io/common.hpp"

namespace mori {
namespace io {

enum OpType { READ_REQ = 0, WRITE_REQ = 1, READ_RESP = 2, WRITE_RESP = 3 };

struct TcpMessageHeader {
  uint8_t opcode;       // 0 = READ_REQ, 1 = WRITE_REQ, 2 = READ_RESP, 3 = WRITE_RESP
  uint8_t reserved{0};  // alignment / future flags
  uint16_t version{1};  // protocol version for evolution
  uint32_t id;          // transfer id
  uint64_t mem_id;      // target memory unique id on remote side
  uint64_t offset;      // remote offset within memory region
  uint64_t size;        // payload size
};

class TcpBackend;  // fwd

class TcpBackendSession;  // fwd

struct BufferBlock {
  char* data{nullptr};
  size_t capacity{0};
  bool pinned{true};
};

// Holds shared state for a batch submission so that we can drive the user-provided
// TransferStatus to completion only after all individual ops complete or on first failure.
struct TcpBatchContext {
  TransferStatus* userStatus{nullptr};
  std::atomic<size_t> remaining{0};
  std::atomic<bool> failed{false};
  // Protected message for first failure.
  std::mutex msgMu;
  std::string failMsg;
  // Constructor
  TcpBatchContext(TransferStatus* us, size_t total) : userStatus(us), remaining(total) {}
};

struct TransferOp {
  MemoryDesc localDest;
  size_t localOffset;
  MemoryDesc remoteDest;
  size_t remoteOffset;
  size_t size;
  TransferStatus* status;
  TransferUniqueId id;
  OpType opType;              // 0 = READ_REQ, 1 = WRITE_REQ, 2 = READ_RESP, 3 = WRITE_RESP
  BufferBlock stagingBuffer;  // used for CPU<->GPU staging if needed
  // Optional batch aggregation context. When non-null, per-op completions update the
  // shared context which owns the original user provided TransferStatus.
  struct TcpBatchContext* batchCtx{nullptr};
};

class ConnectionState {
 public:
  using EndpointHandle = application::TCPEndpointHandle;

  ConnectionState() = default;
  ~ConnectionState() { Close(); }

  EndpointHandle handle{-1, {}};
  application::TCPContext* listener{nullptr};
  enum class RecvState { PARSING_HEADER, PARSING_PAYLOAD };
  RecvState recvState{RecvState::PARSING_HEADER};
  TcpMessageHeader pendingHeader{};  // zero-init
  std::optional<TransferOp> recvOp;

  // Connection lifecycle flags
  std::atomic<bool> connecting{false};  // true after non-blocking connect issued until SO_ERROR==0
  std::atomic<bool> ready{false};  // true after epoll registration & (if outbound) connect success

  std::deque<TransferOp> sendQueue;                             // guarded by mu
  std::unordered_map<TransferUniqueId, TransferOp> pendingOps;  // guarded by mu (WRITE ops)

  // Queue a transfer (caller sets status->IN_PROGRESS beforehand).
  void SubmitTransfer(TransferOp op) {
    std::scoped_lock lk(mu);
    sendQueue.emplace_back(std::move(op));
  }

  // Pop next transfer (by value). Returns empty optional if none.
  std::optional<TransferOp> PopTransfer() {
    std::scoped_lock lk(mu);
    if (sendQueue.empty()) return std::nullopt;
    TransferOp op = std::move(sendQueue.front());
    sendQueue.pop_front();
    return op;
  }

  void Reset() noexcept {
    std::scoped_lock lk(mu);
    recvState = RecvState::PARSING_HEADER;
    pendingHeader = TcpMessageHeader{};
    headerBytesRead = 0;
    payloadBytesRead = 0;
    expectedPayloadSize = 0;
  }

  void Close() noexcept {
    std::scoped_lock lk(mu);
    if (listener) {
      listener->CloseEndpoint(handle);
      handle.fd = -1;
      listener = nullptr;
    } else if (handle.fd >= 0) {
      ::close(handle.fd);
      handle.fd = -1;
    }
    ready.store(false, std::memory_order_release);
  }

  std::mutex mu;

  // New incremental parsing state (non-blocking ET)
  size_t headerBytesRead{0};
  size_t payloadBytesRead{0};
  size_t expectedPayloadSize{0};
  BufferBlock inboundPayload;  // Only used for WRITE_REQ or READ_RESP
  int lastEpollFd{-1};

  // Outgoing (write side) non-blocking send state
  std::optional<TransferOp> activeSendOp;  // op currently being sent (header and maybe payload)
  char outgoingHeader[sizeof(TcpMessageHeader)];
  size_t headerBytesSent{0};
  const char* payloadPtr{nullptr};
  size_t payloadBytesTotal{0};
  size_t payloadBytesSent{0};
};

class ConnectionPool {
 public:
  ConnectionPool() = default;
  ~ConnectionPool() { Shutdown(); }

  // Non-blocking acquire (returns nullptr if none available)
  ConnectionState* GetNextConnection() {
    std::lock_guard<std::mutex> lk(mu);
    if (ready.empty()) return nullptr;
    ConnectionState* conn = ready.front();
    ready.pop_front();
    return conn;
  }

  // Blocking acquire with optional timeout (default: wait indefinitely)
  ConnectionState* AcquireConnection(
      std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
    std::unique_lock<std::mutex> lk(mu);
    auto pred = [this] { return shuttingDown || !ready.empty(); };
    if (timeout == std::chrono::milliseconds::max())
      cv.wait(lk, pred);
    else
      cv.wait_for(lk, timeout, pred);
    if (shuttingDown || ready.empty()) return nullptr;
    ConnectionState* conn = ready.front();
    ready.pop_front();
    return conn;
  }

  std::deque<ConnectionState*> GetAllConnections() {
    std::lock_guard<std::mutex> lk(mu);
    return allConns;
  }

  // Return a leased connection
  void ReleaseConnection(ConnectionState* conn) {
    if (!conn) return;
    std::lock_guard<std::mutex> lk(mu);
    if (shuttingDown) {
      conn->Close();
      return;
    }
    ready.push_back(conn);
    cv.notify_one();
  }

  void SetConnections(const std::vector<ConnectionState*>& conns) {
    std::lock_guard<std::mutex> lk(mu);
    for (auto* c : conns) {
      allConns.push_back(c);
      if (c->ready.load(std::memory_order_acquire)) ready.push_back(c);
    }
    cv.notify_all();
  }

  size_t ConnectionCount() {
    std::lock_guard<std::mutex> lk(mu);
    return allConns.size();
  }

  void RemoveConnection(ConnectionState* conn) {
    if (!conn) return;
    std::lock_guard<std::mutex> lk(mu);
    auto eraseOne = [&](auto& dq) {
      auto it = std::find(dq.begin(), dq.end(), conn);
      if (it != dq.end()) dq.erase(it);
    };
    eraseOne(ready);
    auto it = std::find(allConns.begin(), allConns.end(), conn);
    if (it != allConns.end()) {
      (*it)->Close();
      delete *it;
      allConns.erase(it);
    }
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lk(mu);
    if (!shuttingDown) {
      shuttingDown = true;
      cv.notify_all();
    }
    for (auto* c : allConns) {
      if (c) {
        c->Close();
        delete c;
      }
    }
    allConns.clear();
    ready.clear();
  }

 private:
  std::mutex mu;
  std::condition_variable cv;
  std::deque<ConnectionState*> allConns;
  std::deque<ConnectionState*> ready;  // fully usable connections
  bool shuttingDown{false};
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
      HIP_RUNTIME_CHECK(hipHostFree(b.data));
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

}  // namespace io
}  // namespace mori
