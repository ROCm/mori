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

#include <netinet/tcp.h>
#include <sys/epoll.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

#include "mori/io/common.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

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

struct TransferOp {
  MemoryDesc localDest;
  size_t localOffset;
  MemoryDesc remoteDest;
  size_t remoteOffset;
  size_t size;
  TransferStatus* status;
  TransferUniqueId id;
  bool isRead;
  bool isGpuOp{false};
  BufferBlock stagingBuffer;  // used for CPU<->GPU staging if needed
  hipStream_t stream{nullptr};
  hipEvent_t event{nullptr};
  enum class GpuCopyState { PENDING, IN_PROGRESS, COMPLETED } gpuCopyState;
};

struct GpuCommState {
  hipStream_t stream;
  hipEvent_t event;
  // Pointers to pinned staging buffers, etc.
};

class ConnectionState {
 public:
  using EndpointHandle = application::TCPEndpointHandle;

  ConnectionState() = default;
  ~ConnectionState() { Close(); }

  EndpointHandle handle{-1, {}};
  application::TCPContext* listener{nullptr};

  mutable std::mutex mu;  // protects mutable state below
  enum class RecvState { PARSING_HEADER, PARSING_PAYLOAD };
  RecvState recvstate{RecvState::PARSING_HEADER};
  TcpMessageHeader pendingheader{};  // zero-init
  std::condition_variable recv_cv;
  bool complete{false};
  std::optional<GpuCommState> gpu_state;

  std::deque<TransferOp> pending_ops;  // guarded by mu

  void Reset() noexcept {
    std::scoped_lock lk(mu);
    recvstate = RecvState::PARSING_HEADER;
    pendingheader = TcpMessageHeader{};
    complete = false;
  }

  void MarkComplete() {
    {
      std::scoped_lock lk(mu);
      complete = true;
    }
    recv_cv.notify_all();
  }

  void Close() noexcept {
    std::scoped_lock lk(mu);
    if (listener) {
      listener->CloseEndpoint(handle);
      handle.fd = -1;
      listener = nullptr;
    }
  }
};

class ConnectionPool {
 public:
  ConnectionPool() = default;
  ~ConnectionPool() = default;

  std::shared_ptr<ConnectionState> GetNextConnection() {
    size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
    if (outConns.empty()) return nullptr;
    return outConns[idx % outConns.size()];
  }

  void SetConnections(const std::vector<std::shared_ptr<ConnectionState>>& conns) {
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& c : conns) {
      outConns.push_back(c);
    }
  }

  size_t ConnectionCount() {
    std::lock_guard<std::mutex> lk(mu);
    return outConns.size();
  }

  void RemoveConnection(std::shared_ptr<ConnectionState> conn) {
    auto it = std::find(outConns.begin(), outConns.end(), conn);
    if (it != outConns.end()) {
      it->get()->Close();
      std::lock_guard<std::mutex> lk(mu);
      outConns.erase(it);
    }
  }

  void ClearConnections() {
    std::lock_guard<std::mutex> lk(mu);
    for (auto& c : outConns) {
      c->Close();
    }
    outConns.clear();
  }

 private:
  std::mutex mu;
  std::deque<std::shared_ptr<ConnectionState>> outConns;
  std::atomic<size_t> nextIdx{0};
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

int FullSend(int fd, const void* buf, size_t len);
int FullRecv(int fd, void* buf, size_t len);
int FullWritev(int fd, struct iovec* iov, int iovcnt);

/**
 * Single Producer / Multiple Consumer bounded blocking queue.
 * Thread-safety:
 *  - Exactly one producer thread may call push/emplace/try_push.
 *  - Multiple consumer threads may call pop/try_pop concurrently.
 *  - shutdown() may be called once (typically by the producer / owner).
 */
template <typename T>
class SPMCQueue {
 public:
  explicit SPMCQueue(size_t capacity) : capacity_(capacity), buffer_(capacity) {
    if (capacity_ == 0) throw std::invalid_argument("capacity must be > 0");
  }

  SPMCQueue(const SPMCQueue&) = delete;
  SPMCQueue& operator=(const SPMCQueue&) = delete;
  SPMCQueue(SPMCQueue&&) = delete;
  SPMCQueue& operator=(SPMCQueue&&) = delete;

  ~SPMCQueue() {
    shutdown();  // ensure any waiting threads are released
  }

  // Blocking push (returns false if queue has been shut down before insertion)
  bool push(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this] { return size_ < capacity_ || done_; });
    if (done_) return false;
    write_unlocked(item);
    lock.unlock();
    cv_not_empty_.notify_one();
    return true;
  }

  bool push(T&& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this] { return size_ < capacity_ || done_; });
    if (done_) return false;
    write_unlocked(std::move(item));
    lock.unlock();
    cv_not_empty_.notify_one();
    return true;
  }

  // Variadic emplace (construct T from args) - returns false if closed
  template <class... Args>
  bool emplace(Args&&... args) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this] { return size_ < capacity_ || done_; });
    if (done_) return false;
    buffer_[tail_] = T(std::forward<Args>(args)...);
    advance_tail_unlocked();
    lock.unlock();
    cv_not_empty_.notify_one();
    return true;
  }

  // Non-blocking try_push (returns false if full or closed)
  bool try_push(const T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (done_ || size_ == capacity_) return false;
    write_unlocked(item);
    cv_not_empty_.notify_one();
    return true;
  }

  bool try_push(T&& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (done_ || size_ == capacity_) return false;
    write_unlocked(std::move(item));
    cv_not_empty_.notify_one();
    return true;
  }

  // Blocking pop. Returns false when queue is empty AND has been shut down.
  bool pop(T& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this] { return size_ > 0 || done_; });
    if (size_ == 0 && done_) return false;
    out = std::move(buffer_[head_]);
    head_ = (head_ + 1) % capacity_;
    --size_;
    lock.unlock();
    cv_not_full_.notify_one();
    return true;
  }

  // Non-blocking pop. Returns false if empty (even if not closed yet).
  bool try_pop(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (size_ == 0) return false;
    out = std::move(buffer_[head_]);
    head_ = (head_ + 1) % capacity_;
    --size_;
    cv_not_full_.notify_one();
    return true;
  }

  void shutdown() {
    bool notify = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!done_) {
        done_ = true;
        notify = true;
      }
    }
    if (notify) {
      cv_not_empty_.notify_all();
      cv_not_full_.notify_all();  // wake producer if waiting
    }
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ == 0;
  }

  size_t capacity() const noexcept { return capacity_; }

 private:
  void write_unlocked(const T& v) {
    buffer_[tail_] = v;
    advance_tail_unlocked();
  }
  void write_unlocked(T&& v) {
    buffer_[tail_] = std::move(v);
    advance_tail_unlocked();
  }
  void advance_tail_unlocked() {
    tail_ = (tail_ + 1) % capacity_;
    ++size_;
  }

  const size_t capacity_;
  std::vector<T> buffer_;

  size_t head_ = 0;
  size_t tail_ = 0;
  size_t size_ = 0;

  mutable std::mutex mutex_;
  std::condition_variable cv_not_full_;
  std::condition_variable cv_not_empty_;
  bool done_ = false;
};

}  // namespace io
}  // namespace mori
