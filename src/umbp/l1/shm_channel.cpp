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
#include "umbp/l1/shm_channel.h"

#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <random>

namespace mori::umbp::l1::shm {

namespace {

constexpr uint32_t kMagic = 0x4c31534d;  // "L1SM"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 4096;
constexpr size_t kErrorText = 248;
constexpr std::chrono::milliseconds kWaitSlice{100};

static_assert(std::atomic<uint32_t>::is_always_lock_free, "futex words must be plain memory");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "ring cursors must be plain memory");

struct ErrorSlot {
  // Odd while the text is being written; each failure adds two.
  std::atomic<uint64_t> seq;
  char text[kErrorText];
};

struct RecordHeader {
  uint32_t size;
  uint32_t method;
  uint64_t call_id;  // 0: one-way
};

size_t Align8(size_t n) { return (n + 7) & ~size_t{7}; }

size_t RoundUpPow2(size_t n) {
  size_t p = 4096;
  while (p < n) p <<= 1;
  return p;
}

inline void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

// Shared, not FUTEX_PRIVATE: the two waiters are different processes.
void FutexWait(std::atomic<uint32_t>* word, uint32_t expected, std::chrono::nanoseconds timeout) {
  timespec ts;
  ts.tv_sec = static_cast<time_t>(timeout.count() / 1000000000);
  ts.tv_nsec = static_cast<long>(timeout.count() % 1000000000);
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT, expected, &ts, nullptr, 0);
}

void FutexWake(std::atomic<uint32_t>* word, int count) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE, count, nullptr, nullptr, 0);
}

std::string Errno(const std::string& what) { return what + ": " + std::strerror(errno); }

}  // namespace

struct Segment {
  uint32_t magic;
  uint32_t version;
  uint64_t nonce;
  uint64_t request_bytes;
  uint64_t response_bytes;
  std::atomic<uint32_t> client_attached;
  std::atomic<uint32_t> closed;
  // Requests: the client owns the tail, the server the head.
  alignas(64) std::atomic<uint64_t> req_tail;
  alignas(64) std::atomic<uint64_t> req_head;
  // 1 while the server sleeps for a request.
  alignas(64) std::atomic<uint32_t> server_sleeping;
  // 1 while a client thread sleeps for a reply or for room in the ring.
  alignas(64) std::atomic<uint32_t> client_waiting;
  // The reply to call `resp_seq`.
  alignas(64) std::atomic<uint64_t> resp_seq;
  uint64_t resp_size;
  uint32_t resp_too_large;
  alignas(64) ErrorSlot errors[kErrorSlots];

  char* ring() { return reinterpret_cast<char*>(this) + kHeaderBytes; }
  char* response() { return ring() + request_bytes; }

  void CopyIn(uint64_t at, const void* src, size_t n) {
    const size_t mask = request_bytes - 1;
    const size_t off = at & mask;
    const size_t first = std::min(n, static_cast<size_t>(request_bytes) - off);
    std::memcpy(ring() + off, src, first);
    if (first < n) std::memcpy(ring(), static_cast<const char*>(src) + first, n - first);
  }
  void CopyOut(uint64_t at, void* dst, size_t n) {
    const size_t mask = request_bytes - 1;
    const size_t off = at & mask;
    const size_t first = std::min(n, static_cast<size_t>(request_bytes) - off);
    std::memcpy(dst, ring() + off, first);
    if (first < n) std::memcpy(static_cast<char*>(dst) + first, ring(), n - first);
  }
};

static_assert(sizeof(Segment) <= kHeaderBytes, "the header must fit its page");

// ------------------------------- server ------------------------------------

std::unique_ptr<ServerChannel> ServerChannel::Create(const std::string& dir, const Sizes& sizes,
                                                     std::string* error) {
  static std::atomic<uint64_t> counter{0};
  std::random_device random;
  const uint64_t nonce = (uint64_t{random()} << 32) | random();
  const size_t request_bytes = RoundUpPow2(sizes.request_bytes);
  const size_t response_bytes = Align8(std::max<size_t>(sizes.response_bytes, 4096));
  const size_t total = kHeaderBytes + request_bytes + response_bytes;

  char name[96];
  std::snprintf(name, sizeof(name), "/umbp_l1_shm_%d_%llu_%016llx", static_cast<int>(getpid()),
                static_cast<unsigned long long>(counter.fetch_add(1)),
                static_cast<unsigned long long>(nonce));
  const std::string path = dir + name;
  const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0) {
    *error = Errno("create " + path);
    return nullptr;
  }
  // Reserve the pages now: a full tmpfs would otherwise fault on first touch.
  const int reserved = posix_fallocate(fd, 0, static_cast<off_t>(total));
  if (reserved != 0) {
    errno = reserved;
    *error = Errno("reserve " + std::to_string(total) + " bytes for " + path);
    close(fd);
    unlink(path.c_str());
    return nullptr;
  }
  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    *error = Errno("map " + path);
    unlink(path.c_str());
    return nullptr;
  }
  auto* segment = static_cast<Segment*>(base);
  segment->version = kVersion;
  segment->nonce = nonce;
  segment->request_bytes = request_bytes;
  segment->response_bytes = response_bytes;
  segment->magic = kMagic;

  std::unique_ptr<ServerChannel> channel(new ServerChannel());
  channel->path_ = path;
  channel->nonce_ = nonce;
  channel->segment_ = segment;
  channel->mapped_bytes_ = total;
  channel->request_bytes_ = request_bytes;
  channel->response_bytes_ = response_bytes;
  channel->linked_ = true;
  return channel;
}

ServerChannel::~ServerChannel() {
  Close();
  Unlink();
  if (segment_ != nullptr) munmap(segment_, mapped_bytes_);
}

bool ServerChannel::Pending() const {
  return segment_->req_tail.load(std::memory_order_seq_cst) !=
         segment_->req_head.load(std::memory_order_relaxed);
}

bool ServerChannel::ClientAttached() const {
  return segment_->client_attached.load(std::memory_order_acquire) != 0;
}

void ServerChannel::Unlink() {
  std::lock_guard<std::mutex> lock(consume_mutex_);
  if (!linked_) return;
  unlink(path_.c_str());
  linked_ = false;
}

void ServerChannel::Close() {
  segment_->closed.store(1, std::memory_order_seq_cst);
  segment_->client_waiting.store(0, std::memory_order_seq_cst);
  FutexWake(&segment_->client_waiting, INT_MAX);
}

size_t ServerChannel::Drain(const Handler& handler) {
  std::lock_guard<std::mutex> lock(consume_mutex_);
  return DrainLocked(handler);
}

bool ServerChannel::TryDrain(const Handler& handler) {
  std::unique_lock<std::mutex> lock(consume_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return false;
  DrainLocked(handler);
  return true;
}

size_t ServerChannel::DrainLocked(const Handler& handler) {
  Segment* s = segment_;
  size_t handled = 0;
  for (;;) {
    const uint64_t head = s->req_head.load(std::memory_order_relaxed);
    const uint64_t tail = s->req_tail.load(std::memory_order_acquire);
    if (head == tail) break;
    RecordHeader header;
    const uint64_t pending = tail - head;
    bool sane = pending >= sizeof(header) && pending <= request_bytes_;
    size_t total = 0;
    if (sane) {
      s->CopyOut(head, &header, sizeof(header));
      total = sizeof(header) + Align8(header.size);
      sane = header.size <= request_bytes_ && total <= pending;
    }
    if (!sane) {
      // The ring is not one this code wrote. Nothing in it can be trusted.
      s->req_head.store(tail, std::memory_order_seq_cst);
      Close();
      break;
    }
    const size_t offset = (head + sizeof(header)) & (request_bytes_ - 1);
    const char* data = s->ring() + offset;
    if (offset + header.size > request_bytes_) {
      scratch_.resize(header.size);
      s->CopyOut(head + sizeof(header), scratch_.data(), header.size);
      data = scratch_.data();
    }
    if (header.call_id != 0) {
      reply_.clear();
      handler(header.method, data, header.size, &reply_);
      if (reply_.size() > response_bytes_) {
        s->resp_too_large = 1;
        s->resp_size = reply_.size();
      } else {
        std::memcpy(s->response(), reply_.data(), reply_.size());
        s->resp_too_large = 0;
        s->resp_size = reply_.size();
      }
      s->resp_seq.store(header.call_id, std::memory_order_seq_cst);
    } else {
      const std::string failure = handler(header.method, data, header.size, nullptr);
      if (!failure.empty() && header.method < kErrorSlots) {
        ErrorSlot& slot = s->errors[header.method];
        const uint64_t seq = slot.seq.load(std::memory_order_relaxed);
        slot.seq.store(seq + 1, std::memory_order_seq_cst);
        const size_t n = std::min(failure.size(), kErrorText - 1);
        std::memcpy(slot.text, failure.data(), n);
        slot.text[n] = '\0';
        slot.seq.store(seq + 2, std::memory_order_seq_cst);
      }
    }
    s->req_head.store(head + total, std::memory_order_seq_cst);
    // A reply is ready, or the ring has room again: wake whoever sleeps for it.
    if (s->client_waiting.load(std::memory_order_seq_cst) != 0) {
      s->client_waiting.store(0, std::memory_order_seq_cst);
      FutexWake(&s->client_waiting, INT_MAX);
    }
    ++handled;
  }
  return handled;
}

bool ServerChannel::Wait(std::chrono::microseconds spin, std::chrono::milliseconds timeout) {
  if (Pending()) return true;
  if (spin.count() > 0) {
    const auto until = std::chrono::steady_clock::now() + spin;
    do {
      for (int i = 0; i < 16; ++i) {
        if (Pending()) return true;
        CpuRelax();
      }
    } while (std::chrono::steady_clock::now() < until);
  }
  Segment* s = segment_;
  s->server_sleeping.store(1, std::memory_order_seq_cst);
  if (!Pending()) FutexWait(&s->server_sleeping, 1, timeout);
  s->server_sleeping.store(0, std::memory_order_seq_cst);
  return Pending();
}

// ------------------------------- client ------------------------------------

std::unique_ptr<ClientChannel> ClientChannel::Open(const std::string& path, uint64_t nonce,
                                                   std::string* error) {
  const int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    *error = Errno("open " + path);
    return nullptr;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < kHeaderBytes) {
    *error = "segment " + path + " is too small";
    close(fd);
    return nullptr;
  }
  const size_t total = static_cast<size_t>(st.st_size);
  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    *error = Errno("map " + path);
    return nullptr;
  }
  auto* segment = static_cast<Segment*>(base);
  const bool matches = segment->magic == kMagic && segment->version == kVersion &&
                       segment->nonce == nonce && segment->request_bytes != 0 &&
                       (segment->request_bytes & (segment->request_bytes - 1)) == 0 &&
                       kHeaderBytes + segment->request_bytes + segment->response_bytes == total;
  if (!matches) {
    *error = "segment " + path + " is not the one the server announced";
    munmap(base, total);
    return nullptr;
  }
  std::unique_ptr<ClientChannel> channel(new ClientChannel());
  channel->segment_ = segment;
  channel->mapped_bytes_ = total;
  channel->request_bytes_ = segment->request_bytes;
  channel->response_bytes_ = segment->response_bytes;
  segment->client_attached.store(1, std::memory_order_release);
  return channel;
}

ClientChannel::~ClientChannel() {
  if (segment_ != nullptr) munmap(segment_, mapped_bytes_);
}

size_t ClientChannel::max_request_bytes() const {
  return request_bytes_ / 2 - sizeof(RecordHeader);
}

void ClientChannel::WakeServer() {
  Segment* s = segment_;
  if (s->server_sleeping.load(std::memory_order_seq_cst) != 1) return;
  if (s->server_sleeping.exchange(0, std::memory_order_seq_cst) == 1) {
    FutexWake(&s->server_sleeping, 1);
  }
}

ClientChannel::Room ClientChannel::PutLocked(uint32_t method, uint64_t call_id, const char* data,
                                             size_t size) {
  Segment* s = segment_;
  const size_t total = sizeof(RecordHeader) + Align8(size);
  if (total > request_bytes_ / 2) return Room::kTooLarge;
  const uint64_t tail = s->req_tail.load(std::memory_order_relaxed);
  const uint64_t head = s->req_head.load(std::memory_order_acquire);
  if (tail - head + total > request_bytes_) return Room::kFull;
  RecordHeader header{static_cast<uint32_t>(size), method, call_id};
  s->CopyIn(tail, &header, sizeof(header));
  s->CopyIn(tail + sizeof(header), data, size);
  s->req_tail.store(tail + total, std::memory_order_seq_cst);
  return Room::kOk;
}

SendResult ClientChannel::WaitFor(const std::function<bool()>& ready,
                                  std::chrono::microseconds spin,
                                  std::chrono::steady_clock::time_point deadline) {
  Segment* s = segment_;
  if (spin.count() > 0) {
    const auto until = std::chrono::steady_clock::now() + spin;
    do {
      for (int i = 0; i < 16; ++i) {
        if (ready()) return SendResult::kOk;
        CpuRelax();
      }
    } while (std::chrono::steady_clock::now() < until);
  }
  // From here on the thread sleeps.
  struct Blocking {
    explicit Blocking(ClientChannel* c) : channel(c) {
      if (channel->enter_) channel->enter_();
    }
    ~Blocking() {
      if (channel->leave_) channel->leave_();
    }
    ClientChannel* channel;
  } blocking(this);
  auto checked = std::chrono::steady_clock::now();
  for (;;) {
    s->client_waiting.store(1, std::memory_order_seq_cst);
    if (ready()) return SendResult::kOk;
    if (s->closed.load(std::memory_order_seq_cst) != 0) return SendResult::kClosed;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return SendResult::kTimeout;
    if (alive_ && now - checked >= kWaitSlice) {
      checked = now;
      if (!alive_()) return SendResult::kClosed;
    }
    FutexWait(&s->client_waiting, 1,
              std::min<std::chrono::nanoseconds>(kWaitSlice, deadline - now));
  }
}

void ClientChannel::Lock(std::unique_lock<std::mutex>* lock) {
  if (lock->try_lock()) return;
  // Another thread of this process is in its call. Sleeping for it with the
  // caller's lock held could deadlock: that thread may be waiting to take
  // the caller's lock back.
  if (enter_) enter_();
  lock->lock();
  if (leave_) leave_();
}

SendResult ClientChannel::SendWith(uint32_t method, uint64_t call_id, const char* data, size_t size,
                                   std::chrono::steady_clock::time_point deadline) {
  Segment* s = segment_;
  if (s->closed.load(std::memory_order_acquire) != 0) return SendResult::kClosed;
  {
    std::unique_lock<std::mutex> lock(send_mutex_, std::defer_lock);
    Lock(&lock);
    for (;;) {
      const Room room = PutLocked(method, call_id, data, size);
      if (room == Room::kOk) break;
      if (room == Room::kTooLarge) return SendResult::kTooLarge;
      WakeServer();
      const size_t total = sizeof(RecordHeader) + Align8(size);
      const SendResult waited = WaitFor(
          [&] {
            return s->req_tail.load(std::memory_order_relaxed) -
                       s->req_head.load(std::memory_order_seq_cst) + total <=
                   request_bytes_;
          },
          std::chrono::microseconds(0), deadline);
      if (waited != SendResult::kOk) return waited;
    }
  }
  WakeServer();
  return SendResult::kOk;
}

SendResult ClientChannel::Send(uint32_t method, const char* data, size_t size,
                               std::chrono::milliseconds timeout) {
  return SendWith(method, 0, data, size, std::chrono::steady_clock::now() + timeout);
}

SendResult ClientChannel::Call(uint32_t method, const char* data, size_t size, std::string* reply,
                               std::chrono::milliseconds timeout) {
  Segment* s = segment_;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::unique_lock<std::mutex> lock(call_mutex_, std::defer_lock);
  Lock(&lock);
  const uint64_t id = ++next_call_;
  const SendResult sent = SendWith(method, id, data, size, deadline);
  if (sent != SendResult::kOk) return sent;
  const SendResult waited =
      WaitFor([&] { return s->resp_seq.load(std::memory_order_seq_cst) == id; }, spin_, deadline);
  if (waited != SendResult::kOk) return waited;
  if (s->resp_too_large != 0) return SendResult::kReplyTooLarge;
  reply->assign(s->response(), s->resp_size);
  return SendResult::kOk;
}

bool ClientChannel::TakeError(uint32_t method, std::string* text) {
  if (method >= kErrorSlots) return false;
  ErrorSlot& slot = segment_->errors[method];
  for (;;) {
    const uint64_t before = slot.seq.load(std::memory_order_seq_cst);
    uint64_t seen = seen_error_[method].load(std::memory_order_relaxed);
    if (before == seen) return false;
    if (before & 1) return false;  // being written; the next call collects it
    char copy[kErrorText];
    std::memcpy(copy, slot.text, sizeof(copy));
    if (slot.seq.load(std::memory_order_seq_cst) != before) continue;
    copy[kErrorText - 1] = '\0';
    // Two threads may see the same failure; one of them reports it.
    if (!seen_error_[method].compare_exchange_strong(seen, before)) return false;
    *text = copy;
    return true;
  }
}

}  // namespace mori::umbp::l1::shm
