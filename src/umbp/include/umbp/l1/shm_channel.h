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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mori::umbp::l1::shm {

// A message channel between one client process and the server, in a memory
// segment both map. It carries opaque (method, bytes) messages and knows
// nothing of what they mean.
//
//   client -> server   a byte ring of requests, in the order they were sent.
//                      A request is either one-way (the sender does not wait;
//                      a failure is parked in the segment and collected by a
//                      later call) or a call (the sender waits for a reply).
//   server -> client   one reply buffer: a channel has one call outstanding
//                      at a time.
//
// Neither side makes a system call on the fast path. A side that has nothing
// to do sleeps on a futex word in the segment, and the other side wakes it
// only when that word says it sleeps.
//
// The segment is a file (tmpfs by default) the server creates and the client
// opens by path, so the two need only share that directory.

struct Sizes {
  size_t request_bytes = size_t{1} << 20;   // rounded up to a power of two
  size_t response_bytes = size_t{1} << 20;  // the largest reply a call can get
};

// One-way requests of at most this many distinct methods can park a failure.
constexpr uint32_t kErrorSlots = 8;

struct Segment;  // the mapped layout, private to the implementation

// What the server does with one request. For a call (`reply` non-null) it
// fills `*reply`. For a one-way request it returns an empty string, or what
// went wrong.
using Handler =
    std::function<std::string(uint32_t method, const char* data, size_t size, std::string* reply)>;

class ServerChannel {
 public:
  // Create the segment as a new file in `dir`. Null, with `*error`, on failure.
  static std::unique_ptr<ServerChannel> Create(const std::string& dir, const Sizes& sizes,
                                               std::string* error);
  ~ServerChannel();

  ServerChannel(const ServerChannel&) = delete;
  ServerChannel& operator=(const ServerChannel&) = delete;

  const std::string& path() const { return path_; }
  uint64_t nonce() const { return nonce_; }
  size_t request_bytes() const { return request_bytes_; }
  size_t response_bytes() const { return response_bytes_; }

  // Whether a request is waiting.
  bool Pending() const;
  // Handle every waiting request, in order. Any thread may call this; callers
  // are serialized. Returns how many were handled.
  size_t Drain(const Handler& handler);
  // As Drain, unless another thread is draining: then that thread has the
  // requests in hand and this returns at once, with false.
  bool TryDrain(const Handler& handler);
  // Return once a request is waiting or `timeout` has passed: poll for `spin`,
  // then sleep. True when a request is waiting.
  bool Wait(std::chrono::microseconds spin, std::chrono::milliseconds timeout);
  // Whether the client has mapped the segment.
  bool ClientAttached() const;
  // Remove the file; the mappings live on. Idempotent.
  void Unlink();
  // Tell the client the server has gone; its waits end.
  void Close();

 private:
  ServerChannel() = default;
  size_t DrainLocked(const Handler& handler);

  std::string path_;
  uint64_t nonce_ = 0;
  Segment* segment_ = nullptr;
  size_t mapped_bytes_ = 0;
  size_t request_bytes_ = 0;
  size_t response_bytes_ = 0;
  bool linked_ = false;
  std::mutex consume_mutex_;
  std::string scratch_;  // a request that wraps the ring, made contiguous
  std::string reply_;
};

enum class SendResult {
  kOk,
  kTooLarge,  // does not fit the ring: send it another way
  kClosed,    // the server has gone
  kTimeout,
  kReplyTooLarge,  // Call only: the server ran it but its reply exceeds the reply buffer
};

class ClientChannel {
 public:
  // Map the segment at `path`, which must carry `nonce`.
  static std::unique_ptr<ClientChannel> Open(const std::string& path, uint64_t nonce,
                                             std::string* error);
  ~ClientChannel();

  ClientChannel(const ClientChannel&) = delete;
  ClientChannel& operator=(const ClientChannel&) = delete;

  // How long a call polls for its reply before it sleeps.
  void set_spin(std::chrono::microseconds spin) { spin_ = spin; }
  // Asked every ~100 ms of waiting; a wait ends with kClosed once it is false.
  void set_alive(std::function<bool()> alive) { alive_ = std::move(alive); }
  // Called on the calling thread around every sleep -- for a reply, for room
  // in the ring, for another thread of this process to finish its call --
  // and around nothing else: `enter` before, `leave` after. For a caller
  // that holds a lock it must not sleep under, such as the Python GIL.
  void set_blocking(std::function<void()> enter, std::function<void()> leave) {
    enter_ = std::move(enter);
    leave_ = std::move(leave);
  }

  // Queue a one-way request. Waits for room when the ring is full.
  SendResult Send(uint32_t method, const char* data, size_t size,
                  std::chrono::milliseconds timeout);
  // Send a request and wait for its reply.
  SendResult Call(uint32_t method, const char* data, size_t size, std::string* reply,
                  std::chrono::milliseconds timeout);
  // The failure of an earlier one-way request of `method`, once. False when
  // there is none.
  bool TakeError(uint32_t method, std::string* text);
  // The largest request the ring takes.
  size_t max_request_bytes() const;
  size_t response_bytes() const { return response_bytes_; }

 private:
  ClientChannel() = default;

  enum class Room { kOk, kTooLarge, kFull };
  // With send_mutex_ held.
  Room PutLocked(uint32_t method, uint64_t call_id, const char* data, size_t size);
  SendResult SendWith(uint32_t method, uint64_t call_id, const char* data, size_t size,
                      std::chrono::steady_clock::time_point deadline);
  // Take `mutex`, sleeping for it only between `enter_` and `leave_`.
  void Lock(std::unique_lock<std::mutex>* lock);
  void WakeServer();
  // Sleep until `ready()` holds. kOk, kClosed or kTimeout.
  SendResult WaitFor(const std::function<bool()>& ready, std::chrono::microseconds spin,
                     std::chrono::steady_clock::time_point deadline);

  Segment* segment_ = nullptr;
  size_t mapped_bytes_ = 0;
  size_t request_bytes_ = 0;
  size_t response_bytes_ = 0;
  std::chrono::microseconds spin_{200};
  std::function<bool()> alive_;
  std::function<void()> enter_;
  std::function<void()> leave_;
  std::mutex send_mutex_;  // one writer of the request ring at a time
  std::mutex call_mutex_;  // one call outstanding at a time
  uint64_t next_call_ = 0;
  std::atomic<uint64_t> seen_error_[kErrorSlots] = {};
};

}  // namespace mori::umbp::l1::shm
