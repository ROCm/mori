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
// Bounded batch queue: many sampler threads produce, one writer consumes.
//
// Ownership moves with the batch (no copies). Each stream has a hard quota of queued batches
// so a stalled writer bounds memory. Push never blocks: a full stream gets kFull and the
// sampler drops that batch's data (and reuses the batch). After Close(), every Push returns
// kClosed, which is how late submissions from abandoned threads are rejected.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace netmon {

struct Batch {
  Batch(int stream_id, size_t record_words, size_t record_capacity)
      : stream(stream_id),
        width(record_words),
        capacity(record_capacity),
        words(record_words * record_capacity) {}
  uint64_t* Slot() { return &words[count * width]; }
  bool Full() const { return count >= capacity; }
  const uint64_t* Record(size_t i) const { return &words[i * width]; }

  int stream;
  size_t width;
  size_t capacity;
  size_t count = 0;
  std::vector<uint64_t> words;
};

class BatchQueue {
 public:
  enum class PushResult { kOk, kFull, kClosed };

  BatchQueue(size_t streams, size_t per_stream_quota)
      : per_stream_(streams, 0), quota_(per_stream_quota) {}

  // On kOk, takes ownership (b becomes null). Otherwise b is left with the caller.
  PushResult Push(std::unique_ptr<Batch>& b) {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) return PushResult::kClosed;
    size_t& q = per_stream_[static_cast<size_t>(b->stream)];
    if (q >= quota_) return PushResult::kFull;
    q++;
    items_.push_back(std::move(b));
    cv_.notify_one();
    return PushResult::kOk;
  }

  // Waits up to timeout. Returns null on timeout, or when closed and empty (*drained=true).
  std::unique_ptr<Batch> Pop(std::chrono::nanoseconds timeout, bool* drained) {
    std::unique_lock<std::mutex> lk(mu_);
    *drained = false;
    cv_.wait_for(lk, timeout, [&] { return !items_.empty() || closed_; });
    if (items_.empty()) {
      *drained = closed_;
      return nullptr;
    }
    std::unique_ptr<Batch> b = std::move(items_.front());
    items_.pop_front();
    per_stream_[static_cast<size_t>(b->stream)]--;
    return b;
  }

  void Close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  // Failure path: discard everything still queued (data that will never reach disk).
  size_t DiscardAll() {
    std::deque<std::unique_ptr<Batch>> dead;
    {
      std::lock_guard<std::mutex> lk(mu_);
      dead.swap(items_);
      for (auto& q : per_stream_) q = 0;
    }
    return dead.size();  // freed outside the lock
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::unique_ptr<Batch>> items_;
  std::vector<size_t> per_stream_;
  size_t quota_;
  bool closed_ = false;
};

}  // namespace netmon
