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
// Bounded single-producer single-consumer ring. Push never blocks and never allocates:
// when full it returns false and the caller counts the drop.
#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace xgmi {

template <typename T>
class SpscRing {
 public:
  explicit SpscRing(size_t capacity) : buf_(capacity + 1) {}

  bool Push(const T& v) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t next = (h + 1) % buf_.size();
    if (next == tail_.load(std::memory_order_acquire)) return false;
    buf_[h] = v;
    head_.store(next, std::memory_order_release);
    return true;
  }

  // Moves up to `max` items into `out`; returns how many.
  size_t PopMany(T* out, size_t max) {
    size_t t = tail_.load(std::memory_order_relaxed);
    const size_t h = head_.load(std::memory_order_acquire);
    size_t n = 0;
    while (t != h && n < max) {
      out[n++] = buf_[t];
      t = (t + 1) % buf_.size();
    }
    tail_.store(t, std::memory_order_release);
    return n;
  }

 private:
  std::vector<T> buf_;
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};

}  // namespace xgmi
