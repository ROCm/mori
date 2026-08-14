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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace mori::umbp {

// Run fn(i) for i in [0, n), on up to `threads` workers, work-stealing so an
// uneven index cost cannot strand a worker.  Each index runs exactly once, so
// fn may write result[i] without locking.  The caller's thread takes part and
// the call returns only once every index is done.
template <typename Fn>
void ParallelFor(size_t n, int threads, Fn&& fn) {
  if (n == 0) return;
  const size_t workers = std::min<size_t>(n, static_cast<size_t>(std::max(threads, 1)));
  if (workers <= 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < n;
         i = next.fetch_add(1, std::memory_order_relaxed)) {
      fn(i);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (size_t w = 0; w + 1 < workers; ++w) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
}

}  // namespace mori::umbp
