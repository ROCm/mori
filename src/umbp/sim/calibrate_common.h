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
#include <cstdint>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mori::umbp::calibration {

inline size_t Positive(const std::string& text) {
  size_t parsed = 0;
  const auto value = std::stoull(text, &parsed);
  if (text.empty() || text.front() < '0' || text.front() > '9' || value == 0 ||
      parsed != text.size() || value > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("expected a positive integer");
  return static_cast<size_t>(value);
}

inline std::vector<size_t> Sizes(const std::string& text) {
  if (text.empty() || text.back() == ',') throw std::invalid_argument("empty size list entry");
  std::istringstream input(text);
  std::string item;
  std::vector<size_t> out;
  while (std::getline(input, item, ',')) out.push_back(Positive(item));
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Start callers together; thread creation and wake-up stay outside transfer timers.
template <class Fn>
std::vector<int64_t> Concurrent(size_t count, Fn run) {
  if (count == 1) return {run(0)};
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::atomic<size_t> ready{0};
  std::vector<std::future<int64_t>> pending;
  pending.reserve(count);
  try {
    for (size_t c = 0; c < count; ++c) {
      pending.push_back(std::async(std::launch::async, [&, c] {
        ++ready;
        gate.wait();
        return run(c);
      }));
    }
    while (ready.load() != count) std::this_thread::yield();
  } catch (...) {
    start.set_value();  // Drain existing callers if thread creation fails.
    throw;
  }
  start.set_value();
  std::vector<int64_t> times;
  for (auto& call : pending) times.push_back(call.get());
  return times;
}

}  // namespace mori::umbp::calibration
