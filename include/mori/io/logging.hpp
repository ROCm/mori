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

#include <chrono>
#include <string>

#include "spdlog/spdlog.h"

namespace mori {
namespace io {

#define MORI_IO_TRACE spdlog::trace
#define MORI_IO_DEBUG spdlog::debug
#define MORI_IO_INFO spdlog::info
#define MORI_IO_WARN spdlog::warn
#define MORI_IO_ERROR spdlog::error
#define MORI_IO_CRITICAL spdlog::critical

// trace / debug / info / warning / error / critical
inline void SetLogLevel(const std::string& strLevel) {
  spdlog::level::level_enum level = spdlog::level::from_str(strLevel);
  spdlog::set_level(level);
  MORI_IO_INFO("Set MORI-IO log level to {}", spdlog::level::to_string_view(level));
}

class ScopedTimer {
 public:
  using Clock = std::chrono::steady_clock;

  explicit ScopedTimer(const std::string& n) : name(n), start(Clock::now()) {}

  ~ScopedTimer() {
    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    MORI_IO_DEBUG("ScopedTimer [{}] took {} ns", name, duration);
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  std::string name;
  Clock::time_point start;
};

#define MORI_IO_TIMER(message) ScopedTimer instance(message)
#define MORI_IO_FUNCTION_TIMER ScopedTimer instance(__PRETTY_FUNCTION__)
}  // namespace io
}  // namespace mori
