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

// Stage-by-stage timing for the SSD batch put/get path, enabled by UMBP_SSD_TIMING.
// When off every helper is a cached bool test and no clock is read, so this can
// stay in the hot path.  Emitted at INFO (module UMBP) as [SsdPerf/tier] (one
// drive: lookup | IO | CRC), [SsdPerf/shard] (per-drive fan-out), [SsdPerf/peer]
// and [SsdPerf/remote], each with keys, bytes, ms and the implied GB/s.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace mori::umbp::ssdperf {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Env is read once; the hot-path check is a load of a cached bool.
inline bool Enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("UMBP_SSD_TIMING");
    if (v == nullptr || *v == '\0') return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
  }();
  return enabled;
}

// Meaningless when timing is off, but only ever consumed inside an Enabled() branch.
inline TimePoint Now() { return Enabled() ? Clock::now() : TimePoint{}; }

inline double MsBetween(TimePoint a, TimePoint b) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count() * 1000.0;
}

inline double MsSince(TimePoint a) { return Enabled() ? MsBetween(a, Clock::now()) : 0.0; }

// Guarded against a zero denominator so a log line never prints inf.
inline double GbPerSec(uint64_t bytes, double ms) {
  return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1.0e6) : 0.0;
}

inline double Pct(double part_ms, double total_ms) {
  return total_ms > 0.0 ? part_ms * 100.0 / total_ms : 0.0;
}

}  // namespace mori::umbp::ssdperf
