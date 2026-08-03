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

// Stage-by-stage timing for the SSD batch put/get path.
//
// Enabled by setting UMBP_SSD_TIMING (any non-empty value other than 0/false/off).
// When off, every helper here compiles down to a single cached bool test and no
// clock is read, so the instrumentation can stay in the hot path permanently.
//
// The breakdown is emitted at INFO level (module UMBP) at four layers, so a slow
// batch can be attributed without a profiler:
//
//   [SsdPerf/tier]   one physical drive: index lookup | device IO | CRC verify
//   [SsdPerf/shard]  the multi-drive fan-out: per-drive key counts and wall time
//   [SsdPerf/peer]   PeerSsdManager: metadata passes vs. backend IO
//   [SsdPerf/remote] PoolClient: prepare RPC | RDMA | lease release
//
// Every line reports keys, bytes, elapsed_ms and the implied GB/s for the stage
// that moved the bytes, so the drive-count scaling question is answerable
// directly from the logs.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace mori::umbp::ssdperf {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Cached once: the env is read on first use and never re-read, so the check on
// the hot path is a relaxed load of a bool.
inline bool Enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("UMBP_SSD_TIMING");
    if (v == nullptr || *v == '\0') return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
  }();
  return enabled;
}

// Clock reads are skipped entirely when timing is off; the returned value is
// then meaningless but is only ever consumed inside an Enabled() branch.
inline TimePoint Now() { return Enabled() ? Clock::now() : TimePoint{}; }

inline double MsBetween(TimePoint a, TimePoint b) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count() * 1000.0;
}

inline double MsSince(TimePoint a) { return Enabled() ? MsBetween(a, Clock::now()) : 0.0; }

// GB/s (10^9) for `bytes` moved in `ms`; 0 when no time elapsed, so a log line
// never prints inf.
inline double GbPerSec(uint64_t bytes, double ms) {
  return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1.0e6) : 0.0;
}

// Percentage of `total_ms` taken by `part_ms`, guarded against a zero total.
inline double Pct(double part_ms, double total_ms) {
  return total_ms > 0.0 ? part_ms * 100.0 / total_ms : 0.0;
}

// Accumulates elapsed time across repeated start/stop pairs (used where a stage
// runs once per key inside a loop).  No-op when timing is disabled.
class Accumulator {
 public:
  void Start() {
    if (Enabled()) start_ = Clock::now();
  }
  void Stop() {
    if (Enabled()) ms_ += MsBetween(start_, Clock::now());
  }
  double ms() const { return ms_; }

 private:
  TimePoint start_{};
  double ms_ = 0.0;
};

}  // namespace mori::umbp::ssdperf
