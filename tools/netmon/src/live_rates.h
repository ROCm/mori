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

// Online version of the offline report's interval rules (ainic_netmon_report.Rates; design:
// NETMON_LIVE_PUSH_DESIGN §3). Samples go in one at a time, in order; intervals and gaps come
// out through a sink, at most one sample period after the wall-clock placement of a sample is
// certain. Pure computation: no I/O, no locks, no clocks.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace netmon {

// Median of the most recent `cap` values. Recomputed on every read while the window is small,
// then at most once per kRecomputeEvery additions (an nth_element over 1024 values per
// sample and counter would dominate the exporter's CPU time).
class WindowMedian {
 public:
  explicit WindowMedian(size_t cap = 1024) : cap_(cap) {}
  void Add(double v);
  void Clear();
  static constexpr size_t kExactBelow = 64, kRecomputeEvery = 32;
  size_t size() const { return n_; }
  double Get();  // NaN if empty

 private:
  size_t cap_;
  std::vector<double> ring_;
  size_t next_ = 0, n_ = 0;
  size_t since_ = 0;  // additions since the last recompute
  bool dirty_ = true;
  double cached_ = 0;
  std::vector<double> scratch_;
};

class LiveSink {
 public:
  virtual ~LiveSink() = default;
  // A valid interval of counter `c` ending at wall-clock `end_ms`.
  virtual void Interval(size_t c, bool low_res, int64_t end_ms, double rate_per_s, double increase,
                        double dt_s) = 0;
  // An unusable interval: `clock` = no certain wall-clock placement, else a counter reset.
  virtual void Gap(size_t c, bool clock, int64_t end_ms, double dt_s) = 0;
};

struct LiveStats {
  uint64_t samples = 0;
  uint64_t unconfirmed_dropped = 0;   // pending buffer overflow (§3.3)
  uint64_t clock_back_intervals = 0;  // dropped while paused after a backward step (§3.4)
  uint64_t ms_merged = 0;             // samples folded into the next interval (§3.4)
  uint64_t gaps_dropped = 0;          // gap points whose ms timestamp did not advance
};

class LiveRates {
 public:
  static constexpr size_t kWarmupSamples = 32;
  static constexpr double kSlowFactor = 1.5;
  static constexpr double kLowResFactor = 3.0;
  static constexpr uint64_t kAnchorMaxErrNs = 50000;
  static constexpr int64_t kClockStepNs = 1000000;

  // `optional[c]`: counter c is emitted only from its first change on (§3.5).
  LiveRates(uint64_t interval_ns, double snapshot_fraction, size_t counters,
            std::vector<bool> optional, size_t pending_cap);

  // rec = [t0_mono, t1_mono, real, real_err, v0 .. v(counters-1)].
  void Add(const uint64_t* rec, LiveSink* sink);
  // End of the stream: samples after the last trusted anchor keep its segment (offline:
  // no later anchor = not uncertain).
  void Finish(LiveSink* sink);

  // Largest per-counter baseline (ns) once warmed up, else 0. min_step = ceil(3 x this).
  double BaselineNs();
  bool WarmedUp() const { return stats_.samples >= kWarmupSamples; }
  const LiveStats& stats() const { return stats_; }

 private:
  struct Sample {
    double t = 0;     // monotonic sample time (ns, t0 + f * rtt)
    double wall = 0;  // set when confirmed
    int64_t seg = -1;
    bool slow = false;
    bool uncertain = false;
    std::vector<uint64_t> v;
  };
  struct Counter {
    bool optional = false, active = true;
    bool have_raw = false;
    uint64_t raw = 0, first = 0;
    bool reset_pending = false;
    bool have_base = false;
    double base_t = 0, base_wall = 0;
    int64_t base_seg = -1;
    uint64_t base_v = 0;
    bool clock_break = false;  // uncertain / dropped samples since the base
    int64_t last_end_ms = INT64_MIN;
    int64_t last_seg = -1;  // segment of the last emitted interval end
    int64_t last_gap_ms[2] = {INT64_MIN, INT64_MIN};
    WindowMedian dt{1024};
  };

  void Confirm(Sample& s, LiveSink* sink);
  void Step(const Sample& s, LiveSink* sink);
  void EmitGap(Counter& k, size_t c, bool clock, int64_t end_ms, double dt_ns, LiveSink* sink);

  uint64_t interval_ns_;
  double f_;
  size_t pending_cap_;
  std::vector<Counter> ctr_;
  WindowMedian rtt_{1024};
  WindowMedian off_{1024};
  std::deque<Sample> pending_;  // pending_[0] is the last trusted anchor once one was seen
  bool have_anchor_ = false;
  double last_off_ = 0;
  int64_t seg_ = 0;
  LiveStats stats_;
};

}  // namespace netmon
