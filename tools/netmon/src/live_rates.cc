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
#include "live_rates.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace netmon {

// ------------------------------------------------------------------ WindowMedian

void WindowMedian::Add(double v) {
  if (ring_.size() < cap_) {
    ring_.push_back(v);
  } else {
    ring_[next_] = v;
  }
  next_ = (next_ + 1) % cap_;
  n_ = ring_.size();
  dirty_ = true;
  since_++;
}

void WindowMedian::Clear() {
  ring_.clear();
  next_ = n_ = since_ = 0;
  dirty_ = true;
}

double WindowMedian::Get() {
  if (n_ == 0) return std::numeric_limits<double>::quiet_NaN();
  if (!dirty_) return cached_;
  if (n_ >= kExactBelow && since_ < kRecomputeEvery) return cached_;
  since_ = 0;
  scratch_.assign(ring_.begin(), ring_.end());
  const size_t mid = n_ / 2;
  std::nth_element(scratch_.begin(), scratch_.begin() + mid, scratch_.end());
  double m = scratch_[mid];
  if (n_ % 2 == 0) {  // numpy: mean of the two middle values
    const double lo = *std::max_element(scratch_.begin(), scratch_.begin() + mid);
    m = (lo + m) / 2.0;
  }
  cached_ = m;
  dirty_ = false;
  return m;
}

// ------------------------------------------------------------------ LiveRates

namespace {
// numpy's np.round: round half to even (the default floating-point rounding mode).
int64_t RoundMs(double wall_ns) { return static_cast<int64_t>(std::nearbyint(wall_ns / 1e6)); }
}  // namespace

LiveRates::LiveRates(uint64_t interval_ns, double snapshot_fraction, size_t counters,
                     std::vector<bool> optional, size_t pending_cap)
    : interval_ns_(interval_ns),
      f_(snapshot_fraction),
      pending_cap_(std::max<size_t>(pending_cap, 2)),
      ctr_(counters) {
  for (size_t c = 0; c < counters && c < optional.size(); c++) {
    ctr_[c].optional = optional[c];
    ctr_[c].active = !optional[c];
  }
}

void LiveRates::Add(const uint64_t* rec, LiveSink* sink) {
  stats_.samples++;
  // Same arithmetic as the offline code (float64 throughout), so results match bit for bit
  // when the medians agree.
  const double t0 = static_cast<double>(rec[0]);
  const double rtt = static_cast<double>(rec[1]) - t0;
  rtt_.Add(rtt);
  const double med = rtt_.Get();
  Sample s;
  s.t = t0 + f_ * rtt;
  s.slow = WarmedUp() && med > 0 && rtt > kSlowFactor * med;
  s.v.assign(rec + 4, rec + 4 + ctr_.size());

  const bool trusted = rec[3] <= kAnchorMaxErrNs;
  if (!trusted) {
    pending_.push_back(std::move(s));
    while (pending_.size() > pending_cap_) {  // §3.3: never output an unconfirmed sample
      Sample& old = pending_.front();
      old.uncertain = true;
      Confirm(old, sink);
      pending_.pop_front();
      stats_.unconfirmed_dropped++;
    }
    return;
  }
  const double real = static_cast<double>(rec[2]);
  const double err = static_cast<double>(rec[3]);
  const double off = real - (t0 - err / 2.0);
  if (!have_anchor_) {
    // Samples before the first trusted anchor take its segment (offline: segment after).
    have_anchor_ = true;
    off_.Add(off);
  } else if (std::fabs(off - last_off_) > static_cast<double>(kClockStepNs)) {
    // A clock step between the previous trusted anchor and this one: every sample in
    // between (the previous anchor included) has no certain wall-clock placement.
    for (auto& p : pending_) p.uncertain = true;
    seg_++;
    for (auto& p : pending_) Confirm(p, sink);
    pending_.clear();
    off_.Clear();
    off_.Add(off);
    last_off_ = off;
    pending_.push_back(std::move(s));
    return;
  } else {
    off_.Add(off);
  }
  last_off_ = off;
  for (auto& p : pending_) Confirm(p, sink);
  pending_.clear();
  pending_.push_back(std::move(s));  // placed once the next trusted anchor arrives
}

void LiveRates::Finish(LiveSink* sink) {
  if (have_anchor_) {
    for (auto& p : pending_) Confirm(p, sink);
  }
  pending_.clear();
}

void LiveRates::Confirm(Sample& s, LiveSink* sink) {
  if (s.uncertain) {
    // Still part of the full series for reset detection (offline finds decreases on every
    // sample), but never a base and never the end of a usable interval.
    for (size_t c = 0; c < ctr_.size(); c++) {
      Counter& k = ctr_[c];
      const uint64_t v = s.v[c];
      if (k.have_raw && v < k.raw) k.reset_pending = true;
      k.raw = v;
      if (!k.have_raw) k.first = v;
      k.have_raw = true;
      if (k.optional && !k.active && v != k.first) k.active = true;
      if (k.have_base) k.clock_break = true;
    }
    return;
  }
  s.wall = s.t + off_.Get();
  s.seg = seg_;
  Step(s, sink);
}

void LiveRates::Step(const Sample& s, LiveSink* sink) {
  const int64_t end_ms = RoundMs(s.wall);
  for (size_t c = 0; c < ctr_.size(); c++) {
    Counter& k = ctr_[c];
    const uint64_t v = s.v[c];
    // 1. reset detection on the full series (slow samples included)
    if (k.have_raw && v < k.raw) k.reset_pending = true;
    if (!k.have_raw) k.first = v;
    k.raw = v;
    k.have_raw = true;
    if (k.optional && !k.active && v != k.first) k.active = true;
    // 2. a slow sample has no reliable time: never a base, never an interval end
    if (s.slow) continue;
    auto set_base = [&] {
      k.have_base = true;
      k.base_t = s.t;
      k.base_wall = s.wall;
      k.base_seg = s.seg;
      k.base_v = v;
      k.reset_pending = false;
      k.clock_break = false;
    };
    if (!k.have_base) {
      set_base();
      continue;
    }
    const double dt = s.t - k.base_t;
    if (k.reset_pending || v < k.base_v) {  // reset takes precedence over clock (offline)
      if (k.active) EmitGap(k, c, false, end_ms, dt, sink);
      set_base();
      continue;
    }
    if (!(dt > 0)) {  // offline: not valid, not a reset, no gap
      set_base();
      continue;
    }
    if (k.clock_break || k.base_seg != s.seg) {
      if (k.active) EmitGap(k, c, true, end_ms, dt, sink);
      set_base();
      continue;
    }
    // 3. millisecond timestamps must increase per series (§3.4)
    if (end_ms <= k.last_end_ms) {
      if (s.seg == k.last_seg) {
        stats_.ms_merged++;  // rounding collision: this sample is skipped, the base stays
        continue;
      }
      stats_.clock_back_intervals++;  // wall clock went back: paused until it passes
      set_base();
      continue;
    }
    k.dt.Add(dt);
    const double base = std::max(static_cast<double>(interval_ns_), k.dt.Get());
    const bool low = base > 0 && dt > kLowResFactor * base;
    const double dv = static_cast<double>(v - k.base_v);
    if (k.active) sink->Interval(c, low, end_ms, dv / dt * 1e9, dv, dt / 1e9);
    k.last_end_ms = end_ms;
    k.last_seg = s.seg;
    set_base();
  }
}

void LiveRates::EmitGap(Counter& k, size_t c, bool clock, int64_t end_ms, double dt_ns,
                        LiveSink* sink) {
  int64_t& last = k.last_gap_ms[clock ? 1 : 0];
  if (end_ms <= last) {
    stats_.gaps_dropped++;
    return;
  }
  last = end_ms;
  sink->Gap(c, clock, end_ms, dt_ns / 1e9);
}

double LiveRates::BaselineNs() {
  if (!WarmedUp()) return 0;
  double b = 0;
  for (auto& k : ctr_) {
    if (k.dt.size() == 0) continue;
    b = std::max(b, std::max(static_cast<double>(interval_ns_), k.dt.Get()));
  }
  return b;
}

}  // namespace netmon
