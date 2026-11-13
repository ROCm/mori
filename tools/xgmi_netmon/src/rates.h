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
// Interval rules, shared by the live push and the importer so both write the same points.
//
// For consecutive snapshots of one GPU:
//   dt = delta firmware_timestamp * 10 ns   (no point if dt <= 0)
//   per link and direction: bytes = delta counter * 1024
//     - a decrease (driver reload) gives no point for that link and direction
//     - a rate above 2x the link max is not physical: no point (counted as implausible)
// Point time = real_ns of the read that first saw the snapshot, in ms.
//
// Series (Prometheus text with timestamps), labels experiment, host, gpu, and peer, dir per link
// (dir = wr: sent by this GPU on the link, rd: received):
//   xgmi_bytes, xgmi_rate_bytes_per_second  per link and direction
//   xgmi_interval_seconds                   per GPU
#pragma once

#include <cstdint>
#include <string>

#include "recording.h"

namespace xgmi {

struct RateStats {
  uint64_t intervals = 0;
  uint64_t points = 0;       // lines written
  uint64_t stale = 0;        // firmware_timestamp did not advance
  uint64_t decreases = 0;    // link-direction intervals dropped: counter went down
  uint64_t implausible = 0;  // link-direction intervals dropped: rate > 2x link max
};

// Prometheus label value escaping (\, ", newline).
std::string EscapeLabel(const std::string& v);

class RateCalc {
 public:
  RateCalc(const std::string& experiment, const std::string& host, int gpu,
           const int peer[kNumLinks], double link_max);
  // Appends the lines of the interval ending at `r` (none for the first record).
  void Add(const Record& r, std::string* out);
  const RateStats& stats() const { return stats_; }

 private:
  std::string gpu_labels_;
  std::string link_labels_[kNumLinks][2];  // [port][0 = rd, 1 = wr]; empty if not connected
  double link_max_;
  bool have_prev_ = false;
  Snapshot prev_;
  RateStats stats_;
};

}  // namespace xgmi
