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
// `xgmi-netmon record`: one sampler thread per GPU, a writer thread, an optional push thread.
//
// Sampler: wakes on the wall-clock grid (multiples of the period in CLOCK_REALTIME, like
// ainic-netmon) with clock_nanosleep(TIMER_ABSTIME), preads gpu_metrics and, when
// firmware_timestamp changed, offers the record to the writer ring and the push ring. It never
// blocks and never allocates: a full ring drops the record and counts it.
// Writer: drains the rings into DIR/gpu<N>.bin every 100 ms.
// Push: every 1 s turns new records into interval points (rates.h) and POSTs them; a failed POST
// drops its points (counted) and is never resent.
#pragma once

#include <cstdint>
#include <string>

namespace xgmi {

struct RecordOptions {
  std::string out;        // may contain {host}
  double duration_s = 0;  // 0: until SIGINT / SIGTERM
  uint64_t period_ns = 2000000;
  std::string push_url;    // empty: no live push
  std::string experiment;  // empty: name of the output's parent directory
  std::string host_id;     // empty: first DNS label of the hostname
  std::string drm_root = "/sys/class/drm";
};

int RunRecord(const RecordOptions& opt);

}  // namespace xgmi
