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
#include "rates.h"

#include <cstdio>

namespace xgmi {

std::string EscapeLabel(const std::string& v) {
  std::string out;
  for (char c : v) {
    if (c == '\\' || c == '"') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

RateCalc::RateCalc(const std::string& experiment, const std::string& host, int gpu,
                   const int peer[kNumLinks], double link_max)
    : link_max_(link_max) {
  const std::string base = "experiment=\"" + EscapeLabel(experiment) + "\",host=\"" +
                           EscapeLabel(host) + "\",gpu=\"" + std::to_string(gpu) + "\"";
  gpu_labels_ = "{" + base + "}";
  for (int p = 0; p < kNumLinks; ++p) {
    if (peer[p] < 0) continue;
    const std::string link = "{" + base + ",peer=\"" + std::to_string(peer[p]) + "\",dir=\"";
    link_labels_[p][0] = link + "rd\"}";
    link_labels_[p][1] = link + "wr\"}";
  }
}

void RateCalc::Add(const Record& r, std::string* out) {
  if (!have_prev_) {
    prev_ = r.s;
    have_prev_ = true;
    return;
  }
  if (r.s.fw <= prev_.fw) {
    ++stats_.stale;
    return;  // keep prev_: the next advancing snapshot spans from it
  }
  const double dt = static_cast<double>(r.s.fw - prev_.fw) * 1e-8;
  const unsigned long long ts = r.real_ns / 1000000;
  char line[320];
  ++stats_.intervals;
  std::snprintf(line, sizeof(line), "xgmi_interval_seconds%s %.9f %llu\n", gpu_labels_.c_str(), dt,
                ts);
  *out += line;
  ++stats_.points;
  for (int p = 0; p < kNumLinks; ++p) {
    if (link_labels_[p][0].empty()) continue;
    const uint64_t cur[2] = {r.s.rd[p], r.s.wr[p]};
    const uint64_t old[2] = {prev_.rd[p], prev_.wr[p]};
    for (int d = 0; d < 2; ++d) {
      if (cur[d] < old[d]) {
        ++stats_.decreases;
        continue;
      }
      const unsigned long long bytes = (cur[d] - old[d]) * 1024ull;
      const double rate = static_cast<double>(bytes) / dt;
      if (link_max_ > 0 && rate > 2 * link_max_) {
        ++stats_.implausible;
        continue;
      }
      std::snprintf(line, sizeof(line),
                    "xgmi_bytes%s %llu %llu\nxgmi_rate_bytes_per_second%s %.6g %llu\n",
                    link_labels_[p][d].c_str(), bytes, ts, link_labels_[p][d].c_str(), rate, ts);
      *out += line;
      stats_.points += 2;
    }
  }
  prev_ = r.s;
}

}  // namespace xgmi
