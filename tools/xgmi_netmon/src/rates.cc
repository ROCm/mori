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

std::string JsonString(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", c);
      out += b;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

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
  const std::string base = ",\"experiment\":" + JsonString(experiment) +
                           ",\"host\":" + JsonString(host) + ",\"gpu\":\"" + std::to_string(gpu) +
                           "\"";
  interval_.head = "{\"metric\":{\"__name__\":\"xgmi_interval_seconds\"" + base + "},\"values\":[";
  static const char* const kDir[2] = {"rd", "wr"};
  static const char* const kName[2] = {"xgmi_bytes", "xgmi_rate_bytes_per_second"};
  for (int p = 0; p < kNumLinks; ++p) {
    if (peer[p] < 0) continue;
    connected_[p] = true;
    for (int d = 0; d < 2; ++d)
      for (int m = 0; m < 2; ++m)
        link_[p][d][m].head = std::string("{\"metric\":{\"__name__\":\"") + kName[m] + "\"" + base +
                              ",\"peer\":\"" + std::to_string(peer[p]) + "\",\"dir\":\"" + kDir[d] +
                              "\"},\"values\":[";
  }
}

void RateCalc::Point(Series* s, const char* value, const char* ts) {
  if (!s->values.empty()) {
    s->values += ',';
    s->timestamps += ',';
  }
  s->values += value;
  s->timestamps += ts;
  ++pending_;
  ++stats_.points;
}

void RateCalc::Add(const Record& r) {
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
  char ts[24], v[40];
  std::snprintf(ts, sizeof(ts), "%llu", static_cast<unsigned long long>(r.real_ns / 1000000));
  ++stats_.intervals;
  std::snprintf(v, sizeof(v), "%.9f", dt);
  Point(&interval_, v, ts);
  for (int p = 0; p < kNumLinks; ++p) {
    if (!connected_[p]) continue;
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
      std::snprintf(v, sizeof(v), "%llu", bytes);
      Point(&link_[p][d][0], v, ts);
      std::snprintf(v, sizeof(v), "%.6g", rate);
      Point(&link_[p][d][1], v, ts);
    }
  }
  prev_ = r.s;
}

void RateCalc::Flush(std::string* out) {
  auto emit = [out](Series* s) {
    if (s->values.empty()) return;
    *out += s->head;
    *out += s->values;
    *out += "],\"timestamps\":[";
    *out += s->timestamps;
    *out += "]}\n";
    s->values.clear();
    s->timestamps.clear();
  };
  emit(&interval_);
  for (int p = 0; p < kNumLinks; ++p)
    for (int d = 0; d < 2; ++d)
      for (int m = 0; m < 2; ++m) emit(&link_[p][d][m]);
  pending_ = 0;
}

}  // namespace xgmi
