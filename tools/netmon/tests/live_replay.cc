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
//
// Test tool: replays records through LiveRates and prints what it emits, so the Python tests
// (test_live.py) and the live-vs-import comparison can check it against the offline Rates.
//
//   netmon_live_replay INTERVAL_NS SNAPSHOT_FRACTION COUNTERS OPTIONAL_MASK PENDING_CAP
//     stdin:  u64le records [t0, t1, real, real_err, v0 .. v(COUNTERS-1)]
//     stdout: "I,c,low,end_ms,rate,increase,dt_s" / "G,c,clock,end_ms,dt_s" lines (%.17g),
//             then "S,samples,unconfirmed_dropped,clock_back,ms_merged,gaps_dropped,baseline_ns"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "live_rates.h"

namespace {

class Printer : public netmon::LiveSink {
 public:
  void Interval(size_t c, bool low, int64_t end_ms, double rate, double inc, double dt) override {
    printf("I,%zu,%d,%lld,%.17g,%.17g,%.17g\n", c, low ? 1 : 0, static_cast<long long>(end_ms),
           rate, inc, dt);
  }
  void Gap(size_t c, bool clock, int64_t end_ms, double dt) override {
    printf("G,%zu,%d,%lld,%.17g\n", c, clock ? 1 : 0, static_cast<long long>(end_ms), dt);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s INTERVAL_NS F COUNTERS OPTIONAL_MASK PENDING_CAP < records\n",
            argv[0]);
    return 2;
  }
  const uint64_t interval = strtoull(argv[1], nullptr, 10);
  const double f = strtod(argv[2], nullptr);
  const size_t n = strtoull(argv[3], nullptr, 10);
  const std::string mask = argv[4];
  const size_t cap = strtoull(argv[5], nullptr, 10);
  std::vector<bool> optional(n, false);
  for (size_t c = 0; c < n && c < mask.size(); c++) optional[c] = mask[c] == '1';
  netmon::LiveRates lr(interval, f, n, optional, cap);
  Printer p;
  std::vector<uint64_t> rec(4 + n);
  while (fread(rec.data(), sizeof(uint64_t), rec.size(), stdin) == rec.size()) {
    lr.Add(rec.data(), &p);
  }
  lr.Finish(&p);
  const auto& st = lr.stats();
  printf("S,%llu,%llu,%llu,%llu,%llu,%.17g\n", static_cast<unsigned long long>(st.samples),
         static_cast<unsigned long long>(st.unconfirmed_dropped),
         static_cast<unsigned long long>(st.clock_back_intervals),
         static_cast<unsigned long long>(st.ms_merged),
         static_cast<unsigned long long>(st.gaps_dropped), lr.BaselineNs());
  return 0;
}
