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

// Shared rendering for the batch fan-out ("where did this batch land") log
// lines emitted by the master's RoutePut strategy and by PoolClient's
// BatchPut/BatchGet.  All three answer the same question -- is the batch spread
// over every node (and therefore every NVMe drive), or piling onto one -- so
// they share one format and one notion of "share".

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>

namespace mori::umbp {

/// Target ("node/TIER") -> {keys, bytes}.
using BatchDistMap = std::map<std::string, std::pair<uint64_t, uint64_t>>;

/// "node/TIER:keys=12,bytes=48.0MiB,share=25.0% ..." sorted by target, so
/// successive lines line up column-wise when eyeballing a log.  @p total_keys is
/// the denominator for share; 0 renders share as "n/a".
inline std::string FormatBatchDist(const BatchDistMap& dist, uint64_t total_keys) {
  if (dist.empty()) return "<none>";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  bool first = true;
  for (const auto& [target, counts] : dist) {
    if (!first) oss << ' ';
    first = false;
    oss << target << ":keys=" << counts.first
        << ",bytes=" << static_cast<double>(counts.second) / (1024.0 * 1024.0) << "MiB,share=";
    if (total_keys == 0) {
      oss << "n/a";
    } else {
      oss << (100.0 * static_cast<double>(counts.first) / static_cast<double>(total_keys)) << '%';
    }
  }
  return oss.str();
}

/// Largest single-target key share, in [0, 100].  100/N means perfectly even
/// over N targets; 100 means everything landed on one node/tier.
inline double BatchDistMaxShare(const BatchDistMap& dist, uint64_t total_keys) {
  if (dist.empty() || total_keys == 0) return 0.0;
  uint64_t max_keys = 0;
  for (const auto& [target, counts] : dist) max_keys = std::max(max_keys, counts.first);
  return 100.0 * static_cast<double>(max_keys) / static_cast<double>(total_keys);
}

/// Add @p src into @p dst target-by-target.
inline void AccumulateBatchDist(const BatchDistMap& src, BatchDistMap* dst) {
  for (const auto& [target, counts] : src) {
    auto& total = (*dst)[target];
    total.first += counts.first;
    total.second += counts.second;
  }
}

/// Total keys across every target.
inline uint64_t BatchDistTotalKeys(const BatchDistMap& dist) {
  uint64_t total = 0;
  for (const auto& [target, counts] : dist) total += counts.first;
  return total;
}

/// Total bytes across every target.
inline uint64_t BatchDistTotalBytes(const BatchDistMap& dist) {
  uint64_t total = 0;
  for (const auto& [target, counts] : dist) total += counts.second;
  return total;
}

}  // namespace mori::umbp
