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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace mori::umbp::bench {

inline std::string ToLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline bool FilterMatches(const std::string& filter, const std::string& scenario_name) {
  if (filter.empty()) return true;
  return ToLowerCopy(scenario_name).find(ToLowerCopy(filter)) != std::string::npos;
}

inline size_t CountTrue(const std::vector<bool>& values) {
  return static_cast<size_t>(std::count(values.begin(), values.end(), true));
}

inline size_t CountSuccessfulPages(const std::vector<bool>& key_results, size_t keys_per_page) {
  if (keys_per_page == 0) return 0;
  size_t ok_pages = 0;
  for (size_t i = 0; i < key_results.size(); i += keys_per_page) {
    bool page_ok = true;
    for (size_t j = 0; j < keys_per_page && i + j < key_results.size(); ++j) {
      page_ok = page_ok && key_results[i + j];
    }
    if (page_ok) ++ok_pages;
  }
  return ok_pages;
}

inline double ThroughputOpsPerSec(size_t successful_ops, double elapsed_sec) {
  return elapsed_sec > 0.0 ? static_cast<double>(successful_ops) / elapsed_sec : 0.0;
}

inline double ThroughputMbPerSec(size_t successful_bytes, double elapsed_sec) {
  return elapsed_sec > 0.0 ? static_cast<double>(successful_bytes) / (1024.0 * 1024.0) / elapsed_sec
                           : 0.0;
}

struct ResultTally {
  size_t requested_ops = 0;
  size_t successful_ops = 0;
  size_t requested_bytes = 0;
  size_t successful_bytes = 0;
  std::vector<double> latencies_us;

  void ReserveLatencySamples(size_t n) { latencies_us.reserve(n); }

  size_t failed_ops() const {
    return (requested_ops >= successful_ops) ? (requested_ops - successful_ops) : 0;
  }

  size_t sample_count() const { return latencies_us.size(); }

  void AddSample(size_t requested, size_t succeeded, size_t req_bytes, size_t ok_bytes,
                 double total_us) {
    requested_ops += requested;
    successful_ops += succeeded;
    requested_bytes += req_bytes;
    successful_bytes += ok_bytes;
    if (requested == 0) return;

    // Record one latency sample per batch call.  Percentiles (p50/p95/p99)
    // reflect batch-call-level variance — the `n` column in the output shows
    // the number of timing samples while `req`/`ok` show per-op counts.
    latencies_us.push_back(total_us);
  }

  void AddCall(size_t requested, size_t succeeded, size_t req_bytes, size_t ok_bytes,
               double total_us) {
    requested_ops += requested;
    successful_ops += succeeded;
    requested_bytes += req_bytes;
    successful_bytes += ok_bytes;
    latencies_us.push_back(total_us);
  }

  void AddOp(bool ok, size_t bytes, double total_us) {
    AddSample(1, ok ? 1 : 0, bytes, ok ? bytes : 0, total_us);
  }

  void Merge(const ResultTally& other) {
    requested_ops += other.requested_ops;
    successful_ops += other.successful_ops;
    requested_bytes += other.requested_bytes;
    successful_bytes += other.successful_bytes;
    latencies_us.insert(latencies_us.end(), other.latencies_us.begin(), other.latencies_us.end());
  }
};

enum class BatchWriteMode {
  Fused,
  Fallback,
};

inline size_t EstimateRecordBytes(size_t key_size, size_t value_size, size_t record_header_size) {
  return record_header_size + key_size + value_size;
}

inline size_t EstimateBatchRecordBytes(size_t keys_in_batch, size_t key_size, size_t value_size,
                                       size_t record_header_size) {
  return keys_in_batch * EstimateRecordBytes(key_size, value_size, record_header_size);
}

inline BatchWriteMode ClassifyBatchWriteMode(size_t keys_in_batch, size_t key_size,
                                             size_t value_size, size_t segment_size,
                                             size_t record_header_size) {
  return EstimateBatchRecordBytes(keys_in_batch, key_size, value_size, record_header_size) <=
                 segment_size
             ? BatchWriteMode::Fused
             : BatchWriteMode::Fallback;
}

struct WorkloadSummary {
  size_t num_keys = 0;
  size_t value_size = 0;
  size_t batch_size = 0;
  size_t dram_capacity = 0;
  size_t ssd_capacity = 0;
  size_t segment_size = 0;
  size_t key_size_hint = 0;
  size_t record_header_size = 0;
};

struct ConfigWarning {
  std::string message;
};

inline std::vector<ConfigWarning> CollectConfigWarnings(const WorkloadSummary& summary) {
  std::vector<ConfigWarning> warnings;
  if (summary.value_size > summary.dram_capacity) {
    warnings.push_back(
        {"value_size exceeds DRAM capacity; DRAM-resident and cold CopyToSSD scenarios will "
         "partially fail or be skipped."});
  }
  if (summary.value_size > summary.ssd_capacity) {
    warnings.push_back(
        {"value_size exceeds SSD capacity; single-value SSD writes cannot succeed."});
  }
  if (summary.num_keys * summary.value_size > summary.ssd_capacity) {
    warnings.push_back(
        {"working set exceeds SSD capacity; warm-cache SSD read scenarios will report mixed "
         "residency rather than full-hit throughput."});
  }
  if (summary.batch_size > 0 &&
      ClassifyBatchWriteMode(summary.batch_size, summary.key_size_hint, summary.value_size,
                             summary.segment_size,
                             summary.record_header_size) == BatchWriteMode::Fallback) {
    warnings.push_back(
        {"batch payload exceeds segment_size; WriteBatch-style scenarios will "
         "exercise fallback/per-key behavior instead of fused writes."});
  }
  return warnings;
}

}  // namespace mori::umbp::bench
