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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mori {
namespace metrics {

// ---------------------------------------------------------------------------
// MetricsServer
//
// Embeds a minimal HTTP server that serves Prometheus text-format metrics on
// GET /metrics.  Designed to be instantiated once at process start and live
// for the duration of the process.
//
// Usage:
//   mori::metrics::MetricsServer srv(9091);   // start once
//
//   // anywhere in mori:
//   srv.set_gauge("mori_rdma_throughput_gbps", "RDMA write throughput", val);
//   srv.add_counter("mori_requests_total", "Completed requests");
//   srv.observe("mori_latency_seconds", "End-to-end latency",
//               {0.001, 0.01, 0.1, 0.5, 1.0, 5.0}, measured_sec);
// ---------------------------------------------------------------------------
class MetricsServer {
 public:
  // Create and immediately start the server on `port`.
  // Throws std::runtime_error if the socket cannot be bound.
  explicit MetricsServer(int port = 9091);

  // Stop the accept loop and join the background thread.
  ~MetricsServer();

  MetricsServer(const MetricsServer&) = delete;
  MetricsServer& operator=(const MetricsServer&) = delete;
  MetricsServer(MetricsServer&&) = delete;
  MetricsServer& operator=(MetricsServer&&) = delete;

  // ------------------------------------------------------------------
  // Utility
  // ------------------------------------------------------------------

  // Sanitize a string for use as part of a Prometheus metric name.
  // Replaces any character outside [a-zA-Z0-9_] with '_'.
  static std::string SanitizeName(std::string_view s);

  // ------------------------------------------------------------------
  // Metric update API (all methods are thread-safe)
  // ------------------------------------------------------------------

  // Overwrite the current value of a gauge.
  void setGauge(std::string_view name, std::string_view help, double value);

  // Add `delta` to a monotonically-increasing counter (default delta = 1).
  void addCounter(std::string_view name, std::string_view help, uint64_t delta = 1);

  // Record one histogram observation.
  // `bounds` is an ascending list of finite upper-bound values; the implicit
  // +Inf bucket is always appended automatically.  On the first call for a
  // given `name`, `bounds` initialises the bucket layout; later calls ignore
  // `bounds` and reuse the stored layout.
  void observe(std::string_view name, std::string_view help, const std::vector<double>& bounds,
               double value);

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------
  int port() const noexcept { return port_; }
  bool running() const noexcept { return running_.load(std::memory_order_relaxed); }

 private:
  // ---- per-metric storage -------------------------------------------

  struct GaugeEntry {
    std::string help;
    double value{0.0};
  };

  struct CounterEntry {
    std::string help;
    uint64_t value{0};
  };

  struct HistogramEntry {
    std::string help;
    std::vector<double> bounds;           // explicit upper bounds (sorted)
    std::vector<uint64_t> bucket_counts;  // cumulative counts per bound
    uint64_t count{0};
    double sum{0.0};
  };

  // ---- internal helpers ---------------------------------------------

  // Serialise all metrics to Prometheus text format (caller must hold mutex_).
  std::string serializeLocked() const;

  // Main loop running in accept_thread_.
  void acceptLoop();

  // Handle a single connected client fd (read request, write response, close).
  void handleClient(int client_fd);

  // ---- member data --------------------------------------------------

  const int port_;
  int server_fd_{-1};
  std::atomic<bool> running_{false};
  std::thread accept_thread_;

  mutable std::mutex mutex_;
  std::map<std::string, GaugeEntry> gauges_;
  std::map<std::string, CounterEntry> counters_;
  std::map<std::string, HistogramEntry> histograms_;
};

}  // namespace metrics
}  // namespace mori
