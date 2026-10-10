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

#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/metrics/metric_sink.h"

namespace mori::metrics {
class MetricsServer;
}

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  PrometheusMetricSink — a node that serves its own numbers.
//
//  The masterless counterpart to MasterClient: instead of buffering samples
//  and shipping them to a master that owns the only MetricsServer in the
//  deployment, this sink writes them straight into a MetricsServer running in
//  THIS process, which a Prometheus job scrapes at http://<node>:<port>/metrics.
//
//  It also stamps a `node` label onto every series it forwards, which is what
//  master used to add on ingest.  Keeping it means a dashboard query written
//  against a master-backed cluster (`sum by (node) (...)`) works unchanged
//  against a standalone server, and two servers scraped into one Prometheus do
//  not silently collapse into one series.
//
//  Owns the MetricsServer: construction binds the port and throws
//  std::runtime_error if it cannot, so a misconfigured port fails loudly at
//  startup instead of producing a server that is quietly unscrapeable.
// ---------------------------------------------------------------------------
class PrometheusMetricSink final : public MetricSink {
 public:
  // Binds `port` immediately.  `node_id` is stamped as a `node` label on every
  // series; an empty node_id stamps nothing.
  PrometheusMetricSink(int port, std::string node_id);
  ~PrometheusMetricSink() override;

  PrometheusMetricSink(const PrometheusMetricSink&) = delete;
  PrometheusMetricSink& operator=(const PrometheusMetricSink&) = delete;

  void AddCounter(std::string name, std::string help, Labels labels, double delta) override;
  void SetGauge(std::string name, std::string help, Labels labels, double value) override;
  void Observe(std::string name, std::string help, Labels labels, const std::vector<double>& bounds,
               double value) override;

  int port() const;
  const std::string& node_id() const { return node_id_; }

  // Exposed for tests that want to scrape without going over a socket.
  mori::metrics::MetricsServer* server() { return server_.get(); }

 private:
  // Appends {node=node_id_} unless the caller already supplied one.
  Labels WithNode(Labels labels) const;

  std::string node_id_;
  std::unique_ptr<mori::metrics::MetricsServer> server_;
};

}  // namespace mori::umbp
