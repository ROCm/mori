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
#include "umbp/distributed/metrics/prometheus_metric_sink.h"

#include <algorithm>
#include <utility>

#include "mori/metrics/prometheus_metrics_server.hpp"

namespace mori::umbp {

PrometheusMetricSink::PrometheusMetricSink(int port, std::string node_id)
    : node_id_(std::move(node_id)), server_(std::make_unique<mori::metrics::MetricsServer>(port)) {}

PrometheusMetricSink::~PrometheusMetricSink() = default;

int PrometheusMetricSink::port() const { return server_->port(); }

MetricSink::Labels PrometheusMetricSink::WithNode(Labels labels) const {
  if (node_id_.empty()) return labels;
  // A component that already knows which node it speaks for keeps its own
  // value: overwriting it here would be this sink inventing a fact about a
  // series it is only forwarding.
  const bool has_node =
      std::any_of(labels.begin(), labels.end(), [](const auto& kv) { return kv.first == "node"; });
  if (!has_node) labels.emplace_back("node", node_id_);
  return labels;
}

void PrometheusMetricSink::AddCounter(std::string name, std::string help, Labels labels,
                                      double delta) {
  server_->addCounter(name, help, WithNode(std::move(labels)), delta);
}

void PrometheusMetricSink::SetGauge(std::string name, std::string help, Labels labels,
                                    double value) {
  server_->setGauge(name, help, WithNode(std::move(labels)), value);
}

void PrometheusMetricSink::Observe(std::string name, std::string help, Labels labels,
                                   const std::vector<double>& bounds, double value) {
  server_->observe(name, help, WithNode(std::move(labels)), bounds, value);
}

}  // namespace mori::umbp
