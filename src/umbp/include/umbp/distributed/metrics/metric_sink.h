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

#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
//  MetricSink — where a node's metrics go.
//
//  Everything UMBP measures on a peer used to have exactly one destination:
//  MasterClient buffered it and shipped it to the master, which merged it into
//  the master's own MetricsServer.  That made observability a property of
//  cluster membership rather than of the process, so a node running WITHOUT a
//  master — an embedded client, and every standalone server, which is the
//  deployment sglang's direct external linker actually uses — measured
//  everything and then dropped it on the floor, with no port to scrape and no
//  flag that could change that.
//
//  This interface is the seam that separates the two.  A component reports
//  into a MetricSink; whether that sink forwards over gRPC to a master
//  (MasterClient) or serves the numbers locally in Prometheus text format
//  (PrometheusMetricSink) is a deployment decision made once at wiring time.
// ---------------------------------------------------------------------------

namespace mori::umbp {

class MetricSink {
 public:
  using Labels = std::vector<std::pair<std::string, std::string>>;

  virtual ~MetricSink() = default;

  // Add `delta` to a monotonically-increasing counter series.
  virtual void AddCounter(std::string name, std::string help, Labels labels, double delta) = 0;

  // Overwrite the current reading of a gauge series.
  virtual void SetGauge(std::string name, std::string help, Labels labels, double value) = 0;

  // Record one histogram observation.  `bounds` is an ascending list of finite
  // upper bounds; the first write for a series wins the layout, matching
  // MetricsServer::observe() so the two sinks stay interchangeable.
  virtual void Observe(std::string name, std::string help, Labels labels,
                       const std::vector<double>& bounds, double value) = 0;
};

}  // namespace mori::umbp
