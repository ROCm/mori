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
#pragma once

#include <memory>
#include <string>

#include "umbp/common/config.h"

namespace mori::umbp {
class PrometheusMetricSink;
}

namespace mori::umbp::standalone {

// Port the standalone server serves Prometheus metrics on when nothing else
// says otherwise.  Deliberately not the master's 9091: a node may run a master
// and a standalone server, and two processes silently contending for one port
// is a worse failure than a second number to remember.
inline constexpr int kDefaultStandaloneMetricsPort = 9092;

// Resolve the metrics port from UMBP_STANDALONE_METRICS_PORT.
//   unset              -> kDefaultStandaloneMetricsPort
//   0 or "off"/"false" -> 0, meaning do not serve metrics
//   1..65535           -> that port
// Any other value is rejected: `*error` is set and 0 is NOT returned, because
// "the operator asked for something impossible" and "the operator asked for
// silence" must not look the same to the caller.
int ResolveMetricsPortFromEnv(bool* ok, std::string* error);

class StandaloneServer {
 public:
  StandaloneServer(UMBPConfig config, std::string address);
  ~StandaloneServer();

  bool Start();
  void Run();
  void Shutdown();

  const std::string& address() const { return address_; }

  // The port /metrics is served on, or 0 when this server publishes no metrics
  // (explicitly disabled, a master is configured, or the bind failed).
  int metrics_port() const;

 private:
  class Impl;

  UMBPConfig config_;
  std::string address_;
  // Declared BEFORE impl_ so it outlives it: the PoolClient inside impl_ holds
  // a borrowed pointer to this sink and publishes into it until its Shutdown.
  // Destruction runs in reverse declaration order, so impl_ dies first.
  std::unique_ptr<PrometheusMetricSink> metric_sink_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mori::umbp::standalone
