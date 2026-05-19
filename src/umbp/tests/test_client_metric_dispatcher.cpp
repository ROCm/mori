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
#include <gtest/gtest.h>

#include <string_view>

#include "umbp.pb.h"
#include "umbp/distributed/master/client_metric_dispatcher.h"

namespace mori::umbp {
namespace {

class CountingConsumer final : public ClientMetricConsumer {
 public:
  explicit CountingConsumer(std::string_view accepted) : accepted_(accepted) {}

  bool Accept(std::string_view metric_name) const override { return metric_name == accepted_; }

  void OnSample(std::string_view node_id, const ::umbp::MetricSample& sample,
                uint64_t now_ns) override {
    ++calls;
    last_node = std::string(node_id.data(), node_id.size());
    last_name = sample.name();
    last_now_ns = now_ns;
  }

  std::string accepted_;
  int calls = 0;
  std::string last_node;
  std::string last_name;
  uint64_t last_now_ns = 0;
};

::umbp::MetricSample Sample(std::string_view name) {
  ::umbp::MetricSample sample;
  sample.set_name(std::string(name));
  sample.set_counter_delta(1.0);
  return sample;
}

TEST(ClientMetricDispatcher, DispatchesToAllMatchingConsumers) {
  ClientMetricDispatcher dispatcher;
  CountingConsumer a("metric_a");
  CountingConsumer b("metric_a");
  CountingConsumer c("metric_c");
  dispatcher.Register(&a);
  dispatcher.Register(&b);
  dispatcher.Register(&c);

  dispatcher.Dispatch("node-1", Sample("metric_a"), 123);

  EXPECT_EQ(a.calls, 1);
  EXPECT_EQ(b.calls, 1);
  EXPECT_EQ(c.calls, 0);
  EXPECT_EQ(a.last_node, "node-1");
  EXPECT_EQ(a.last_name, "metric_a");
  EXPECT_EQ(a.last_now_ns, 123u);
}

TEST(ClientMetricDispatcher, EmptyAndNullConsumersAreNoop) {
  ClientMetricDispatcher dispatcher;
  dispatcher.Register(nullptr);
  EXPECT_NO_THROW(dispatcher.Dispatch("node-1", Sample("metric_a"), 1));
}

}  // namespace
}  // namespace mori::umbp
