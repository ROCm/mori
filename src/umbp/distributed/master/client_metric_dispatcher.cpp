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
#include "umbp/distributed/master/client_metric_dispatcher.h"

namespace mori::umbp {

void ClientMetricDispatcher::Register(ClientMetricConsumer* consumer) {
  if (consumer == nullptr) return;
  consumers_.push_back(consumer);
}

void ClientMetricDispatcher::Dispatch(std::string_view node_id, const ::umbp::MetricSample& sample,
                                      uint64_t now_ns) const {
  const std::string_view metric_name(sample.name().data(), sample.name().size());
  for (ClientMetricConsumer* consumer : consumers_) {
    if (consumer != nullptr && consumer->Accept(metric_name)) {
      consumer->OnSample(node_id, sample, now_ns);
    }
  }
}

}  // namespace mori::umbp
