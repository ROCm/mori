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

#include <cstdint>
#include <string_view>
#include <vector>

#include "umbp.pb.h"

namespace mori::umbp {

class ClientMetricConsumer {
 public:
  virtual ~ClientMetricConsumer() = default;

  virtual bool Accept(std::string_view metric_name) const = 0;
  virtual void OnSample(std::string_view node_id, const ::umbp::MetricSample& sample,
                        uint64_t now_ns) = 0;
};

class ClientMetricDispatcher {
 public:
  void Register(ClientMetricConsumer* consumer);
  void Dispatch(std::string_view node_id, const ::umbp::MetricSample& sample,
                uint64_t now_ns) const;

 private:
  std::vector<ClientMetricConsumer*> consumers_;
};

}  // namespace mori::umbp
