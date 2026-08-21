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
#include "umbp/distributed/metrics/component_metrics.h"

namespace mori::umbp {
namespace {

// Baseline key = source + metric + label set.  '\0' separates the parts so no
// combination of label text can forge another key's identity.
std::string BaselineKey(const std::string& source_id, const char* name,
                        const MetricLabels& source_labels, const MetricLabels& sample_labels) {
  std::string id = source_id;
  id += '\0';
  id += name;
  auto append = [&id](const MetricLabels& labels) {
    for (const auto& [k, v] : labels) {
      id += '\0';
      id += k;
      id += '=';
      id += v;
    }
  };
  append(source_labels);
  append(sample_labels);
  return id;
}

MetricLabels MergeLabels(const MetricLabels& source_labels, const MetricLabels& sample_labels) {
  MetricLabels merged;
  merged.reserve(source_labels.size() + sample_labels.size());
  merged.insert(merged.end(), source_labels.begin(), source_labels.end());
  merged.insert(merged.end(), sample_labels.begin(), sample_labels.end());
  return merged;
}

}  // namespace

void MetricPublisher::Publish(const std::string& source_id, const MetricLabels& source_labels,
                              const MetricSource& source, const Sink& sink) {
  Publish(source_id, source_labels, source.SampleMetrics(), sink);
}

void MetricPublisher::Publish(const std::string& source_id, const MetricLabels& source_labels,
                              const std::vector<MetricSample>& samples, const Sink& sink) {
  for (const MetricSample& s : samples) {
    if (s.name == nullptr) continue;

    if (s.kind == MetricKind::kGauge) {
      // No baseline: a gauge is whatever it reads now.  Shipped every tick so a
      // value that stops changing still keeps its series alive.
      if (sink.gauge) {
        sink.gauge(s.name, s.help, MergeLabels(source_labels, s.labels),
                   static_cast<double>(s.value) * s.scale);
      }
      continue;
    }

    uint64_t& last = last_[BaselineKey(source_id, s.name, source_labels, s.labels)];
    if (s.value > last && sink.counter) {
      sink.counter(s.name, s.help, MergeLabels(source_labels, s.labels),
                   static_cast<double>(s.value - last) * s.scale);
    }
    // Updated even on a zero or negative delta so the next tick stays correct
    // against a counter that was rebuilt underneath us.
    last = s.value;
  }
}

}  // namespace mori::umbp
