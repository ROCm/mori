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

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/prefetch.h"
#include "umbp_standalone.grpc.pb.h"

namespace mori::umbp {

// Retain the Router's wire paths without exposing the standalone data/admin RPCs.
// Callbacks share the owner's queue; they and this service must outlive its server.
class PrefetchService final : public ::umbp::UMBPStandalone::Service {
 public:
  using Submit = std::function<PrefetchSubmitResult(const std::vector<std::string>&,
                                                    PrefetchOptions, std::string)>;
  using Lookup = std::function<PrefetchStatus(const std::string&)>;

  PrefetchService(Submit submit, Lookup lookup)
      : submit_(std::move(submit)), lookup_(std::move(lookup)) {}

  grpc::Status SubmitPrefetch(grpc::ServerContext*, const ::umbp::PrefetchRequest* request,
                              ::umbp::PrefetchSubmitResponse* response) override {
    if (request->lease_ttl_ms() > 300000 || request->timeout_ms() > 300000) {
      response->set_error("prefetch timeout or lease exceeds the accepted bound");
      return grpc::Status::OK;
    }
    PrefetchOptions options;
    options.lease_ttl = std::chrono::milliseconds(request->lease_ttl_ms());
    options.timeout = std::chrono::milliseconds(request->timeout_ms());
    const auto result =
        submit_({request->keys().begin(), request->keys().end()}, options, request->request_id());
    response->set_accepted(result.accepted);
    response->set_request_id(result.request_id);
    response->set_error(result.error);
    return grpc::Status::OK;
  }

  grpc::Status GetPrefetchStatus(grpc::ServerContext*, const ::umbp::PrefetchStatusRequest* request,
                                 ::umbp::PrefetchStatusResponse* response) override {
    const auto status = lookup_(request->request_id());
    response->set_state(static_cast<::umbp::PrefetchState>(status.state));
    response->set_completed_keys(status.completed_keys);
    response->set_total_keys(status.ok.size());
    response->set_error(status.error);
    response->set_queue_latency_us(status.queue_latency_us);
    response->set_execution_latency_us(status.execution_latency_us);
    response->set_total_latency_us(status.total_latency_us);
    for (bool ok : status.ok) response->add_ok(ok);
    return grpc::Status::OK;
  }

 private:
  Submit submit_;
  Lookup lookup_;
};

}  // namespace mori::umbp
