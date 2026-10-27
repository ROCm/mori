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

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sgl_kv_indexer.grpc.pb.h"

namespace mori::umbp {
namespace indexer_pb = ::kv_indexer::v1;

// Synchronous DRAM reporting. Call after commit/eviction; metadata and recovery
// belong to the caller. One writer per worker's DRAM tier. RPC errors throw.
class KvIndexerReporter {
 public:
  KvIndexerReporter(const std::string& endpoint, const std::string& worker_id,
                    const std::string& address, uint32_t components = 1,
                    uint32_t swa_window_tokens = 0, uint32_t timeout_ms = 200)
      : timeout_(timeout_ms) {
    if (endpoint.empty() || worker_id.empty() || address.empty() || !(components & 1) ||
        (components & ~7u) || ((components & 2) && !swa_window_tokens) || !timeout_ms)
      throw std::invalid_argument("invalid KV Indexer identity, cache spec or timeout");
    stub_ = indexer_pb::KVIndexer::NewStub(
        grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
    request_.set_worker_id(worker_id);
    request_.set_worker_address(address);
    auto* spec = request_.mutable_cache_spec();
    spec->set_version(1);
    spec->set_components(components);
    spec->set_swa_window_tokens(swa_window_tokens);
    spec->set_full_tier_mask(6);  // HBM + DRAM are eligible for prefix reuse.
    spec->set_swa_tier_mask(6);
    spec->set_mamba_tier_mask(6);
  }

  // Consecutive canonical hashes; nullopt parent means root. Size is in tokens.
  void Report(const std::vector<int64_t>& hashes, std::optional<int64_t> parent_hash,
              uint32_t block_size, uint32_t component_mask = 1) {
    Apply(indexer_pb::ACTION_REPORT, hashes, parent_hash, block_size, component_mask);
  }
  void Revoke(const std::vector<int64_t>& hashes) { Apply(indexer_pb::ACTION_REVOKE, hashes); }
  void Clear() { Apply(indexer_pb::ACTION_CLEAR_ALL_AT_TIER, {}); }

 private:
  void Apply(indexer_pb::ExternalKvActionType type, const std::vector<int64_t>& hashes,
             std::optional<int64_t> parent = {}, uint32_t size = 0, uint32_t mask = 0) {
    if (hashes.size() > 16384) throw std::invalid_argument("at most 16384 hashes per call");
    std::lock_guard<std::mutex> lock(mutex_);
    if (type == indexer_pb::ACTION_REPORT &&
        (!size || !(mask & 1) || (mask & ~request_.cache_spec().components())))
      throw std::invalid_argument(
          "report requires token count and declared components including FULL");
    if (hashes.empty() && type != indexer_pb::ACTION_CLEAR_ALL_AT_TIER) return;
    request_.set_seq(request_.seq() + 1);
    request_.clear_actions();
    auto* action = request_.add_actions();
    action->set_type(type);
    action->set_tier(indexer_pb::TIER_DRAM);
    for (auto hash : hashes) action->add_hashes(hash);
    if (type == indexer_pb::ACTION_REPORT) {
      if (parent) action->set_parent_block_hash(*parent);
      action->mutable_block_sizes()->Resize(hashes.size(), size);
      action->mutable_component_masks()->Resize(hashes.size(), mask);
    }
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + timeout_);
    indexer_pb::ApplyExternalKvBatchResponse response;
    auto status = stub_->ApplyExternalKvBatch(&context, request_, &response);
    if (!status.ok())
      throw std::runtime_error("KV Indexer RPC " + std::to_string(status.error_code()) + ": " +
                               status.error_message());
  }
  std::unique_ptr<indexer_pb::KVIndexer::Stub> stub_;
  indexer_pb::ApplyExternalKvBatchRequest request_;
  std::chrono::milliseconds timeout_;
  std::mutex mutex_;
};
}  // namespace mori::umbp
