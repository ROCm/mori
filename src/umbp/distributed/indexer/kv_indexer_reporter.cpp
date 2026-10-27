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
#include "umbp/distributed/indexer/kv_indexer_reporter.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <chrono>
#include <stdexcept>
#include <utility>

namespace mori::umbp {
namespace pb = ::kv_indexer::v1;

KvIndexerReporter::KvIndexerReporter(const std::string& endpoint, std::string worker_id,
                                     std::string worker_address, uint32_t components,
                                     uint32_t swa_window_tokens, uint32_t timeout_ms)
    : timeout_ms_(timeout_ms), components_(components) {
  if (endpoint.empty() || worker_id.empty() || worker_address.empty() || !(components & 1) ||
      (components & ~7u) || ((components & 2) && !swa_window_tokens) || !timeout_ms)
    throw std::invalid_argument(
        "invalid KV Indexer endpoint, worker identity, cache spec or timeout");
  stub_ = pb::KVIndexer::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
  request_.set_worker_id(std::move(worker_id));
  request_.set_worker_address(std::move(worker_address));
  auto* spec = request_.mutable_cache_spec();
  spec->set_version(1);
  spec->set_components(components);
  spec->set_swa_window_tokens(swa_window_tokens);
  // Keep this worker's HBM and DRAM eligible for component-aware prefix queries.
  spec->set_full_tier_mask(6);
  spec->set_swa_tier_mask(6);
  spec->set_mamba_tier_mask(6);
}

void KvIndexerReporter::Report(const std::vector<int64_t>& hashes,
                               std::optional<int64_t> parent_hash, uint32_t block_size,
                               uint32_t component_mask) {
  if (!block_size || !(component_mask & 1) || (component_mask & ~components_))
    throw std::invalid_argument(
        "report requires token count and declared components including FULL");
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_REPORT);
  if (parent_hash) action.set_parent_block_hash(*parent_hash);
  for (auto hash : hashes) {
    action.add_hashes(hash);
    action.add_block_sizes(block_size);
    action.add_component_masks(component_mask);
  }
  Apply(std::move(action));
}

void KvIndexerReporter::Revoke(const std::vector<int64_t>& hashes) {
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_REVOKE);
  for (auto hash : hashes) action.add_hashes(hash);
  Apply(std::move(action));
}

void KvIndexerReporter::Clear() {
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_CLEAR_ALL_AT_TIER);
  Apply(std::move(action));
}

void KvIndexerReporter::Apply(pb::ExternalKvAction action) {
  if (action.hashes_size() > 16384)
    throw std::invalid_argument("KV Indexer accepts at most 16384 hashes per call");
  if (action.type() != pb::ACTION_CLEAR_ALL_AT_TIER && action.hashes().empty()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  action.set_tier(pb::TIER_DRAM);
  request_.set_seq(request_.seq() + 1);
  request_.clear_actions();
  *request_.add_actions() = std::move(action);
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms_));
  pb::ApplyExternalKvBatchResponse response;
  auto status = stub_->ApplyExternalKvBatch(&context, request_, &response);
  if (!status.ok())
    throw std::runtime_error("KV Indexer RPC " + std::to_string(status.error_code()) + ": " +
                             status.error_message());
}

}  // namespace mori::umbp
