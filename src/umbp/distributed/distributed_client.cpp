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
#include "umbp/distributed/distributed_client.h"

#include <stdexcept>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"

namespace mori::umbp {

DistributedClient::DistributedClient(const UMBPConfig& config) : config_(config) {
  if (!config.distributed.has_value()) {
    throw std::runtime_error("DistributedClient requires UMBPConfig::distributed to be set");
  }

  const auto& dc = config.distributed.value();
  MasterClientConfig mc_config;
  mc_config.master_address = dc.master_address;
  mc_config.node_id = dc.node_id;
  mc_config.node_address = dc.node_address;
  mc_config.auto_heartbeat = dc.auto_heartbeat;
  master_client_ = std::make_unique<MasterClient>(mc_config);

  // TODO: init IOEngine from dc.io_engine_host / dc.io_engine_port
}

DistributedClient::~DistributedClient() { Close(); }

// ---------------------------------------------------------------------------
// Core KV Operations — stub implementations (master-led flow TODO)
// ---------------------------------------------------------------------------

bool DistributedClient::Put(const std::string& /*key*/, uintptr_t /*src*/, size_t /*size*/) {
  // TODO: RoutePut -> MORI-IO write -> Register
  return false;
}

bool DistributedClient::Get(const std::string& /*key*/, uintptr_t /*dst*/, size_t /*size*/) {
  // TODO: RouteGet -> RDMA/MORI-IO read
  return false;
}

bool DistributedClient::Exists(const std::string& /*key*/) const {
  // TODO: MasterClient::Lookup (read-only, no access count side-effects)
  return false;
}

// ---------------------------------------------------------------------------
// Batch Operations
// ---------------------------------------------------------------------------

std::vector<bool> DistributedClient::BatchPut(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& srcs,
                                              const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Put(keys[i], srcs[i], sizes[i]);
  }
  return results;
}

std::vector<bool> DistributedClient::BatchPutWithDepth(const std::vector<std::string>& keys,
                                                       const std::vector<uintptr_t>& srcs,
                                                       const std::vector<size_t>& sizes,
                                                       const std::vector<int>& /*depths*/) {
  // TODO: forward depths to Master for global eviction decisions
  return BatchPut(keys, srcs, sizes);
}

std::vector<bool> DistributedClient::BatchGet(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& dsts,
                                              const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Get(keys[i], dsts[i], sizes[i]);
  }
  return results;
}

std::vector<bool> DistributedClient::BatchExists(const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Exists(keys[i]);
  }
  return results;
}

size_t DistributedClient::BatchExistsConsecutive(const std::vector<std::string>& keys) const {
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!Exists(keys[i])) return i;
  }
  return keys.size();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void DistributedClient::Clear() {
  // TODO: clear all entries (local and/or global)
}

bool DistributedClient::Flush() {
  // TODO: flush pending operations
  return true;
}

void DistributedClient::Close() {
  if (closed_) return;
  closed_ = true;

  if (master_client_) {
    master_client_->StopHeartbeat();
    if (master_client_->IsRegistered()) {
      master_client_->UnregisterSelf();
    }
  }
  io_engine_.reset();
  master_client_.reset();
}

bool DistributedClient::IsDistributed() const { return true; }

bool DistributedClient::ReportExternalKvBlocks(const std::vector<std::string>& hashes,
                                               TierType tier) {
  if (!master_client_) return false;
  const std::string& node_id = config_.distributed->node_id;
  auto status = master_client_->ReportExternalKvBlocks(node_id, hashes, tier);
  return status.ok();
}

bool DistributedClient::RevokeExternalKvBlocks(const std::vector<std::string>& hashes) {
  if (!master_client_) return false;
  const std::string& node_id = config_.distributed->node_id;
  auto status = master_client_->RevokeExternalKvBlocks(node_id, hashes);
  return status.ok();
}

std::vector<IUMBPClient::ExternalKvMatch> DistributedClient::MatchExternalKv(
    const std::vector<std::string>& hashes) {
  if (!master_client_) return {};

  std::vector<MasterClient::ExternalKvNodeMatch> raw;
  auto status = master_client_->MatchExternalKv(hashes, &raw);
  if (!status.ok()) return {};

  std::vector<IUMBPClient::ExternalKvMatch> result;
  result.reserve(raw.size());
  for (auto& r : raw) {
    IUMBPClient::ExternalKvMatch m;
    m.node_id = std::move(r.node_id);
    m.peer_address = std::move(r.peer_address);
    m.matched_hashes = std::move(r.matched_hashes);
    m.tier = r.tier;
    result.push_back(std::move(m));
  }
  return result;
}

}  // namespace mori::umbp
