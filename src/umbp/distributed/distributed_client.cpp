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

namespace mori::umbp {

DistributedClient::DistributedClient(const UMBPConfig& config) : config_(config) {
  if (!config.distributed.has_value()) {
    throw std::runtime_error("DistributedClient requires UMBPConfig::distributed to be set");
  }
  // TODO: init MasterClient, IOEngine, PeerServiceServer
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
  // TODO: Lookup via Master's GlobalBlockIndex
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
  // TODO: shutdown MasterClient heartbeat, IOEngine, PeerServiceServer
}

bool DistributedClient::IsDistributed() const { return true; }

}  // namespace mori::umbp
