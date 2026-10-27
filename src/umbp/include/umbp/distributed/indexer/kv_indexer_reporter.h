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

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "sgl_kv_indexer.grpc.pb.h"

namespace mori::umbp {

// Explicit, synchronous DRAM reporting. The caller owns logical metadata and
// invokes these after commit/eviction/clear. No snapshots, threads, or retries.
class KvIndexerReporter {
 public:
  KvIndexerReporter(const std::string& endpoint, std::string worker_id, std::string worker_address,
                    uint32_t components = 1, uint32_t swa_window_tokens = 0,
                    uint32_t timeout_ms = 200);

  // One consecutive chain, at most 16,384 canonical int64 hashes. The parent
  // is the predecessor of hashes[0]; nullopt denotes a root. Masks replace the
  // placement, not OR it. All required physical shards must already be present.
  void Report(const std::vector<int64_t>& hashes, std::optional<int64_t> parent_hash,
              uint32_t block_size, uint32_t component_mask = 1);
  void Revoke(const std::vector<int64_t>& hashes);
  void Clear();  // This worker's DRAM only; caller must exclusively own it.

 private:
  // Throws on invalid input or RPC failure. A failed RPC may have committed;
  // the caller must reconcile from current state instead of replaying old events.
  void Apply(kv_indexer::v1::ExternalKvAction action);
  std::unique_ptr<kv_indexer::v1::KVIndexer::Stub> stub_;
  kv_indexer::v1::ApplyExternalKvBatchRequest request_;
  uint32_t timeout_ms_;
  const uint32_t components_;
  std::mutex mutex_;
};

}  // namespace mori::umbp
