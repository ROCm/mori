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

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

/// Lightweight index for externally-managed KV blocks (e.g. sglang L1/L2
/// cache).  Unlike GlobalBlockIndex, this index has no Location, no
/// BlockMetrics, and no ClientRegistry back-pointer.  It maps each hash to the
/// set of nodes that hold the block and the storage tier they reported.
class ExternalKvBlockIndex {
 public:
  ExternalKvBlockIndex() = default;
  ~ExternalKvBlockIndex() = default;

  ExternalKvBlockIndex(const ExternalKvBlockIndex&) = delete;
  ExternalKvBlockIndex& operator=(const ExternalKvBlockIndex&) = delete;

  // Register (or overwrite) a batch of hashes for node_id at the given tier.
  void Register(const std::string& node_id, const std::vector<std::string>& hashes, TierType tier);

  // Remove specific hashes for a node.
  void Unregister(const std::string& node_id, const std::vector<std::string>& hashes);

  // Remove all hashes for a node (bulk, called on node expiry/unregister).
  void UnregisterByNode(const std::string& node_id);

  struct NodeMatch {
    std::string node_id;
    std::vector<std::string> matched_hashes;
    TierType tier = TierType::UNKNOWN;
  };

  // Return per-node matches across all queried hashes.
  std::vector<NodeMatch> Match(const std::vector<std::string>& hashes) const;

 private:
  // hash -> (node_id -> tier)
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unordered_map<std::string, TierType>> entries_;
};

}  // namespace mori::umbp
