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
#include "umbp/distributed/master/external_kv_block_index.h"

#include <mutex>
#include <set>
#include <shared_mutex>
#include <unordered_map>

namespace mori::umbp {

namespace {

TierType BestTier(const std::set<TierType>& tiers) {
  if (tiers.count(TierType::HBM)) return TierType::HBM;
  if (tiers.count(TierType::DRAM)) return TierType::DRAM;
  if (tiers.count(TierType::SSD)) return TierType::SSD;
  return TierType::UNKNOWN;
}

}  // namespace

void ExternalKvBlockIndex::Register(const std::string& node_id,
                                    const std::vector<std::string>& hashes, TierType tier) {
  std::unique_lock lock(mutex_);
  for (const auto& hash : hashes) {
    entries_[hash][node_id].insert(tier);
  }
}

void ExternalKvBlockIndex::Unregister(const std::string& node_id,
                                      const std::vector<std::string>& hashes, TierType tier) {
  std::unique_lock lock(mutex_);
  for (const auto& hash : hashes) {
    auto it = entries_.find(hash);
    if (it == entries_.end()) {
      continue;
    }
    auto node_it = it->second.find(node_id);
    if (node_it == it->second.end()) {
      continue;
    }
    node_it->second.erase(tier);
    if (node_it->second.empty()) {
      it->second.erase(node_it);
    }
    if (it->second.empty()) {
      entries_.erase(it);
    }
  }
}

void ExternalKvBlockIndex::UnregisterByNode(const std::string& node_id) {
  std::unique_lock lock(mutex_);
  auto it = entries_.begin();
  while (it != entries_.end()) {
    it->second.erase(node_id);
    if (it->second.empty()) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<ExternalKvBlockIndex::NodeMatch> ExternalKvBlockIndex::Match(
    const std::vector<std::string>& hashes) const {
  std::shared_lock lock(mutex_);

  std::unordered_map<std::string, NodeMatch> acc;

  for (const auto& hash : hashes) {
    auto it = entries_.find(hash);
    if (it == entries_.end()) {
      continue;
    }
    for (const auto& [node_id, tiers] : it->second) {
      auto& entry = acc[node_id];
      entry.node_id = node_id;
      entry.matched_hashes.push_back(hash);
      entry.tiers.insert(tiers.begin(), tiers.end());
    }
  }

  std::vector<NodeMatch> result;
  result.reserve(acc.size());
  for (auto& [_, match] : acc) {
    match.tier = BestTier(match.tiers);
    result.push_back(std::move(match));
  }
  return result;
}

size_t ExternalKvBlockIndex::GetKvCount(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  size_t count = 0;
  for (const auto& [hash, nodes] : entries_) {
    if (nodes.count(node_id)) {
      ++count;
    }
  }
  return count;
}

}  // namespace mori::umbp
