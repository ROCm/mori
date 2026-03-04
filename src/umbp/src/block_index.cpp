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
#include "umbp/block_index.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include "umbp/client_registry.h"

namespace mori::umbp {

void BlockIndex::SetClientRegistry(ClientRegistry* registry) {
  std::unique_lock lock(mutex_);
  registry_ = registry;
}

void BlockIndex::Register(const std::string& node_id, const std::string& key,
                          const Location& location) {
  (void)BatchRegister(node_id, {{key, location}});
}

bool BlockIndex::Unregister(const std::string& node_id, const std::string& key,
                            const Location& location) {
  return BatchUnregister(node_id, {{key, location}}) > 0;
}

size_t BlockIndex::UnregisterByNode(const std::string& key, const std::string& node_id) {
  size_t removed = 0;
  bool should_untrack = false;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }

    auto& locs = it->second.locations;
    const size_t original_size = locs.size();
    locs.erase(std::remove_if(locs.begin(), locs.end(),
                              [&node_id](const Location& loc) { return loc.node_id == node_id; }),
               locs.end());
    removed = original_size - locs.size();
    should_untrack =
        removed > 0 && std::none_of(locs.begin(), locs.end(), [&node_id](const Location& loc) {
          return loc.node_id == node_id;
        });

    if (locs.empty()) {
      entries_.erase(it);
    }
  }

  if (removed > 0 && should_untrack && registry != nullptr) {
    registry->UntrackKey(node_id, key);
  }

  return removed;
}

size_t BlockIndex::BatchRegister(const std::string& node_id,
                                 const std::vector<std::pair<std::string, Location>>& entries) {
  if (entries.empty()) {
    return 0;
  }

  size_t inserted = 0;
  std::vector<std::string> keys_to_track;
  std::unordered_set<std::string> keys_seen;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    for (const auto& [key, location] : entries) {
      auto& entry = entries_[key];
      const auto now = std::chrono::steady_clock::now();
      if (entry.locations.empty()) {
        entry.metrics.created_at = now;
        entry.metrics.last_accessed_at = now;
        entry.metrics.access_count = 0;
      }

      auto it = std::find(entry.locations.begin(), entry.locations.end(), location);
      if (it != entry.locations.end()) {
        continue;
      }

      entry.locations.push_back(location);
      entry.metrics.last_accessed_at = now;
      ++entry.metrics.access_count;
      ++inserted;

      if (keys_seen.insert(key).second) {
        keys_to_track.push_back(key);
      }
    }
  }

  if (registry != nullptr) {
    for (const auto& key : keys_to_track) {
      registry->TrackKey(node_id, key);
    }
  }

  return inserted;
}

size_t BlockIndex::BatchUnregister(const std::string& node_id,
                                   const std::vector<std::pair<std::string, Location>>& entries) {
  if (entries.empty()) {
    return 0;
  }

  size_t removed_count = 0;
  std::vector<std::string> keys_to_untrack;
  std::unordered_set<std::string> keys_seen;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    for (const auto& [key, location] : entries) {
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        continue;
      }

      auto& locs = it->second.locations;
      const size_t original_size = locs.size();
      locs.erase(std::remove(locs.begin(), locs.end(), location), locs.end());

      if (locs.size() == original_size) {
        continue;
      }

      ++removed_count;
      const bool has_remaining_for_client =
          std::any_of(locs.begin(), locs.end(),
                      [&node_id](const Location& loc) { return loc.node_id == node_id; });
      if (!has_remaining_for_client && keys_seen.insert(key).second) {
        keys_to_untrack.push_back(key);
      }

      if (locs.empty()) {
        entries_.erase(it);
      }
    }
  }

  if (registry != nullptr) {
    for (const auto& key : keys_to_untrack) {
      registry->UntrackKey(node_id, key);
    }
  }

  return removed_count;
}

void BlockIndex::RecordAccess(const std::string& key) {
  std::unique_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }

  it->second.metrics.last_accessed_at = std::chrono::steady_clock::now();
  ++it->second.metrics.access_count;
}

std::vector<Location> BlockIndex::Lookup(const std::string& key) const {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return {};
  }

  return it->second.locations;
}

std::optional<BlockMetrics> BlockIndex::GetMetrics(const std::string& key) const {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }

  return it->second.metrics;
}

}  // namespace mori::umbp
