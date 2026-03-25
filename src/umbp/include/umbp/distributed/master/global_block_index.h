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

#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/common/types.h"

namespace mori::umbp {

class ClientRegistry;

struct BlockEntry {
  std::vector<Location> locations;
  BlockMetrics metrics;
};

class GlobalBlockIndex {
 public:
  GlobalBlockIndex() = default;
  ~GlobalBlockIndex() = default;

  GlobalBlockIndex(const GlobalBlockIndex&) = delete;
  GlobalBlockIndex& operator=(const GlobalBlockIndex&) = delete;

  void SetClientRegistry(ClientRegistry* registry);

  // --- Mutators ---
  void Register(const std::string& node_id, const std::string& key, const Location& location);

  bool Unregister(const std::string& node_id, const std::string& key, const Location& location);

  size_t UnregisterByNode(const std::string& key, const std::string& node_id);

  // Batch variants — single lock acquisition for the entire batch.
  size_t BatchRegister(const std::string& node_id,
                       const std::vector<std::pair<std::string, Location>>& entries);
  size_t BatchUnregister(const std::string& node_id,
                         const std::vector<std::pair<std::string, Location>>& entries);

  // Bump last_accessed_at and access_count. Called by Router on RouteGet.
  void RecordAccess(const std::string& key);

  // --- Queries ---
  std::vector<Location> Lookup(const std::string& key) const;

  // Returns metrics for a key, or nullopt if the key doesn't exist.
  std::optional<BlockMetrics> GetMetrics(const std::string& key) const;

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, BlockEntry> entries_;
  ClientRegistry* registry_ = nullptr;
};

}  // namespace mori::umbp
