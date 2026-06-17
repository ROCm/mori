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

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mori::umbp {

// Per-hash cumulative hit counter for external KV placement matches.
// Entries are created only for hashes that were actually matched by
// MatchExternalKv(count_as_hit=true); revoke does not remove them.
class ExternalKvHitIndex {
 public:
  static constexpr size_t kShards = 256;

  ExternalKvHitIndex() = default;
  ~ExternalKvHitIndex() = default;

  ExternalKvHitIndex(const ExternalKvHitIndex&) = delete;
  ExternalKvHitIndex& operator=(const ExternalKvHitIndex&) = delete;

  // Caller owns request-level de-duplication. Each input hash is incremented
  // exactly once by this call.
  void IncrementHits(const std::vector<std::string>& unique_hashes, uint64_t now_ns);

  // Sparse lookup. Missing hashes are skipped, and duplicate query hashes
  // produce at most one output entry.
  size_t Lookup(const std::vector<std::string>& hashes,
                std::vector<std::pair<std::string, uint64_t>>* out) const;

  // Drop entries whose last activity is older than cutoff_ns.
  size_t GarbageCollect(uint64_t cutoff_ns);

  size_t Size() const;

 private:
  struct Entry {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> last_seen_ns{0};
  };

  struct Shard {
    mutable std::shared_mutex mu;
    std::unordered_map<std::string, std::unique_ptr<Entry>> entries;
  };

  static size_t ShardIdx(std::string_view hash);
  static void UpdateLastSeen(Entry* entry, uint64_t now_ns);

  std::array<Shard, kShards> shards_;
};

}  // namespace mori::umbp
