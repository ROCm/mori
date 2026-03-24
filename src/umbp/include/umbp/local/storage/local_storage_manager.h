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

#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/storage/tier_backend.h"

namespace mori::umbp {

class LocalStorageManager {
 public:
  // Callback fired after a key changes tier or is fully evicted.
  // to_tier == std::nullopt means the key was evicted (no destination tier).
  // WARNING: Must NOT call back into LocalStorageManager (deadlock risk).
  using TierChangeCallback = std::function<void(const std::string& key, StorageTier from_tier,
                                                std::optional<StorageTier> to_tier)>;

  // index may be nullptr if index updates are not needed (testing).
  explicit LocalStorageManager(const UMBPConfig& config,
                               mori::umbp::LocalBlockIndex* index = nullptr);

  // Write to the specified tier.
  // When writing to DRAM and space is insufficient, automatically demotes
  // LRU keys to the next-slower tier to make room.
  bool Write(const std::string& key, const void* data, size_t size,
             StorageTier tier = StorageTier::CPU_DRAM);
  bool WriteFromPtr(const std::string& key, uintptr_t src, size_t size,
                    StorageTier tier = StorageTier::CPU_DRAM);
  // depth == -1 means no metadata (degenerates to plain LRU eviction).
  bool WriteFromPtrWithDepth(const std::string& key, uintptr_t src, size_t size, int depth,
                             StorageTier tier = StorageTier::CPU_DRAM);

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size);
  std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes);
  bool Exists(const std::string& key) const;
  bool Evict(const std::string& key);
  std::pair<size_t, size_t> Capacity(StorageTier tier) const;

  bool Demote(const std::string& key);                        // Move to next-slower tier
  bool Promote(const std::string& key);                       // Move to next-faster tier
  bool CopyToSSD(const std::string& key);                     // Non-destructive DRAM→SSD copy
  bool CopyToSSDBatch(const std::vector<std::string>& keys);  // Batched DRAM→SSD copy
  void Clear();

  // Install a callback invoked after MoveKey() or Evict() completes.
  // Must be called before any concurrent Put/Get/Evict (typically in constructor).
  void SetOnTierChange(TierChangeCallback cb);

  // Access tiers generically
  TierBackend* GetTier(StorageTier tier);
  const TierBackend* GetTier(StorageTier tier) const;

  // Typed access (returns nullptr if tier not present or wrong type)
  template <typename T>
  T* GetTierAs(StorageTier tier) {
    return dynamic_cast<T*>(GetTier(tier));
  }

 private:
  UMBPConfig config_;
  UMBPRole role_;
  mori::umbp::LocalBlockIndex* index_;  // non-owning, may be nullptr
  TierChangeCallback on_tier_change_;   // set once before concurrent access

  // Ordered fastest-to-slowest: [{CPU_DRAM, dram}, {LOCAL_SSD, ssd}, ...]
  struct TierEntry {
    StorageTier id;
    std::unique_ptr<TierBackend> backend;
  };
  std::vector<TierEntry> tiers_;

  // -----------------------------------------------------------------------
  // Depth and group metadata (protected by depth_mu_)
  //
  // LocalStorageManager has no class-level mutex; tier backends use their own
  // internal locks.  depth_map_ and group_map_ are manager-owned state accessed
  // from concurrent Put and eviction paths, so they require their own lock.
  // -----------------------------------------------------------------------
  mutable std::shared_mutex depth_mu_;
  std::unordered_map<std::string, int> depth_map_;                       // key → chain depth
  std::unordered_map<std::string, std::vector<std::string>> group_map_;  // base_hash → keys

  // depth_map_ + group_map_ helpers (all acquire depth_mu_ internally)
  void RecordDepth(const std::string& key, int depth);
  int GetDepth(const std::string& key) const;  // returns -1 if unknown
  void RemoveDepthAndGroup(const std::string& key);
  void RecordGroup(const std::string& key);
  std::vector<std::string> GetGroup(const std::string& key) const;

  // Strip _k/_v and rank suffixes to recover the SHA256 base hash.
  // Returns key unchanged if parsing fails (safe fallback).
  static std::string ExtractBaseHash(const std::string& key);

  // Select a victim from |tier| using the configured eviction policy.
  // For "prefix_aware_lru": inspect up to eviction_candidate_window candidates,
  // score by depth, pick the deepest (suffix block). Falls back to plain LRU
  // if no metadata is available.
  // Returns empty string if the tier is empty.
  std::string SelectVictim(TierBackend* tier);

  // -----------------------------------------------------------------------
  // Existing helpers
  // -----------------------------------------------------------------------
  TierBackend* FindTierHolding(const std::string& key);
  const TierBackend* FindTierHolding(const std::string& key) const;
  TierBackend* NextSlowerTier(StorageTier current);
  TierBackend* NextFasterTier(StorageTier current);
  bool MoveKey(const std::string& key, TierBackend* from, TierBackend* to);
  bool DemoteLRUForSpace(TierBackend* tier);
  bool InsertReadCacheNoWriteback(const std::string& key);
  void UpsertIndexTier(const std::string& key, StorageTier tier, size_t size_hint);

  void MaybeAutoPromote(const std::string& key);
};

}  // namespace mori::umbp
