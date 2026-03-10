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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "umbp/block_index/block_index.h"
#include "umbp/common/config.h"
#include "umbp/storage/tier_backend.h"

class LocalStorageManager {
 public:
  // index may be nullptr if index updates are not needed (testing).
  explicit LocalStorageManager(const UMBPConfig& config, BlockIndexClient* index = nullptr);

  // Write to the specified tier.
  // When writing to DRAM and space is insufficient, automatically demotes
  // LRU keys to the next-slower tier to make room.
  bool Write(const std::string& key, const void* data, size_t size,
             StorageTier tier = StorageTier::CPU_DRAM);
  bool WriteFromPtr(const std::string& key, uintptr_t src, size_t size,
                    StorageTier tier = StorageTier::CPU_DRAM);

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size);
  bool Exists(const std::string& key) const;
  bool Evict(const std::string& key);
  std::pair<size_t, size_t> Capacity(StorageTier tier) const;

  bool Demote(const std::string& key);     // Move to next-slower tier
  bool Promote(const std::string& key);    // Move to next-faster tier
  bool CopyToSSD(const std::string& key);  // Non-destructive DRAM→SSD copy
  void Clear();

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
  BlockIndexClient* index_;  // non-owning, may be nullptr

  // Ordered fastest-to-slowest: [{CPU_DRAM, dram}, {LOCAL_SSD, ssd}, ...]
  struct TierEntry {
    StorageTier id;
    std::unique_ptr<TierBackend> backend;
  };
  std::vector<TierEntry> tiers_;

  // Helpers
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
