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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

// Fan one logical SSD tier across N drives (one sub-backend per mount point), so
// a batch is written to and read from every drive at once for their AGGREGATE
// bandwidth.
//
// Not SPDK RAID0: that stripes at the block layer and needs the whole set behind
// SPDK, uniform drives, and a device-wide rebuild to change membership.  This
// sits one level up - works over ordinary mounts, tolerates differently-sized
// drives, and keeps each key whole on one drive so a single-key read costs one
// drive's IO instead of a strip-split.  The two compose; a shard may be a RAID0.
//
// Placement is balanced, not hashed: each write goes to the shard with the most
// free space, degenerating to round-robin for uniform values on uniform drives.
// key_shard_ records the choice, so a re-put lands back on the same drive.
//
// Concurrency: key_shard_ is guarded by a shared_mutex; sub-backends are called
// WITHOUT it held, so shard IO runs fully in parallel.
class ShardedSsdTier : public TierBackend {
 public:
  // @p shards must be non-empty; every element non-null.  @p io_threads caps
  // the worker threads used by the batch paths (<= shard count); 0 means "one
  // per shard", which is what saturates N drives.
  explicit ShardedSsdTier(std::vector<std::unique_ptr<TierBackend>> shards, int io_threads = 0);
  ~ShardedSsdTier() override;

  ShardedSsdTier(const ShardedSsdTier&) = delete;
  ShardedSsdTier& operator=(const ShardedSsdTier&) = delete;

  size_t ShardCount() const { return shards_.size(); }

  // Test/observability: which shard currently holds @p key, or -1 if unknown.
  int ShardOf(const std::string& key) const;

  // Test/observability: per-shard (used, total) — the balance check.
  std::vector<std::pair<size_t, size_t>> PerShardCapacity() const;

  bool Write(const std::string& key, const void* data, size_t size) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  std::vector<char> Read(const std::string& key) override;
  TierCapabilities Capabilities() const override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;
  std::optional<std::string> GetLocationId(const std::string& key) const override;
  bool Flush() override;
  void SetColdRead(bool enable) override;

  // Parallel across shards: keys are bucketed by their (assigned or resolved)
  // shard and each bucket is handed to its own worker.  This is the whole point
  // of the class — N drives move bytes concurrently.
  std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data_ptrs,
                               const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;
  bool WriteBatch(const std::vector<std::string>& keys, const std::vector<const void*>& data_ptrs,
                  const std::vector<size_t>& sizes) override;
  std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;

 private:
  // Shard for a new key of @p size: the one with the most projected free space
  // that can still fit it, or -1 when none can.  Caller holds map_mu_ exclusively
  // and passes the projected reservation through free_hint_ (see ReserveLocked).
  int PickShardLocked(size_t size) const;

  // Resolve an existing key's shard; -1 when the key is unknown here.  Caller
  // holds at least a shared lock on map_mu_.
  int LookupLocked(const std::string& key) const;

  // Record/undo a placement decision.  used_hint_ is a placement hint only:
  // Capacity() always reports the sub-backends' own numbers, so drift costs
  // balance, never correctness.
  void ReserveLocked(int shard, size_t size);
  void ReleaseLocked(int shard, size_t size);

  // Shard assignment + the byte count needed to release the placement hint on
  // evict, so eviction never has to re-read a key just to learn its size.
  struct KeyLoc {
    uint32_t shard = 0;
    uint64_t size = 0;
  };

  std::vector<std::unique_ptr<TierBackend>> shards_;
  std::vector<size_t> shard_total_;  // immutable per-shard capacity
  int io_threads_;

  mutable std::shared_mutex map_mu_;
  std::unordered_map<std::string, KeyLoc> key_shard_;
  std::vector<size_t> used_hint_;  // guarded by map_mu_
};

}  // namespace mori::umbp
