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
#include "umbp/local/tiers/sharded_ssd_tier.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Run fn(shard_bucket_index) for every non-empty bucket, one worker per bucket
// up to `max_threads`, executing the last bucket inline on the calling thread.
// Buckets touch disjoint shards and disjoint result indices, so no locking is
// needed inside fn.
template <typename Fn>
void RunPerShard(const std::vector<int>& busy_shards, int max_threads, Fn&& fn) {
  if (busy_shards.empty()) return;
  if (busy_shards.size() == 1 || max_threads <= 1) {
    for (int s : busy_shards) fn(s);
    return;
  }
  const size_t spawn =
      std::min<size_t>(busy_shards.size(), static_cast<size_t>(std::max(max_threads, 1))) - 1;
  std::vector<std::thread> workers;
  workers.reserve(spawn);
  for (size_t i = 0; i < spawn; ++i) {
    workers.emplace_back([&fn, s = busy_shards[i]]() { fn(s); });
  }
  // Remaining buckets (including everything past the thread cap) run here.
  for (size_t i = spawn; i < busy_shards.size(); ++i) fn(busy_shards[i]);
  for (auto& t : workers) t.join();
}

}  // namespace

ShardedSsdTier::ShardedSsdTier(std::vector<std::unique_ptr<TierBackend>> shards, int io_threads)
    : TierBackend(StorageTier::LOCAL_SSD), shards_(std::move(shards)) {
  if (shards_.empty()) {
    throw std::runtime_error("[ShardedSsdTier] requires at least one shard");
  }
  for (const auto& s : shards_) {
    if (!s) throw std::runtime_error("[ShardedSsdTier] null shard backend");
  }
  shard_total_.reserve(shards_.size());
  used_hint_.assign(shards_.size(), 0);
  for (size_t i = 0; i < shards_.size(); ++i) {
    auto cap = shards_[i]->Capacity();
    used_hint_[i] = cap.first;
    shard_total_.push_back(cap.second);
  }
  io_threads_ = io_threads > 0 ? io_threads : static_cast<int>(shards_.size());
  MORI_UMBP_INFO("[ShardedSsdTier] {} shard(s), io_threads={}", shards_.size(), io_threads_);
}

ShardedSsdTier::~ShardedSsdTier() = default;

int ShardedSsdTier::LookupLocked(const std::string& key) const {
  auto it = key_shard_.find(key);
  return it == key_shard_.end() ? -1 : static_cast<int>(it->second.shard);
}

int ShardedSsdTier::PickShardLocked(size_t size) const {
  int best = -1;
  size_t best_free = 0;
  for (size_t i = 0; i < shards_.size(); ++i) {
    const size_t used = used_hint_[i];
    if (used >= shard_total_[i]) continue;
    const size_t free_bytes = shard_total_[i] - used;
    if (free_bytes < size) continue;
    if (best < 0 || free_bytes > best_free) {
      best = static_cast<int>(i);
      best_free = free_bytes;
    }
  }
  return best;
}

void ShardedSsdTier::ReserveLocked(int shard, size_t size) { used_hint_[shard] += size; }

void ShardedSsdTier::ReleaseLocked(int shard, size_t size) {
  used_hint_[shard] = used_hint_[shard] > size ? used_hint_[shard] - size : 0;
}

int ShardedSsdTier::ShardOf(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(map_mu_);
  return LookupLocked(key);
}

std::vector<std::pair<size_t, size_t>> ShardedSsdTier::PerShardCapacity() const {
  std::vector<std::pair<size_t, size_t>> out;
  out.reserve(shards_.size());
  for (const auto& s : shards_) out.push_back(s->Capacity());
  return out;
}

bool ShardedSsdTier::Write(const std::string& key, const void* data, size_t size) {
  int shard;
  bool is_new;
  {
    std::unique_lock<std::shared_mutex> lock(map_mu_);
    shard = LookupLocked(key);
    is_new = shard < 0;
    if (is_new) {
      shard = PickShardLocked(size);
      if (shard < 0) {
        MORI_UMBP_WARN("[ShardedSsdTier] Write key={} size={}: no shard has room", key, size);
        return false;
      }
      // Reserve before dropping the lock so concurrent writers see the fill and
      // spread out instead of all piling onto the same "most free" shard.
      key_shard_[key] = KeyLoc{static_cast<uint32_t>(shard), size};
      ReserveLocked(shard, size);
    }
  }

  const bool ok = shards_[shard]->Write(key, data, size);
  if (!ok && is_new) {
    std::unique_lock<std::shared_mutex> lock(map_mu_);
    key_shard_.erase(key);
    ReleaseLocked(shard, size);
  }
  return ok;
}

bool ShardedSsdTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  int shard;
  {
    std::shared_lock<std::shared_mutex> lock(map_mu_);
    shard = LookupLocked(key);
  }
  if (shard < 0) return false;
  return shards_[shard]->ReadIntoPtr(key, dst_ptr, size);
}

bool ShardedSsdTier::Exists(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(map_mu_);
  return LookupLocked(key) >= 0;
}

bool ShardedSsdTier::Evict(const std::string& key) {
  int shard;
  size_t freed = 0;
  {
    std::shared_lock<std::shared_mutex> lock(map_mu_);
    auto it = key_shard_.find(key);
    if (it == key_shard_.end()) return false;
    shard = static_cast<int>(it->second.shard);
    freed = it->second.size;
  }
  if (!shards_[shard]->Evict(key)) return false;
  std::unique_lock<std::shared_mutex> lock(map_mu_);
  key_shard_.erase(key);
  ReleaseLocked(shard, freed);
  return true;
}

std::pair<size_t, size_t> ShardedSsdTier::Capacity() const {
  size_t used = 0, total = 0;
  for (const auto& s : shards_) {
    auto cap = s->Capacity();
    used += cap.first;
    total += cap.second;
  }
  return {used, total};
}

void ShardedSsdTier::Clear() {
  for (auto& s : shards_) s->Clear();
  std::unique_lock<std::shared_mutex> lock(map_mu_);
  key_shard_.clear();
  std::fill(used_hint_.begin(), used_hint_.end(), 0);
}

std::vector<char> ShardedSsdTier::Read(const std::string& key) {
  int shard;
  {
    std::shared_lock<std::shared_mutex> lock(map_mu_);
    shard = LookupLocked(key);
  }
  if (shard < 0) return {};
  return shards_[shard]->Read(key);
}

TierCapabilities ShardedSsdTier::Capabilities() const {
  // Batch capability is this class's own (it parallelizes across shards) even
  // when a sub-backend only offers the serial default.  Zero-copy read is only
  // advertised when every shard can do it.
  TierCapabilities caps;
  caps.batch_write = true;
  caps.batch_read = true;
  caps.zero_copy_read = true;
  for (const auto& s : shards_) {
    if (!s->Capabilities().zero_copy_read) caps.zero_copy_read = false;
  }
  return caps;
}

std::string ShardedSsdTier::GetLRUKey() const {
  // Shards keep independent recency lists, so there is no globally exact LRU
  // here.  Take the oldest key of the fullest shard: that is the shard eviction
  // must relieve anyway.  PeerSsdManager drives distributed eviction from its
  // own global LRU and does not call this.
  int fullest = -1;
  size_t most_used = 0;
  for (size_t i = 0; i < shards_.size(); ++i) {
    auto cap = shards_[i]->Capacity();
    if (fullest < 0 || cap.first > most_used) {
      fullest = static_cast<int>(i);
      most_used = cap.first;
    }
  }
  return fullest < 0 ? std::string() : shards_[fullest]->GetLRUKey();
}

std::vector<std::string> ShardedSsdTier::GetLRUCandidates(size_t max_candidates) const {
  // Round-robin merge of the per-shard tails, so victims are spread across
  // drives instead of emptying one.
  std::vector<std::vector<std::string>> per_shard;
  per_shard.reserve(shards_.size());
  for (const auto& s : shards_) per_shard.push_back(s->GetLRUCandidates(max_candidates));

  std::vector<std::string> out;
  out.reserve(max_candidates);
  for (size_t round = 0; out.size() < max_candidates; ++round) {
    bool progressed = false;
    for (auto& cands : per_shard) {
      if (round >= cands.size()) continue;
      out.push_back(cands[round]);
      progressed = true;
      if (out.size() >= max_candidates) break;
    }
    if (!progressed) break;
  }
  return out;
}

std::optional<std::string> ShardedSsdTier::GetLocationId(const std::string& key) const {
  int shard;
  {
    std::shared_lock<std::shared_mutex> lock(map_mu_);
    shard = LookupLocked(key);
  }
  if (shard < 0) return std::nullopt;
  auto inner = shards_[shard]->GetLocationId(key);
  if (!inner) return std::nullopt;
  // Prefix the drive so a location string identifies which device holds the key.
  return "s" + std::to_string(shard) + ":" + *inner;
}

bool ShardedSsdTier::Flush() {
  bool ok = true;
  for (auto& s : shards_) ok = s->Flush() && ok;
  return ok;
}

void ShardedSsdTier::SetColdRead(bool enable) {
  for (auto& s : shards_) s->SetColdRead(enable);
}

std::vector<bool> ShardedSsdTier::BatchWrite(const std::vector<std::string>& keys,
                                             const std::vector<const void*>& data_ptrs,
                                             const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;
  if (data_ptrs.size() != keys.size() || sizes.size() != keys.size()) {
    MORI_UMBP_ERROR("[ShardedSsdTier] BatchWrite: mismatched input lengths ({} / {} / {})",
                    keys.size(), data_ptrs.size(), sizes.size());
    return results;
  }

  // Assign every key to a shard under one lock, applying the projected
  // reservation as we go so a batch of same-size values spreads evenly.
  std::vector<std::vector<size_t>> buckets(shards_.size());
  std::vector<char> is_new(keys.size(), 0);
  {
    std::unique_lock<std::shared_mutex> lock(map_mu_);
    for (size_t i = 0; i < keys.size(); ++i) {
      int shard = LookupLocked(keys[i]);
      if (shard < 0) {
        shard = PickShardLocked(sizes[i]);
        if (shard < 0) continue;  // leaves results[i] == false
        key_shard_[keys[i]] = KeyLoc{static_cast<uint32_t>(shard), sizes[i]};
        ReserveLocked(shard, sizes[i]);
        is_new[i] = 1;
      }
      buckets[shard].push_back(i);
    }
  }

  std::vector<int> busy;
  for (size_t s = 0; s < buckets.size(); ++s) {
    if (!buckets[s].empty()) busy.push_back(static_cast<int>(s));
  }

  RunPerShard(busy, io_threads_, [&](int s) {
    const auto& idxs = buckets[s];
    std::vector<std::string> k;
    std::vector<const void*> d;
    std::vector<size_t> z;
    k.reserve(idxs.size());
    d.reserve(idxs.size());
    z.reserve(idxs.size());
    for (size_t i : idxs) {
      k.push_back(keys[i]);
      d.push_back(data_ptrs[i]);
      z.push_back(sizes[i]);
    }
    auto shard_res = shards_[s]->BatchWrite(k, d, z);
    for (size_t j = 0; j < idxs.size(); ++j) {
      results[idxs[j]] = j < shard_res.size() && shard_res[j];
    }
  });

  // Roll back the map/hint for newly-assigned keys whose write failed, so a
  // retry is free to pick a different drive.
  std::unique_lock<std::shared_mutex> lock(map_mu_);
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!results[i] && is_new[i]) {
      auto it = key_shard_.find(keys[i]);
      if (it != key_shard_.end()) {
        ReleaseLocked(static_cast<int>(it->second.shard), it->second.size);
        key_shard_.erase(it);
      }
    }
  }
  return results;
}

std::vector<bool> ShardedSsdTier::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                                   const std::vector<uintptr_t>& dst_ptrs,
                                                   const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;
  if (dst_ptrs.size() != keys.size() || sizes.size() != keys.size()) {
    MORI_UMBP_ERROR("[ShardedSsdTier] BatchReadIntoPtr: mismatched input lengths ({} / {} / {})",
                    keys.size(), dst_ptrs.size(), sizes.size());
    return results;
  }

  std::vector<std::vector<size_t>> buckets(shards_.size());
  {
    std::shared_lock<std::shared_mutex> lock(map_mu_);
    for (size_t i = 0; i < keys.size(); ++i) {
      int shard = LookupLocked(keys[i]);
      if (shard >= 0) buckets[shard].push_back(i);
    }
  }

  std::vector<int> busy;
  for (size_t s = 0; s < buckets.size(); ++s) {
    if (!buckets[s].empty()) busy.push_back(static_cast<int>(s));
  }

  RunPerShard(busy, io_threads_, [&](int s) {
    const auto& idxs = buckets[s];
    std::vector<std::string> k;
    std::vector<uintptr_t> d;
    std::vector<size_t> z;
    k.reserve(idxs.size());
    d.reserve(idxs.size());
    z.reserve(idxs.size());
    for (size_t i : idxs) {
      k.push_back(keys[i]);
      d.push_back(dst_ptrs[i]);
      z.push_back(sizes[i]);
    }
    auto shard_res = shards_[s]->BatchReadIntoPtr(k, d, z);
    for (size_t j = 0; j < idxs.size(); ++j) {
      results[idxs[j]] = j < shard_res.size() && shard_res[j];
    }
  });
  return results;
}

bool ShardedSsdTier::WriteBatch(const std::vector<std::string>& keys,
                                const std::vector<const void*>& data_ptrs,
                                const std::vector<size_t>& sizes) {
  auto res = BatchWrite(keys, data_ptrs, sizes);
  return std::all_of(res.begin(), res.end(), [](bool ok) { return ok; });
}

std::vector<bool> ShardedSsdTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                                   const std::vector<uintptr_t>& dst_ptrs,
                                                   const std::vector<size_t>& sizes) {
  return BatchReadIntoPtr(keys, dst_ptrs, sizes);
}

}  // namespace mori::umbp
