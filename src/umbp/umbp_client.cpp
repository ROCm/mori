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
#include "umbp/umbp_client.h"

#include <algorithm>
#include <future>

UMBPConfig UMBPClient::NormalizeConfig(const UMBPConfig& config) {
  UMBPConfig normalized = config;
  normalized.role = config.ResolveRole();
  normalized.follower_mode = (normalized.role == UMBPRole::SharedSSDFollower);
  normalized.force_ssd_copy_on_write = (normalized.role == UMBPRole::SharedSSDLeader);
  return normalized;
}

UMBPClient::UMBPClient(const UMBPConfig& config)
    : config_(NormalizeConfig(config)), role_(config_.ResolveRole()), storage_(config_, &index_) {
  if (role_ == UMBPRole::SharedSSDLeader && config_.copy_to_ssd_async) {
    const size_t n_workers = std::max<size_t>(1, config_.ssd_writer_threads);
    copy_workers_.reserve(n_workers);
    for (size_t i = 0; i < n_workers; ++i) {
      copy_workers_.emplace_back(&UMBPClient::CopyWorkerLoop, this);
    }
  }
}

UMBPClient::~UMBPClient() {
  if (!copy_workers_.empty()) {
    {
      std::lock_guard<std::mutex> lock(copy_mu_);
      stop_copy_worker_.store(true);
    }
    copy_cv_.notify_all();
    for (auto& w : copy_workers_) {
      if (w.joinable()) w.join();
    }
  }
}

bool UMBPClient::EnqueueCopyToSSD(const std::string& key) {
  std::unique_lock<std::mutex> lock(copy_mu_);
  if (copy_queue_.size() >= config_.copy_to_ssd_queue_depth) {
    return false;
  }
  copy_queue_.push_back({key});
  lock.unlock();
  copy_cv_.notify_one();
  return true;
}

void UMBPClient::CopyWorkerLoop() {
  const size_t batch_max = std::max<size_t>(1, config_.ssd_batch_max_ops);
  while (true) {
    std::vector<std::string> batch;
    {
      std::unique_lock<std::mutex> lock(copy_mu_);
      copy_cv_.wait(lock, [&]() { return stop_copy_worker_.load() || !copy_queue_.empty(); });
      if (stop_copy_worker_.load() && copy_queue_.empty()) return;
      // Drain up to batch_max entries.
      size_t n = std::min(batch_max, copy_queue_.size());
      batch.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        batch.push_back(std::move(copy_queue_.front().key));
        copy_queue_.pop_front();
      }
    }
    if (batch.size() == 1) {
      storage_.CopyToSSD(batch[0]);
    } else {
      storage_.CopyToSSDBatch(batch);
    }
  }
}

bool UMBPClient::MaybeCopyToSharedSSD(const std::string& key) {
  if (role_ != UMBPRole::SharedSSDLeader) return true;
  if (!config_.copy_to_ssd_async) {
    storage_.CopyToSSD(key);
    return true;
  }
  if (!EnqueueCopyToSSD(key)) {
    // Queue is full, back to sync path to preserve write-through behavior.
    storage_.CopyToSSD(key);
  }
  return true;
}

size_t UMBPClient::EnqueueCopyToSSDBatch(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(copy_mu_);
  size_t remaining = (config_.copy_to_ssd_queue_depth > copy_queue_.size())
                         ? config_.copy_to_ssd_queue_depth - copy_queue_.size()
                         : 0;
  size_t to_enqueue = std::min(keys.size(), remaining);
  for (size_t i = 0; i < to_enqueue; ++i) {
    copy_queue_.push_back({keys[i]});
  }
  lock.unlock();
  if (to_enqueue > 0) {
    copy_cv_.notify_all();  // wake all workers for parallel drain
  }
  return to_enqueue;
}

void UMBPClient::MaybeBatchCopyToSharedSSD(const std::vector<std::string>& keys) {
  if (role_ != UMBPRole::SharedSSDLeader || keys.empty()) return;

  if (!config_.copy_to_ssd_async) {
    storage_.CopyToSSDBatch(keys);
    return;
  }
  size_t enqueued = EnqueueCopyToSSDBatch(keys);
  if (enqueued < keys.size()) {
    // Overflow: sync fallback for remaining keys.
    std::vector<std::string> overflow(keys.begin() + enqueued, keys.end());
    storage_.CopyToSSDBatch(overflow);
  }
}

namespace {
template <typename Fn>
void RunParallelFor(size_t n, size_t workers, Fn&& fn) {
  if (n == 0) return;
  if (workers <= 1 || n == 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  size_t chunks = std::min(workers, n);
  size_t chunk_size = (n + chunks - 1) / chunks;
  std::vector<std::future<void>> futures;
  futures.reserve(chunks);
  for (size_t c = 0; c < chunks; ++c) {
    size_t begin = c * chunk_size;
    if (begin >= n) break;
    size_t end = std::min(n, begin + chunk_size);
    futures.push_back(std::async(std::launch::async, [begin, end, &fn]() {
      for (size_t i = begin; i < end; ++i) fn(i);
    }));
  }
  for (auto& f : futures) f.get();
}
}  // namespace

bool UMBPClient::Put(const std::string& key, const void* data, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  // Content-addressed dedup: same key = same data (SHA-256 of token IDs).
  // This matches SGLang/MooncakeStore semantics where KV cache blocks are
  // immutable — once written, the same hash always maps to the same content.
  if (index_.MayExist(key)) return true;

  if (!storage_.Write(key, data, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  MaybeCopyToSharedSSD(key);
  return true;
}

bool UMBPClient::PutFromPtr(const std::string& key, uintptr_t src, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;
  if (index_.MayExist(key)) return true;  // content-addressed dedup

  if (!storage_.WriteFromPtr(key, src, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  MaybeCopyToSharedSSD(key);
  return true;
}

bool UMBPClient::GetIntoPtr(const std::string& key, uintptr_t dst, size_t size) {
  bool in_index = index_.MayExist(key);

  if (!in_index && role_ != UMBPRole::SharedSSDFollower) return false;

  bool ok = storage_.ReadIntoPtr(key, dst, size);

  if (role_ == UMBPRole::SharedSSDFollower) {
    if (ok) {
      StorageTier tier = StorageTier::LOCAL_SSD;
      auto* dram = storage_.GetTier(StorageTier::CPU_DRAM);
      if (dram && dram->Exists(key)) {
        tier = StorageTier::CPU_DRAM;
      }
      if (!index_.UpdateTier(key, tier)) {
        index_.Insert(key, {tier, 0, size});
      }
    } else {
      // In follower mode, the in-memory index can become stale if leader
      // evicts files from shared SSD. Remove stale hints.
      if (in_index && !storage_.Exists(key)) {
        index_.Remove(key);
      }
    }
  } else if (!ok && in_index && !storage_.Exists(key)) {
    // If the key was indexed but has been evicted from all tiers, clean up.
    index_.Remove(key);
  }

  if (!ok && role_ == UMBPRole::SharedSSDFollower && !in_index) {
    // Best effort stale-hint cleanup for keys first observed via filesystem
    // fallback but missing at read time.
    if (!storage_.Exists(key)) {
      index_.Remove(key);
    }
  }

  return ok;
}

bool UMBPClient::Exists(const std::string& key) const {
  if (role_ == UMBPRole::SharedSSDFollower) {
    // In follower mode, always verify underlying tiers. The index is only a
    // performance hint and may be stale across ranks.
    return storage_.Exists(key);
  }
  return index_.MayExist(key);
}

bool UMBPClient::Remove(const std::string& key) {
  auto loc = index_.Remove(key);
  if (!loc) return false;

  storage_.Evict(key);
  return true;
}

std::vector<bool> UMBPClient::BatchPutFromPtr(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& ptrs,
                                              const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);

  // Phase 1 (serial): write to DRAM + update index.
  for (size_t i = 0; i < keys.size(); ++i) {
    if (role_ == UMBPRole::SharedSSDFollower) continue;
    if (index_.MayExist(keys[i])) {
      results[i] = true;
      continue;
    }
    if (!storage_.WriteFromPtr(keys[i], ptrs[i], sizes[i])) continue;
    index_.Insert(keys[i], {StorageTier::CPU_DRAM, 0, sizes[i]});
    results[i] = true;
  }

  // Phase 2: batch copy to shared SSD (Leader only).
  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    MaybeBatchCopyToSharedSSD(ssd_keys);
  }
  return results;
}

std::vector<bool> UMBPClient::BatchPutFromPtrWithDepth(const std::vector<std::string>& keys,
                                                       const std::vector<uintptr_t>& ptrs,
                                                       const std::vector<size_t>& sizes,
                                                       const std::vector<int>& depths) {
  std::vector<bool> results(keys.size(), false);

  // Phase 1 (serial): write to DRAM + update index.
  for (size_t i = 0; i < keys.size(); ++i) {
    if (role_ == UMBPRole::SharedSSDFollower) continue;
    if (index_.MayExist(keys[i])) {
      results[i] = true;  // content-addressed dedup
      continue;
    }
    int depth = (i < depths.size()) ? depths[i] : -1;
    if (!storage_.WriteFromPtrWithDepth(keys[i], ptrs[i], sizes[i], depth)) continue;
    index_.Insert(keys[i], {StorageTier::CPU_DRAM, 0, sizes[i]});
    results[i] = true;
  }

  // Phase 2: batch copy to shared SSD (Leader only).
  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    MaybeBatchCopyToSharedSSD(ssd_keys);
  }
  return results;
}

std::vector<bool> UMBPClient::BatchGetIntoPtr(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& ptrs,
                                              const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = GetIntoPtr(keys[i], ptrs[i], sizes[i]);
  }
  return results;
}

std::vector<bool> UMBPClient::BatchExists(const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Exists(keys[i]);
  }
  return results;
}

size_t UMBPClient::BatchExistsConsecutive(const std::vector<std::string>& keys) const {
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!Exists(keys[i])) return i;
  }
  return keys.size();
}

void UMBPClient::Clear() {
  index_.Clear();
  storage_.Clear();
}

BlockIndexClient& UMBPClient::Index() { return index_; }

LocalStorageManager& UMBPClient::Storage() { return storage_; }
