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

UMBPConfig UMBPClient::NormalizeConfig(const UMBPConfig& config) {
  UMBPConfig normalized = config;
  normalized.role = config.ResolveRole();
  normalized.follower_mode = (normalized.role == UMBPRole::SharedSSDFollower);
  normalized.force_ssd_copy_on_write = (normalized.role == UMBPRole::SharedSSDLeader);
  return normalized;
}

UMBPClient::UMBPClient(const UMBPConfig& config)
    : config_(NormalizeConfig(config)), role_(config_.ResolveRole()), storage_(config_, &index_) {}

bool UMBPClient::Put(const std::string& key, const void* data, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  // Content-addressed dedup: same key = same data (SHA-256 of token IDs).
  // This matches SGLang/MooncakeStore semantics where KV cache blocks are
  // immutable — once written, the same hash always maps to the same content.
  if (index_.MayExist(key)) return true;

  if (!storage_.Write(key, data, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});

  // Leader mode: also copy to shared SSD so followers can discover it
  if (role_ == UMBPRole::SharedSSDLeader) {
    storage_.CopyToSSD(key);  // best-effort, failure is non-fatal
  }
  return true;
}

bool UMBPClient::PutFromPtr(const std::string& key, uintptr_t src, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;
  if (index_.MayExist(key)) return true;  // content-addressed dedup

  if (!storage_.WriteFromPtr(key, src, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});

  if (role_ == UMBPRole::SharedSSDLeader) {
    storage_.CopyToSSD(key);
  }
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
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = PutFromPtr(keys[i], ptrs[i], sizes[i]);
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

void UMBPClient::Clear() {
  index_.Clear();
  storage_.Clear();
}

BlockIndexClient& UMBPClient::Index() { return index_; }

LocalStorageManager& UMBPClient::Storage() { return storage_; }
