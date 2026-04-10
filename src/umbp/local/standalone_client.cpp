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
#include "umbp/local/standalone_client.h"

#include <stdexcept>
#include <string>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/dram_tier.h"

namespace mori::umbp {

UMBPConfig StandaloneClient::NormalizeConfig(const UMBPConfig& config) {
  UMBPConfig normalized = config;
  normalized.role = config.ResolveRole();
  normalized.follower_mode = (normalized.role == UMBPRole::SharedSSDFollower);
  normalized.force_ssd_copy_on_write = (normalized.role == UMBPRole::SharedSSDLeader);
  std::string error_message;
  if (!normalized.Validate(&error_message)) {
    throw std::runtime_error("invalid UMBP config: " + error_message);
  }
  return normalized;
}

StandaloneClient::StandaloneClient(const UMBPConfig& config)
    : config_(NormalizeConfig(config)), role_(config_.ResolveRole()), storage_(config_, &index_) {
  copy_pipeline_ = std::make_unique<CopyPipeline>(storage_, config_.copy_pipeline, role_);
}

StandaloneClient::~StandaloneClient() { Close(); }

// ---------------------------------------------------------------------------
// Core KV Operations (IUMBPClient interface)
// ---------------------------------------------------------------------------

bool StandaloneClient::Put(const std::string& key, uintptr_t src, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;
  if (index_.MayExist(key)) return true;

  if (!storage_.WriteFromPtr(key, src, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  copy_pipeline_->MaybeCopyToSharedSSD(key);
  return true;
}

bool StandaloneClient::Get(const std::string& key, uintptr_t dst, size_t size) {
  bool in_index = index_.MayExist(key);

  if (!in_index && role_ != UMBPRole::SharedSSDFollower) {
    return false;
  }

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
      if (in_index && !storage_.Exists(key)) {
        index_.Remove(key);
      }
    }
  } else if (!ok && in_index && !storage_.Exists(key)) {
    index_.Remove(key);
  }

  if (!ok && role_ == UMBPRole::SharedSSDFollower && !in_index) {
    if (!storage_.Exists(key)) {
      index_.Remove(key);
    }
  }

  return ok;
}

bool StandaloneClient::Exists(const std::string& key) const {
  if (role_ == UMBPRole::SharedSSDFollower) {
    return storage_.Exists(key);
  }
  return index_.MayExist(key);
}

// ---------------------------------------------------------------------------
// Batch Operations
// ---------------------------------------------------------------------------

std::vector<bool> StandaloneClient::BatchPut(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& srcs,
                                             const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);

  for (size_t i = 0; i < keys.size(); ++i) {
    if (role_ == UMBPRole::SharedSSDFollower) continue;
    if (index_.MayExist(keys[i])) {
      results[i] = true;
      continue;
    }
    if (!storage_.WriteFromPtr(keys[i], srcs[i], sizes[i])) continue;
    index_.Insert(keys[i], {StorageTier::CPU_DRAM, 0, sizes[i]});
    results[i] = true;
  }

  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    copy_pipeline_->MaybeBatchCopyToSharedSSD(ssd_keys);
  }
  return results;
}

std::vector<bool> StandaloneClient::BatchPutWithDepth(const std::vector<std::string>& keys,
                                                      const std::vector<uintptr_t>& srcs,
                                                      const std::vector<size_t>& sizes,
                                                      const std::vector<int>& depths) {
  std::vector<bool> results(keys.size(), false);

  for (size_t i = 0; i < keys.size(); ++i) {
    if (role_ == UMBPRole::SharedSSDFollower) continue;
    if (index_.MayExist(keys[i])) {
      results[i] = true;
      continue;
    }
    int depth = (i < depths.size()) ? depths[i] : -1;
    if (!storage_.WriteFromPtrWithDepth(keys[i], srcs[i], sizes[i], depth)) continue;
    index_.Insert(keys[i], {StorageTier::CPU_DRAM, 0, sizes[i]});
    results[i] = true;
  }

  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    copy_pipeline_->MaybeBatchCopyToSharedSSD(ssd_keys);
  }
  return results;
}

std::vector<bool> StandaloneClient::BatchGet(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dsts,
                                             const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;

  std::vector<size_t> read_indices;
  std::vector<bool> was_in_index(keys.size(), false);
  read_indices.reserve(keys.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    was_in_index[i] = index_.MayExist(keys[i]);
    if (!was_in_index[i] && role_ != UMBPRole::SharedSSDFollower) {
      continue;
    }
    read_indices.push_back(i);
  }

  if (read_indices.empty()) return results;

  std::vector<std::string> batch_keys;
  std::vector<uintptr_t> batch_ptrs;
  std::vector<size_t> batch_sizes;
  batch_keys.reserve(read_indices.size());
  batch_ptrs.reserve(read_indices.size());
  batch_sizes.reserve(read_indices.size());
  for (size_t idx : read_indices) {
    batch_keys.push_back(keys[idx]);
    batch_ptrs.push_back(dsts[idx]);
    batch_sizes.push_back(sizes[idx]);
  }

  auto batch_results = storage_.ReadBatchIntoPtr(batch_keys, batch_ptrs, batch_sizes);

  for (size_t j = 0; j < read_indices.size(); ++j) {
    size_t i = read_indices[j];
    bool local_hit = batch_results[j];

    if (role_ == UMBPRole::SharedSSDFollower) {
      if (local_hit) {
        StorageTier tier = StorageTier::LOCAL_SSD;
        auto* dram = storage_.GetTier(StorageTier::CPU_DRAM);
        if (dram && dram->Exists(keys[i])) {
          tier = StorageTier::CPU_DRAM;
        }
        if (!index_.UpdateTier(keys[i], tier)) {
          index_.Insert(keys[i], {tier, 0, sizes[i]});
        }
      } else if (!storage_.Exists(keys[i])) {
        index_.Remove(keys[i]);
      }
    } else if (!local_hit && was_in_index[i] && !storage_.Exists(keys[i])) {
      index_.Remove(keys[i]);
    }

    results[i] = local_hit;
  }

  return results;
}

std::vector<bool> StandaloneClient::BatchExists(const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    results[i] = Exists(keys[i]);
  }
  return results;
}

size_t StandaloneClient::BatchExistsConsecutive(const std::vector<std::string>& keys) const {
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!Exists(keys[i])) return i;
  }
  return keys.size();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void StandaloneClient::Clear() {
  index_.Clear();
  storage_.Clear();
}

bool StandaloneClient::Flush() { return storage_.Flush(); }

void StandaloneClient::Close() {
  if (closed_) return;
  closed_ = true;
  // CopyPipeline and storage are cleaned up by their own destructors,
  // which run after this in the member destruction order.
}

bool StandaloneClient::IsDistributed() const { return false; }

// ---------------------------------------------------------------------------
// Extra methods (not in IUMBPClient)
// ---------------------------------------------------------------------------

bool StandaloneClient::Put(const std::string& key, const void* data, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;
  if (index_.MayExist(key)) return true;

  if (!storage_.Write(key, data, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  copy_pipeline_->MaybeCopyToSharedSSD(key);
  return true;
}

bool StandaloneClient::Remove(const std::string& key) {
  auto loc = index_.Remove(key);
  if (!loc) return false;
  storage_.Evict(key);
  return true;
}

mori::umbp::LocalBlockIndex& StandaloneClient::Index() { return index_; }

LocalStorageManager& StandaloneClient::Storage() { return storage_; }

}  // namespace mori::umbp
