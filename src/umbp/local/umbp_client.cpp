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
#include "umbp/local/umbp_client.h"

#include <stdexcept>
#include <string>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/types.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/storage/dram_tier.h"

namespace mori::umbp {

UMBPConfig UMBPClient::NormalizeConfig(const UMBPConfig& config) {
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

UMBPClient::UMBPClient(const UMBPConfig& config)
    : config_(NormalizeConfig(config)), role_(config_.ResolveRole()), storage_(config_, &index_) {
  copy_pipeline_ = std::make_unique<CopyPipeline>(storage_, config_.copy_pipeline, role_);

  // Phase 1: connect to Master and start heartbeat. No delegation, no DRAM/SSD
  // export, no changes to Put/Get/Remove yet.
  MORI_UMBP_INFO("[UMBPClient] ctor: distributed.has_value()={}", config_.distributed.has_value());
  if (config_.distributed.has_value()) {
    const auto& dist = config_.distributed.value();

    PoolClientConfig pc_config;
    pc_config.master_config.master_address = dist.master_address;
    pc_config.master_config.node_id = dist.node_id;
    pc_config.master_config.node_address = dist.node_address;
    pc_config.master_config.auto_heartbeat = dist.auto_heartbeat;
    pc_config.io_engine_host = dist.io_engine_host;
    pc_config.io_engine_port = dist.io_engine_port;
    pc_config.staging_buffer_size = dist.staging_buffer_size;
    pc_config.peer_service_port = dist.peer_service_port;

    // Export DramTier buffer so PoolClient registers it with the master at
    // Init() time, enabling remote RDMA reads into this node's DRAM.
    auto* dram = storage_.GetTierAs<DRAMTier>(StorageTier::CPU_DRAM);
    if (dram) {
      auto [used, total] = storage_.Capacity(StorageTier::CPU_DRAM);
      pc_config.dram_buffers.push_back({dram->GetBasePtr(), total});
      pc_config.tier_capacities[TierType::DRAM] = {total, total};
      MORI_UMBP_INFO("[UMBPClient] ctor: exporting DRAM buffer ptr={}, size={}", dram->GetBasePtr(),
                     total);
    }

    MORI_UMBP_INFO("[UMBPClient] ctor: creating PoolClient for node '{}', master='{}'",
                   dist.node_id, dist.master_address);
    pool_client_ = std::make_unique<PoolClient>(std::move(pc_config));
    if (!pool_client_->Init()) {
      MORI_UMBP_ERROR("PoolClient init failed for node '{}', falling back to local mode",
                      dist.node_id);
      pool_client_.reset();
    }
  }

  // Register DRAM for local zero-copy RDMA and install tier-change callback.
  if (pool_client_) {
    auto* dram = storage_.GetTierAs<DRAMTier>(StorageTier::CPU_DRAM);
    if (dram) {
      auto [used, total] = storage_.Capacity(StorageTier::CPU_DRAM);
      pool_client_->RegisterMemory(dram->GetBasePtr(), total);
    }

    // On DRAM→SSD demotion or full eviction from DRAM, unregister the block
    // from Master so remote nodes stop attempting RDMA reads to a stale offset.
    storage_.SetOnTierChange(
        [this](const std::string& key, StorageTier from, std::optional<StorageTier> /*to*/) {
          if (from == StorageTier::CPU_DRAM && pool_client_) {
            pool_client_->UnregisterFromMaster(key);
          }
        });
  }
}

UMBPClient::~UMBPClient() {
  // Shut down PoolClient before storage_/index_ are destroyed, since later
  // phases will give PoolClient pointers into those members.
  if (pool_client_) {
    pool_client_->Shutdown();
    pool_client_.reset();
  }
}

void UMBPClient::MaybeRegisterWithMaster(const std::string& key, size_t size) {
  if (!pool_client_) return;
  auto* dram = storage_.GetTierAs<DRAMTier>(StorageTier::CPU_DRAM);
  if (!dram) return;
  auto offset = dram->GetSlotOffset(key);
  if (!offset) return;
  std::string location_id = "0:" + std::to_string(*offset);
  pool_client_->RegisterWithMaster(key, size, location_id, TierType::DRAM);
}

bool UMBPClient::Put(const std::string& key, const void* data, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  // Content-addressed dedup: same key = same data (SHA-256 of token IDs).
  // This matches SGLang/MooncakeStore semantics where KV cache blocks are
  // immutable — once written, the same hash always maps to the same content.
  if (index_.MayExist(key)) return true;

  if (!storage_.Write(key, data, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  MaybeRegisterWithMaster(key, size);
  copy_pipeline_->MaybeCopyToSharedSSD(key);
  return true;
}

bool UMBPClient::PutFromPtr(const std::string& key, uintptr_t src, size_t size) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;
  if (index_.MayExist(key)) return true;  // content-addressed dedup

  if (!storage_.WriteFromPtr(key, src, size)) return false;

  index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
  MaybeRegisterWithMaster(key, size);
  copy_pipeline_->MaybeCopyToSharedSSD(key);
  return true;
}

bool UMBPClient::GetIntoPtr(const std::string& key, uintptr_t dst, size_t size) {
  bool in_index = index_.MayExist(key);

  MORI_UMBP_DEBUG("[UMBPClient] GetIntoPtr: key='{}' in_index={} role={} pool_client_={}", key,
                  in_index, static_cast<int>(role_), pool_client_ != nullptr);

  if (!in_index && role_ != UMBPRole::SharedSSDFollower && !pool_client_) {
    MORI_UMBP_DEBUG(
        "[UMBPClient] GetIntoPtr: early return false — not in index, not follower, no pool_client");
    return false;
  }

  bool ok = storage_.ReadIntoPtr(key, dst, size);
  MORI_UMBP_DEBUG("[UMBPClient] GetIntoPtr: local ReadIntoPtr for key '{}' returned {}", key, ok);

  // Phase 3: on local miss, try fetching from a remote node's DRAM via RDMA.
  if (!ok && pool_client_) {
    MORI_UMBP_DEBUG(
        "[UMBPClient] GetIntoPtr: local miss for key '{}', attempting GetRemote (size={})", key,
        size);
    ok = pool_client_->GetRemote(key, reinterpret_cast<void*>(dst), size);
    MORI_UMBP_DEBUG("[UMBPClient] GetIntoPtr: GetRemote for key '{}' returned {}", key, ok);
  }

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
  if (index_.MayExist(key)) return true;

  // Phase 3: key not in local index — check if any remote node holds it.
  // SGLang calls Exists() before Get(), so without this remote check a
  // cluster-wide key would appear missing and Get() would never be called.
  if (pool_client_) {
    return pool_client_->ExistsRemote(key);
  }
  return false;
}

bool UMBPClient::Remove(const std::string& key) {
  auto loc = index_.Remove(key);
  if (!loc) return false;

  storage_.Evict(key);
  // Note: the on_tier_change_ callback inside Evict() already calls
  // UnregisterFromMaster for DRAM-resident blocks. Call explicitly here
  // to cover cases where the block may have already left DRAM (e.g. on SSD)
  // but is still registered with Master (Phase 6 scenario).
  if (pool_client_) {
    pool_client_->UnregisterFromMaster(key);
  }
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
    MaybeRegisterWithMaster(keys[i], sizes[i]);
    results[i] = true;
  }

  // Phase 2: batch copy to shared SSD (Leader only).
  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    copy_pipeline_->MaybeBatchCopyToSharedSSD(ssd_keys);
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
    MaybeRegisterWithMaster(keys[i], sizes[i]);
    results[i] = true;
  }

  // Phase 2: batch copy to shared SSD (Leader only).
  if (role_ == UMBPRole::SharedSSDLeader) {
    std::vector<std::string> ssd_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (results[i]) ssd_keys.push_back(keys[i]);
    }
    copy_pipeline_->MaybeBatchCopyToSharedSSD(ssd_keys);
  }
  return results;
}

std::vector<bool> UMBPClient::BatchGetIntoPtr(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& ptrs,
                                              const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;

  // Phase 1: Index pre-check — filter out keys that cannot possibly exist.
  std::vector<size_t> read_indices;  // indices into keys/ptrs/sizes to actually read
  std::vector<bool> was_in_index(keys.size(), false);
  read_indices.reserve(keys.size());

  MORI_UMBP_DEBUG("[UMBPClient] BatchGetIntoPtr: {} keys, role={} pool_client_={}", keys.size(),
                  static_cast<int>(role_), pool_client_ != nullptr);

  for (size_t i = 0; i < keys.size(); ++i) {
    was_in_index[i] = index_.MayExist(keys[i]);
    if (!was_in_index[i] && role_ != UMBPRole::SharedSSDFollower && !pool_client_) {
      // Non-follower, non-distributed: key not in index → guaranteed miss.
      MORI_UMBP_DEBUG(
          "[UMBPClient] BatchGetIntoPtr: skipping key '{}' — not in index, not follower, no "
          "pool_client",
          keys[i]);
      continue;
    }
    read_indices.push_back(i);
  }

  if (read_indices.empty()) return results;

  // Phase 2: Batch storage read.
  std::vector<std::string> batch_keys;
  std::vector<uintptr_t> batch_ptrs;
  std::vector<size_t> batch_sizes;
  batch_keys.reserve(read_indices.size());
  batch_ptrs.reserve(read_indices.size());
  batch_sizes.reserve(read_indices.size());
  for (size_t idx : read_indices) {
    batch_keys.push_back(keys[idx]);
    batch_ptrs.push_back(ptrs[idx]);
    batch_sizes.push_back(sizes[idx]);
  }

  auto batch_results = storage_.ReadBatchIntoPtr(batch_keys, batch_ptrs, batch_sizes);

  // Phase 3: Update local index based on local storage results only.
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
  }

  // Phase 4: Try remote DRAM for local misses and set final results.
  for (size_t j = 0; j < read_indices.size(); ++j) {
    size_t i = read_indices[j];
    bool ok = batch_results[j];

    if (!ok && pool_client_) {
      MORI_UMBP_DEBUG(
          "[UMBPClient] BatchGetIntoPtr: local miss for key '{}', attempting GetRemote (size={})",
          keys[i], sizes[i]);
      ok = pool_client_->GetRemote(keys[i], reinterpret_cast<void*>(ptrs[i]), sizes[i]);
      MORI_UMBP_DEBUG("[UMBPClient] BatchGetIntoPtr: GetRemote for key '{}' returned {}", keys[i],
                      ok);
    }
    results[i] = ok;
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

mori::umbp::LocalBlockIndex& UMBPClient::Index() { return index_; }

LocalStorageManager& UMBPClient::Storage() { return storage_; }

bool UMBPClient::IsDistributed() const { return pool_client_ != nullptr; }

}  // namespace mori::umbp
