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
#include "umbp/storage/local_storage_manager.h"

#include <stdexcept>

#include "umbp/storage/dram_tier.h"
#include "umbp/storage/ssd_tier.h"

LocalStorageManager::LocalStorageManager(const UMBPConfig& config, BlockIndexClient* index)
    : config_(config), role_(config.ResolveRole()), index_(index) {
  // DRAM tier is always present (fastest)
  tiers_.push_back({StorageTier::CPU_DRAM,
                    std::make_unique<DRAMTier>(config_.dram_capacity_bytes,
                                               config_.use_shared_memory, config_.shm_name)});

  // SSD tier is optional (slower)
  if (config_.ssd_enabled) {
    SSDAccessMode ssd_access_mode = SSDAccessMode::ReadWrite;
    if (role_ == UMBPRole::SharedSSDFollower) {
      ssd_access_mode = SSDAccessMode::ReadOnlyShared;
    }
    tiers_.push_back({StorageTier::LOCAL_SSD,
                      std::make_unique<SSDTier>(config_.ssd_storage_dir, config_.ssd_capacity_bytes,
                                                ssd_access_mode)});
  }
}

TierBackend* LocalStorageManager::GetTier(StorageTier tier) {
  for (auto& entry : tiers_) {
    if (entry.id == tier) return entry.backend.get();
  }
  return nullptr;
}

const TierBackend* LocalStorageManager::GetTier(StorageTier tier) const {
  for (const auto& entry : tiers_) {
    if (entry.id == tier) return entry.backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::FindTierHolding(const std::string& key) {
  for (auto& entry : tiers_) {
    if (entry.backend->Exists(key)) return entry.backend.get();
  }
  return nullptr;
}

const TierBackend* LocalStorageManager::FindTierHolding(const std::string& key) const {
  for (const auto& entry : tiers_) {
    if (entry.backend->Exists(key)) return entry.backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::NextSlowerTier(StorageTier current) {
  for (size_t i = 0; i + 1 < tiers_.size(); ++i) {
    if (tiers_[i].id == current) return tiers_[i + 1].backend.get();
  }
  return nullptr;
}

TierBackend* LocalStorageManager::NextFasterTier(StorageTier current) {
  for (size_t i = 1; i < tiers_.size(); ++i) {
    if (tiers_[i].id == current) return tiers_[i - 1].backend.get();
  }
  return nullptr;
}

bool LocalStorageManager::MoveKey(const std::string& key, TierBackend* from, TierBackend* to) {
  auto data = from->Read(key);
  if (data.empty()) return false;

  if (!to->Write(key, data.data(), data.size())) return false;

  from->Evict(key);

  UpsertIndexTier(key, to->tier_id(), data.size());
  return true;
}

bool LocalStorageManager::DemoteLRUForSpace(TierBackend* tier) {
  TierBackend* slower = NextSlowerTier(tier->tier_id());
  if (!slower) return false;
  if (role_ == UMBPRole::SharedSSDFollower && slower->tier_id() == StorageTier::LOCAL_SSD) {
    return false;
  }

  std::string victim = tier->GetLRUKey();
  if (victim.empty()) return false;

  return MoveKey(victim, tier, slower);
}

bool LocalStorageManager::Write(const std::string& key, const void* data, size_t size,
                                StorageTier tier) {
  TierBackend* target = GetTier(tier);
  if (!target) return false;

  // Try direct write
  if (target->Write(key, data, size)) return true;

  // Target full — try demoting LRU keys to next-slower tier
  while (DemoteLRUForSpace(target)) {
    if (target->Write(key, data, size)) return true;
  }

  // No slower tier or it's also full — last-resort eviction (data loss).
  // The caller's new key still gets stored.
  while (true) {
    std::string victim = target->GetLRUKey();
    if (victim.empty()) return false;
    target->Evict(victim);
    if (index_) index_->Remove(victim);
    if (target->Write(key, data, size)) return true;
  }
}

bool LocalStorageManager::WriteFromPtr(const std::string& key, uintptr_t src, size_t size,
                                       StorageTier tier) {
  return Write(key, reinterpret_cast<const void*>(src), size, tier);
}

bool LocalStorageManager::ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) {
  for (size_t i = 0; i < tiers_.size(); ++i) {
    if (tiers_[i].backend->Exists(key)) {
      bool ok = tiers_[i].backend->ReadIntoPtr(key, dst, size);
      // Auto-promote if read from a slower tier
      if (ok && i > 0 && config_.auto_promote_on_read) {
        MaybeAutoPromote(key);
      }
      return ok;
    }
  }
  return false;
}

bool LocalStorageManager::Exists(const std::string& key) const {
  return FindTierHolding(key) != nullptr;
}

bool LocalStorageManager::Evict(const std::string& key) {
  TierBackend* tier = FindTierHolding(key);
  if (!tier) {
    if (index_) index_->Remove(key);
    return false;
  }
  bool ok = tier->Evict(key);
  if (ok && index_) index_->Remove(key);
  return ok;
}

std::pair<size_t, size_t> LocalStorageManager::Capacity(StorageTier tier) const {
  const TierBackend* t = GetTier(tier);
  if (!t) return {0, 0};
  return t->Capacity();
}

bool LocalStorageManager::Demote(const std::string& key) {
  TierBackend* from = FindTierHolding(key);
  if (!from) return false;

  TierBackend* to = NextSlowerTier(from->tier_id());
  if (!to) return false;

  return MoveKey(key, from, to);
}

bool LocalStorageManager::Promote(const std::string& key) {
  TierBackend* from = FindTierHolding(key);
  if (!from) return false;

  TierBackend* to = NextFasterTier(from->tier_id());
  if (!to) return false;

  // Already in the faster tier
  if (to->Exists(key)) return true;

  // Try direct move
  auto data = from->Read(key);
  if (data.empty()) return false;

  // Shared SSD follower mode is read-only against SSD. Promotion into DRAM
  // must never demote DRAM victims into SSD.
  if (role_ == UMBPRole::SharedSSDFollower && from->tier_id() == StorageTier::LOCAL_SSD &&
      to->tier_id() == StorageTier::CPU_DRAM) {
    if (!to->Write(key, data.data(), data.size())) {
      while (true) {
        std::string victim = to->GetLRUKey();
        if (victim.empty()) return false;
        to->Evict(victim);
        if (index_) index_->Remove(victim);
        if (to->Write(key, data.data(), data.size())) break;
      }
    }
    // Drop read-tracking metadata in source tier while preserving shared file.
    from->Evict(key);
    UpsertIndexTier(key, to->tier_id(), data.size());
    return true;
  }

  if (!to->Write(key, data.data(), data.size())) {
    // Target full — demote LRU keys from target to make room
    while (DemoteLRUForSpace(to)) {
      if (to->Write(key, data.data(), data.size())) {
        from->Evict(key);
        UpsertIndexTier(key, to->tier_id(), data.size());
        return true;
      }
    }
    return false;  // Cannot promote
  }

  from->Evict(key);
  UpsertIndexTier(key, to->tier_id(), data.size());
  return true;
}

bool LocalStorageManager::CopyToSSD(const std::string& key) {
  if (role_ == UMBPRole::SharedSSDFollower) return false;

  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  TierBackend* ssd = GetTier(StorageTier::LOCAL_SSD);
  if (!dram || !ssd) return false;

  // Already on SSD
  if (ssd->Exists(key)) return true;

  auto data = dram->Read(key);
  if (data.empty()) return false;

  if (ssd->Write(key, data.data(), data.size())) return true;

  // SSD full — evict SSD LRU entries to make room.
  // If a victim is only on SSD (no DRAM copy), we must also update the index.
  while (true) {
    std::string victim = ssd->GetLRUKey();
    if (victim.empty()) return false;

    // Keep the index consistent: if the victim is only on SSD, evicting it
    // is data loss and the index must be updated.
    if (index_) {
      auto loc = index_->Lookup(victim);
      if (loc && (loc->tier == StorageTier::LOCAL_SSD || !dram->Exists(victim))) {
        index_->Remove(victim);
      }
    }

    ssd->Evict(victim);
    if (ssd->Write(key, data.data(), data.size())) return true;
  }
}

void LocalStorageManager::MaybeAutoPromote(const std::string& key) {
  if (!config_.auto_promote_on_read) return;

  // Only promote if key is not in the fastest tier
  if (tiers_.empty()) return;
  TierBackend* fastest = tiers_.front().backend.get();
  if (fastest->Exists(key)) return;

  // Check fastest tier watermark before promoting
  auto [used, total] = fastest->Capacity();
  double utilization = static_cast<double>(used) / total;
  if (utilization >= config_.dram_high_watermark) return;

  // Best-effort promote
  if (role_ == UMBPRole::SharedSSDFollower) {
    InsertReadCacheNoWriteback(key);
  } else {
    Promote(key);
  }
}

bool LocalStorageManager::InsertReadCacheNoWriteback(const std::string& key) {
  TierBackend* source = GetTier(StorageTier::LOCAL_SSD);
  TierBackend* dram = GetTier(StorageTier::CPU_DRAM);
  if (!source || !dram) return false;

  if (dram->Exists(key)) {
    UpsertIndexTier(key, StorageTier::CPU_DRAM, 0);
    return true;
  }
  if (!source->Exists(key)) return false;

  auto data = source->Read(key);
  if (data.empty()) return false;

  if (!dram->Write(key, data.data(), data.size())) {
    // Follower mode never writes back to SSD. Evict DRAM-only victims.
    while (true) {
      std::string victim = dram->GetLRUKey();
      if (victim.empty()) return false;
      dram->Evict(victim);
      if (index_) index_->Remove(victim);
      if (dram->Write(key, data.data(), data.size())) break;
    }
  }

  UpsertIndexTier(key, StorageTier::CPU_DRAM, data.size());
  return true;
}

void LocalStorageManager::UpsertIndexTier(const std::string& key, StorageTier tier,
                                          size_t size_hint) {
  if (!index_) return;
  if (!index_->UpdateTier(key, tier)) {
    index_->Insert(key, {tier, 0, size_hint});
  }
}

void LocalStorageManager::Clear() {
  for (auto& entry : tiers_) {
    entry.backend->Clear();
  }
  if (index_) index_->Clear();
}
