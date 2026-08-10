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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/peer/medium_backend.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Second registered backend proving BackendRegistry dispatch (design doc
// Phase 2, default option (b)): "MockBackend as the second registered
// backend. Cheap, proves the registry and dispatch paths, keeps the local
// path honestly single-medium."
//
// Deliberately NOT wired into live routing: Capacity() always reports zero,
// so master never picks this node's MockBackend-fronted tier as a Put/Get
// target — the DRAM backend's behavior stays byte-for-byte unchanged with
// this second backend registered alongside it.  The point of this class is
// to be a real, in-memory, fully-functional MediumBackend implementation
// that a test can drive through BackendRegistry — not a no-op stub.
class MockBackend : public MediumBackend {
 public:
  explicit MockBackend(TierType tier) : tier_(tier) {}

  TierType Tier() const override { return tier_; }
  const char* Name() const override { return "MockBackend"; }
  BackendProperties Properties() const override {
    BackendProperties p;
    p.read_rank = 100;  // lowest priority; irrelevant since Capacity() is 0
    p.put_eligible = true;
    return p;
  }

  bool Init(TransferEngine* /*engine*/) override {
    initialized_ = true;
    return true;
  }
  void Shutdown() override {
    std::lock_guard<std::mutex> lock(mutex_);
    owned_.clear();
    pending_.clear();
    initialized_ = false;
  }

  // Always zero: this backend holds real bytes (for dispatch tests) but
  // never advertises capacity, so master-side routing never selects it.
  TierCapacity Capacity() const override { return TierCapacity{}; }

  uint64_t OwnedKeyCount() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return owned_.size();
  }

  std::vector<KvEvent> DrainPendingEvents() override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KvEvent> out;
    out.swap(pending_events_);
    return out;
  }
  std::vector<KvEvent> SnapshotOwnedKeys() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return SnapshotLocked();
  }
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto out = SnapshotLocked();
    pending_events_.clear();
    return out;
  }

  void ClearLocal() override {
    std::lock_guard<std::mutex> lock(mutex_);
    owned_.clear();
    pending_.clear();
    pending_events_.clear();
  }
  void ClearFullSyncAcked() override {}
  bool IsClearFullSyncPending() const override { return false; }

  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& entries) override {
    std::vector<AllocateResult> out(entries.size());
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < entries.size(); ++i) {
      const auto& req = entries[i];
      if (req.size == 0) {
        out[i].outcome = AllocateOutcome::kFailed;
        continue;
      }
      if (owned_.find(req.key) != owned_.end()) {
        out[i].outcome = AllocateOutcome::kSuccessAlreadyExists;
        continue;
      }
      const uint64_t slot_id = next_slot_id_++;
      pending_[slot_id] = {req.key, std::vector<uint8_t>(req.size)};
      out[i].outcome = AllocateOutcome::kSuccessAllocated;
      out[i].slot_id = slot_id;
      out[i].size = req.size;
      out[i].page_size = req.size;
      out[i].pending_ttl_ms = 0;
    }
    return out;
  }

  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& entries) override {
    std::vector<CommitResult> out(entries.size());
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < entries.size(); ++i) {
      auto it = pending_.find(entries[i].slot_id);
      if (it == pending_.end()) continue;
      const uint64_t committed_size = it->second.bytes.size();
      owned_[entries[i].key] = std::move(it->second.bytes);
      pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, entries[i].key, tier_, committed_size});
      out[i].success = true;
      out[i].bytes_committed = committed_size;
      pending_.erase(it);
    }
    return out;
  }

  std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids) override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<bool> out(slot_ids.size(), true);
    for (uint64_t id : slot_ids) pending_.erase(id);
    return out;
  }

  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                          bool /*include_descs*/) override {
    std::vector<ResolvedEntry> out(keys.size());
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = owned_.find(keys[i]);
      if (it == owned_.end()) continue;
      out[i].found = true;
      out[i].size = it->second.size();
    }
    return out;
  }

  std::vector<EvictResult> Evict(const std::vector<std::string>& keys) override {
    std::vector<EvictResult> out;
    out.reserve(keys.size());
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& key : keys) {
      EvictResult r;
      r.key = key;
      auto it = owned_.find(key);
      if (it != owned_.end()) {
        r.bytes_freed = it->second.size();
        pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, tier_, 0});
        owned_.erase(it);
      }
      out.push_back(std::move(r));
    }
    return out;
  }

  uint64_t PageSize() const override { return 1; }
  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override { return {}; }

 private:
  struct PendingSlot {
    std::string key;
    std::vector<uint8_t> bytes;
  };

  std::vector<KvEvent> SnapshotLocked() const {
    std::vector<KvEvent> out;
    out.reserve(owned_.size());
    for (const auto& [key, bytes] : owned_) {
      out.push_back(KvEvent{KvEvent::Kind::ADD, key, tier_, bytes.size()});
    }
    return out;
  }

  TierType tier_;
  bool initialized_ = false;
  mutable std::mutex mutex_;
  uint64_t next_slot_id_ = 1;
  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, std::vector<uint8_t>> owned_;
  std::vector<KvEvent> pending_events_;
};

}  // namespace mori::umbp
