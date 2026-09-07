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
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
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
  // The registry requires every backend on a peer to agree on a page size, so
  // a mock sharing a registry with a real page-based backend must be told which
  // one to report.
  explicit MockBackend(TierType tier, uint64_t page_size = 1)
      : tier_(tier), page_size_(page_size) {}

  TierType Tier() const override { return tier_; }
  const char* Name() const override { return "MockBackend"; }

  bool Init(MemoryRegistrar* /*registrar*/) override {
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

  void SetAutoFlushHook(size_t threshold, std::function<void()> cb) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_flush_threshold_ = threshold;
    auto_flush_cb_ = std::move(cb);
  }

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
      QueueEventLocked(KvEvent{KvEvent::Kind::ADD, entries[i].key, tier_, committed_size});
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

  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys, bool include_descs,
                                          bool allow_file_refs = false) override {
    (void)allow_file_refs;  // this mock stages everything; it publishes no file refs
    std::vector<ResolvedEntry> out(keys.size());
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = owned_.find(keys[i]);
      if (it == owned_.end()) continue;
      out[i].outcome = ResolveOutcome::kFound;
      out[i].found = true;
      out[i].size = it->second.size();
      if (published_buffers_ == 0) continue;
      // Every published mock hands out buffer 0, which is the collision a real
      // peer has too (each backend publishes exactly one buffer today).
      out[i].pages = {PageLocation{0, 0}};
      out[i].page_size = PageSize();
      if (include_descs) out[i].descs = AllBufferDescsLocked();
    }
    return out;
  }

  bool Contains(const std::string& key) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return owned_.find(key) != owned_.end();
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
        QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, tier_, 0});
        owned_.erase(it);
      }
      out.push_back(std::move(r));
    }
    return out;
  }

  uint64_t PageSize() const override { return page_size_; }

  // Publish `count` fake buffers, numbered from 0 — like every real backend,
  // and the reason a bare buffer_index is not an address: two mocks that both
  // publish will both claim buffer 0.  `desc_tag` is the first descriptor byte
  // so a test can tell WHOSE buffer answered.  Off by default, so a mock that
  // does not call this behaves exactly as before.
  void PublishBuffers(uint32_t count, uint8_t desc_tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    published_buffers_ = count;
    desc_tag_ = desc_tag;
  }

  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return AllBufferDescsLocked();
  }

  // Zero unless PublishBuffers() was called; then each key resolves to a page in
  // buffer 0.  A backend keeps each key in its own std::vector rather than in
  // paged buffers, so with no published buffers a caller that tries to transfer
  // against it gets an invalid ref and must treat the key as unservable here —
  // which is exactly the shape a real medium takes when a tier has no live
  // backend on this node.
  size_t BufferCount() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return published_buffers_;
  }
  TransferRef BufferRef(uint32_t /*buffer_index*/) const override { return TransferRef{}; }

 private:
  // Backends leave backend_id at 0 — the peer service stamps the owner at the
  // wire boundary (see BufferMemoryDescBytes).  The mock does the same so a
  // test exercises the real stamping path rather than pre-tagging for it.
  std::vector<BufferMemoryDescBytes> AllBufferDescsLocked() const {
    std::vector<BufferMemoryDescBytes> out;
    out.reserve(published_buffers_);
    for (uint32_t i = 0; i < published_buffers_; ++i) {
      out.push_back(
          {i, std::vector<uint8_t>{desc_tag_, static_cast<uint8_t>(i)}, /*backend_id=*/0});
    }
    return out;
  }

  uint32_t published_buffers_ = 0;
  uint8_t desc_tag_ = 0;

  struct PendingSlot {
    std::string key;
    std::vector<uint8_t> bytes;
  };

  // Mirrors PageBackend::QueueEventLocked: the hook fires under the lock, so
  // the callback must not re-enter.  Caller MUST hold `mutex_`.
  void QueueEventLocked(KvEvent event) {
    pending_events_.push_back(std::move(event));
    if (auto_flush_cb_ && pending_events_.size() >= auto_flush_threshold_) auto_flush_cb_();
  }

  std::vector<KvEvent> SnapshotLocked() const {
    std::vector<KvEvent> out;
    out.reserve(owned_.size());
    for (const auto& [key, bytes] : owned_) {
      out.push_back(KvEvent{KvEvent::Kind::ADD, key, tier_, bytes.size()});
    }
    return out;
  }

  TierType tier_;
  uint64_t page_size_ = 1;
  bool initialized_ = false;
  mutable std::mutex mutex_;
  uint64_t next_slot_id_ = 1;
  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, std::vector<uint8_t>> owned_;
  std::vector<KvEvent> pending_events_;
  size_t auto_flush_threshold_ = SIZE_MAX;  // events before a flush; default = never
  std::function<void()> auto_flush_cb_;
};

}  // namespace mori::umbp
