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
#include "umbp/sim/ssd.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "umbp/common/range_utils.h"
#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/segment/segment_index.h"
#include "umbp/local/tiers/sharded_ssd_tier.h"
#include "umbp/sim/sim.h"

namespace mori::umbp {
namespace {

// Reuse the real segment index; only payload I/O and file identities are virtual.
class SimSsdTier final : public TierBackend {
 public:
  SimSsdTier(UMBPSsdConfig cfg, SimTransferEngine& engine)
      : TierBackend(StorageTier::LOCAL_SSD),
        cfg_(std::move(cfg)),
        engine_(engine),
        index_(cfg_.capacity_bytes),
        segment_size_(cfg_.segment_size_bytes) {
    std::string error;
    if (!cfg_.Validate(&error)) throw std::invalid_argument(error);
    const uint64_t slice = (cfg_.capacity_bytes / 8) & ~(segment::kRecordAlign - 1);
    if (slice >= segment::kRecordAlign) segment_size_ = std::min(segment_size_, slice);
    if (!NewSegment()) throw std::runtime_error("sim SSD file identities exhausted");
  }

  bool Write(const std::string& key, const void* data, size_t size) override {
    return BatchWrite({key}, {data}, {size})[0];
  }
  bool WriteBatch(const std::vector<std::string>& keys, const std::vector<const void*>& data,
                  const std::vector<size_t>& sizes) override {
    const auto ok = BatchWrite(keys, data, sizes);
    return std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
  }
  std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data,
                               const std::vector<size_t>& sizes) override {
    std::vector<bool> out(keys.size(), false);
    if (keys.empty() || data.size() != keys.size() || sizes.size() != keys.size()) return out;
    uint64_t total = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!ValidRecord(keys[i], sizes[i]) ||
          !AddBytes(segment::RecordBytes(keys[i].size(), sizes[i]), &total))
        return out;
    }
    // Match SSDTier's fallback when the batch spans more than one segment.
    if (keys.size() > 1 && total > segment_size_) {
      for (size_t i = 0; i < keys.size(); ++i) out[i] = Write(keys[i], data[i], sizes[i]);
      return out;
    }
    std::vector<Pending> writes;
    std::vector<size_t> positions;
    std::vector<int> retired;
    uint64_t bytes = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (EnsureActive(total, &retired)) {
        for (size_t i = 0; i < keys.size(); ++i) {
          Pending pending;
          if (!Prepare(keys[i], sizes[i], &pending)) continue;
          bytes += pending.write.meta.disk_bytes;
          writes.push_back(std::move(pending));
          positions.push_back(i);
        }
      }
    }
    Retire(retired);
    if (writes.empty()) return out;
    const bool ok = engine_.SimulateStorage(false, bytes, writes.size());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ok)
        for (auto it = writes.rbegin(); it != writes.rend(); ++it) index_.RollbackWrite(it->write);
      for (const auto& pending : writes) Release(pending, &retired);
    }
    Retire(retired);
    if (ok)
      for (size_t i : positions) out[i] = true;
    return out;
  }

  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    return ReadBatchIntoPtr({key}, {dst}, {size})[0];
  }
  std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dsts,
                                     const std::vector<size_t>& sizes) override {
    return ReadBatchIntoPtr(keys, dsts, sizes);
  }
  std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dsts,
                                     const std::vector<size_t>& sizes) override {
    if (dsts.size() != keys.size() || sizes.size() != keys.size())
      return std::vector<bool>(keys.size(), false);
    std::vector<std::vector<uintptr_t>> ptrs;
    std::vector<std::vector<size_t>> lengths, offsets(keys.size(), {0});
    for (size_t i = 0; i < keys.size(); ++i) {
      ptrs.push_back({dsts[i]});
      lengths.push_back({sizes[i]});
    }
    return ReadRanges(keys, ptrs, lengths, offsets, true);
  }
  std::vector<bool> ReadBatchRangesIntoPtr(
      const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dsts,
      const std::vector<std::vector<size_t>>& sizes,
      const std::vector<std::vector<size_t>>& offsets) override {
    return ReadRanges(keys, dsts, sizes, offsets, false);
  }
  std::vector<bool> ReadRanges(const std::vector<std::string>& keys,
                               const std::vector<std::vector<uintptr_t>>& dsts,
                               const std::vector<std::vector<size_t>>& sizes,
                               const std::vector<std::vector<size_t>>& offsets, bool require_full) {
    std::vector<bool> out(keys.size(), false);
    if (!RangeBatchShapeValid(keys.size(), dsts, sizes, offsets)) return out;
    std::vector<uint64_t> pinned;
    uint64_t bytes = 0;
    size_t operations = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < keys.size(); ++i) {
        const auto* meta = index_.FindKey(keys[i]);
        if (!meta || sizes[i].empty()) continue;
        if (require_full && sizes[i][0] != meta->size) continue;
        index_.TouchLRU(keys[i]);
        bool valid = true;
        for (size_t j = 0; j < sizes[i].size(); ++j)
          if (!dsts[i][j] || !sizes[i][j] ||
              IsObjectRangeOverflow(offsets[i][j], sizes[i][j], meta->size))
            valid = false;
        if (!valid) continue;
        std::vector<size_t> order(sizes[i].size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return offsets[i][a] < offsets[i][b]; });
        // Consecutive ranges share a device read; overlaps remain separate runs.
        uint64_t key_bytes = 0;
        size_t runs = 0;
        for (size_t start = 0; start < order.size();) {
          size_t end = start + 1;
          uint64_t begin = offsets[i][order[start]];
          uint64_t finish = begin + sizes[i][order[start]];
          while (end < order.size() && offsets[i][order[end]] == finish)
            finish += sizes[i][order[end++]];
          if (cfg_.direct_io) {
            begin &= ~(segment::kRecordAlign - 1);
            finish = segment::AlignUp(finish);
          }
          valid = AddBytes(finish - begin, &key_bytes) && valid;
          ++runs;
          start = end;
        }
        if (!valid || !AddBytes(key_bytes, &bytes)) continue;
        operations += runs;
        index_.PinSegment(meta->segment_id);
        pinned.push_back(meta->segment_id);
        out[i] = true;
      }
    }
    if (pinned.empty()) return out;
    const bool ok = engine_.SimulateStorage(true, bytes, operations);
    std::vector<int> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (uint64_t id : pinned) index_.UnpinSegment(id);
      for (uint64_t id : pinned) Reclaim(id, &retired);
    }
    Retire(retired);
    if (!ok) std::fill(out.begin(), out.end(), false);
    return out;
  }

  bool Exists(const std::string& key) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.HasKey(key);
  }
  bool Evict(const std::string& key) override {
    std::vector<int> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto* meta = index_.FindKey(key);
      if (!meta) return false;
      const uint64_t id = meta->segment_id;
      index_.EraseKey(key);
      Reclaim(id, &retired);
    }
    Retire(retired);
    return true;
  }
  std::pair<size_t, size_t> Capacity() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.Capacity();
  }
  TierCapabilities Capabilities() const override { return {false, true, true, true, true}; }
  std::string GetLRUKey() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.GetLRUKey();
  }
  std::vector<std::string> GetLRUCandidates(size_t count) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.GetLRUCandidates(count);
  }
  std::vector<EvictionHint> EvictionOrder(size_t count) override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<EvictionHint> out;
    for (auto& [key, bytes] : index_.EvictionOrder(count)) out.push_back({std::move(key), bytes});
    return out;
  }
  TierSpaceStats SpaceStats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return {index_.live_bytes(), index_.Segments().size(), deleted_files_, deleted_bytes_, 0};
  }
  void SetRetiredFileHook(std::function<void(int)> hook, std::chrono::milliseconds) override {
    std::lock_guard<std::mutex> lock(mutex_);
    retired_hook_ = std::move(hook);
  }
  void Clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto next_segment = index_.next_segment_id();
    index_.ResetAll();
    index_.AdvanceNextSegmentId(next_segment);
    index_.ObserveGeneration(last_generation_);
    pending_.clear();
    NewSegment();  // Callers quiesce I/O and clear cached file refs before Clear.
  }
  std::optional<RecordLocation> LocateRecord(const std::string& key) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* meta = index_.FindKey(key);
    if (!meta) return std::nullopt;
    index_.TouchLRU(key);
    return Location(*meta);
  }
  std::optional<RecordReservation> ReserveRecord(const std::string& key, size_t size) override {
    if (!size || !ValidRecord(key, size)) return std::nullopt;
    Pending pending;
    std::vector<int> retired;
    bool prepared;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      prepared = EnsureActive(segment::RecordBytes(key.size(), size), &retired) &&
                 Prepare(key, size, &pending);
    }
    Retire(retired);
    if (!prepared) return std::nullopt;
    const bool ok = engine_.SimulateStorage(false, segment::PrefixBytes(key.size()));
    std::optional<RecordReservation> out;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto* current = index_.FindKey(key);
      if (ok && current && current->generation == pending.write.meta.generation &&
          Location(*current).fd == pending.location.fd) {
        const auto token = pending.write.meta.generation;
        out = RecordReservation{pending.location, token};
        pending_.emplace(token, std::move(pending));
      } else {
        index_.RollbackWrite(pending.write);
        Release(pending, &retired);
      }
    }
    Retire(retired);
    return out;
  }
  bool CommitRecord(const std::string& key, const RecordReservation& r) override {
    Pending pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = pending_.find(r.token);
      if (it == pending_.end() || !Matches(it->second, key, r)) return false;
      pending = std::move(it->second);
      pending_.erase(it);
    }
    const bool ok = engine_.SimulateStorage(false, segment::PrefixBytes(key.size()));
    std::vector<int> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ok) index_.RollbackWrite(pending.write);
      Release(pending, &retired);
    }
    Retire(retired);
    return ok;
  }
  void AbortRecord(const std::string& key, const RecordReservation& r) override {
    std::vector<int> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = pending_.find(r.token);
      if (it == pending_.end() || !Matches(it->second, key, r)) return;
      index_.RollbackWrite(it->second.write);
      Release(it->second, &retired);
      pending_.erase(it);
    }
    Retire(retired);
  }

 private:
  struct Pending {
    segment::WriteReservation write;
    RecordLocation location;
    std::vector<uint64_t> pins;
  };
  static bool ValidRecord(const std::string& key, size_t size) {
    return key.size() <= UINT32_MAX && size <= UINT32_MAX &&
           segment::RecordBytes(key.size(), size) <= UINT32_MAX;
  }
  static bool AddBytes(uint64_t bytes, uint64_t* total) {
    if (bytes > std::numeric_limits<uint64_t>::max() - *total) return false;
    *total += bytes;
    return true;
  }
  bool NewSegment() {
    static std::atomic<uint64_t> next_fd{1};
    const uint64_t fd = next_fd.fetch_add(1);
    if (fd > INT32_MAX) return false;
    segment::Meta seg;
    seg.id = index_.next_segment_id();
    seg.fd = static_cast<int>(fd);  // Opaque identity, never opened/closed or reused.
    index_.InsertSegment(seg);
    index_.set_active_segment_id(seg.id);
    index_.AdvanceNextSegmentId(seg.id + 1);
    return true;
  }
  bool EnsureActive(uint64_t bytes, std::vector<int>* retired) {
    const auto* seg = index_.FindSegment(index_.active_segment_id());
    if (seg && (seg->write_offset == 0 ||
                (seg->write_offset <= segment_size_ && bytes <= segment_size_ - seg->write_offset)))
      return true;
    const uint64_t old = index_.active_segment_id();
    if (!NewSegment()) return false;
    Reclaim(old, retired);
    return true;
  }
  bool Prepare(const std::string& key, size_t size, Pending* pending) {
    const auto [used, capacity] = index_.Capacity();
    if (segment::RecordBytes(key.size(), size) > capacity - used) return false;
    auto* seg = index_.FindSegment(index_.active_segment_id());
    if (!seg || !index_.PrepareWrite(key, size, key.size(), 0, seg, &pending->write, false))
      return false;
    last_generation_ = pending->write.meta.generation;
    pending->location = Location(pending->write.meta);
    pending->pins.push_back(seg->id);
    if (pending->write.had_previous)
      pending->pins.push_back(pending->write.previous_meta.segment_id);
    for (uint64_t id : pending->pins) index_.PinSegment(id);
    return true;
  }
  RecordLocation Location(const segment::KeyMeta& meta) const {
    const auto* seg = index_.FindSegment(meta.segment_id);
    return {seg ? seg->fd : -1, meta.value_offset, segment::PaddedValueBytes(meta.size), meta.size,
            cfg_.direct_io};
  }
  bool Matches(const Pending& pending, const std::string& key,
               const RecordReservation& reservation) const {
    return pending.write.key == key && pending.location.fd == reservation.location.fd;
  }
  void Reclaim(uint64_t id, std::vector<int>* retired) {
    if (id == index_.active_segment_id() && index_.ActiveSegmentEmptied()) NewSegment();
    segment::Meta seg;
    if (!index_.TakeSegment(id, &seg)) return;
    ++deleted_files_;
    deleted_bytes_ += seg.write_offset;
    retired->push_back(seg.fd);
  }
  void Release(const Pending& pending, std::vector<int>* retired) {
    for (uint64_t id : pending.pins) index_.UnpinSegment(id);
    for (uint64_t id : pending.pins) Reclaim(id, retired);
  }
  void Retire(std::vector<int>& retired) {
    if (retired.empty()) return;
    std::function<void(int)> hook;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hook = retired_hook_;
    }
    // No fd reuse or physical close: outstanding virtual file refs remain valid.
    if (hook)
      for (int fd : retired) hook(fd);
    retired.clear();
  }
  UMBPSsdConfig cfg_;
  SimTransferEngine& engine_;
  mutable std::mutex mutex_;
  segment::Index index_;
  uint64_t segment_size_;
  std::unordered_map<uint64_t, Pending> pending_;
  std::function<void(int)> retired_hook_;
  uint64_t last_generation_ = 0;
  uint64_t deleted_files_ = 0, deleted_bytes_ = 0;
};
}  // namespace

std::unique_ptr<MediumBackend> MakeSimSsdBackend(SsdBackend::Config cfg,
                                                 SimTransferEngine& engine) {
  if (!cfg.ssd.enabled || !cfg.ssd.ssd.enabled || cfg.ssd.ssd.ssd_backend != "file" ||
      cfg.ssd.ssd.layout_mode != UMBPSsdLayoutMode::SegmentedLog) {
    throw std::invalid_argument("sim SSD requires an enabled segmented file backend");
  }
  std::vector<std::unique_ptr<TierBackend>> shards;
  const auto dirs = cfg.ssd.ssd.StorageDirs();
  for (size_t i = 0; i < dirs.size(); ++i) {
    auto shard_cfg = cfg.ssd.ssd;
    shard_cfg.capacity_bytes = cfg.ssd.ssd.capacity_bytes / dirs.size();
    if (i == 0) shard_cfg.capacity_bytes += cfg.ssd.ssd.capacity_bytes % dirs.size();
    shards.push_back(std::make_unique<SimSsdTier>(std::move(shard_cfg), engine));
  }
  std::unique_ptr<TierBackend> tier;
  if (shards.size() == 1)
    tier = std::move(shards[0]);
  else
    tier = std::make_unique<ShardedSsdTier>(std::move(shards), cfg.ssd.ssd.shard_io_threads);
  auto manager = std::make_unique<PeerSsdManager>(
      std::move(tier), cfg.ssd.ssd.single_flight_reads,
      [&engine](void* dst, const void* src, size_t size) {
        std::vector<size_t> failed;
        return engine.Transfer({{TransferRef::HostBytes(const_cast<void*>(src), size), 0,
                                 TransferRef::HostBytes(dst, size), 0, size, 0}},
                               &failed);
      });
  return std::make_unique<SsdBackend>(
      std::move(cfg), std::move(manager),
      std::make_unique<SimPageMemorySource>(mori::io::MemoryLocationType::CPU, -1));
}
}  // namespace mori::umbp
