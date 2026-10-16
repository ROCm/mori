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
#include "umbp/local/tiers/segment/segment_index.h"

#include <cmath>

#include "umbp/local/tiers/segment/segment_format.h"

namespace mori::umbp::segment {

void Index::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();

  auto meta = key_meta_.find(key);
  if (meta == key_meta_.end()) return;
  const uint64_t tick = ++access_tick_;
  meta->second.last_access = tick;
  auto seg = segments_.find(meta->second.segment_id);
  if (seg != segments_.end()) seg->second.last_access = tick;
}

void Index::RemoveLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it == lru_map_.end()) return;
  lru_list_.erase(it->second);
  lru_map_.erase(it);
}

void Index::LinkKey(const KeyEntry* entry) {
  segment_keys_[entry->second.segment_id].insert(entry);
}

void Index::UnlinkKey(const KeyEntry* entry) {
  auto it = segment_keys_.find(entry->second.segment_id);
  if (it == segment_keys_.end()) return;
  it->second.erase(entry);
  if (it->second.empty()) segment_keys_.erase(it);
}

bool Index::PrepareWrite(const std::string& key, size_t size, size_t key_len, uint32_t crc32,
                         Meta* seg, WriteReservation* out, bool crc_valid) {
  if (!seg || !out) return false;

  // Admission is decided in on-disk bytes, the padded footprint, because that
  // is what capacity_bytes is meant to bound.  Every record is appended, so a
  // write grows the disk by its whole footprint whatever it replaces, and a
  // dead record keeps its bytes until its segment is deleted.  Charging live
  // bytes only is what let the tier outgrow its capacity on disk (#755).
  const size_t on_disk = static_cast<size_t>(RecordBytes(key_len, size));
  if (disk_used_ + on_disk > capacity_) return false;

  const uint64_t generation = generation_counter_++;
  auto existing = key_meta_.find(key);
  out->key = key;
  out->had_previous = (existing != key_meta_.end());
  if (out->had_previous) out->previous_meta = existing->second;

  // v3 layout: the append cursor is always on a kRecordAlign boundary, so
  // starting the record at write_offset already gives an aligned record_offset;
  // the padded prefix then puts the value on an aligned boundary too.  Both are
  // preconditions for reading or writing this record with O_DIRECT.
  out->record_offset = seg->write_offset;
  out->previous_write_offset = seg->write_offset;
  seg->write_offset += on_disk;
  disk_used_ += on_disk;

  out->meta.segment_id = seg->id;
  out->meta.value_offset = out->record_offset + PrefixBytes(key_len);
  out->meta.size = static_cast<uint32_t>(size);
  out->meta.crc32 = crc32;
  out->meta.generation = generation;
  out->meta.crc_valid = crc_valid;
  out->meta.disk_bytes = static_cast<uint32_t>(on_disk);

  if (existing != key_meta_.end()) {
    auto old_seg = segments_.find(existing->second.segment_id);
    if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.disk_bytes) {
      old_seg->second.live_bytes -= existing->second.disk_bytes;
    }
    used_ -= existing->second.disk_bytes;
    UnlinkKey(&*existing);
  }

  seg->live_bytes += on_disk;
  auto entry = key_meta_.insert_or_assign(key, out->meta).first;
  LinkKey(&*entry);
  used_ += on_disk;
  TouchLRU(key);
  return true;
}

void Index::RollbackWrite(const WriteReservation& reservation) {
  auto it = key_meta_.find(reservation.key);
  // Only this reservation's own record is undone.  If the key has moved on --
  // evicted, or rewritten by a later write -- its current state is newer than
  // anything this reservation knew, so there is nothing to restore either.
  if (it == key_meta_.end() || it->second.generation != reservation.meta.generation) return;

  auto seg = segments_.find(reservation.meta.segment_id);
  if (seg != segments_.end()) {
    if (seg->second.live_bytes >= reservation.meta.disk_bytes) {
      seg->second.live_bytes -= reservation.meta.disk_bytes;
    }
    // Give the bytes back only if nothing was reserved after this record.
    // Rewinding past a later reservation would hand its bytes to the next
    // writer; the hole instead stays dead until the segment is deleted.
    const uint64_t end = reservation.record_offset + reservation.meta.disk_bytes;
    if (seg->second.write_offset == end && end >= reservation.previous_write_offset) {
      const uint64_t released = end - reservation.previous_write_offset;
      disk_used_ = disk_used_ >= released ? disk_used_ - released : 0;
      seg->second.write_offset = reservation.previous_write_offset;
    }
  }
  if (used_ >= reservation.meta.disk_bytes) used_ -= reservation.meta.disk_bytes;
  UnlinkKey(&*it);
  key_meta_.erase(it);
  RemoveLRU(reservation.key);

  if (!reservation.had_previous) return;
  // The previous record's segment is pinned for the write's duration, so it is
  // still there; without it the key simply stays dropped.
  auto old_seg = segments_.find(reservation.previous_meta.segment_id);
  if (old_seg == segments_.end()) return;
  auto entry = key_meta_.insert_or_assign(reservation.key, reservation.previous_meta).first;
  LinkKey(&*entry);
  old_seg->second.live_bytes += reservation.previous_meta.disk_bytes;
  used_ += reservation.previous_meta.disk_bytes;
  TouchLRU(reservation.key);
}

void Index::RecordRecoveredEntry(const std::string& key, const KeyMeta& meta) {
  auto existing = key_meta_.find(key);
  if (existing != key_meta_.end()) {
    if (existing->second.generation >= meta.generation) return;
    auto old_seg = segments_.find(existing->second.segment_id);
    if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.disk_bytes) {
      old_seg->second.live_bytes -= existing->second.disk_bytes;
    }
    if (used_ >= existing->second.disk_bytes) used_ -= existing->second.disk_bytes;
    UnlinkKey(&*existing);
  }

  auto seg = segments_.find(meta.segment_id);
  if (seg != segments_.end()) seg->second.live_bytes += meta.disk_bytes;
  auto entry = key_meta_.insert_or_assign(key, meta).first;
  LinkKey(&*entry);
  used_ += meta.disk_bytes;
  TouchLRU(key);
  ObserveGeneration(meta.generation);
}

bool Index::EraseKey(const std::string& key) {
  auto it = key_meta_.find(key);
  if (it == key_meta_.end()) return false;

  auto seg = segments_.find(it->second.segment_id);
  if (seg != segments_.end() && seg->second.live_bytes >= it->second.disk_bytes) {
    seg->second.live_bytes -= it->second.disk_bytes;
  }
  if (used_ >= it->second.disk_bytes) used_ -= it->second.disk_bytes;
  UnlinkKey(&*it);
  key_meta_.erase(it);
  RemoveLRU(key);
  return true;
}

void Index::ResetMetadata() {
  segment_keys_.clear();
  key_meta_.clear();
  lru_list_.clear();
  lru_map_.clear();
  used_ = 0;
  generation_counter_ = 1;
  for (auto& kv : segments_) {
    kv.second.scanned_offset = 0;
    kv.second.live_bytes = 0;
  }
}

void Index::ResetAll() {
  ResetMetadata();
  segments_.clear();
  disk_used_ = 0;
  known_segment_ids_.clear();
  next_segment_id_ = 0;
  active_segment_id_ = 0;
}

void Index::InsertSegment(const Meta& seg) {
  auto it = segments_.find(seg.id);
  if (it != segments_.end()) {
    disk_used_ = disk_used_ >= it->second.write_offset ? disk_used_ - it->second.write_offset : 0;
    it->second = seg;
  } else {
    segments_.emplace(seg.id, seg);
  }
  disk_used_ += seg.write_offset;
}

void Index::RecomputeDiskBytes() {
  disk_used_ = 0;
  for (const auto& kv : segments_) disk_used_ += kv.second.write_offset;
}

void Index::PinSegment(uint64_t segment_id) {
  auto it = segments_.find(segment_id);
  if (it != segments_.end()) ++it->second.io_refs;
}

void Index::UnpinSegment(uint64_t segment_id) {
  auto it = segments_.find(segment_id);
  if (it != segments_.end() && it->second.io_refs > 0) --it->second.io_refs;
}

bool Index::Reclaimable(uint64_t segment_id) const {
  if (segment_id == active_segment_id_) return false;
  auto it = segments_.find(segment_id);
  if (it == segments_.end() || it->second.io_refs != 0) return false;
  return segment_keys_.count(segment_id) == 0;
}

bool Index::ActiveSegmentEmptied() const {
  auto it = segments_.find(active_segment_id_);
  if (it == segments_.end()) return false;
  const Meta& seg = it->second;
  return seg.write_offset > 0 && seg.io_refs == 0 && segment_keys_.count(seg.id) == 0;
}

bool Index::TakeSegment(uint64_t segment_id, Meta* out) {
  if (!out || !Reclaimable(segment_id)) return false;
  auto it = segments_.find(segment_id);
  *out = std::move(it->second);
  disk_used_ = disk_used_ >= out->write_offset ? disk_used_ - out->write_offset : 0;
  segments_.erase(it);
  known_segment_ids_.erase(segment_id);
  return true;
}

std::vector<std::pair<std::string, uint64_t>> Index::EvictionOrder(size_t max_keys) const {
  std::vector<std::pair<std::string, uint64_t>> out;
  if (max_keys == 0) return out;

  struct Ranked {
    const Meta* seg;
    const std::unordered_set<const KeyEntry*>* keys;
    double score;
  };
  std::vector<Ranked> ranked;
  ranked.reserve(segment_keys_.size());
  const Meta* active = nullptr;
  const std::unordered_set<const KeyEntry*>* active_keys = nullptr;
  for (const auto& [segment_id, keys] : segment_keys_) {
    auto seg_it = segments_.find(segment_id);
    if (seg_it == segments_.end()) continue;
    const Meta& seg = seg_it->second;
    if (segment_id == active_segment_id_) {
      active = &seg;
      active_keys = &keys;
      continue;
    }
    const double live = static_cast<double>(std::max<size_t>(seg.live_bytes, 1));
    const double fraction =
        seg.write_offset > 0 ? std::min(1.0, live / static_cast<double>(seg.write_offset)) : 1.0;
    const double age =
        static_cast<double>(access_tick_ - std::min(seg.last_access, access_tick_)) + 1.0;
    ranked.push_back({&seg, &keys, age / fraction});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.seg->id < b.seg->id;
  });
  if (active != nullptr) ranked.push_back({active, active_keys, 0.0});

  std::vector<const KeyEntry*> picks;
  for (const auto& r : ranked) {
    if (out.size() >= max_keys) break;
    picks.assign(r.keys->begin(), r.keys->end());
    const size_t take = std::min(picks.size(), max_keys - out.size());
    std::partial_sort(picks.begin(), picks.begin() + static_cast<std::ptrdiff_t>(take), picks.end(),
                      [](const KeyEntry* a, const KeyEntry* b) {
                        if (a->second.last_access != b->second.last_access) {
                          return a->second.last_access < b->second.last_access;
                        }
                        return a->first < b->first;
                      });
    // Dead records are paid for by whoever empties the segment, so each live
    // key carries its share: live disk_bytes scaled up to the file's size.
    const double scale = r.seg->live_bytes > 0
                             ? std::max(1.0, static_cast<double>(r.seg->write_offset) /
                                                 static_cast<double>(r.seg->live_bytes))
                             : 1.0;
    for (size_t i = 0; i < take; ++i) {
      const KeyEntry* e = picks[i];
      const auto credit =
          static_cast<uint64_t>(std::llround(static_cast<double>(e->second.disk_bytes) * scale));
      out.emplace_back(e->first, credit);
    }
  }
  return out;
}

}  // namespace mori::umbp::segment
