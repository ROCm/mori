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
#include "umbp/storage/segment/segment_index.h"

#include "umbp/storage/segment/segment_format.h"

namespace segment {

void Index::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void Index::RemoveLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it == lru_map_.end()) return;
  lru_list_.erase(it->second);
  lru_map_.erase(it);
}

bool Index::PrepareWrite(const std::string& key, size_t size, size_t key_len, uint32_t crc32,
                         Meta* seg, WriteReservation* out) {
  if (!seg || !out) return false;

  auto existing = key_meta_.find(key);
  size_t existing_size = (existing == key_meta_.end()) ? 0 : existing->second.size;
  size_t effective_used = (used_ >= existing_size) ? (used_ - existing_size) : 0;
  if (effective_used + size > capacity_) return false;

  const uint64_t generation = generation_counter_++;
  out->key = key;
  out->had_previous = (existing != key_meta_.end());
  if (out->had_previous) out->previous_meta = existing->second;

  const size_t record_size = sizeof(RecordHeader) + key_len + size;
  out->record_offset = seg->write_offset;
  out->previous_write_offset = seg->write_offset;
  seg->write_offset += static_cast<uint64_t>(record_size);

  out->meta.segment_id = seg->id;
  out->meta.value_offset = out->record_offset + sizeof(RecordHeader) + key_len;
  out->meta.size = static_cast<uint32_t>(size);
  out->meta.crc32 = crc32;
  out->meta.generation = generation;

  if (existing != key_meta_.end()) {
    auto old_seg = segments_.find(existing->second.segment_id);
    if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.size) {
      old_seg->second.live_bytes -= existing->second.size;
    }
    used_ -= existing->second.size;
  }

  seg->live_bytes += size;
  key_meta_[key] = out->meta;
  used_ += size;
  TouchLRU(key);
  return true;
}

void Index::RollbackWrite(const WriteReservation& reservation) {
  auto it = key_meta_.find(reservation.key);
  if (it != key_meta_.end() && it->second.generation == reservation.meta.generation) {
    auto seg = segments_.find(reservation.meta.segment_id);
    if (seg != segments_.end()) {
      if (seg->second.live_bytes >= reservation.meta.size) {
        seg->second.live_bytes -= reservation.meta.size;
      }
      seg->second.write_offset = reservation.previous_write_offset;
    }
    if (used_ >= reservation.meta.size) used_ -= reservation.meta.size;
    key_meta_.erase(it);
    RemoveLRU(reservation.key);
  }

  if (reservation.had_previous) {
    key_meta_[reservation.key] = reservation.previous_meta;
    auto old_seg = segments_.find(reservation.previous_meta.segment_id);
    if (old_seg != segments_.end()) old_seg->second.live_bytes += reservation.previous_meta.size;
    used_ += reservation.previous_meta.size;
    TouchLRU(reservation.key);
  }
}

void Index::RecordRecoveredEntry(const std::string& key, const KeyMeta& meta) {
  auto existing = key_meta_.find(key);
  if (existing != key_meta_.end()) {
    if (existing->second.generation >= meta.generation) return;
    auto old_seg = segments_.find(existing->second.segment_id);
    if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.size) {
      old_seg->second.live_bytes -= existing->second.size;
    }
    if (used_ >= existing->second.size) used_ -= existing->second.size;
  }

  auto seg = segments_.find(meta.segment_id);
  if (seg != segments_.end()) seg->second.live_bytes += meta.size;
  key_meta_[key] = meta;
  used_ += meta.size;
  TouchLRU(key);
  ObserveGeneration(meta.generation);
}

bool Index::EraseKey(const std::string& key) {
  auto it = key_meta_.find(key);
  if (it == key_meta_.end()) return false;

  auto seg = segments_.find(it->second.segment_id);
  if (seg != segments_.end() && seg->second.live_bytes >= it->second.size) {
    seg->second.live_bytes -= it->second.size;
  }
  if (used_ >= it->second.size) used_ -= it->second.size;
  key_meta_.erase(it);
  RemoveLRU(key);
  return true;
}

void Index::ResetMetadata() {
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
  known_segment_ids_.clear();
  next_segment_id_ = 0;
  active_segment_id_ = 0;
}

}  // namespace segment
