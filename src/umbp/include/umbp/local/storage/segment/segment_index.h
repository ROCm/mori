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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/local/storage/segment/segment_types.h"

namespace segment {

class Index {
 public:
  explicit Index(size_t capacity = 0) : capacity_(capacity) {}

  void set_capacity(size_t capacity) { capacity_ = capacity; }
  size_t capacity_bytes() const { return capacity_; }

  bool HasKey(const std::string& key) const { return key_meta_.count(key) > 0; }

  const KeyMeta* FindKey(const std::string& key) const {
    auto it = key_meta_.find(key);
    return (it == key_meta_.end()) ? nullptr : &it->second;
  }

  KeyMeta* FindMutableKey(const std::string& key) {
    auto it = key_meta_.find(key);
    return (it == key_meta_.end()) ? nullptr : &it->second;
  }

  Meta* FindSegment(uint64_t segment_id) {
    auto it = segments_.find(segment_id);
    return (it == segments_.end()) ? nullptr : &it->second;
  }

  const Meta* FindSegment(uint64_t segment_id) const {
    auto it = segments_.find(segment_id);
    return (it == segments_.end()) ? nullptr : &it->second;
  }

  std::unordered_map<uint64_t, Meta>& MutableSegments() { return segments_; }
  const std::unordered_map<uint64_t, Meta>& Segments() const { return segments_; }

  bool HasKnownSegment(uint64_t segment_id) const {
    return known_segment_ids_.count(segment_id) > 0;
  }
  void MarkKnownSegment(uint64_t segment_id) { known_segment_ids_.insert(segment_id); }
  void ClearKnownSegments() { known_segment_ids_.clear(); }

  uint64_t next_segment_id() const { return next_segment_id_; }
  void AdvanceNextSegmentId(uint64_t next_segment_id) {
    next_segment_id_ = std::max(next_segment_id_, next_segment_id);
  }

  uint64_t active_segment_id() const { return active_segment_id_; }
  void set_active_segment_id(uint64_t active_segment_id) { active_segment_id_ = active_segment_id; }

  void TouchLRU(const std::string& key);
  void RemoveLRU(const std::string& key);

  bool PrepareWrite(const std::string& key, size_t size, size_t key_len, uint32_t crc32, Meta* seg,
                    WriteReservation* out);
  void RollbackWrite(const WriteReservation& reservation);
  void RecordRecoveredEntry(const std::string& key, const KeyMeta& meta);
  bool EraseKey(const std::string& key);

  void ResetMetadata();
  void ResetAll();

  void ObserveGeneration(uint64_t generation) {
    generation_counter_ = std::max(generation_counter_, generation + 1);
  }

  std::pair<size_t, size_t> Capacity() const { return {used_, capacity_}; }

  std::string GetLRUKey() const {
    if (lru_list_.empty()) return "";
    return lru_list_.back();
  }

  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const {
    if (max_candidates == 0) max_candidates = 1;
    std::vector<std::string> result;
    result.reserve(std::min(max_candidates, lru_list_.size()));
    auto it = lru_list_.rbegin();
    for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
      result.push_back(*it);
    }
    return result;
  }

 private:
  size_t capacity_ = 0;
  size_t used_ = 0;
  uint64_t next_segment_id_ = 0;
  uint64_t active_segment_id_ = 0;
  uint64_t generation_counter_ = 1;
  std::unordered_map<std::string, KeyMeta> key_meta_;
  std::unordered_map<uint64_t, Meta> segments_;
  std::unordered_set<uint64_t> known_segment_ids_;
  std::list<std::string> lru_list_;
  std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;
};

}  // namespace segment
