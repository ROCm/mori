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
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/storage/io_backend.h"
#include "umbp/storage/tier_backend.h"

enum class SegmentedSsdAccessMode : int {
  ReadWrite = 0,
  ReadOnlyShared = 1,
};

class SegmentedSsdTier : public TierBackend {
 public:
  SegmentedSsdTier(const std::string& dir, size_t capacity, const UMBPConfig& config,
                   SegmentedSsdAccessMode access_mode = SegmentedSsdAccessMode::ReadWrite);
  ~SegmentedSsdTier() override;

  SegmentedSsdTier(const SegmentedSsdTier&) = delete;
  SegmentedSsdTier& operator=(const SegmentedSsdTier&) = delete;

  bool Write(const std::string& key, const void* data, size_t size) override;
  bool WriteBatch(const std::vector<std::string>& keys, const std::vector<const void*>& data_ptrs,
                  const std::vector<size_t>& sizes);
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  std::vector<char> Read(const std::string& key) override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

 private:
  struct SegmentRecordHeader {
    uint32_t magic = 0;
    uint16_t version = 1;
    uint16_t flags = 0;
    uint32_t key_len = 0;
    uint32_t value_size = 0;
    uint32_t crc32 = 0;
    uint32_t reserved = 0;  // explicit padding for uint64_t alignment
    uint64_t generation = 0;
  };
  static_assert(sizeof(SegmentRecordHeader) == 32, "unexpected padding in SegmentRecordHeader");

  struct KeyMeta {
    uint64_t segment_id = 0;
    uint64_t value_offset = 0;
    uint32_t size = 0;
    uint32_t crc32 = 0;
    uint64_t generation = 0;
  };

  struct SegmentMeta {
    uint64_t id = 0;
    std::string path;
    int fd = -1;
    uint64_t write_offset = 0;
    uint64_t scanned_offset = 0;
    size_t live_bytes = 0;
  };

  bool IsReadOnlyShared() const { return access_mode_ == SegmentedSsdAccessMode::ReadOnlyShared; }
  bool ShouldSyncOnWrite() const {
    return config_.ssd_durability_mode == UMBPDurabilityMode::Strict;
  }

  bool EnsureActiveSegment(size_t need_bytes);
  bool AppendRecord(const std::string& key, const void* data, size_t size, KeyMeta* out_meta);
  bool ParseSegmentFromOffset(SegmentMeta* seg);
  bool RefreshFromDiskLocked(bool force_full_rescan);
  bool LoadSegmentsLocked(bool force_full_rescan);
  bool OpenOrCreateSegmentLocked(uint64_t segment_id);

  // Const-safe wrapper for follower refresh, localizes const_cast.
  bool RefreshFollowerLocked() const;
  SegmentMeta* GetSegmentLocked(uint64_t segment_id);
  const SegmentMeta* GetSegmentLocked(uint64_t segment_id) const;
  void TouchLRULocked(const std::string& key);
  void RemoveLRULocked(const std::string& key);
  uint32_t CrcUpdate(const void* data, size_t size, uint32_t crc = 0xFFFFFFFFu) const;
  uint32_t ComputeRecordCrc32(const std::string& key, const void* value, size_t value_size) const;

  std::string dir_;
  size_t capacity_;
  mutable size_t used_;
  UMBPConfig config_;
  SegmentedSsdAccessMode access_mode_;

  mutable std::mutex mu_;     // protects metadata: key_meta_, segments_, LRU, used_, etc.
  mutable std::mutex io_mu_;  // protects IoBackend calls when backend is not thread-safe (io_uring)
  bool needs_io_lock_ = false;  // true when IoUringBackend is in use
  mutable std::unordered_map<std::string, KeyMeta> key_meta_;
  mutable std::unordered_map<uint64_t, SegmentMeta> segments_;
  mutable std::unordered_set<uint64_t> known_segment_ids_;
  mutable uint64_t next_segment_id_ = 0;
  mutable uint64_t active_segment_id_ = 0;
  mutable uint64_t generation_counter_ = 1;
  mutable std::list<std::string> lru_list_;
  mutable std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;
  mutable std::unique_ptr<IoBackend> io_backend_;
};
