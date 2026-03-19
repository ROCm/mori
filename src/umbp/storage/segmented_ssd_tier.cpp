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
#include "umbp/storage/segmented_ssd_tier.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr uint32_t kRecordMagic = 0x554D4250;  // "UMBP"
constexpr uint16_t kRecordVersion = 1;
constexpr uint16_t kFlagCommitted = 1;
}  // namespace

SegmentedSsdTier::SegmentedSsdTier(const std::string& dir, size_t capacity,
                                   const UMBPConfig& config, SegmentedSsdAccessMode access_mode)
    : TierBackend(StorageTier::LOCAL_SSD),
      dir_(dir),
      capacity_(capacity),
      used_(0),
      config_(config),
      access_mode_(access_mode),
      io_backend_(
          CreateIoBackend(config.ssd_io_backend, static_cast<uint32_t>(config.ssd_queue_depth))) {
  if (config.ssd_io_backend == UMBPIoBackend::IoUring &&
      dynamic_cast<IoUringBackend*>(io_backend_.get()) == nullptr) {
    throw std::runtime_error("UMBP io_uring backend requested but initialization failed");
  }
  needs_io_lock_ = (dynamic_cast<IoUringBackend*>(io_backend_.get()) != nullptr);
  fs::create_directories(dir_);
  std::lock_guard<std::mutex> lock(mu_);
  RefreshFromDiskLocked(true);
  if (!IsReadOnlyShared() && segments_.empty()) {
    OpenOrCreateSegmentLocked(0);
  }
}

SegmentedSsdTier::~SegmentedSsdTier() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : segments_) {
    if (kv.second.fd >= 0) {
      close(kv.second.fd);
      kv.second.fd = -1;
    }
  }
}

uint32_t SegmentedSsdTier::CrcUpdate(const void* data, size_t size, uint32_t crc) const {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    crc ^= static_cast<uint32_t>(p[i]);
    for (int j = 0; j < 8; ++j) {
      const uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return crc;
}

uint32_t SegmentedSsdTier::ComputeRecordCrc32(const std::string& key, const void* value,
                                              size_t value_size) const {
  uint32_t crc = CrcUpdate(key.data(), key.size());
  return ~CrcUpdate(value, value_size, crc);
}

void SegmentedSsdTier::TouchLRULocked(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void SegmentedSsdTier::RemoveLRULocked(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
    lru_map_.erase(it);
  }
}

SegmentedSsdTier::SegmentMeta* SegmentedSsdTier::GetSegmentLocked(uint64_t segment_id) {
  auto it = segments_.find(segment_id);
  if (it == segments_.end()) return nullptr;
  return &it->second;
}

const SegmentedSsdTier::SegmentMeta* SegmentedSsdTier::GetSegmentLocked(uint64_t segment_id) const {
  auto it = segments_.find(segment_id);
  if (it == segments_.end()) return nullptr;
  return &it->second;
}

bool SegmentedSsdTier::OpenOrCreateSegmentLocked(uint64_t segment_id) {
  SegmentMeta seg;
  seg.id = segment_id;
  seg.path = dir_ + "/" + BuildSegmentFileName(segment_id);

  int flags = IsReadOnlyShared() ? O_RDONLY : (O_RDWR | O_CREAT);
  seg.fd = open(seg.path.c_str(), flags, 0644);
  if (seg.fd < 0) return false;

  struct stat st;
  if (fstat(seg.fd, &st) != 0) {
    close(seg.fd);
    return false;
  }
  seg.write_offset = static_cast<uint64_t>(st.st_size);
  seg.scanned_offset = 0;
  seg.live_bytes = 0;

  segments_[segment_id] = seg;
  known_segment_ids_.insert(segment_id);
  next_segment_id_ = std::max(next_segment_id_, segment_id + 1);
  if (!IsReadOnlyShared()) {
    active_segment_id_ = std::max(active_segment_id_, segment_id);
  }
  return true;
}

bool SegmentedSsdTier::EnsureActiveSegment(size_t need_bytes) {
  SegmentMeta* seg = GetSegmentLocked(active_segment_id_);
  if (!seg) {
    if (!OpenOrCreateSegmentLocked(next_segment_id_)) return false;
    active_segment_id_ = next_segment_id_ - 1;
    seg = GetSegmentLocked(active_segment_id_);
  }
  if (!seg) return false;

  if (seg->write_offset + need_bytes <= config_.ssd_segment_size_bytes) {
    return true;
  }

  uint64_t new_id = next_segment_id_;
  if (!OpenOrCreateSegmentLocked(new_id)) return false;
  active_segment_id_ = new_id;
  return true;
}

bool SegmentedSsdTier::AppendRecord(const std::string& key, const void* data, size_t size,
                                    KeyMeta* out_meta) {
  const size_t record_size = sizeof(SegmentRecordHeader) + key.size() + size;
  if (!EnsureActiveSegment(record_size)) return false;
  SegmentMeta* seg = GetSegmentLocked(active_segment_id_);
  if (!seg || seg->fd < 0) return false;

  SegmentRecordHeader hdr;
  hdr.magic = kRecordMagic;
  hdr.version = kRecordVersion;
  hdr.flags = kFlagCommitted;
  hdr.key_len = static_cast<uint32_t>(key.size());
  hdr.value_size = static_cast<uint32_t>(size);
  hdr.crc32 = ComputeRecordCrc32(key, data, size);
  hdr.generation = generation_counter_++;

  std::vector<char> record(record_size);
  std::memcpy(record.data(), &hdr, sizeof(hdr));
  std::memcpy(record.data() + sizeof(hdr), key.data(), key.size());
  std::memcpy(record.data() + sizeof(hdr) + key.size(), data, size);

  uint64_t record_offset = seg->write_offset;
  if (!io_backend_->PWriteAll(seg->fd, record.data(), record.size(), record_offset)) return false;
  if (ShouldSyncOnWrite() && !io_backend_->Sync(seg->fd)) return false;
  seg->write_offset += static_cast<uint64_t>(record_size);

  out_meta->segment_id = seg->id;
  out_meta->value_offset = record_offset + sizeof(hdr) + key.size();
  out_meta->size = static_cast<uint32_t>(size);
  out_meta->crc32 = hdr.crc32;
  out_meta->generation = hdr.generation;
  return true;
}

bool SegmentedSsdTier::ParseSegmentFromOffset(SegmentMeta* seg) {
  if (!seg || seg->fd < 0) return false;

  struct stat st;
  if (fstat(seg->fd, &st) != 0) return false;
  uint64_t file_size = static_cast<uint64_t>(st.st_size);
  uint64_t offset = seg->scanned_offset;

  while (offset + sizeof(SegmentRecordHeader) <= file_size) {
    SegmentRecordHeader hdr;
    if (!io_backend_->PReadAll(seg->fd, &hdr, sizeof(hdr), offset)) break;
    if (hdr.magic != kRecordMagic || hdr.version != kRecordVersion || hdr.key_len == 0) break;
    const uint64_t rec_size =
        sizeof(SegmentRecordHeader) + static_cast<uint64_t>(hdr.key_len) + hdr.value_size;
    if (offset + rec_size > file_size) break;
    if ((hdr.flags & kFlagCommitted) == 0) {
      offset += rec_size;
      continue;
    }

    std::string key;
    key.resize(hdr.key_len);
    if (!io_backend_->PReadAll(seg->fd, key.data(), hdr.key_len, offset + sizeof(hdr))) break;

    KeyMeta meta;
    meta.segment_id = seg->id;
    meta.value_offset = offset + sizeof(hdr) + hdr.key_len;
    meta.size = hdr.value_size;
    meta.crc32 = hdr.crc32;
    meta.generation = hdr.generation;

    auto existing = key_meta_.find(key);
    if (existing != key_meta_.end()) {
      if (existing->second.generation < meta.generation) {
        auto old_seg = segments_.find(existing->second.segment_id);
        if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.size) {
          old_seg->second.live_bytes -= existing->second.size;
        }
        if (used_ >= existing->second.size) used_ -= existing->second.size;
        key_meta_[key] = meta;
        seg->live_bytes += meta.size;
        used_ += meta.size;
      }
    } else {
      key_meta_[key] = meta;
      seg->live_bytes += meta.size;
      used_ += meta.size;
    }
    TouchLRULocked(key);
    generation_counter_ = std::max(generation_counter_, hdr.generation + 1);
    offset += rec_size;
  }
  seg->scanned_offset = offset;
  seg->write_offset = std::max(seg->write_offset, file_size);
  return true;
}

bool SegmentedSsdTier::LoadSegmentsLocked(bool force_full_rescan) {
  if (force_full_rescan) {
    for (auto& kv : segments_) {
      if (kv.second.fd >= 0) close(kv.second.fd);
    }
    segments_.clear();
    key_meta_.clear();
    lru_list_.clear();
    lru_map_.clear();
    known_segment_ids_.clear();
    used_ = 0;
    next_segment_id_ = 0;
    active_segment_id_ = 0;
    generation_counter_ = 1;
  }

  std::vector<uint64_t> ids;
  if (fs::exists(dir_)) {
    for (const auto& entry : fs::directory_iterator(dir_)) {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (name.rfind("segment_", 0) != 0) continue;
      if (name.size() <= 12 || name.substr(name.size() - 4) != ".log") continue;
      const std::string id_s = name.substr(8, name.size() - 12);
      if (id_s.empty()) continue;
      uint64_t sid = static_cast<uint64_t>(std::stoull(id_s));
      ids.push_back(sid);
    }
  }
  std::sort(ids.begin(), ids.end());

  for (uint64_t sid : ids) {
    if (known_segment_ids_.count(sid) == 0) {
      if (!OpenOrCreateSegmentLocked(sid)) return false;
    }
  }
  return true;
}

bool SegmentedSsdTier::RefreshFromDiskLocked(bool force_full_rescan) {
  if (!LoadSegmentsLocked(force_full_rescan)) return false;
  for (auto& kv : segments_) {
    if (!ParseSegmentFromOffset(&kv.second)) return false;
  }
  return true;
}

bool SegmentedSsdTier::RefreshFollowerLocked() const {
  return const_cast<SegmentedSsdTier*>(this)->RefreshFromDiskLocked(false);
}

bool SegmentedSsdTier::Write(const std::string& key, const void* data, size_t size) {
  // Phase 1 (under mu_): validate capacity, build record, reserve offset, update metadata.
  int write_fd = -1;
  uint64_t record_offset = 0;
  std::vector<char> record;
  bool should_sync = false;
  KeyMeta meta;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (IsReadOnlyShared()) return false;

    auto existing = key_meta_.find(key);
    size_t existing_size = (existing == key_meta_.end()) ? 0 : existing->second.size;
    if (used_ - existing_size + size > capacity_) {
      return false;
    }

    // Inline AppendRecord logic: reserve offset under mu_, defer I/O.
    const size_t record_size = sizeof(SegmentRecordHeader) + key.size() + size;
    if (!EnsureActiveSegment(record_size)) return false;
    SegmentMeta* seg = GetSegmentLocked(active_segment_id_);
    if (!seg || seg->fd < 0) return false;

    SegmentRecordHeader hdr;
    hdr.magic = kRecordMagic;
    hdr.version = kRecordVersion;
    hdr.flags = kFlagCommitted;
    hdr.key_len = static_cast<uint32_t>(key.size());
    hdr.value_size = static_cast<uint32_t>(size);
    hdr.crc32 = ComputeRecordCrc32(key, data, size);
    hdr.generation = generation_counter_++;

    record.resize(record_size);
    std::memcpy(record.data(), &hdr, sizeof(hdr));
    std::memcpy(record.data() + sizeof(hdr), key.data(), key.size());
    std::memcpy(record.data() + sizeof(hdr) + key.size(), data, size);

    // Reserve the offset atomically under mu_.
    record_offset = seg->write_offset;
    seg->write_offset += static_cast<uint64_t>(record_size);
    write_fd = seg->fd;
    should_sync = ShouldSyncOnWrite();

    meta.segment_id = seg->id;
    meta.value_offset = record_offset + sizeof(hdr) + key.size();
    meta.size = static_cast<uint32_t>(size);
    meta.crc32 = hdr.crc32;
    meta.generation = hdr.generation;

    // Update metadata under mu_.
    if (existing != key_meta_.end()) {
      auto old_seg = segments_.find(existing->second.segment_id);
      if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.size) {
        old_seg->second.live_bytes -= existing->second.size;
      }
      used_ -= existing->second.size;
    }
    seg->live_bytes += meta.size;
    key_meta_[key] = meta;
    used_ += meta.size;
    TouchLRULocked(key);
  }

  // Phase 2 (under io_mu_ if needed, mu_ released): perform I/O.
  bool io_ok;
  if (needs_io_lock_) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    io_ok = io_backend_->PWriteAll(write_fd, record.data(), record.size(), record_offset);
    if (io_ok && should_sync) io_ok = io_backend_->Sync(write_fd);
  } else {
    io_ok = io_backend_->PWriteAll(write_fd, record.data(), record.size(), record_offset);
    if (io_ok && should_sync) io_ok = io_backend_->Sync(write_fd);
  }

  if (!io_ok) {
    // Rollback metadata on I/O failure.
    std::lock_guard<std::mutex> lock(mu_);
    auto it = key_meta_.find(key);
    if (it != key_meta_.end() && it->second.generation == meta.generation) {
      auto seg = segments_.find(meta.segment_id);
      if (seg != segments_.end() && seg->second.live_bytes >= meta.size) {
        seg->second.live_bytes -= meta.size;
      }
      if (used_ >= meta.size) used_ -= meta.size;
      key_meta_.erase(it);
      RemoveLRULocked(key);
    }
    return false;
  }

  return true;
}

bool SegmentedSsdTier::WriteBatch(const std::vector<std::string>& keys,
                                  const std::vector<const void*>& data_ptrs,
                                  const std::vector<size_t>& sizes) {
  if (keys.empty()) return true;

  // Phase 1 (under mu_): validate, build records, reserve offsets, update metadata.
  struct PreparedRecord {
    std::vector<char> record;
    int fd;
    uint64_t offset;
    KeyMeta meta;
    std::string key;
  };
  std::vector<PreparedRecord> prepared;
  prepared.reserve(keys.size());
  bool should_sync = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (IsReadOnlyShared()) return false;
    should_sync = ShouldSyncOnWrite();

    for (size_t i = 0; i < keys.size(); ++i) {
      const auto& key = keys[i];
      const void* data = data_ptrs[i];
      size_t size = sizes[i];

      auto existing = key_meta_.find(key);
      size_t existing_size = (existing == key_meta_.end()) ? 0 : existing->second.size;
      if (used_ - existing_size + size > capacity_) continue;

      const size_t record_size = sizeof(SegmentRecordHeader) + key.size() + size;
      if (!EnsureActiveSegment(record_size)) continue;
      SegmentMeta* seg = GetSegmentLocked(active_segment_id_);
      if (!seg || seg->fd < 0) continue;

      SegmentRecordHeader hdr;
      hdr.magic = kRecordMagic;
      hdr.version = kRecordVersion;
      hdr.flags = kFlagCommitted;
      hdr.key_len = static_cast<uint32_t>(key.size());
      hdr.value_size = static_cast<uint32_t>(size);
      hdr.crc32 = ComputeRecordCrc32(key, data, size);
      hdr.generation = generation_counter_++;

      PreparedRecord pr;
      pr.record.resize(record_size);
      std::memcpy(pr.record.data(), &hdr, sizeof(hdr));
      std::memcpy(pr.record.data() + sizeof(hdr), key.data(), key.size());
      std::memcpy(pr.record.data() + sizeof(hdr) + key.size(), data, size);

      pr.offset = seg->write_offset;
      seg->write_offset += static_cast<uint64_t>(record_size);
      pr.fd = seg->fd;
      pr.key = key;
      pr.meta.segment_id = seg->id;
      pr.meta.value_offset = pr.offset + sizeof(hdr) + key.size();
      pr.meta.size = static_cast<uint32_t>(size);
      pr.meta.crc32 = hdr.crc32;
      pr.meta.generation = hdr.generation;

      // Update metadata.
      if (existing != key_meta_.end()) {
        auto old_seg = segments_.find(existing->second.segment_id);
        if (old_seg != segments_.end() && old_seg->second.live_bytes >= existing->second.size) {
          old_seg->second.live_bytes -= existing->second.size;
        }
        used_ -= existing->second.size;
      }
      seg->live_bytes += pr.meta.size;
      key_meta_[key] = pr.meta;
      used_ += pr.meta.size;
      TouchLRULocked(key);

      prepared.push_back(std::move(pr));
    }
  }

  if (prepared.empty()) return true;

  // Phase 2 (under io_mu_ if needed): batch I/O.
  std::vector<IoOp> ops;
  ops.reserve(prepared.size());
  for (const auto& pr : prepared) {
    ops.push_back({pr.fd, pr.record.data(), pr.record.size(), pr.offset});
  }

  bool io_ok;
  if (needs_io_lock_) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    io_ok = io_backend_->PWriteBatch(ops);
    if (io_ok && should_sync) {
      // Sync unique fds.
      std::unordered_set<int> fds;
      for (const auto& op : ops) fds.insert(op.fd);
      for (int fd : fds) {
        if (!io_backend_->Sync(fd)) {
          io_ok = false;
          break;
        }
      }
    }
  } else {
    io_ok = io_backend_->PWriteBatch(ops);
    if (io_ok && should_sync) {
      std::unordered_set<int> fds;
      for (const auto& op : ops) fds.insert(op.fd);
      for (int fd : fds) {
        if (!io_backend_->Sync(fd)) {
          io_ok = false;
          break;
        }
      }
    }
  }

  if (!io_ok) {
    // Rollback all prepared metadata.
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& pr : prepared) {
      auto it = key_meta_.find(pr.key);
      if (it != key_meta_.end() && it->second.generation == pr.meta.generation) {
        auto seg = segments_.find(pr.meta.segment_id);
        if (seg != segments_.end() && seg->second.live_bytes >= pr.meta.size) {
          seg->second.live_bytes -= pr.meta.size;
        }
        if (used_ >= pr.meta.size) used_ -= pr.meta.size;
        key_meta_.erase(it);
        RemoveLRULocked(pr.key);
      }
    }
    return false;
  }

  return true;
}

bool SegmentedSsdTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  // Phase 1 (under mu_): look up metadata.
  int read_fd = -1;
  uint64_t value_offset = 0;
  uint32_t expected_crc = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = key_meta_.find(key);
    if (it == key_meta_.end() && IsReadOnlyShared()) {
      RefreshFromDiskLocked(false);
      it = key_meta_.find(key);
    }
    if (it == key_meta_.end()) return false;
    if (size != it->second.size) return false;

    auto* seg = GetSegmentLocked(it->second.segment_id);
    if (!seg || seg->fd < 0) return false;
    read_fd = seg->fd;
    value_offset = it->second.value_offset;
    expected_crc = it->second.crc32;
    TouchLRULocked(key);
  }

  // Phase 2 (under io_mu_ if needed): perform I/O.
  bool io_ok;
  if (needs_io_lock_) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    io_ok = io_backend_->PReadAll(read_fd, reinterpret_cast<void*>(dst_ptr), size, value_offset);
  } else {
    io_ok = io_backend_->PReadAll(read_fd, reinterpret_cast<void*>(dst_ptr), size, value_offset);
  }
  if (!io_ok) return false;

  // Phase 3 (no lock): verify CRC.
  if (ComputeRecordCrc32(key, reinterpret_cast<const void*>(dst_ptr), size) != expected_crc) {
    return false;
  }
  return true;
}

bool SegmentedSsdTier::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (key_meta_.count(key) > 0) return true;
  if (!IsReadOnlyShared()) return false;
  RefreshFollowerLocked();
  return key_meta_.count(key) > 0;
}

bool SegmentedSsdTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = key_meta_.find(key);
  if (it == key_meta_.end()) return false;

  auto seg = segments_.find(it->second.segment_id);
  if (seg != segments_.end() && seg->second.live_bytes >= it->second.size) {
    seg->second.live_bytes -= it->second.size;
  }
  if (used_ >= it->second.size) used_ -= it->second.size;
  key_meta_.erase(it);
  RemoveLRULocked(key);
  return true;
}

std::pair<size_t, size_t> SegmentedSsdTier::Capacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return {used_, capacity_};
}

void SegmentedSsdTier::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : segments_) {
    if (kv.second.fd >= 0) {
      close(kv.second.fd);
      kv.second.fd = -1;
    }
    if (!IsReadOnlyShared()) {
      std::remove(kv.second.path.c_str());
    }
  }
  segments_.clear();
  key_meta_.clear();
  lru_list_.clear();
  lru_map_.clear();
  known_segment_ids_.clear();
  used_ = 0;
  next_segment_id_ = 0;
  active_segment_id_ = 0;
  generation_counter_ = 1;

  if (!IsReadOnlyShared()) {
    OpenOrCreateSegmentLocked(0);
  } else {
    RefreshFromDiskLocked(true);
  }
}

std::vector<char> SegmentedSsdTier::Read(const std::string& key) {
  // Phase 1 (under mu_): look up metadata.
  int read_fd = -1;
  uint64_t value_offset = 0;
  uint32_t read_size = 0;
  uint32_t expected_crc = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = key_meta_.find(key);
    if (it == key_meta_.end() && IsReadOnlyShared()) {
      RefreshFromDiskLocked(false);
      it = key_meta_.find(key);
    }
    if (it == key_meta_.end()) return {};

    auto* seg = GetSegmentLocked(it->second.segment_id);
    if (!seg || seg->fd < 0) return {};
    read_fd = seg->fd;
    value_offset = it->second.value_offset;
    read_size = it->second.size;
    expected_crc = it->second.crc32;
    TouchLRULocked(key);
  }

  // Phase 2 (under io_mu_ if needed): perform I/O.
  std::vector<char> out(read_size);
  bool io_ok;
  if (needs_io_lock_) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    io_ok = io_backend_->PReadAll(read_fd, out.data(), out.size(), value_offset);
  } else {
    io_ok = io_backend_->PReadAll(read_fd, out.data(), out.size(), value_offset);
  }
  if (!io_ok) return {};

  // Phase 3 (no lock): verify CRC.
  if (ComputeRecordCrc32(key, out.data(), out.size()) != expected_crc) return {};
  return out;
}

std::string SegmentedSsdTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::vector<std::string> SegmentedSsdTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  auto it = lru_list_.rbegin();
  for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
    result.push_back(*it);
  }
  return result;
}
