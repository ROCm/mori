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
#include "umbp/storage/ssd_tier.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "umbp/storage/segment/segment_format.h"

namespace fs = std::filesystem;

SSDTier::SSDTier(const std::string& dir, size_t capacity, const UMBPConfig& config,
                 SSDAccessMode access_mode)
    : TierBackend(StorageTier::LOCAL_SSD),
      dir_(dir),
      capacity_(capacity),
      config_(config),
      access_mode_(access_mode),
      io_driver_(CreateStorageIoDriver(config.ssd.io.backend,
                                       static_cast<uint32_t>(config.ssd.io.queue_depth))),
      index_(capacity) {
  std::string error_message;
  if (!config_.Validate(&error_message)) {
    throw std::runtime_error("invalid UMBP config: " + error_message);
  }
  if (config_.ssd.io.backend == UMBPIoBackend::IoUring &&
      !io_driver_->Capabilities().native_async) {
    throw std::runtime_error("UMBP io_uring backend requested but initialization failed");
  }
  writer_ = std::make_unique<segment::Writer>(*io_driver_);
  fs::create_directories(dir_);
  std::lock_guard<std::mutex> lock(mu_);
  RefreshFromDiskLocked(true);
  if (!IsReadOnlyShared() && index_.Segments().empty()) {
    OpenOrCreateSegmentLocked(0);
  }
}

SSDTier::~SSDTier() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : index_.MutableSegments()) {
    if (kv.second.fd >= 0) {
      close(kv.second.fd);
      kv.second.fd = -1;
    }
  }
}

segment::Meta* SSDTier::GetSegmentLocked(uint64_t segment_id) {
  return index_.FindSegment(segment_id);
}

const segment::Meta* SSDTier::GetSegmentLocked(uint64_t segment_id) const {
  return index_.FindSegment(segment_id);
}

void SSDTier::RememberStatus(IoStatus status) const { last_io_status_ = std::move(status); }

bool SSDTier::OpenOrCreateSegmentLocked(uint64_t segment_id) {
  segment::Meta seg;
  seg.id = segment_id;
  seg.path = dir_ + "/" + segment::BuildFileName(segment_id);

  int flags = IsReadOnlyShared() ? O_RDONLY : (O_RDWR | O_CREAT);
  seg.fd = open(seg.path.c_str(), flags, 0644);
  if (seg.fd < 0) return false;

  struct stat st;
  if (fstat(seg.fd, &st) != 0) {
    close(seg.fd);
    return false;
  }

  seg.write_offset = static_cast<uint64_t>(st.st_size);
  index_.MutableSegments()[segment_id] = seg;
  index_.MarkKnownSegment(segment_id);
  index_.AdvanceNextSegmentId(segment_id + 1);
  if (!IsReadOnlyShared()) {
    index_.set_active_segment_id(std::max(index_.active_segment_id(), segment_id));
  }
  return true;
}

bool SSDTier::EnsureActiveSegment(size_t need_bytes) {
  auto* seg = GetSegmentLocked(index_.active_segment_id());
  if (!seg) {
    if (!OpenOrCreateSegmentLocked(index_.next_segment_id())) return false;
    index_.set_active_segment_id(index_.next_segment_id() - 1);
    seg = GetSegmentLocked(index_.active_segment_id());
  }
  if (!seg) return false;

  if (seg->write_offset + need_bytes <= config_.ssd.segment_size_bytes) return true;

  uint64_t new_id = index_.next_segment_id();
  if (!OpenOrCreateSegmentLocked(new_id)) return false;
  index_.set_active_segment_id(new_id);
  return true;
}

bool SSDTier::RefreshFromDiskLocked(bool force_full_rescan) {
  if (force_full_rescan) {
    for (auto& kv : index_.MutableSegments()) {
      if (kv.second.fd >= 0) {
        close(kv.second.fd);
        kv.second.fd = -1;
      }
    }
    index_.ResetAll();
  }

  std::string error_message;
  bool ok = scanner_.RefreshFromDisk(dir_, *io_driver_, index_, IsReadOnlyShared(),
                                     force_full_rescan, &error_message);
  if (!ok && !error_message.empty()) {
    RememberStatus(IoStatus::IoError(error_message));
  }
  return ok;
}

bool SSDTier::RefreshFollowerLocked() const {
  return const_cast<SSDTier*>(this)->RefreshFromDiskLocked(false);
}

bool SSDTier::Write(const std::string& key, const void* data, size_t size) {
  if (IsReadOnlyShared()) return false;

  const size_t record_size = sizeof(segment::RecordHeader) + key.size() + size;
  segment::PreparedRecord pr;
  int write_fd = -1;
  {
    // Phase 1: reserve index space and build record buffer under mu_
    std::lock_guard<std::mutex> lock(mu_);
    if (!EnsureActiveSegment(record_size)) return false;
    auto* seg = GetSegmentLocked(index_.active_segment_id());
    if (!seg) return false;
    if (!writer_->Prepare(key, data, size, seg, index_, &pr)) return false;
    write_fd = seg->fd;
  }

  // Phase 2: perform I/O outside mu_ (io_mu_ serializes non-thread-safe backends)
  const bool needs_io_lock = !io_driver_->Capabilities().thread_safe;
  IoStatus status;
  if (needs_io_lock) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    status = writer_->WriteRecord(write_fd, pr, ShouldSyncOnWrite());
  } else {
    status = writer_->WriteRecord(write_fd, pr, ShouldSyncOnWrite());
  }

  if (!status.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    index_.RollbackWrite(pr.reservation);
    RememberStatus(std::move(status));
    return false;
  }
  return true;
}

bool SSDTier::WriteBatch(const std::vector<std::string>& keys,
                         const std::vector<const void*>& data_ptrs,
                         const std::vector<size_t>& sizes) {
  if (keys.empty()) return true;
  if (IsReadOnlyShared()) return false;

  size_t total_bytes = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    total_bytes += sizeof(segment::RecordHeader) + keys[i].size() + sizes[i];
  }
  if (total_bytes > config_.ssd.segment_size_bytes) {
    bool all_ok = true;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!Write(keys[i], data_ptrs[i], sizes[i])) all_ok = false;
    }
    return all_ok;
  }

  std::vector<segment::PreparedRecord> prepared;
  int write_fd = -1;
  {
    // Phase 1: reserve index space and build record buffers under mu_
    std::lock_guard<std::mutex> lock(mu_);
    if (!EnsureActiveSegment(total_bytes)) return false;
    auto* seg = GetSegmentLocked(index_.active_segment_id());
    if (!seg) return false;
    write_fd = seg->fd;

    prepared.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      segment::PreparedRecord pr;
      if (!writer_->Prepare(keys[i], data_ptrs[i], sizes[i], seg, index_, &pr)) continue;
      prepared.push_back(std::move(pr));
    }
  }

  if (prepared.empty()) return true;

  // Phase 2: perform I/O outside mu_
  const bool needs_io_lock = !io_driver_->Capabilities().thread_safe;
  IoStatus status;
  if (needs_io_lock) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    status = writer_->WriteRecords(write_fd, prepared, ShouldSyncOnWrite());
  } else {
    status = writer_->WriteRecords(write_fd, prepared, ShouldSyncOnWrite());
  }

  if (!status.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& pr : prepared) index_.RollbackWrite(pr.reservation);
    RememberStatus(std::move(status));
    return false;
  }
  return true;
}

bool SSDTier::ReadRecordLocked(const std::string& key, void* dst, size_t size,
                               uint32_t expected_crc, uint64_t value_offset, int read_fd) const {
  const bool needs_external_lock = !io_driver_->Capabilities().thread_safe;
  IoStatus status;
  if (needs_external_lock) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    status = io_driver_->ReadAt(read_fd, dst, size, value_offset);
  } else {
    status = io_driver_->ReadAt(read_fd, dst, size, value_offset);
  }
  if (!status.ok()) {
    RememberStatus(std::move(status));
    return false;
  }

  if (segment::ComputeRecordCrc32(key, dst, size) != expected_crc) {
    RememberStatus(IoStatus::Corruption("segment CRC mismatch"));
    return false;
  }
  return true;
}

bool SSDTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  int read_fd = -1;
  uint64_t value_offset = 0;
  uint32_t expected_crc = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto* meta = index_.FindKey(key);
    if (!meta && IsReadOnlyShared()) {
      RefreshFromDiskLocked(false);
      meta = index_.FindMutableKey(key);
    }
    if (!meta) return false;
    if (size != meta->size) return false;
    auto* seg = GetSegmentLocked(meta->segment_id);
    if (!seg || seg->fd < 0) return false;
    read_fd = seg->fd;
    value_offset = meta->value_offset;
    expected_crc = meta->crc32;
    index_.TouchLRU(key);
  }
  return ReadRecordLocked(key, reinterpret_cast<void*>(dst_ptr), size, expected_crc, value_offset,
                          read_fd);
}

std::vector<bool> SSDTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                            const std::vector<uintptr_t>& dst_ptrs,
                                            const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;

  // Per-key lookup result for Phase 2/3.
  struct ReadLookup {
    size_t orig_idx;
    int fd;
    uint64_t offset;
    uint32_t expected_crc;
    size_t size;
    void* dst;
  };

  std::vector<ReadLookup> lookups;
  lookups.reserve(keys.size());

  // Phase 1 (mu_ held): batch index lookup + metadata extraction.
  {
    std::lock_guard<std::mutex> lock(mu_);

    // Follower: do a single refresh if any key is missing.
    if (IsReadOnlyShared()) {
      bool any_missing = false;
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!index_.FindKey(keys[i])) {
          any_missing = true;
          break;
        }
      }
      if (any_missing) {
        RefreshFromDiskLocked(false);
      }
    }

    for (size_t i = 0; i < keys.size(); ++i) {
      auto* meta = index_.FindKey(keys[i]);
      if (!meta) continue;
      if (sizes[i] != meta->size) continue;
      auto* seg = GetSegmentLocked(meta->segment_id);
      if (!seg || seg->fd < 0) continue;
      index_.TouchLRU(keys[i]);
      lookups.push_back({i, seg->fd, meta->value_offset, meta->crc32, sizes[i],
                         reinterpret_cast<void*>(dst_ptrs[i])});
    }
  }

  if (lookups.empty()) return results;

  // Phase 2 (io_mu_ if needed): batch I/O.
  const bool needs_io_lock = !io_driver_->Capabilities().thread_safe;
  const bool use_batch = io_driver_->Capabilities().batch_read && lookups.size() > 1;

  std::vector<bool> io_ok(lookups.size(), false);

  if (use_batch) {
    std::vector<IoReadOp> ops;
    ops.reserve(lookups.size());
    for (const auto& lk : lookups) {
      ops.push_back({lk.fd, lk.dst, lk.size, lk.offset});
    }

    IoStatus status;
    if (needs_io_lock) {
      std::lock_guard<std::mutex> io_lock(io_mu_);
      status = io_driver_->ReadBatch(ops);
    } else {
      status = io_driver_->ReadBatch(ops);
    }

    if (status.ok()) {
      // All I/O succeeded; mark all as ok for CRC check.
      std::fill(io_ok.begin(), io_ok.end(), true);
    } else {
      // Batch failed — fall back to per-key reads.
      RememberStatus(std::move(status));
      for (size_t j = 0; j < lookups.size(); ++j) {
        const auto& lk = lookups[j];
        IoStatus s;
        if (needs_io_lock) {
          std::lock_guard<std::mutex> io_lock(io_mu_);
          s = io_driver_->ReadAt(lk.fd, lk.dst, lk.size, lk.offset);
        } else {
          s = io_driver_->ReadAt(lk.fd, lk.dst, lk.size, lk.offset);
        }
        io_ok[j] = s.ok();
        if (!s.ok()) RememberStatus(std::move(s));
      }
    }
  } else {
    // Serial path (single key or no batch_read capability).
    for (size_t j = 0; j < lookups.size(); ++j) {
      const auto& lk = lookups[j];
      IoStatus s;
      if (needs_io_lock) {
        std::lock_guard<std::mutex> io_lock(io_mu_);
        s = io_driver_->ReadAt(lk.fd, lk.dst, lk.size, lk.offset);
      } else {
        s = io_driver_->ReadAt(lk.fd, lk.dst, lk.size, lk.offset);
      }
      io_ok[j] = s.ok();
      if (!s.ok()) RememberStatus(std::move(s));
    }
  }

  // Phase 3 (no lock): per-key CRC verification.
  for (size_t j = 0; j < lookups.size(); ++j) {
    if (!io_ok[j]) continue;
    const auto& lk = lookups[j];
    if (segment::ComputeRecordCrc32(keys[lk.orig_idx], lk.dst, lk.size) != lk.expected_crc) {
      RememberStatus(IoStatus::Corruption("segment CRC mismatch"));
      continue;
    }
    results[lk.orig_idx] = true;
  }

  return results;
}

bool SSDTier::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (index_.HasKey(key)) return true;
  if (!IsReadOnlyShared()) return false;
  RefreshFollowerLocked();
  return index_.HasKey(key);
}

bool SSDTier::Evict(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.EraseKey(key);
}

std::pair<size_t, size_t> SSDTier::Capacity() const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.Capacity();
}

void SSDTier::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : index_.MutableSegments()) {
    if (kv.second.fd >= 0) {
      close(kv.second.fd);
      kv.second.fd = -1;
    }
    if (!IsReadOnlyShared()) {
      std::remove(kv.second.path.c_str());
    }
  }
  index_.ResetAll();
  if (!IsReadOnlyShared()) {
    OpenOrCreateSegmentLocked(0);
  } else {
    RefreshFromDiskLocked(true);
  }
}

std::vector<char> SSDTier::Read(const std::string& key) {
  int read_fd = -1;
  uint64_t value_offset = 0;
  uint32_t read_size = 0;
  uint32_t expected_crc = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto* meta = index_.FindKey(key);
    if (!meta && IsReadOnlyShared()) {
      RefreshFromDiskLocked(false);
      meta = index_.FindMutableKey(key);
    }
    if (!meta) return {};
    auto* seg = GetSegmentLocked(meta->segment_id);
    if (!seg || seg->fd < 0) return {};
    read_fd = seg->fd;
    value_offset = meta->value_offset;
    read_size = meta->size;
    expected_crc = meta->crc32;
    index_.TouchLRU(key);
  }

  std::vector<char> out(read_size);
  if (!ReadRecordLocked(key, out.data(), out.size(), expected_crc, value_offset, read_fd))
    return {};
  return out;
}

TierCapabilities SSDTier::Capabilities() const {
  TierCapabilities caps;
  caps.batch_write = true;
  caps.batch_read = true;
  return caps;
}

std::string SSDTier::GetLRUKey() const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.GetLRUKey();
}

std::vector<std::string> SSDTier::GetLRUCandidates(size_t max_candidates) const {
  std::lock_guard<std::mutex> lock(mu_);
  return index_.GetLRUCandidates(max_candidates);
}
