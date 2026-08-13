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
#include "umbp/local/tiers/ssd_tier.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/aligned_buffer.h"
#include "umbp/common/parallel_for.h"
#include "umbp/common/ssd_perf.h"
#include "umbp/local/tiers/segment/segment_format.h"

namespace fs = std::filesystem;

namespace mori::umbp {

SSDTier::SSDTier(const std::string& dir, size_t capacity, const UMBPSsdConfig& ssd_config,
                 SSDAccessMode access_mode)
    : TierBackend(StorageTier::LOCAL_SSD),
      dir_(dir),
      capacity_(capacity),
      ssd_config_(ssd_config),
      access_mode_(access_mode),
      io_driver_(CreateStorageIoDriver(ssd_config.io.backend,
                                       static_cast<uint32_t>(ssd_config.io.queue_depth))),
      index_(capacity) {
  std::string error_message;
  if (!ssd_config_.Validate(&error_message)) {
    throw std::runtime_error("invalid UMBP SSD config: " + error_message);
  }
  if (ssd_config_.io.backend == UMBPIoBackend::IoUring &&
      !io_driver_->Capabilities().native_async) {
    MORI_UMBP_WARN(
        "SSDTier: io_uring backend requested but unavailable (kernel missing io_uring "
        "support or restricted by seccomp/capabilities); falling back to POSIX backend");
  }
  writer_ = std::make_unique<segment::Writer>(*io_driver_, ssd_config_.verify_crc);
  fs::create_directories(dir_);

  if (ssd_config_.direct_io) {
    direct_io_ = ProbeDirectIo();
    if (!direct_io_) {
      MORI_UMBP_WARN(
          "[SSDTier] {}: direct I/O requested but this filesystem rejects O_DIRECT at {}B "
          "alignment (tmpfs/overlayfs cannot do it); falling back to buffered I/O. Reads will "
          "be served from the page cache, so reported tier bandwidth will NOT be the device's",
          dir_, segment::kRecordAlign);
    }
  }
  MORI_UMBP_INFO("[SSDTier] {}: direct_io={} verify_crc={} tier_io_threads={} io_backend={}", dir_,
                 direct_io_, ssd_config_.verify_crc, TierThreads(),
                 ssd_config_.io.backend == UMBPIoBackend::IoUring ? "io_uring" : "posix");
  if (direct_io_ && ssd_config_.io.backend != UMBPIoBackend::IoUring) {
    MORI_UMBP_WARN(
        "[SSDTier] {}: direct I/O on the POSIX backend issues one blocking pread per key "
        "(queue depth 1). Every read is now a synchronous device round trip, which will look "
        "far slower than it needs to — prefer ssd.io.backend=io_uring for direct I/O",
        dir_);
  }

  std::lock_guard<std::mutex> lock(mu_);
  RefreshFromDiskLocked(true);
  DropUnparsedTailsLocked();
  if (!IsReadOnlyShared() && index_.Segments().empty()) {
    OpenOrCreateSegmentLocked(0);
  }
}

int SSDTier::TierThreads() const {
  int t = ssd_config_.tier_io_threads;
  if (t < 1) t = 1;
  const unsigned hc = std::thread::hardware_concurrency();
  if (hc > 0 && t > static_cast<int>(hc)) t = static_cast<int>(hc);
  return t;
}

int SSDTier::SegmentOpenFlags() const { return direct_io_ ? O_DIRECT : 0; }

bool SSDTier::ProbeDirectIo() const {
  const std::string path = dir_ + "/.umbp_odirect_probe";
  // O_DIRECT is rejected at open() time on filesystems that cannot do it, but a
  // successful open does not prove the alignment is accepted, so round-trip one
  // real block before believing it.
  int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) return false;

  bool ok = false;
  try {
    AlignedBuffer block(static_cast<size_t>(segment::kRecordAlign));
    std::memset(block.data(), 0xA5, block.padded_size());
    const ssize_t w = pwrite(fd, block.data(), block.padded_size(), 0);
    if (w == static_cast<ssize_t>(block.padded_size())) {
      AlignedBuffer back(static_cast<size_t>(segment::kRecordAlign));
      const ssize_t r = pread(fd, back.data(), back.padded_size(), 0);
      ok = r == static_cast<ssize_t>(back.padded_size()) &&
           std::memcmp(block.data(), back.data(), block.padded_size()) == 0;
    }
  } catch (const std::bad_alloc&) {
    ok = false;
  }

  close(fd);
  unlink(path.c_str());
  return ok;
}

void SSDTier::SetColdRead(bool enable) {
  if (enable == direct_io_) return;
  if (enable && !ProbeDirectIo()) {
    MORI_UMBP_WARN("[SSDTier] {}: SetColdRead(true) ignored — filesystem rejects O_DIRECT", dir_);
    return;
  }
  direct_io_ = enable;
  MORI_UMBP_INFO("[SSDTier] {}: direct I/O {} for segments opened from here on", dir_,
                 enable ? "enabled" : "disabled");
}

void SSDTier::DropUnparsedTailsLocked() {
  // A follower must never rewrite the shared log; the owner does this once at
  // startup, before any write can land.
  if (IsReadOnlyShared()) return;

  for (auto& kv : index_.MutableSegments()) {
    auto& seg = kv.second;
    if (seg.fd < 0) continue;

    struct stat st;
    if (fstat(seg.fd, &st) != 0) continue;
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    if (seg.scanned_offset >= file_size) continue;  // fully parsed, nothing to do

    // Bytes past the last parsed record boundary are permanently unreachable:
    // the scanner stops at the first header it cannot read and every later
    // restart stops at the same place.  Two ways to get here --
    //   * records written by an older kRecordVersion (e.g. the CRC-32/ISO-HDLC
    //     v1 format that preceded CRC-32C), and
    //   * a torn tail record from a crash mid-write.
    // Either way, appending after them silently loses the new records on the
    // next restart, so truncate instead.  Safe by construction: the SSD tier is
    // a cache whose contents are re-fetchable.
    const uint64_t dropped = file_size - seg.scanned_offset;
    MORI_UMBP_WARN(
        "[SSDTier] {}: dropping {}B of unreadable tail (stale record version or torn write); "
        "segment truncated to the last valid record at offset {}",
        seg.path, dropped, seg.scanned_offset);
    if (ftruncate(seg.fd, static_cast<off_t>(seg.scanned_offset)) != 0) {
      MORI_UMBP_ERROR("[SSDTier] {}: ftruncate to {} failed; segment left as-is", seg.path,
                      seg.scanned_offset);
      continue;
    }
    // Reset the append cursor too, otherwise the next write leaves a hole that
    // the scanner would stop at all over again.
    seg.write_offset = seg.scanned_offset;
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

  int flags = (IsReadOnlyShared() ? O_RDONLY : (O_RDWR | O_CREAT)) | SegmentOpenFlags();
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

  if (seg->write_offset + need_bytes <= ssd_config_.segment_size_bytes) return true;

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
                                     force_full_rescan, &error_message, SegmentOpenFlags());
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

  const size_t record_size = static_cast<size_t>(segment::RecordBytes(key.size(), size));
  segment::PreparedRecord pr;

  // Phase 1a (no lock): checksum + assemble, same rationale as WriteBatch.
  writer_->Build(key, data, size, &pr);

  int write_fd = -1;
  {
    // Phase 1b: reserve index space under mu_
    std::lock_guard<std::mutex> lock(mu_);
    if (!EnsureActiveSegment(record_size)) return false;
    auto* seg = GetSegmentLocked(index_.active_segment_id());
    if (!seg) return false;
    if (!writer_->Reserve(key, size, seg, index_, &pr)) return false;
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

  // Padded size: what the batch will actually occupy on disk, and therefore what
  // the segment-roll check has to be made against.
  size_t total_bytes = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    total_bytes += static_cast<size_t>(segment::RecordBytes(keys[i].size(), sizes[i]));
  }
  if (total_bytes > ssd_config_.segment_size_bytes) {
    bool all_ok = true;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!Write(keys[i], data_ptrs[i], sizes[i])) all_ok = false;
    }
    return all_ok;
  }

  // Stage timers for the [SsdPerf/tier] PUT breakdown (no-ops unless
  // UMBP_SSD_TIMING is set).  `prepare` covers the CRC + record memcpy done
  // under mu_, which is the stage that blocks concurrent reads on this drive.
  const auto t_begin = ssdperf::Now();
  double build_ms = 0.0, lock_ms = 0.0, reserve_ms = 0.0, io_ms = 0.0, sync_ms = 0.0;

  // Phase 1a (NO lock): checksum + assemble every record.  This is the expensive
  // half of the old Prepare() and it touches no shared state, so keeping it out
  // of mu_ stops a write batch from blocking concurrent reads on this drive for
  // the whole of its CRC + copy time.
  std::vector<segment::PreparedRecord> built(keys.size());
  ParallelFor(keys.size(), TierThreads(),
              [&](size_t i) { writer_->Build(keys[i], data_ptrs[i], sizes[i], &built[i]); });
  build_ms = ssdperf::MsSince(t_begin);

  std::vector<segment::PreparedRecord> prepared;
  int write_fd = -1;
  {
    // Phase 1b: reserve index/segment space under mu_ (cheap: no CRC, no copy).
    const auto t_lock = ssdperf::Now();
    std::lock_guard<std::mutex> lock(mu_);
    const auto t_locked = ssdperf::Now();
    lock_ms = ssdperf::MsBetween(t_lock, t_locked);
    if (!EnsureActiveSegment(total_bytes)) return false;
    auto* seg = GetSegmentLocked(index_.active_segment_id());
    if (!seg) return false;
    write_fd = seg->fd;

    prepared.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!writer_->Reserve(keys[i], sizes[i], seg, index_, &built[i])) continue;
      prepared.push_back(std::move(built[i]));
    }
    reserve_ms = ssdperf::MsSince(t_locked);
  }

  if (prepared.empty()) return true;

  // Phase 2: perform I/O outside mu_
  const bool needs_io_lock = !io_driver_->Capabilities().thread_safe;
  const auto t_io = ssdperf::Now();
  IoStatus status;
  if (needs_io_lock) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    status = writer_->WriteRecords(write_fd, prepared, ShouldSyncOnWrite(), &sync_ms);
  } else {
    status = writer_->WriteRecords(write_fd, prepared, ShouldSyncOnWrite(), &sync_ms);
  }
  io_ms = ssdperf::MsSince(t_io) - sync_ms;

  if (ssdperf::Enabled()) {
    const double total_ms = ssdperf::MsSince(t_begin);
    MORI_UMBP_INFO(
        "[SsdPerf/tier] PUT dir={} keys={} bytes={} total_ms={:.3f} GB_s={:.2f} | "
        "build_crc_memcpy={:.3f}ms ({:.0f}%, lock-free) mu_wait={:.3f}ms reserve={:.3f}ms "
        "({:.0f}%, under mu_) dev_write={:.3f}ms ({:.0f}%) fsync={:.3f}ms ({:.0f}%) | "
        "crc_GB_s={:.2f} dev_GB_s={:.2f} sync_on_write={}",
        dir_, prepared.size(), total_bytes, total_ms, ssdperf::GbPerSec(total_bytes, total_ms),
        build_ms, ssdperf::Pct(build_ms, total_ms), lock_ms, reserve_ms,
        ssdperf::Pct(reserve_ms, total_ms), io_ms, ssdperf::Pct(io_ms, total_ms), sync_ms,
        ssdperf::Pct(sync_ms, total_ms), ssdperf::GbPerSec(total_bytes, build_ms),
        ssdperf::GbPerSec(total_bytes, io_ms), ShouldSyncOnWrite());
  }

  if (!status.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& pr : prepared) index_.RollbackWrite(pr.reservation);
    RememberStatus(std::move(status));
    return false;
  }
  return true;
}

IoStatus SSDTier::ReadValueInto(int fd, void* dst, size_t size, uint64_t value_offset) const {
  // Value offsets are alignment multiples by construction in the v3 layout, so
  // only the caller's buffer and length can disqualify a direct read.  When they
  // do, read the padded value (always present on disk) into an aligned bounce
  // buffer and hand back just the value bytes.
  if (direct_io_ && !IsDirectIoCompatible(dst, size)) {
    AlignedBuffer bounce(size);
    IoStatus status = io_driver_->ReadAt(fd, bounce.data(), bounce.padded_size(), value_offset);
    if (!status.ok()) return status;
    std::memcpy(dst, bounce.data(), size);
    return IoStatus::Ok();
  }
  return io_driver_->ReadAt(fd, dst, size, value_offset);
}

bool SSDTier::ReadRecordLocked(const std::string& key, void* dst, size_t size,
                               uint32_t expected_crc, uint64_t value_offset, int read_fd,
                               bool crc_valid) const {
  const bool needs_external_lock = !io_driver_->Capabilities().thread_safe;
  IoStatus status;
  if (needs_external_lock) {
    std::lock_guard<std::mutex> io_lock(io_mu_);
    status = ReadValueInto(read_fd, dst, size, value_offset);
  } else {
    status = ReadValueInto(read_fd, dst, size, value_offset);
  }
  if (!status.ok()) {
    RememberStatus(std::move(status));
    return false;
  }

  if (ShouldVerifyCrc(crc_valid) && segment::ComputeRecordCrc32(key, dst, size) != expected_crc) {
    RememberStatus(IoStatus::Corruption("segment CRC mismatch"));
    return false;
  }
  return true;
}

bool SSDTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  int read_fd = -1;
  uint64_t value_offset = 0;
  uint32_t expected_crc = 0;
  bool crc_valid = true;
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
    crc_valid = meta->crc_valid;
    index_.TouchLRU(key);
  }
  return ReadRecordLocked(key, reinterpret_cast<void*>(dst_ptr), size, expected_crc, value_offset,
                          read_fd, crc_valid);
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
    void* dst;       // final destination (caller's buffer)
    void* io_dst;    // where the I/O lands: dst, or `bounce` when misaligned
    size_t io_size;  // bytes to transfer: size, or the padded value length
    bool crc_valid;
    AlignedBuffer bounce;  // only allocated when a bounce is actually needed
  };

  std::vector<ReadLookup> lookups;
  lookups.reserve(keys.size());

  // Stage timers for the [SsdPerf/tier] GET breakdown (no-ops unless
  // UMBP_SSD_TIMING is set).  The three stages are disjoint and cover the whole
  // call, so their percentages answer "device or CPU?" directly.
  const auto t_begin = ssdperf::Now();
  double lock_ms = 0.0, lookup_ms = 0.0, io_ms = 0.0, crc_ms = 0.0;

  // Phase 1 (mu_ held): batch index lookup + metadata extraction.
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto t_locked = ssdperf::Now();
    lock_ms = ssdperf::MsBetween(t_begin, t_locked);

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

      ReadLookup lk;
      lk.orig_idx = i;
      lk.fd = seg->fd;
      lk.offset = meta->value_offset;
      lk.expected_crc = meta->crc32;
      lk.size = sizes[i];
      lk.dst = reinterpret_cast<void*>(dst_ptrs[i]);
      lk.crc_valid = meta->crc_valid;
      lk.io_dst = lk.dst;
      lk.io_size = lk.size;
      // Direct I/O demands an aligned buffer and length.  KV page buffers are
      // normally both (page sizes here are exact 4 KiB multiples out of
      // hugepage-backed allocations), so this bounce is the exception, not the
      // rule — but it has to exist, because the failure mode without it is an
      // EINVAL that the per-key fallback below would simply repeat, surfacing as
      // a silent 100% miss rather than an error.
      if (direct_io_ && !IsDirectIoCompatible(lk.dst, lk.size)) {
        lk.bounce.Resize(lk.size);
        lk.io_dst = lk.bounce.data();
        lk.io_size = lk.bounce.padded_size();
      }
      lookups.push_back(std::move(lk));
    }
    lookup_ms = ssdperf::MsSince(t_locked);
  }

  if (lookups.empty()) return results;

  // Phase 2 (io_mu_ if needed): batch I/O.
  const bool needs_io_lock = !io_driver_->Capabilities().thread_safe;
  const bool use_batch = io_driver_->Capabilities().batch_read && lookups.size() > 1;
  const auto t_io = ssdperf::Now();

  std::vector<bool> io_ok(lookups.size(), false);

  if (use_batch) {
    std::vector<IoReadOp> ops;
    ops.reserve(lookups.size());
    for (const auto& lk : lookups) {
      ops.push_back({lk.fd, lk.io_dst, lk.io_size, lk.offset});
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
          s = io_driver_->ReadAt(lk.fd, lk.io_dst, lk.io_size, lk.offset);
        } else {
          s = io_driver_->ReadAt(lk.fd, lk.io_dst, lk.io_size, lk.offset);
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
        s = io_driver_->ReadAt(lk.fd, lk.io_dst, lk.io_size, lk.offset);
      } else {
        s = io_driver_->ReadAt(lk.fd, lk.io_dst, lk.io_size, lk.offset);
      }
      io_ok[j] = s.ok();
      if (!s.ok()) RememberStatus(std::move(s));
    }
  }

  io_ms = ssdperf::MsSince(t_io);

  // Phase 3 (no lock): copy out of any bounce buffers, then verify checksums.
  // Both are pure CPU over disjoint indices, so they fan out across
  // tier_io_threads — this is the phase that was single-threaded while the DRAM
  // tier's equivalent memcpy ran on read_threads_ workers.
  const auto t_crc = ssdperf::Now();
  uint64_t verified_bytes = 0;
  for (size_t j = 0; j < lookups.size(); ++j) {
    if (io_ok[j]) verified_bytes += lookups[j].size;
  }

  std::vector<char> key_ok(lookups.size(), 0);
  ParallelFor(lookups.size(), TierThreads(), [&](size_t j) {
    if (!io_ok[j]) return;
    auto& lk = lookups[j];
    if (lk.io_dst != lk.dst) std::memcpy(lk.dst, lk.bounce.data(), lk.size);
    // CRC covers exactly the value, never the record's alignment padding.
    if (ShouldVerifyCrc(lk.crc_valid) &&
        segment::ComputeRecordCrc32(keys[lk.orig_idx], lk.dst, lk.size) != lk.expected_crc) {
      return;  // key_ok stays 0
    }
    key_ok[j] = 1;
  });

  for (size_t j = 0; j < lookups.size(); ++j) {
    if (!io_ok[j]) continue;
    if (!key_ok[j]) {
      RememberStatus(IoStatus::Corruption("segment CRC mismatch"));
      continue;
    }
    results[lookups[j].orig_idx] = true;
  }
  crc_ms = ssdperf::MsSince(t_crc);

  if (ssdperf::Enabled()) {
    const double total_ms = lock_ms + lookup_ms + io_ms + crc_ms;
    MORI_UMBP_INFO(
        "[SsdPerf/tier] GET dir={} keys={} served={} bytes={} total_ms={:.3f} GB_s={:.2f} | "
        "mu_wait={:.3f}ms index={:.3f}ms ({:.0f}%) dev_read={:.3f}ms ({:.0f}%) crc={:.3f}ms "
        "({:.0f}%) | dev_GB_s={:.2f} crc_GB_s={:.2f} batched={}",
        dir_, keys.size(), lookups.size(), verified_bytes, total_ms,
        ssdperf::GbPerSec(verified_bytes, total_ms), lock_ms, lookup_ms,
        ssdperf::Pct(lookup_ms, total_ms), io_ms, ssdperf::Pct(io_ms, total_ms), crc_ms,
        ssdperf::Pct(crc_ms, total_ms), ssdperf::GbPerSec(verified_bytes, io_ms),
        ssdperf::GbPerSec(verified_bytes, crc_ms), use_batch);
  }

  return results;
}

std::vector<bool> SSDTier::BatchWrite(const std::vector<std::string>& keys,
                                      const std::vector<const void*>& data_ptrs,
                                      const std::vector<size_t>& sizes) {
  bool ok = WriteBatch(keys, data_ptrs, sizes);
  return std::vector<bool>(keys.size(), ok);
}

std::vector<bool> SSDTier::BatchReadIntoPtr(const std::vector<std::string>& keys,
                                            const std::vector<uintptr_t>& dst_ptrs,
                                            const std::vector<size_t>& sizes) {
  return ReadBatchIntoPtr(keys, dst_ptrs, sizes);
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
  bool crc_valid = true;
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
    crc_valid = meta->crc_valid;
    index_.TouchLRU(key);
  }

  // The returned vector is only max_align_t-aligned, so under direct I/O
  // ReadValueInto bounces it; that is the whole cost of keeping this API's
  // std::vector<char> return type.
  std::vector<char> out(read_size);
  if (!ReadRecordLocked(key, out.data(), out.size(), expected_crc, value_offset, read_fd,
                        crc_valid))
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

std::optional<std::string> SSDTier::GetLocationId(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto* meta = index_.FindKey(key);
  if (!meta && IsReadOnlyShared()) {
    const_cast<SSDTier*>(this)->RefreshFromDiskLocked(false);
    meta = index_.FindKey(key);
  }
  if (!meta) {
    return std::nullopt;
  }
  return "seg" + std::to_string(meta->segment_id) + ":" + std::to_string(meta->value_offset);
}

}  // namespace mori::umbp
