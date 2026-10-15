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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/local/tiers/segment/segment_index.h"
#include "umbp/local/tiers/segment/segment_scanner.h"
#include "umbp/local/tiers/segment/segment_writer.h"
#include "umbp/local/tiers/tier_backend.h"
#include "umbp/storage/io/status.h"
#include "umbp/storage/io/storage_io_driver.h"

namespace mori::umbp {

enum class SSDAccessMode : int {
  ReadWrite = 0,
  ReadOnlyShared = 1,
};

// An append-only log of segment files.  A record is never rewritten in place;
// evicting a key only drops it from the index, so its bytes stay on disk until
// every record in its segment is gone and the segment file itself is deleted.
// Space therefore comes back a segment at a time, which is why:
//   * Capacity() and write admission count bytes on disk, dead records included;
//   * EvictionOrder() hands out whole segments' keys, emptiest and coldest
//     first (see segment::Index::EvictionOrder);
//   * a segment is deleted as soon as it is sealed, empty and has no read or
//     write in flight on its fd.
class SSDTier : public TierBackend {
 public:
  SSDTier(const std::string& dir, size_t capacity, const UMBPSsdConfig& ssd_config,
          SSDAccessMode access_mode = SSDAccessMode::ReadWrite);
  ~SSDTier() override;

  SSDTier(const SSDTier&) = delete;
  SSDTier& operator=(const SSDTier&) = delete;

  bool Write(const std::string& key, const void* data, size_t size) override;
  bool WriteBatch(const std::vector<std::string>& keys, const std::vector<const void*>& data_ptrs,
                  const std::vector<size_t>& sizes) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  std::vector<bool> ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data_ptrs,
                               const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;
  // Scattered sub-object reads: each key supplies several disjoint
  // object-relative ranges, each with its own destination buffer.  The v3
  // layout makes this cheap -- a value is one contiguous, alignment-anchored
  // extent, so a range is just `value_offset + object_offset` handed to pread.
  //
  // CHECKSUMS ARE NOT VERIFIED HERE, whatever ssd.verify_crc says.  A record's
  // CRC covers the whole value, so a partial read has nothing to check itself
  // against; the alternative (per-block checksums) is a record-format change.
  // Whole-object reads keep verifying, so this narrows the guarantee only for
  // callers that opted into ranged I/O.
  std::vector<bool> ReadBatchRangesIntoPtr(
      const std::vector<std::string>& keys, const std::vector<std::vector<uintptr_t>>& dst_ptrs,
      const std::vector<std::vector<size_t>>& sizes,
      const std::vector<std::vector<size_t>>& src_offsets) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  // {bytes on disk, capacity}: the segments' total size, not just live records.
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  std::vector<char> Read(const std::string& key) override;
  TierCapabilities Capabilities() const override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;
  std::vector<EvictionHint> EvictionOrder(size_t max_keys) override;
  void SetRetiredFileHook(std::function<void(int fd)> on_close,
                          std::chrono::milliseconds grace) override;
  const IoStatus& LastIoStatus() const { return last_io_status_; }

  TierSpaceStats SpaceStats() const override;
  // Segment size in effect: ssd.segment_size_bytes, capped for a small tier.
  uint64_t segment_size() const { return segment_size_; }
  std::optional<std::string> GetLocationId(const std::string& key) const override;
  std::optional<RecordLocation> LocateRecord(const std::string& key) const override;
  // Enables O_DIRECT for segments opened from here on.  Takes effect for
  // segments opened after the call; existing fds keep their current mode, so
  // the intended use is at construction (ssd.direct_io) rather than mid-run.
  void SetColdRead(bool enable) override;

  // True when segment I/O is actually bypassing the page cache.  False if
  // direct I/O was never requested, or was requested and the probe failed.
  bool direct_io_active() const { return direct_io_; }

 private:
  bool IsReadOnlyShared() const { return access_mode_ == SSDAccessMode::ReadOnlyShared; }
  bool ShouldSyncOnWrite() const {
    return ssd_config_.durability.mode == UMBPDurabilityMode::Strict;
  }
  int SegmentOpenFlags() const;
  // Empirical check that this directory's filesystem supports O_DIRECT at
  // kRecordAlign: opens a probe file, writes and reads back one aligned block.
  // Cheaper to trust than a reported capability, and it catches tmpfs/overlayfs
  // (which reject the open) as well as devices needing coarser alignment.
  bool ProbeDirectIo() const;

  // Makes sure the active segment can take `need_bytes`, rolling to a new one
  // if not.  A segment sealed by the roll that is already empty goes into
  // `doomed`.
  bool EnsureActiveSegment(size_t need_bytes, std::vector<segment::Meta>* doomed);
  bool RollActiveSegmentLocked();
  // A new segment is only opened while the filesystem still has a segment's
  // worth of free space, so a drive something else is filling fails writes
  // cleanly instead of with ENOSPC partway through a record.
  bool HasRoomForNewSegment() const;

  // If `segment_id` can be deleted, moves it from the index into `doomed`.  An
  // emptied active segment is rolled past first.  Owner only.
  void CollectReclaimableLocked(uint64_t segment_id, std::vector<segment::Meta>* doomed);
  // Drops the pins a read or write took, collecting segments that were waiting
  // only on them.
  void UnpinLocked(const std::vector<uint64_t>& pinned, std::vector<segment::Meta>* doomed);
  void ReleasePins(const std::vector<uint64_t>& pinned);
  // Unlinks the doomed segments and closes (or parks) their fds.  Runs without
  // mu_: deleting a multi-GiB file frees its extents, which takes milliseconds.
  void DisposeSegments(std::vector<segment::Meta>* doomed);
  // Closes a deleted segment's fd now, or after the retired-file grace.
  void RetireFd(int fd);
  // Closes parked fds whose grace has run out (all of them when `all`).
  void SweepGraveyard(bool all, bool run_hook = true) const;

  bool RefreshFromDiskLocked(bool force_full_rescan);
  // Startup repair: truncate each segment to the last record the scanner could
  // parse.  Removes records from an older kRecordVersion and torn tails, both of
  // which would otherwise make every subsequently appended record unreadable
  // after a restart.  Owner-only; no-op for a fully-parsed segment.
  void DropUnparsedTailsLocked();
  bool OpenOrCreateSegmentLocked(uint64_t segment_id);

  bool RefreshFollowerLocked() const;
  segment::Meta* GetSegmentLocked(uint64_t segment_id);
  const segment::Meta* GetSegmentLocked(uint64_t segment_id) const;
  bool ReadRecordLocked(const std::string& key, void* dst, size_t size, uint32_t expected_crc,
                        uint64_t value_offset, int read_fd, bool crc_valid) const;
  void RememberStatus(IoStatus status) const;
  // Issue one value read, transparently bouncing through an aligned buffer when
  // direct I/O is on and `dst`/`size` do not meet the alignment rules.
  IoStatus ReadValueInto(int fd, void* dst, size_t size, uint64_t value_offset) const;
  bool ShouldVerifyCrc(bool crc_valid) const { return ssd_config_.verify_crc && crc_valid; }
  int TierThreads() const;

  std::string dir_;
  size_t capacity_;
  UMBPSsdConfig ssd_config_;
  SSDAccessMode access_mode_;
  bool direct_io_ = false;
  uint64_t segment_size_ = 0;

  // Deleted segments whose fds wait out the retired-file grace.
  struct RetiredFd {
    int fd = -1;
    std::chrono::steady_clock::time_point close_at;
  };
  mutable std::mutex graveyard_mu_;
  mutable std::deque<RetiredFd> graveyard_;     // guarded by graveyard_mu_
  std::function<void(int fd)> retired_hook_;    // guarded by graveyard_mu_
  std::chrono::milliseconds retired_grace_{0};  // guarded by graveyard_mu_
  std::atomic<uint64_t> segments_reclaimed_{0};
  std::atomic<uint64_t> bytes_reclaimed_{0};
  mutable std::atomic<uint64_t> no_room_refusals_{0};

  mutable std::mutex mu_;
  mutable std::mutex io_mu_;
  std::unique_ptr<StorageIoDriver> io_driver_;
  segment::Index index_;
  segment::Scanner scanner_;
  std::unique_ptr<segment::Writer> writer_;
  mutable IoStatus last_io_status_;
};

}  // namespace mori::umbp
