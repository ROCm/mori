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
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  std::vector<char> Read(const std::string& key) override;
  TierCapabilities Capabilities() const override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;
  const IoStatus& LastIoStatus() const { return last_io_status_; }
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

  bool EnsureActiveSegment(size_t need_bytes);
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

  mutable std::mutex mu_;
  mutable std::mutex io_mu_;
  std::unique_ptr<StorageIoDriver> io_driver_;
  segment::Index index_;
  segment::Scanner scanner_;
  std::unique_ptr<segment::Writer> writer_;
  mutable IoStatus last_io_status_;
};

}  // namespace mori::umbp
