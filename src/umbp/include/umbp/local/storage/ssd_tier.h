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
#include "umbp/local/storage/io/status.h"
#include "umbp/local/storage/io/storage_io_driver.h"
#include "umbp/local/storage/segment/segment_index.h"
#include "umbp/local/storage/segment/segment_scanner.h"
#include "umbp/local/storage/segment/segment_writer.h"
#include "umbp/local/storage/tier_backend.h"

namespace mori::umbp {

enum class SSDAccessMode : int {
  ReadWrite = 0,
  ReadOnlyShared = 1,
};

class SSDTier : public TierBackend {
 public:
  SSDTier(const std::string& dir, size_t capacity, const UMBPConfig& config,
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
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;
  std::vector<char> Read(const std::string& key) override;
  TierCapabilities Capabilities() const override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;
  const IoStatus& LastIoStatus() const { return last_io_status_; }

 private:
  bool IsReadOnlyShared() const { return access_mode_ == SSDAccessMode::ReadOnlyShared; }
  bool ShouldSyncOnWrite() const {
    return config_.ssd.durability.mode == UMBPDurabilityMode::Strict;
  }

  bool EnsureActiveSegment(size_t need_bytes);
  bool RefreshFromDiskLocked(bool force_full_rescan);
  bool OpenOrCreateSegmentLocked(uint64_t segment_id);

  bool RefreshFollowerLocked() const;
  segment::Meta* GetSegmentLocked(uint64_t segment_id);
  const segment::Meta* GetSegmentLocked(uint64_t segment_id) const;
  bool ReadRecordLocked(const std::string& key, void* dst, size_t size, uint32_t expected_crc,
                        uint64_t value_offset, int read_fd) const;
  void RememberStatus(IoStatus status) const;

  std::string dir_;
  size_t capacity_;
  UMBPConfig config_;
  SSDAccessMode access_mode_;

  mutable std::mutex mu_;
  mutable std::mutex io_mu_;
  std::unique_ptr<StorageIoDriver> io_driver_;
  segment::Index index_;
  segment::Scanner scanner_;
  std::unique_ptr<segment::Writer> writer_;
  mutable IoStatus last_io_status_;
};

}  // namespace mori::umbp
