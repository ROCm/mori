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

#include <cstring>
#include <string>
#include <vector>

#include "umbp/common/aligned_buffer.h"
#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/segment/segment_index.h"
#include "umbp/storage/io/storage_io_driver.h"

namespace mori::umbp::segment {

struct PreparedRecord {
  // Aligned rather than std::vector<char>: with O_DIRECT the source buffer of a
  // write must sit on a kRecordAlign boundary, which the default allocator does
  // not guarantee.  Build() sizes it to RecordBytes(), already an alignment
  // multiple, so the whole buffer can go to the device in one op.
  AlignedBuffer record;
  WriteReservation reservation;
  uint32_t crc32 = 0;     // set by Build, consumed by Reserve
  bool crc_valid = true;  // false when checksumming was skipped
};

class Writer {
 public:
  // `compute_crc` false skips checksumming on this store's writes and stamps
  // kFlagNoCrc, so readers know to skip verification for these records.
  explicit Writer(StorageIoDriver& io_driver, bool compute_crc = true)
      : io_driver_(io_driver), compute_crc_(compute_crc) {}

  // Phase 1a (NO lock held): checksum the record and assemble its on-disk bytes.
  // Pure CPU over caller-owned memory — it touches no Index or Meta state, so it
  // must run outside the tier mutex.  This is the expensive half (a CRC and a
  // full copy of the value); keeping it under the lock would block every
  // concurrent reader on the same drive for the duration of the batch.
  // `generation` is left zero and stamped by Reserve.
  void Build(const std::string& key, const void* data, size_t size, PreparedRecord* out) const;

  // Phase 1b (caller holds mu_): reserve index/segment space for an already-built
  // record and stamp the header field that depends on the reservation.  Cheap:
  // no checksum, no payload copy.  Returns false if capacity is exhausted.
  bool Reserve(const std::string& key, size_t size, Meta* segment_meta, Index& index,
               PreparedRecord* out) const;

  // Build + Reserve in one call, for callers already holding mu_.  Prefer the
  // split form on any path where the lock is contended.
  bool Prepare(const std::string& key, const void* data, size_t size, Meta* segment_meta,
               Index& index, PreparedRecord* out) const;

  // Phase 2 (caller holds io_mu_ only): write the prepared record to disk.
  IoStatus WriteRecord(int fd, const PreparedRecord& pr, bool should_sync) const;

  // Phase 2 batch variant: write multiple prepared records to disk.
  IoStatus WriteRecords(int fd, const std::vector<PreparedRecord>& records, bool should_sync) const;

 private:
  StorageIoDriver& io_driver_;
  bool compute_crc_ = true;
};

}  // namespace mori::umbp::segment
