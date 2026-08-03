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
#include "umbp/local/tiers/segment/segment_writer.h"

#include <cstddef>

#include "umbp/common/ssd_perf.h"

namespace mori::umbp::segment {

void Writer::Build(const std::string& key, const void* data, size_t size,
                   PreparedRecord* out) const {
  if (!out) return;

  const size_t record_size = sizeof(RecordHeader) + key.size() + size;
  out->crc32 = ComputeRecordCrc32(key, data, size);

  RecordHeader hdr;
  hdr.magic = kRecordMagic;
  hdr.version = kRecordVersion;
  hdr.flags = kFlagCommitted;
  hdr.key_len = static_cast<uint32_t>(key.size());
  hdr.value_size = static_cast<uint32_t>(size);
  hdr.crc32 = out->crc32;
  hdr.generation = 0;  // stamped by Reserve, which allocates the generation

  out->record.resize(record_size);
  std::memcpy(out->record.data(), &hdr, sizeof(hdr));
  std::memcpy(out->record.data() + sizeof(hdr), key.data(), key.size());
  std::memcpy(out->record.data() + sizeof(hdr) + key.size(), data, size);
}

bool Writer::Reserve(const std::string& key, size_t size, Meta* segment_meta, Index& index,
                     PreparedRecord* out) const {
  if (!segment_meta || !out) return false;
  if (out->record.size() < sizeof(RecordHeader)) return false;  // Build() not run

  if (!index.PrepareWrite(key, size, key.size(), out->crc32, segment_meta, &out->reservation))
    return false;

  // Patch the one header field the reservation produces, in place, instead of
  // rebuilding the whole record.
  const uint64_t generation = out->reservation.meta.generation;
  std::memcpy(out->record.data() + offsetof(RecordHeader, generation), &generation,
              sizeof(generation));
  return true;
}

bool Writer::Prepare(const std::string& key, const void* data, size_t size, Meta* segment_meta,
                     Index& index, PreparedRecord* out) const {
  if (!segment_meta || !out) return false;
  Build(key, data, size, out);
  return Reserve(key, size, segment_meta, index, out);
}

IoStatus Writer::WriteRecord(int fd, const PreparedRecord& pr, bool should_sync,
                             double* sync_ms_out) const {
  IoStatus status =
      io_driver_.WriteAt(fd, pr.record.data(), pr.record.size(), pr.reservation.record_offset);
  if (!status.ok()) return status;
  if (!should_sync) return IoStatus::Ok();
  const auto t_sync = ssdperf::Now();
  status = io_driver_.Sync(fd);
  if (sync_ms_out) *sync_ms_out += ssdperf::MsSince(t_sync);
  return status;
}

IoStatus Writer::WriteRecords(int fd, const std::vector<PreparedRecord>& records, bool should_sync,
                              double* sync_ms_out) const {
  std::vector<IoWriteOp> ops;
  ops.reserve(records.size());
  for (const auto& pr : records) {
    ops.push_back({fd, pr.record.data(), pr.record.size(), pr.reservation.record_offset});
  }

  IoStatus status = io_driver_.WriteBatch(ops);
  if (!status.ok()) return status;
  if (!should_sync) return IoStatus::Ok();
  const auto t_sync = ssdperf::Now();
  status = io_driver_.SyncMany({fd});
  if (sync_ms_out) *sync_ms_out += ssdperf::MsSince(t_sync);
  return status;
}

}  // namespace mori::umbp::segment
