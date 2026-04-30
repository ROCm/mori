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

namespace mori::umbp::segment {

bool Writer::Prepare(const std::string& key, const void* data, size_t size, Meta* segment_meta,
                     Index& index, PreparedRecord* out) const {
  if (!segment_meta || !out) return false;

  const size_t record_size = sizeof(RecordHeader) + key.size() + size;
  const uint32_t crc32 = ComputeRecordCrc32(key, data, size);

  if (!index.PrepareWrite(key, size, key.size(), crc32, segment_meta, &out->reservation))
    return false;

  RecordHeader hdr;
  hdr.magic = kRecordMagic;
  hdr.version = kRecordVersion;
  hdr.flags = kFlagCommitted;
  hdr.key_len = static_cast<uint32_t>(key.size());
  hdr.value_size = static_cast<uint32_t>(size);
  hdr.crc32 = crc32;
  hdr.generation = out->reservation.meta.generation;

  out->record.resize(record_size);
  std::memcpy(out->record.data(), &hdr, sizeof(hdr));
  std::memcpy(out->record.data() + sizeof(hdr), key.data(), key.size());
  std::memcpy(out->record.data() + sizeof(hdr) + key.size(), data, size);
  return true;
}

IoStatus Writer::WriteRecord(int fd, const PreparedRecord& pr, bool should_sync) const {
  IoStatus status =
      io_driver_.WriteAt(fd, pr.record.data(), pr.record.size(), pr.reservation.record_offset);
  if (!status.ok()) return status;
  if (should_sync) return io_driver_.Sync(fd);
  return IoStatus::Ok();
}

IoStatus Writer::WriteRecords(int fd, const std::vector<PreparedRecord>& records,
                              bool should_sync) const {
  std::vector<IoWriteOp> ops;
  ops.reserve(records.size());
  for (const auto& pr : records) {
    ops.push_back({fd, pr.record.data(), pr.record.size(), pr.reservation.record_offset});
  }

  IoStatus status = io_driver_.WriteBatch(ops);
  if (!status.ok()) return status;
  if (should_sync) return io_driver_.SyncMany({fd});
  return IoStatus::Ok();
}

}  // namespace mori::umbp::segment
