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
#include "umbp/local/tiers/segment/segment_scanner.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <string>
#include <vector>

#include "umbp/local/tiers/segment/segment_format.h"

namespace fs = std::filesystem;

namespace mori::umbp::segment {

bool Scanner::RefreshFromDisk(const std::string& dir, StorageIoDriver& io_driver, Index& index,
                              bool read_only_shared, bool force_full_rescan,
                              std::string* error_message) const {
  if (force_full_rescan) index.ResetMetadata();

  std::vector<uint64_t> ids;
  if (fs::exists(dir)) {
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (name.rfind("segment_", 0) != 0) continue;
      if (name.size() <= 12 || name.substr(name.size() - 4) != ".log") continue;
      const std::string id_s = name.substr(8, name.size() - 12);
      if (id_s.empty()) continue;
      uint64_t sid = 0;
      auto [ptr, ec] = std::from_chars(id_s.data(), id_s.data() + id_s.size(), sid);
      if (ec != std::errc{} || ptr != id_s.data() + id_s.size()) continue;
      ids.push_back(sid);
    }
  }
  std::sort(ids.begin(), ids.end());

  for (uint64_t sid : ids) {
    if (index.HasKnownSegment(sid)) continue;
    Meta seg;
    seg.id = sid;
    seg.path = dir + "/" + BuildFileName(sid);
    seg.fd = open(seg.path.c_str(), read_only_shared ? O_RDONLY : (O_RDWR | O_CREAT), 0644);
    if (seg.fd < 0) {
      if (error_message) *error_message = "failed to open segment " + seg.path;
      return false;
    }

    struct stat st;
    if (fstat(seg.fd, &st) != 0) {
      close(seg.fd);
      if (error_message) *error_message = "failed to stat segment " + seg.path;
      return false;
    }
    seg.write_offset = static_cast<uint64_t>(st.st_size);
    index.MutableSegments()[sid] = seg;
    index.MarkKnownSegment(sid);
    index.AdvanceNextSegmentId(sid + 1);
  }

  for (auto& kv : index.MutableSegments()) {
    auto& seg = kv.second;
    struct stat st;
    if (fstat(seg.fd, &st) != 0) {
      if (error_message) *error_message = "failed to stat segment during refresh";
      return false;
    }
    uint64_t file_size = static_cast<uint64_t>(st.st_size);
    uint64_t offset = force_full_rescan ? 0 : seg.scanned_offset;

    while (offset + sizeof(RecordHeader) <= file_size) {
      RecordHeader hdr;
      IoStatus hdr_status = io_driver.ReadAt(seg.fd, &hdr, sizeof(hdr), offset);
      if (!hdr_status.ok()) break;
      if (hdr.magic != kRecordMagic || hdr.version != kRecordVersion || hdr.key_len == 0) break;
      const uint64_t rec_size =
          sizeof(RecordHeader) + static_cast<uint64_t>(hdr.key_len) + hdr.value_size;
      if (offset + rec_size > file_size) break;
      if ((hdr.flags & kFlagCommitted) == 0) {
        offset += rec_size;
        continue;
      }

      std::string key;
      key.resize(hdr.key_len);
      IoStatus key_status = io_driver.ReadAt(seg.fd, key.data(), hdr.key_len, offset + sizeof(hdr));
      if (!key_status.ok()) break;

      KeyMeta meta;
      meta.segment_id = seg.id;
      meta.value_offset = offset + sizeof(hdr) + hdr.key_len;
      meta.size = hdr.value_size;
      meta.crc32 = hdr.crc32;
      meta.generation = hdr.generation;
      index.RecordRecoveredEntry(key, meta);
      offset += rec_size;
    }

    seg.scanned_offset = offset;
    seg.write_offset = std::max(seg.write_offset, file_size);
  }

  return true;
}

}  // namespace mori::umbp::segment
