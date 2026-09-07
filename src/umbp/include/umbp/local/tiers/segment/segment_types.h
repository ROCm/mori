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
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mori::umbp::segment {

struct KeyMeta {
  uint64_t segment_id = 0;
  uint64_t value_offset = 0;
  uint32_t size = 0;
  uint32_t crc32 = 0;
  uint64_t generation = 0;
  // Bytes this record actually occupies in the segment, i.e. `size` plus the
  // header, key and v3 alignment padding.  Capacity is accounted in these rather
  // than in `size` so a store of small values cannot overrun capacity_bytes on
  // disk: padding rounds every record up to a kRecordAlign multiple, which for a
  // 4 KiB value is 2x its size (and for the multi-MiB KV pages this tier
  // normally holds, under 0.1%).
  uint32_t disk_bytes = 0;
  // False for records written with checksumming disabled (kFlagNoCrc).  Readers
  // must skip verification for those regardless of their own verify_crc setting,
  // otherwise every such key reads back as corrupt.
  bool crc_valid = true;
};

struct Meta {
  uint64_t id = 0;
  std::string path;
  int fd = -1;
  uint64_t write_offset = 0;
  uint64_t scanned_offset = 0;
  size_t live_bytes = 0;
};

struct WriteReservation {
  std::string key;
  KeyMeta meta;
  bool had_previous = false;
  KeyMeta previous_meta;
  uint64_t record_offset = 0;
  uint64_t previous_write_offset = 0;
};

}  // namespace mori::umbp::segment
