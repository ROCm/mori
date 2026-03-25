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

namespace segment {

struct KeyMeta {
  uint64_t segment_id = 0;
  uint64_t value_offset = 0;
  uint32_t size = 0;
  uint32_t crc32 = 0;
  uint64_t generation = 0;
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

}  // namespace segment
