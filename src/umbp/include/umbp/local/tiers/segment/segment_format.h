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
#include <string>

namespace mori::umbp::segment {

constexpr uint32_t kRecordMagic = 0x554D4250;  // "UMBP"
constexpr uint16_t kRecordVersion = 1;
constexpr uint16_t kFlagCommitted = 1;

struct RecordHeader {
  uint32_t magic = 0;
  uint16_t version = 1;
  uint16_t flags = 0;
  uint32_t key_len = 0;
  uint32_t value_size = 0;
  uint32_t crc32 = 0;
  uint32_t reserved = 0;
  uint64_t generation = 0;
};
static_assert(sizeof(RecordHeader) == 32, "unexpected padding in RecordHeader");

uint32_t CrcUpdate(const void* data, size_t size, uint32_t crc = 0xFFFFFFFFu);
uint32_t ComputeRecordCrc32(const std::string& key, const void* value, size_t value_size);
std::string BuildFileName(uint64_t segment_id);

}  // namespace mori::umbp::segment
