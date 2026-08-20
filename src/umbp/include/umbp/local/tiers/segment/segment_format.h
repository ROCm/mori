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
// v2 switched the record checksum from CRC-32/ISO-HDLC to CRC-32C so it can use
// the SSE4.2 crc32 instruction. The checksum values differ, so v1 records must
// not be verified with the v2 routine; the scanner drops them on version
// mismatch and the segment is refilled.
// v3 padded the layout to kRecordAlign so every record starts, and every value
// starts and ends, on an alignment boundary — the precondition for O_DIRECT.
// The padding is unconditional (not gated on direct_io) so one directory can be
// read back with either setting; only the reader's buffer alignment differs.
// v2 records are dropped by the scanner on version mismatch, same as v1.
constexpr uint16_t kRecordVersion = 3;
constexpr uint16_t kFlagCommitted = 1;
// Set when the writer skipped checksumming (UMBP_SSD_VERIFY_CRC=0).  Recorded
// per record so a store written with checksums off is still readable by a
// process with them on, instead of reporting every key as corrupt.
constexpr uint16_t kFlagNoCrc = 2;

// On-disk alignment for record and value boundaries.  Deliberately a fixed
// constant rather than the device's reported DIO requirement: the layout must
// not change meaning when the same directory moves between devices.  4096
// covers every realistic requirement (512 and 4096 both divide it), and the
// O_DIRECT probe only has to confirm the device needs no more than this.
constexpr uint64_t kRecordAlign = 4096;

constexpr uint64_t AlignUp(uint64_t v, uint64_t align = kRecordAlign) {
  return (v + align - 1) & ~(align - 1);
}

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

// v3 record layout, all quantities multiples of kRecordAlign:
//
//   [ prefix: RecordHeader | key | pad ][ value | pad ]
//   ^ record_offset                     ^ value_offset
//
// A record therefore occupies PrefixBytes() + PaddedValueBytes() and the next
// record starts on an aligned boundary by construction.
constexpr uint64_t PrefixBytes(size_t key_len) {
  return AlignUp(sizeof(RecordHeader) + static_cast<uint64_t>(key_len));
}
constexpr uint64_t PaddedValueBytes(size_t value_size) {
  return AlignUp(static_cast<uint64_t>(value_size));
}
constexpr uint64_t RecordBytes(size_t key_len, size_t value_size) {
  return PrefixBytes(key_len) + PaddedValueBytes(value_size);
}

// Streaming CRC-32C update; hardware-accelerated where SSE4.2 is available and
// bit-identical on the portable fallback. Not complemented -- callers finish a
// digest by inverting the returned value, as ComputeRecordCrc32 does.
uint32_t CrcUpdate(const void* data, size_t size, uint32_t crc = 0xFFFFFFFFu);
uint32_t ComputeRecordCrc32(const std::string& key, const void* value, size_t value_size);
std::string BuildFileName(uint64_t segment_id);

}  // namespace mori::umbp::segment
