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
#include "umbp/local/tiers/segment/segment_format.h"

#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(__i386__)
#include <nmmintrin.h>
#define UMBP_SEGMENT_CRC32C_X86 1
#endif

namespace mori::umbp::segment {

namespace {

// CRC-32C (Castagnoli), reflected polynomial. This is the polynomial the SSE4.2
// crc32 instruction implements, so the table path and the hardware path below
// produce identical checksums: a segment written on a host with hardware support
// still verifies on one without it.
constexpr uint32_t kCrc32cPoly = 0x82F63B78u;

struct Crc32cTable {
  uint32_t entry[256];

  constexpr Crc32cTable() : entry() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (kCrc32cPoly & (~(c & 1u) + 1u));
      entry[i] = c;
    }
  }
};

constexpr Crc32cTable kCrc32cTable{};

uint32_t CrcUpdateTable(const uint8_t* p, size_t size, uint32_t crc) {
  for (size_t i = 0; i < size; ++i) crc = (crc >> 8) ^ kCrc32cTable.entry[(crc ^ p[i]) & 0xFFu];
  return crc;
}

#ifdef UMBP_SEGMENT_CRC32C_X86
__attribute__((target("sse4.2"))) uint32_t CrcUpdateHw(const uint8_t* p, size_t size,
                                                       uint32_t crc) {
  uint64_t acc = crc;
  while (size >= sizeof(uint64_t)) {
    uint64_t chunk;
    std::memcpy(&chunk, p, sizeof(chunk));
    acc = _mm_crc32_u64(acc, chunk);
    p += sizeof(uint64_t);
    size -= sizeof(uint64_t);
  }
  uint32_t tail = static_cast<uint32_t>(acc);
  while (size--) tail = _mm_crc32_u8(tail, *p++);
  return tail;
}

bool DetectSse42() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("sse4.2");
}

const bool kHasSse42 = DetectSse42();
#endif  // UMBP_SEGMENT_CRC32C_X86

}  // namespace

uint32_t CrcUpdate(const void* data, size_t size, uint32_t crc) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
#ifdef UMBP_SEGMENT_CRC32C_X86
  if (kHasSse42) return CrcUpdateHw(p, size, crc);
#endif
  return CrcUpdateTable(p, size, crc);
}

uint32_t ComputeRecordCrc32(const std::string& key, const void* value, size_t value_size) {
  uint32_t crc = CrcUpdate(key.data(), key.size());
  return ~CrcUpdate(value, value_size, crc);
}

std::string BuildFileName(uint64_t segment_id) {
  return "segment_" + std::to_string(segment_id) + ".log";
}

}  // namespace mori::umbp::segment
