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

// Pins the segment record checksum to CRC-32C.
//
// CrcUpdate has two implementations selected at runtime (SSE4.2 hardware and a
// portable table).  A segment written on a host with hardware support must
// verify on one without it, so the two MUST agree bit-for-bit -- this test
// compares whichever path this host selects against an independent bitwise
// reference, which is the shared ground truth for both.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/segment/segment_index.h"
#include "umbp/local/tiers/segment/segment_writer.h"
#include "umbp/storage/io/storage_io_driver.h"

using namespace mori::umbp::segment;

namespace {

// Independent bitwise CRC-32C (Castagnoli, reflected).  Deliberately naive and
// written from the polynomial rather than shared with the production code, so a
// change there cannot silently "fix" this reference too.
uint32_t ReferenceCrc32c(const void* data, size_t size, uint32_t crc = 0xFFFFFFFFu) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    crc ^= p[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0x82F63B78u & (~(crc & 1u) + 1u));
  }
  return crc;
}

void test_known_vector() {
  std::cout << "test_known_vector... ";
  // The standard CRC-32C check value: crc("123456789") == 0xE3069283.
  // Catches an accidental revert to CRC-32/ISO-HDLC (0xEDB88320), whose check
  // value is 0xCBF43926.
  const std::string s = "123456789";
  assert(~CrcUpdate(s.data(), s.size()) == 0xE3069283u);
  std::cout << "OK\n";
}

void test_matches_reference_all_tail_lengths() {
  std::cout << "test_matches_reference_all_tail_lengths... ";
  // The hardware path consumes 8 bytes at a time and finishes byte-wise, so
  // every length mod 8 (and every length below one block) is a distinct case.
  std::mt19937 rng(0xC0FFEE);
  std::vector<uint8_t> buf(8192);
  for (auto& b : buf) b = static_cast<uint8_t>(rng());

  for (size_t n = 0; n <= 512; ++n) {
    assert(CrcUpdate(buf.data(), n) == ReferenceCrc32c(buf.data(), n));
  }
  for (size_t n : {1023u, 1024u, 1025u, 4095u, 4096u, 8191u, 8192u}) {
    assert(CrcUpdate(buf.data(), n) == ReferenceCrc32c(buf.data(), n));
  }
  // Unaligned starts: the hardware path must not assume 8-byte alignment.
  for (size_t off = 1; off < 9; ++off) {
    assert(CrcUpdate(buf.data() + off, 777) == ReferenceCrc32c(buf.data() + off, 777));
  }
  std::cout << "OK\n";
}

void test_streaming_equals_one_shot() {
  std::cout << "test_streaming_equals_one_shot... ";
  // ComputeRecordCrc32 seeds the value digest with the key digest, so chunked
  // updates must compose exactly like a single pass.
  std::vector<uint8_t> buf(10007);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(i * 31 + 7);

  const uint32_t one_shot = CrcUpdate(buf.data(), buf.size());
  uint32_t chunked = 0xFFFFFFFFu;
  for (size_t off = 0; off < buf.size();) {
    const size_t take = std::min<size_t>(1 + (off % 97), buf.size() - off);
    chunked = CrcUpdate(buf.data() + off, take, chunked);
    off += take;
  }
  assert(one_shot == chunked);
  std::cout << "OK\n";
}

void test_key_is_bound_to_record() {
  std::cout << "test_key_is_bound_to_record... ";
  // The read path compares against a checksum taken over key+value, which is
  // what turns "bytes are intact" into "these are the bytes for THIS key" --
  // the guard against a stale index entry pointing at another record.
  std::vector<uint8_t> value(4096, 0xAB);
  const uint32_t a = ComputeRecordCrc32("prefix/block/0001", value.data(), value.size());
  const uint32_t b = ComputeRecordCrc32("prefix/block/0002", value.data(), value.size());
  assert(a != b);

  // ...and a single flipped payload byte must change the digest.
  std::vector<uint8_t> corrupted = value;
  corrupted[2048] ^= 0x01;
  assert(ComputeRecordCrc32("prefix/block/0001", corrupted.data(), corrupted.size()) != a);
  std::cout << "OK\n";
}

void test_empty_inputs() {
  std::cout << "test_empty_inputs... ";
  // Zero-length update is the identity on the running digest.
  assert(CrcUpdate(nullptr, 0, 0x12345678u) == 0x12345678u);
  const std::string key = "k";
  assert(ComputeRecordCrc32(key, nullptr, 0) == ~CrcUpdate(key.data(), key.size()));
  std::cout << "OK\n";
}

void test_build_reserve_split() {
  std::cout << "test_build_reserve_split... ";
  // Build() runs outside the tier mutex and Reserve() inside it, so Build must
  // produce the complete record except for `generation`, which only exists once
  // the reservation is taken.  Reserve patches it in place -- if that patch were
  // lost, the scanner would recover every record with generation 0 and
  // RecordRecoveredEntry's ordering check would keep the wrong version.
  auto driver = mori::umbp::CreateStorageIoDriver(mori::umbp::UMBPIoBackend::Posix, 0);
  Writer writer(*driver);

  const std::string key = "block/abc";
  std::vector<uint8_t> value(256, 0x5A);

  PreparedRecord pr;
  writer.Build(key, value.data(), value.size(), &pr);

  RecordHeader built{};
  std::memcpy(&built, pr.record.data(), sizeof(built));
  assert(built.magic == kRecordMagic);
  assert(built.version == kRecordVersion);
  assert(built.flags == kFlagCommitted);
  assert(built.key_len == key.size());
  assert(built.value_size == value.size());
  assert(built.crc32 == ComputeRecordCrc32(key, value.data(), value.size()));
  assert(pr.crc32 == built.crc32);
  assert(built.generation == 0);  // not yet reserved
  // v3 pads the record so the value starts on a kRecordAlign boundary.
  assert(pr.record.size() == RecordBytes(key.size(), value.size()));
  assert(pr.record.size() % kRecordAlign == 0);
  // Key and value bytes are already in place before the lock is taken.
  assert(std::memcmp(pr.record.data() + sizeof(RecordHeader), key.data(), key.size()) == 0);
  assert(std::memcmp(pr.record.data() + PrefixBytes(key.size()), value.data(), value.size()) == 0);

  // Reserve stamps a real generation into the already-built buffer.
  Index index(1 << 20);
  Meta seg;
  seg.id = 0;
  seg.write_offset = 0;
  assert(writer.Reserve(key, value.size(), &seg, index, &pr));

  RecordHeader reserved{};
  std::memcpy(&reserved, pr.record.data(), sizeof(reserved));
  assert(reserved.generation != 0);
  assert(reserved.generation == pr.reservation.meta.generation);
  // Reserve must touch nothing else.
  assert(reserved.crc32 == built.crc32);
  assert(reserved.key_len == built.key_len);
  assert(reserved.value_size == built.value_size);
  assert(std::memcmp(pr.record.data() + PrefixBytes(key.size()), value.data(), value.size()) == 0);
  // The reservation must put the value on an aligned offset -- the precondition
  // that makes the record readable and writable with O_DIRECT.
  assert(pr.reservation.record_offset % kRecordAlign == 0);
  assert(pr.reservation.meta.value_offset % kRecordAlign == 0);
  assert(seg.write_offset % kRecordAlign == 0);

  // Generations strictly advance, so recovery can order two writes of one key.
  PreparedRecord pr2;
  writer.Build(key, value.data(), value.size(), &pr2);
  assert(writer.Reserve(key, value.size(), &seg, index, &pr2));
  assert(pr2.reservation.meta.generation > pr.reservation.meta.generation);

  // Reserve on a record that was never built must fail rather than write a
  // generation past the end of an empty buffer.
  PreparedRecord empty;
  assert(!writer.Reserve(key, value.size(), &seg, index, &empty));

  std::cout << "OK\n";
}

}  // namespace

int main() {
  test_known_vector();
  test_matches_reference_all_tail_lengths();
  test_streaming_equals_one_shot();
  test_key_is_bound_to_record();
  test_empty_inputs();
  test_build_reserve_split();
  std::cout << "All segment CRC tests passed\n";
  return 0;
}
