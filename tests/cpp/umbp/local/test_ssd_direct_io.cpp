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
//
// Covers the direct-I/O SSD path and the two knobs added alongside it
// (verify_crc, tier_io_threads).  The point of direct I/O here is measurement
// honesty: with buffered I/O the tier reads back out of the page cache at many
// times device bandwidth, so any drive-scaling or DRAM-vs-SSD number taken from
// it is meaningless.  These tests pin the correctness properties that make the
// unbuffered path usable, not its speed.

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "umbp/common/aligned_buffer.h"
#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/ssd_tier.h"

namespace fs = std::filesystem;
using mori::umbp::AlignedBuffer;
using mori::umbp::IsDirectIoCompatible;
using mori::umbp::SSDTier;
using mori::umbp::UMBPIoBackend;
using mori::umbp::UMBPSsdConfig;
using namespace mori::umbp::segment;

#define CHECK(cond)                                                                     \
  do {                                                                                  \
    if (!(cond)) {                                                                      \
      std::cerr << "FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

namespace {

std::string MakeDir(const std::string& name) {
  const std::string dir = "/tmp/umbp_test_directio_" + name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

UMBPSsdConfig BaseConfig(const std::string& dir, bool direct_io, bool verify_crc,
                         int tier_io_threads = 4) {
  UMBPSsdConfig cfg;
  cfg.enabled = true;
  cfg.storage_dir = dir;
  cfg.capacity_bytes = 64ULL * 1024 * 1024;
  cfg.segment_size_bytes = 16ULL * 1024 * 1024;
  cfg.io.backend = UMBPIoBackend::Posix;  // no io_uring dependency in unit tests
  cfg.direct_io = direct_io;
  cfg.verify_crc = verify_crc;
  cfg.tier_io_threads = tier_io_threads;
  return cfg;
}

// A value whose size is an exact kRecordAlign multiple, like the KV pages this
// tier actually stores — the case that reads straight into the caller's buffer
// with no bounce.
std::vector<char> AlignedValue(char fill, size_t blocks = 2) {
  return std::vector<char>(static_cast<size_t>(kRecordAlign) * blocks, fill);
}

}  // namespace

// The v3 layout must put every record and every value on an alignment boundary,
// regardless of key length — that is the property O_DIRECT depends on, and it
// has to hold for the buffered path too so one directory works either way.
void test_layout_is_aligned() {
  std::cout << "test_layout_is_aligned... ";
  for (size_t key_len : {1u, 9u, 31u, 32u, 33u, 100u, 4063u, 4064u, 4065u, 9000u}) {
    for (size_t value : {1u, 512u, 4095u, 4096u, 4097u, 1u << 20}) {
      CHECK(PrefixBytes(key_len) % kRecordAlign == 0);
      CHECK(PrefixBytes(key_len) >= sizeof(RecordHeader) + key_len);
      CHECK(PaddedValueBytes(value) % kRecordAlign == 0);
      CHECK(PaddedValueBytes(value) >= value);
      CHECK(RecordBytes(key_len, value) % kRecordAlign == 0);
    }
  }
  std::cout << "OK\n";
}

void test_aligned_buffer() {
  std::cout << "test_aligned_buffer... ";
  AlignedBuffer buf(100);
  CHECK(reinterpret_cast<uintptr_t>(buf.data()) % kRecordAlign == 0);
  CHECK(buf.size() == 100);
  CHECK(buf.padded_size() == kRecordAlign);
  // Padding must be zeroed, so record padding never leaks heap bytes to disk.
  for (size_t i = buf.size(); i < buf.padded_size(); ++i) CHECK(buf.data()[i] == 0);

  CHECK(IsDirectIoCompatible(buf.data(), static_cast<size_t>(kRecordAlign)));
  CHECK(!IsDirectIoCompatible(buf.data(), 100));               // length unaligned
  CHECK(!IsDirectIoCompatible(buf.data() + 1, kRecordAlign));  // address unaligned

  AlignedBuffer moved(std::move(buf));
  CHECK(reinterpret_cast<uintptr_t>(moved.data()) % kRecordAlign == 0);
  std::cout << "OK\n";
}

// Round-trip through a tier, once buffered and once unbuffered.  Both must
// return byte-identical data and survive a reopen (which exercises the scanner's
// aligned prefix reads on an O_DIRECT fd).
void test_round_trip(bool direct_io) {
  std::cout << "test_round_trip(direct_io=" << direct_io << ")... ";
  const std::string dir = MakeDir(direct_io ? "rt_direct" : "rt_buffered");
  auto cfg = BaseConfig(dir, direct_io, /*verify_crc=*/true);

  const std::vector<std::string> keys = {"k/aligned", "k/odd", "k/big"};
  std::vector<std::vector<char>> values = {AlignedValue('A'), std::vector<char>(1234, 'B'),
                                           AlignedValue('C', 5)};

  {
    SSDTier tier(dir, cfg.capacity_bytes, cfg);
    if (direct_io && !tier.direct_io_active()) {
      std::cout << "SKIPPED (filesystem rejects O_DIRECT)\n";
      return;
    }
    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (auto& v : values) {
      ptrs.push_back(v.data());
      sizes.push_back(v.size());
    }
    CHECK(tier.WriteBatch(keys, ptrs, sizes));

    // Read back into an aligned destination (the no-bounce path) and into a
    // deliberately misaligned one (the bounce path); both must be exact.
    AlignedBuffer aligned_dst(values[0].size());
    CHECK(tier.ReadIntoPtr(keys[0], reinterpret_cast<uintptr_t>(aligned_dst.data()),
                           values[0].size()));
    CHECK(std::memcmp(aligned_dst.data(), values[0].data(), values[0].size()) == 0);

    std::vector<char> odd_dst(values[1].size());
    CHECK(tier.ReadIntoPtr(keys[1], reinterpret_cast<uintptr_t>(odd_dst.data()), values[1].size()));
    CHECK(std::memcmp(odd_dst.data(), values[1].data(), values[1].size()) == 0);

    // Batch read, mixing aligned and unaligned destinations in one call.
    AlignedBuffer b0(values[0].size());
    std::vector<char> b1(values[1].size());
    AlignedBuffer b2(values[2].size());
    std::vector<uintptr_t> dsts = {reinterpret_cast<uintptr_t>(b0.data()),
                                   reinterpret_cast<uintptr_t>(b1.data()),
                                   reinterpret_cast<uintptr_t>(b2.data())};
    auto ok =
        tier.ReadBatchIntoPtr(keys, dsts, {values[0].size(), values[1].size(), values[2].size()});
    CHECK(ok.size() == 3);
    for (bool r : ok) CHECK(r);
    CHECK(std::memcmp(b0.data(), values[0].data(), values[0].size()) == 0);
    CHECK(std::memcmp(b1.data(), values[1].data(), values[1].size()) == 0);
    CHECK(std::memcmp(b2.data(), values[2].data(), values[2].size()) == 0);

    CHECK(tier.Read(keys[1]) == values[1]);
  }

  // Reopen: the scanner has to rebuild the index from the padded records, over
  // an O_DIRECT fd when direct I/O is on.  A regression here shows up as a
  // silent empty tier, not an error, so assert every key came back.
  {
    SSDTier tier(dir, cfg.capacity_bytes, cfg);
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(tier.Exists(keys[i]));
      CHECK(tier.Read(keys[i]) == values[i]);
    }
  }

  fs::remove_all(dir);
  std::cout << "OK\n";
}

// verify_crc=0 must round-trip, and — the part that is easy to get wrong — a
// store written with checksums off must stay readable by a tier with checksums
// on.  Without the kFlagNoCrc marker every such key would read back as corrupt.
void test_crc_disabled_round_trip() {
  std::cout << "test_crc_disabled_round_trip... ";
  const std::string dir = MakeDir("crc_off");
  const std::string key = "k/nocrc";
  const auto value = AlignedValue('Z');

  {
    auto cfg = BaseConfig(dir, /*direct_io=*/false, /*verify_crc=*/false);
    SSDTier tier(dir, cfg.capacity_bytes, cfg);
    CHECK(tier.Write(key, value.data(), value.size()));
    CHECK(tier.Read(key) == value);
  }

  // Same directory, checksums back on.
  {
    auto cfg = BaseConfig(dir, /*direct_io=*/false, /*verify_crc=*/true);
    SSDTier tier(dir, cfg.capacity_bytes, cfg);
    CHECK(tier.Exists(key));
    CHECK(tier.Read(key) == value);

    // A record written *with* a checksum is still verified by this tier.
    const std::string key2 = "k/withcrc";
    const auto value2 = AlignedValue('Y');
    CHECK(tier.Write(key2, value2.data(), value2.size()));
    CHECK(tier.Read(key2) == value2);
  }

  fs::remove_all(dir);
  std::cout << "OK\n";
}

// tier_io_threads only changes how the CPU phases are scheduled; results must be
// identical at any thread count.  1 vs 8 also exercises the ParallelFor
// boundary between the inline loop and the worker path.
void test_tier_threads_are_result_invariant() {
  std::cout << "test_tier_threads_are_result_invariant... ";
  std::vector<std::string> keys;
  std::vector<std::vector<char>> values;
  for (int i = 0; i < 17; ++i) {  // deliberately not a multiple of any thread count
    keys.push_back("k/" + std::to_string(i));
    values.push_back(AlignedValue(static_cast<char>('a' + i % 26)));
  }

  for (int threads : {1, 2, 8}) {
    const std::string dir = MakeDir("threads_" + std::to_string(threads));
    auto cfg = BaseConfig(dir, /*direct_io=*/false, /*verify_crc=*/true, threads);
    SSDTier tier(dir, cfg.capacity_bytes, cfg);

    std::vector<const void*> ptrs;
    std::vector<size_t> sizes;
    for (auto& v : values) {
      ptrs.push_back(v.data());
      sizes.push_back(v.size());
    }
    CHECK(tier.WriteBatch(keys, ptrs, sizes));

    std::vector<AlignedBuffer> dst_bufs(keys.size());
    std::vector<uintptr_t> dsts;
    for (size_t i = 0; i < keys.size(); ++i) {
      dst_bufs[i].Resize(values[i].size());
      dsts.push_back(reinterpret_cast<uintptr_t>(dst_bufs[i].data()));
    }
    auto ok = tier.ReadBatchIntoPtr(keys, dsts, sizes);
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(ok[i]);
      CHECK(std::memcmp(dst_bufs[i].data(), values[i].data(), values[i].size()) == 0);
    }
    fs::remove_all(dir);
  }
  std::cout << "OK\n";
}

// Capacity is charged in padded on-disk bytes, so a store of small values cannot
// admit more than the drive can hold.  Before v3 this was accounted in raw value
// bytes, which for a 512 B value under-counted the footprint by 16x.
void test_capacity_charges_padded_bytes() {
  std::cout << "test_capacity_charges_padded_bytes... ";
  const std::string dir = MakeDir("capacity");
  const std::string key_a = "a";
  const size_t value_size = 512;
  const size_t per_record = static_cast<size_t>(RecordBytes(key_a.size(), value_size));
  CHECK(per_record == 2 * kRecordAlign);  // 4 KiB prefix + 4 KiB padded value

  auto cfg = BaseConfig(dir, /*direct_io=*/false, /*verify_crc=*/true);
  cfg.capacity_bytes = 2 * per_record;
  SSDTier tier(dir, cfg.capacity_bytes, cfg);

  const std::vector<char> value(value_size, 'q');
  CHECK(tier.Write("a", value.data(), value.size()));
  CHECK(tier.Write("b", value.data(), value.size()));
  // Third write must be refused: the tier is full in on-disk terms even though
  // only 1536 raw value bytes have been stored.
  CHECK(!tier.Write("c", value.data(), value.size()));
  CHECK(tier.Capacity().first == 2 * per_record);

  fs::remove_all(dir);
  std::cout << "OK\n";
}

int main() {
  test_layout_is_aligned();
  test_aligned_buffer();
  test_round_trip(/*direct_io=*/false);
  test_round_trip(/*direct_io=*/true);
  test_crc_disabled_round_trip();
  test_tier_threads_are_result_invariant();
  test_capacity_charges_padded_bytes();
  std::cout << "All direct-IO SSD tests passed\n";
  return 0;
}
