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
// The file SSD tier gives disk back a segment file at a time (#755): on-disk
// accounting, deletion of emptied segments, and the eviction order that
// empties them.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/sharded_ssd_tier.h"
#include "umbp/local/tiers/ssd_tier.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
// A 60 KiB value plus its 4 KiB prefix is a 64 KiB record, so a 1 MiB segment
// holds exactly 16 and key i of a sequential fill lives in segment i / 16.
constexpr size_t kValue = 60 * kKiB;
constexpr size_t kRecord = 64 * kKiB;
constexpr size_t kPerSegment = 16;

std::string Key(int i) { return "k" + std::to_string(i); }

// Every byte depends on the key, so a read that returns another record's bytes
// cannot pass for this one.
std::vector<char> ValueFor(int i, size_t size = kValue) {
  std::vector<char> v(size);
  for (size_t b = 0; b < size; ++b) v[b] = static_cast<char>((i * 131 + b * 7) & 0xff);
  return v;
}

class SegmentReclaimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("umbp_seg_reclaim_" + std::to_string(getpid()) + "_" +
            ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  UMBPSsdConfig Config(size_t capacity, size_t segment, bool verify_crc = true) const {
    UMBPSsdConfig cfg;
    cfg.enabled = true;
    cfg.storage_dir = dir_.string();
    cfg.capacity_bytes = capacity;
    cfg.segment_size_bytes = segment;
    cfg.direct_io = false;
    cfg.verify_crc = verify_crc;
    cfg.io.backend = UMBPIoBackend::Posix;
    cfg.durability.mode = UMBPDurabilityMode::Relaxed;
    return cfg;
  }

  std::unique_ptr<SSDTier> MakeTier(size_t capacity = 8 * kMiB, size_t segment = 1 * kMiB,
                                    bool verify_crc = true) const {
    return std::make_unique<SSDTier>(dir_.string(), capacity,
                                     Config(capacity, segment, verify_crc));
  }

  size_t SegmentFiles(const fs::path& dir) const {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().filename().string().rfind("segment_", 0) == 0) ++n;
    }
    return n;
  }
  size_t SegmentFiles() const { return SegmentFiles(dir_); }

  uint64_t BytesOnDisk() const {
    uint64_t total = 0;
    for (const auto& e : fs::directory_iterator(dir_)) {
      if (e.path().filename().string().rfind("segment_", 0) == 0) total += e.file_size();
    }
    return total;
  }

  // Descriptors this process holds on files under the tier directory,
  // including deleted ones (readlink then ends in " (deleted)").
  size_t OpenFdsUnderDir() const {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator("/proc/self/fd")) {
      std::error_code ec;
      const auto target = fs::read_symlink(e.path(), ec);
      if (!ec && target.string().rfind(dir_.string(), 0) == 0) ++n;
    }
    return n;
  }

  static bool Write(SSDTier& tier, int i) {
    const auto v = ValueFor(i);
    return tier.Write(Key(i), v.data(), v.size());
  }

  static bool ReadsBack(TierBackend& tier, int i) {
    std::vector<char> buf(kValue);
    if (!tier.ReadIntoPtr(Key(i), reinterpret_cast<uintptr_t>(buf.data()), buf.size())) {
      return false;
    }
    return buf == ValueFor(i);
  }

  fs::path dir_;
};

TEST_F(SegmentReclaimTest, RecordSizesMatchTheLayoutTheseTestsAssume) {
  EXPECT_EQ(segment::RecordBytes(Key(0).size(), kValue), kRecord);
  EXPECT_EQ(segment::RecordBytes(Key(99999).size(), kValue), kRecord);
}

// #755's repro: write and evict far more than the capacity.  Disk use, segment
// files and open fds used to grow with every byte ever written.
TEST_F(SegmentReclaimTest, WriteEvictChurnKeepsDiskBounded) {
  auto tier = MakeTier();
  int next = 0;
  for (int round = 0; round < 40; ++round) {
    const int first = next;
    for (int i = 0; i < 100; ++i, ++next) ASSERT_TRUE(Write(*tier, next)) << "round " << round;
    for (int i = first; i < next; ++i) ASSERT_TRUE(tier->Evict(Key(i)));

    EXPECT_LE(tier->Capacity().first, 8 * kMiB);
    EXPECT_LE(BytesOnDisk(), 1 * kMiB) << "round " << round << ": dead segments left on disk";
    EXPECT_LE(SegmentFiles(), 2u) << "round " << round;
    EXPECT_LE(OpenFdsUnderDir(), 2u) << "round " << round;
    EXPECT_EQ(tier->SpaceStats().live_bytes, 0u);
    EXPECT_EQ(tier->SpaceStats().files, SegmentFiles());
  }
  // 4000 records of 64 KiB went through, ~250 MiB, 31x the capacity.
  EXPECT_GE(tier->SpaceStats().bytes_deleted, 4000 * kRecord - 1 * kMiB);

  // A restart neither resurrects the evicted keys nor starts out full.
  tier.reset();
  auto reopened = MakeTier();
  EXPECT_FALSE(reopened->Exists(Key(0)));
  EXPECT_FALSE(reopened->Exists(Key(2005)));
  EXPECT_TRUE(Write(*reopened, 100000));
  EXPECT_TRUE(ReadsBack(*reopened, 100000));
}

// Admission charges what is on disk.  Evicting a few keys from every segment
// frees no file, so it must not let the tier take more.
TEST_F(SegmentReclaimTest, AdmissionCountsDeadRecordsUntilTheirSegmentGoes) {
  auto tier = MakeTier();
  int n = 0;
  while (Write(*tier, n)) ++n;
  ASSERT_EQ(n, 128) << "8 MiB of 64 KiB records";
  EXPECT_EQ(tier->Capacity().first, 8 * kMiB);

  // One key out of each of the 8 segments: 512 KiB less live, 0 bytes less disk.
  for (int s = 0; s < 8; ++s) ASSERT_TRUE(tier->Evict(Key(s * kPerSegment)));
  EXPECT_EQ(tier->SpaceStats().live_bytes, 8 * kMiB - 8 * kRecord);
  EXPECT_EQ(tier->Capacity().first, 8 * kMiB);
  EXPECT_FALSE(Write(*tier, 1000)) << "admitted past capacity on disk";

  // The order empties one whole segment first; doing so frees its file.
  const auto order = tier->EvictionOrder(kPerSegment - 1);
  ASSERT_EQ(order.size(), kPerSegment - 1);
  for (const auto& hint : order) {
    const int i = std::stoi(hint.key.substr(1));
    EXPECT_EQ(i / static_cast<int>(kPerSegment), 0) << hint.key << " is not in segment 0";
    ASSERT_TRUE(tier->Evict(hint.key));
  }
  EXPECT_EQ(tier->Capacity().first, 7 * kMiB);
  EXPECT_EQ(tier->SpaceStats().files_deleted, 1u);
  for (int i = 0; i < static_cast<int>(kPerSegment); ++i) {
    EXPECT_TRUE(Write(*tier, 1000 + i)) << i;
  }
}

TEST_F(SegmentReclaimTest, EvictionOrderRanksEmptiestAndColdestFirst) {
  auto tier = MakeTier();
  for (int i = 0; i < 64; ++i) ASSERT_TRUE(Write(*tier, i));       // segments 0..3, 3 active
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(ReadsBack(*tier, i));   // segment 0 is now hot
  for (int i = 32; i < 44; ++i) ASSERT_TRUE(tier->Evict(Key(i)));  // segment 2: 4 of 16 live

  const auto order = tier->EvictionOrder(64);
  ASSERT_EQ(order.size(), 52u);
  // Segment 2 first: mostly dead, so its 4 keys buy a whole file...
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(order[j].key, Key(44 + j));
    EXPECT_EQ(order[j].reclaim_bytes, 4 * kRecord) << "each carries a quarter of the file";
  }
  // ...then segment 1 (cold), then the freshly read segment 0, coldest key
  // first, and the active segment last.
  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(order[4 + j].key, Key(16 + j));
    EXPECT_EQ(order[4 + j].reclaim_bytes, kRecord);
    EXPECT_EQ(order[20 + j].key, Key(j));
    EXPECT_EQ(order[36 + j].key, Key(48 + j));
  }
  EXPECT_TRUE(tier->EvictionOrder(0).empty());
  EXPECT_EQ(tier->EvictionOrder(3).size(), 3u);
}

// A batch reports each key: one that does not fit fails alone, and is not
// left looking stored.  Before, a key refused for space came back `true`.
TEST_F(SegmentReclaimTest, BatchWriteReportsKeysThatDidNotFit) {
  auto tier = MakeTier();
  for (int i = 0; i < 120; ++i) ASSERT_TRUE(Write(*tier, i));  // 7.5 MiB

  std::vector<std::string> keys;
  std::vector<std::vector<char>> values;
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  for (int i = 200; i < 212; ++i) {
    keys.push_back(Key(i));
    values.push_back(ValueFor(i));
  }
  for (const auto& v : values) {
    ptrs.push_back(v.data());
    sizes.push_back(v.size());
  }
  const auto results = tier->BatchWrite(keys, ptrs, sizes);
  ASSERT_EQ(results.size(), keys.size());
  for (size_t j = 0; j < keys.size(); ++j) {
    EXPECT_EQ(results[j], j < 8) << keys[j];
    EXPECT_EQ(tier->Exists(keys[j]), j < 8) << keys[j];
  }
  EXPECT_TRUE(ReadsBack(*tier, 207));
  EXPECT_FALSE(tier->WriteBatch(keys, ptrs, sizes));
}

TEST_F(SegmentReclaimTest, OverwritingEveryRecordDeletesTheOldSegment) {
  auto tier = MakeTier();
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(Write(*tier, i));
  ASSERT_EQ(SegmentFiles(), 1u);
  for (int i = 0; i < 16; ++i) {
    const auto v = ValueFor(i + 500);
    ASSERT_TRUE(tier->Write(Key(i), v.data(), v.size()));
  }
  EXPECT_EQ(SegmentFiles(), 1u) << "segment 0 holds nothing live and should be gone";
  EXPECT_EQ(tier->SpaceStats().files_deleted, 1u);
  EXPECT_EQ(tier->Capacity().first, 16 * kRecord);
  std::vector<char> buf(kValue);
  ASSERT_TRUE(tier->ReadIntoPtr(Key(3), reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
  EXPECT_EQ(buf, ValueFor(503));
}

// A reservation's value is written through its segment's fd, so the segment
// outlives everything else in it until the reservation ends.
TEST_F(SegmentReclaimTest, AReservationKeepsItsSegmentUntilItEnds) {
  auto tier = MakeTier();
  auto reserved = tier->ReserveRecord(Key(0), kValue);
  ASSERT_TRUE(reserved.has_value());
  for (int i = 1; i <= 16; ++i) ASSERT_TRUE(Write(*tier, i));  // fills and seals segment 0
  for (int i = 1; i < 16; ++i) ASSERT_TRUE(tier->Evict(Key(i)));
  ASSERT_TRUE(Write(*tier, 0));  // the reserved key moves on to segment 1
  EXPECT_EQ(tier->SpaceStats().files_deleted, 0u) << "deleted under a pending reservation";

  tier->AbortRecord(Key(0), *reserved);
  EXPECT_EQ(tier->SpaceStats().files_deleted, 1u);
  EXPECT_TRUE(ReadsBack(*tier, 0));
}

// A segment whose every record a later segment supersedes -- left behind by a
// build that never deleted segments -- is deleted when the tier starts.
TEST_F(SegmentReclaimTest, StartupDeletesSegmentsWithNothingLive) {
  {
    auto tier = MakeTier();
    for (int i = 0; i < 16; ++i) ASSERT_TRUE(Write(*tier, i));
  }
  // Same records, same generations: whichever copy the scan meets first keeps
  // every key, and the other holds nothing live.
  fs::copy_file(dir_ / "segment_0.log", dir_ / "segment_5.log");
  auto tier = MakeTier();
  EXPECT_NE(fs::exists(dir_ / "segment_0.log"), fs::exists(dir_ / "segment_5.log"))
      << "exactly one copy should be left";
  EXPECT_EQ(BytesOnDisk(), 16 * kRecord);  // plus, if segment 0 went, a fresh empty one
  EXPECT_EQ(tier->Capacity().first, 16 * kRecord);
  EXPECT_EQ(tier->SpaceStats().files_deleted, 1u);
  for (int i = 0; i < 16; ++i) EXPECT_TRUE(ReadsBack(*tier, i)) << i;
}

// A deleted segment's fd may still be in a zero-copy reader's hands, so with a
// grace set it stays open that long, and the hook hears about it just before
// the close.
TEST_F(SegmentReclaimTest, RetiredFileHookWaitsOutTheGrace) {
  auto tier = MakeTier();
  std::mutex mu;
  std::vector<int> closed;
  tier->SetRetiredFileHook(
      [&](int fd) {
        std::lock_guard<std::mutex> lock(mu);
        closed.push_back(fd);
      },
      std::chrono::milliseconds(300));

  for (int i = 0; i <= 16; ++i) ASSERT_TRUE(Write(*tier, i));  // seals segment 0
  const auto loc = tier->LocateRecord(Key(0));
  ASSERT_TRUE(loc.has_value());
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(tier->Evict(Key(i)));
  EXPECT_FALSE(fs::exists(dir_ / "segment_0.log")) << "unlinked at once";
  EXPECT_EQ(tier->Capacity().first, kRecord) << "and no longer counted";
  {
    std::lock_guard<std::mutex> lock(mu);
    EXPECT_TRUE(closed.empty());
  }
  EXPECT_NE(fcntl(loc->fd, F_GETFD), -1) << "the handed-out fd must stay open for the grace";
  std::vector<char> buf(loc->readable_size);
  EXPECT_EQ(pread(loc->fd, buf.data(), buf.size(), static_cast<off_t>(loc->value_offset)),
            static_cast<ssize_t>(buf.size()));
  buf.resize(kValue);
  EXPECT_EQ(buf, ValueFor(0)) << "its bytes are intact until the close";

  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  tier->Capacity();  // sweeps
  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(closed.size(), 1u);
  EXPECT_EQ(closed[0], loc->fd);
}

TEST_F(SegmentReclaimTest, SegmentSizeIsCappedForASmallTier) {
  auto tier = MakeTier(64 * kMiB, 4ULL * 1024 * kMiB);
  EXPECT_EQ(tier->segment_size(), 8 * kMiB);
  auto big =
      std::make_unique<SSDTier>(dir_.string() + "_big", 64 * kMiB, Config(64 * kMiB, 4 * kMiB));
  EXPECT_EQ(big->segment_size(), 4 * kMiB) << "already small enough";
  big.reset();
  fs::remove_all(dir_.string() + "_big");
}

TEST_F(SegmentReclaimTest, ShardedOrderStartsWithTheFullestDrive) {
  const fs::path d0 = dir_ / "d0";
  const fs::path d1 = dir_ / "d1";
  fs::create_directories(d0);
  fs::create_directories(d1);
  std::vector<std::unique_ptr<TierBackend>> shards;
  for (const auto& d : {d0, d1}) {
    auto cfg = Config(8 * kMiB, 1 * kMiB);
    cfg.storage_dir = d.string();
    shards.push_back(std::make_unique<SSDTier>(d.string(), 8 * kMiB, cfg));
  }
  ShardedSsdTier tier(std::move(shards));
  ASSERT_TRUE(tier.Capabilities().orders_eviction);
  for (int i = 0; i < 96; ++i) {
    const auto v = ValueFor(i);
    ASSERT_TRUE(tier.Write(Key(i), v.data(), v.size())) << i;
  }
  // Empty most of shard 1, which frees its sealed segments.
  for (int i = 0; i < 96; ++i) {
    if (tier.ShardOf(Key(i)) == 1 && i < 80) ASSERT_TRUE(tier.Evict(Key(i)));
  }
  const auto per_shard = tier.PerShardCapacity();
  ASSERT_GT(per_shard[0].first, per_shard[1].first);
  const auto order = tier.EvictionOrder(8);
  ASSERT_EQ(order.size(), 8u);
  for (const auto& hint : order) EXPECT_EQ(tier.ShardOf(hint.key), 0) << hint.key;

  // Draining through the order frees disk on the drive it came from.
  const size_t before = per_shard[0].first;
  for (const auto& hint : tier.EvictionOrder(kPerSegment)) ASSERT_TRUE(tier.Evict(hint.key));
  EXPECT_LT(tier.PerShardCapacity()[0].first, before);
  EXPECT_LE(SegmentFiles(d0) + SegmentFiles(d1), 12u);
}

// Reads hold the segment they read from.  Readers, an evictor and a writer
// race while segments are created and deleted under them; with checksums off,
// a read served from a closed-and-reused fd would come back `true` with some
// other record's bytes.
TEST_F(SegmentReclaimTest, ReadsNeverSeeAnotherSegmentsBytes) {
  auto tier = MakeTier(8 * kMiB, 256 * kKiB, /*verify_crc=*/false);  // 4 records a segment
  std::atomic<int> written{0};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> good_reads{0};
  std::atomic<uint64_t> bad_reads{0};

  std::thread writer([&] {
    for (int i = 0; !stop.load(); ++i) {
      if (!Write(*tier, i)) continue;
      written.store(i + 1);
      if (i >= 24) tier->Evict(Key(i - 24));  // keep ~24 live, segments churn
    }
  });
  std::vector<std::thread> readers;
  for (int r = 0; r < 3; ++r) {
    readers.emplace_back([&, r] {
      std::mt19937 rng(r);
      std::vector<char> buf(kValue);
      while (!stop.load()) {
        const int hi = written.load();
        if (hi == 0) continue;
        const int i = std::max(0, hi - 1 - static_cast<int>(rng() % 32));
        if (r == 0) {
          std::vector<std::string> keys{Key(i)};
          std::vector<uintptr_t> dsts{reinterpret_cast<uintptr_t>(buf.data())};
          std::vector<size_t> sizes{buf.size()};
          if (!tier->ReadBatchIntoPtr(keys, dsts, sizes)[0]) continue;
        } else if (!tier->ReadIntoPtr(Key(i), reinterpret_cast<uintptr_t>(buf.data()),
                                      buf.size())) {
          continue;
        }
        (buf == ValueFor(i) ? good_reads : bad_reads).fetch_add(1);
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::seconds(2));
  stop.store(true);
  writer.join();
  for (auto& t : readers) t.join();

  EXPECT_EQ(bad_reads.load(), 0u);
  EXPECT_GT(good_reads.load(), 0u);
  EXPECT_GT(tier->SpaceStats().files_deleted, 10u) << "the test did not churn segments";
  EXPECT_LE(tier->Capacity().first, 8 * kMiB);
}

}  // namespace
}  // namespace mori::umbp
