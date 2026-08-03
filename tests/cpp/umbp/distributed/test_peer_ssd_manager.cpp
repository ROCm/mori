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
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

// Unique temp dir per fixture instance; backend uses Posix I/O to avoid
// io_uring availability differences inside the build container.
class PeerSsdManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = fs::temp_directory_path() / ("umbp_ssd_test_" + std::to_string(::getpid()) + "_" +
                                        std::to_string(counter.fetch_add(1)));
    fs::remove_all(dir_);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  PeerSsdConfig MakeConfig(size_t capacity = 64ULL * 1024 * 1024) const {
    PeerSsdConfig cfg;
    cfg.enabled = true;
    cfg.ssd.enabled = true;
    cfg.ssd.storage_dir = dir_.string();
    cfg.ssd.capacity_bytes = capacity;
    cfg.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness
    return cfg;
  }

  static std::vector<std::pair<const void*, size_t>> OneSegment(const std::string& s) {
    return {{s.data(), s.size()}};
  }

  fs::path dir_;
};

TEST_F(PeerSsdManagerTest, WriteRecordsOwnershipAndQueuesAddEvent) {
  PeerSsdManager mgr(MakeConfig());
  const std::string key = "key-1";
  const std::string value = "hello-ssd-payload";

  ASSERT_TRUE(mgr.Write(key, OneSegment(value), value.size()));
  EXPECT_TRUE(mgr.Exists(key));

  auto events = mgr.DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].key, key);
  EXPECT_EQ(events[0].tier, TierType::SSD);
  EXPECT_EQ(events[0].size, value.size());

  // Drain is destructive.
  EXPECT_TRUE(mgr.DrainPendingEvents().empty());

  auto snap = mgr.SnapshotOwnedKeys();
  ASSERT_EQ(snap.size(), 1u);
  EXPECT_EQ(snap[0].key, key);
  EXPECT_EQ(snap[0].tier, TierType::SSD);
  EXPECT_EQ(snap[0].size, value.size());
}

TEST_F(PeerSsdManagerTest, WriteAssemblesNonContiguousSegments) {
  PeerSsdManager mgr(MakeConfig());
  const std::string a = "abc";
  const std::string b = "defgh";
  std::vector<std::pair<const void*, size_t>> segs = {{a.data(), a.size()}, {b.data(), b.size()}};

  ASSERT_TRUE(mgr.Write("multi", segs, a.size() + b.size()));
  EXPECT_TRUE(mgr.Exists("multi"));
  auto snap = mgr.SnapshotOwnedKeys();
  ASSERT_EQ(snap.size(), 1u);
  EXPECT_EQ(snap[0].size, a.size() + b.size());
}

TEST_F(PeerSsdManagerTest, CapacityReportsTotalAndGrowsWithWrites) {
  const size_t cap = 32ULL * 1024 * 1024;
  PeerSsdManager mgr(MakeConfig(cap));

  auto [used_before, total_before] = mgr.Capacity();
  EXPECT_EQ(total_before, cap);

  std::string value(4096, 'x');
  ASSERT_TRUE(mgr.Write("big", OneSegment(value), value.size()));

  auto [used_after, total_after] = mgr.Capacity();
  EXPECT_EQ(total_after, cap);
  EXPECT_GE(used_after, used_before);
}

TEST_F(PeerSsdManagerTest, EvictRemovesOwnershipAndQueuesRemoveEvent) {
  PeerSsdManager mgr(MakeConfig());
  const std::string key = "key-evict";
  const std::string value = "payload";
  ASSERT_TRUE(mgr.Write(key, OneSegment(value), value.size()));
  mgr.DrainPendingEvents();  // discard the ADD

  EXPECT_TRUE(mgr.Evict(key));
  EXPECT_FALSE(mgr.Exists(key));

  auto events = mgr.DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::REMOVE);
  EXPECT_EQ(events[0].key, key);
  EXPECT_EQ(events[0].tier, TierType::SSD);

  // Evicting an unknown key is a no-op (no event, returns false).
  EXPECT_FALSE(mgr.Evict("never-written"));
  EXPECT_TRUE(mgr.DrainPendingEvents().empty());
}

TEST_F(PeerSsdManagerTest, PrepareReadReturnsBytesForOwnedKey) {
  PeerSsdManager mgr(MakeConfig());
  const std::string key = "key-read";
  const std::string value = "hello-ssd-read-path";
  ASSERT_TRUE(mgr.Write(key, OneSegment(value), value.size()));

  std::vector<char> staging(value.size());
  auto out = mgr.PrepareRead(key, staging.data(), staging.size());
  EXPECT_EQ(out.status, SsdReadStatus::kOk);
  EXPECT_EQ(out.size, value.size());
  EXPECT_EQ(std::string(staging.data(), out.size), value);
}

TEST_F(PeerSsdManagerTest, PrepareReadUnknownKeyIsNotFound) {
  PeerSsdManager mgr(MakeConfig());
  std::vector<char> staging(64);
  auto out = mgr.PrepareRead("never-written", staging.data(), staging.size());
  EXPECT_EQ(out.status, SsdReadStatus::kNotFound);
}

TEST_F(PeerSsdManagerTest, PrepareReadRejectsOverCapBeforeIo) {
  PeerSsdManager mgr(MakeConfig());
  const std::string key = "key-big";
  const std::string value(4096, 'z');
  ASSERT_TRUE(mgr.Write(key, OneSegment(value), value.size()));

  // Capacity smaller than the actual size must be rejected as kSizeTooLarge
  // (and the reported size is the real size) without reading into the buffer.
  std::vector<char> staging(value.size() / 2);
  auto out = mgr.PrepareRead(key, staging.data(), staging.size());
  EXPECT_EQ(out.status, SsdReadStatus::kSizeTooLarge);
  EXPECT_EQ(out.size, value.size());
}

// ---- Unified owned-location source aggregation ------------------------------

// Minimal OwnedLocationSource that replays a fixed event list, used to verify
// MasterClient's multi-source concat logic without a live master.
class FakeSource : public OwnedLocationSource {
 public:
  explicit FakeSource(std::vector<KvEvent> events) : events_(std::move(events)) {}
  std::vector<KvEvent> DrainPendingEvents() override {
    auto out = events_;
    drained_ = true;
    return out;
  }
  std::vector<KvEvent> SnapshotOwnedKeys() const override { return events_; }
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override {
    drained_ = true;  // outbox dropped by the authoritative full-sync snapshot
    return events_;
  }
  bool drained_ = false;

 private:
  std::vector<KvEvent> events_;
};

TEST(OwnedLocationSourceAgg, DrainAndSnapshotConcatAcrossSourcesInOrder) {
  FakeSource dram({{KvEvent::Kind::ADD, "d1", TierType::DRAM, 10},
                   {KvEvent::Kind::ADD, "d2", TierType::DRAM, 20}});
  FakeSource ssd({{KvEvent::Kind::ADD, "s1", TierType::SSD, 30}});

  std::vector<OwnedLocationSource*> sources = {&dram, &ssd};

  auto drained = DrainAllSources(sources);
  ASSERT_EQ(drained.size(), 3u);
  EXPECT_EQ(drained[0].key, "d1");
  EXPECT_EQ(drained[0].tier, TierType::DRAM);
  EXPECT_EQ(drained[1].key, "d2");
  EXPECT_EQ(drained[2].key, "s1");
  EXPECT_EQ(drained[2].tier, TierType::SSD);
  EXPECT_TRUE(dram.drained_);
  EXPECT_TRUE(ssd.drained_);

  auto snap = SnapshotAllSourcesForFullSync(sources);
  ASSERT_EQ(snap.size(), 3u);
  EXPECT_EQ(snap[2].tier, TierType::SSD);
}

TEST(OwnedLocationSourceAgg, NullSourcesAreSkipped) {
  FakeSource only({{KvEvent::Kind::ADD, "x", TierType::SSD, 1}});
  std::vector<OwnedLocationSource*> sources = {nullptr, &only, nullptr};
  auto drained = DrainAllSources(sources);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].key, "x");
  EXPECT_TRUE(SnapshotAllSourcesForFullSync({nullptr}).empty());
}

// ---------------------------------------------------------------------------
//  Batched direct-SSD paths (WriteBatch / PrepareReadBatch)
// ---------------------------------------------------------------------------

TEST_F(PeerSsdManagerTest, WriteBatchRecordsEveryKeyAndQueuesOneAddEach) {
  PeerSsdManager mgr(MakeConfig());
  constexpr int kN = 32;
  std::vector<std::string> keys;
  std::vector<std::string> payloads;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (int i = 0; i < kN; ++i) {
    keys.push_back("bk-" + std::to_string(i));
    payloads.push_back(std::string(4096, static_cast<char>('a' + (i % 26))));
  }
  for (int i = 0; i < kN; ++i) {
    srcs.push_back(payloads[i].data());
    sizes.push_back(payloads[i].size());
  }

  auto ok = mgr.WriteBatch(keys, srcs, sizes);
  ASSERT_EQ(ok.size(), static_cast<size_t>(kN));
  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(ok[i]) << "key " << i;
    EXPECT_TRUE(mgr.Exists(keys[i]));
  }

  auto events = mgr.DrainPendingEvents();
  EXPECT_EQ(events.size(), static_cast<size_t>(kN));
  for (const auto& e : events) {
    EXPECT_EQ(e.kind, KvEvent::Kind::ADD);
    EXPECT_EQ(e.tier, TierType::SSD);
    EXPECT_EQ(e.size, 4096u);
  }
}

// An already-resident key must succeed with no device write and no second ADD:
// re-putting a live key is the common case on a warm cache.
TEST_F(PeerSsdManagerTest, WriteBatchDedupsAlreadyOwnedKeys) {
  PeerSsdManager mgr(MakeConfig());
  const std::string value(2048, 'D');
  ASSERT_TRUE(mgr.Write("dup", OneSegment(value), value.size()));
  ASSERT_EQ(mgr.DrainPendingEvents().size(), 1u);

  std::vector<std::string> keys = {"dup", "fresh"};
  const std::string fresh(2048, 'F');
  std::vector<const void*> srcs = {value.data(), fresh.data()};
  std::vector<size_t> sizes = {value.size(), fresh.size()};

  auto ok = mgr.WriteBatch(keys, srcs, sizes);
  EXPECT_TRUE(ok[0]);
  EXPECT_TRUE(ok[1]);

  auto events = mgr.DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);  // only "fresh"
  EXPECT_EQ(events[0].key, "fresh");
}

TEST_F(PeerSsdManagerTest, WriteBatchRejectsMismatchedLengths) {
  PeerSsdManager mgr(MakeConfig());
  const std::string value(64, 'X');
  auto ok = mgr.WriteBatch({"a", "b"}, {value.data()}, {value.size()});
  ASSERT_EQ(ok.size(), 2u);
  EXPECT_FALSE(ok[0]);
  EXPECT_FALSE(ok[1]);
}

TEST_F(PeerSsdManagerTest, PrepareReadBatchServesEveryKey) {
  PeerSsdManager mgr(MakeConfig());
  constexpr int kN = 16;
  constexpr size_t kSize = 8192;
  std::vector<std::string> keys, payloads;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (int i = 0; i < kN; ++i) {
    keys.push_back("rk-" + std::to_string(i));
    payloads.push_back(std::string(kSize, static_cast<char>(i)));
  }
  for (int i = 0; i < kN; ++i) {
    srcs.push_back(payloads[i].data());
    sizes.push_back(kSize);
  }
  ASSERT_EQ(mgr.WriteBatch(keys, srcs, sizes).size(), static_cast<size_t>(kN));

  std::vector<std::string> outs(kN, std::string(kSize, '\0'));
  std::vector<void*> dsts;
  std::vector<size_t> caps(kN, kSize);
  for (auto& o : outs) dsts.push_back(o.data());

  auto res = mgr.PrepareReadBatch(keys, dsts, caps);
  ASSERT_EQ(res.size(), static_cast<size_t>(kN));
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(res[i].status, SsdReadStatus::kOk) << "key " << i;
    EXPECT_EQ(res[i].size, kSize);
    EXPECT_EQ(outs[i], payloads[i]);
  }
}

// Per-key outcomes must be independent: a miss and an over-cap key must not
// spoil the keys around them.
TEST_F(PeerSsdManagerTest, PrepareReadBatchMixesHitMissAndTooLarge) {
  PeerSsdManager mgr(MakeConfig());
  const std::string small(1024, 'S');
  const std::string big(8192, 'B');
  ASSERT_TRUE(mgr.Write("hit", OneSegment(small), small.size()));
  ASSERT_TRUE(mgr.Write("toobig", OneSegment(big), big.size()));

  std::string out_hit(1024, '\0');
  std::string out_miss(1024, '\0');
  std::string out_big(1024, '\0');
  std::vector<std::string> keys = {"hit", "absent", "toobig"};
  std::vector<void*> dsts = {out_hit.data(), out_miss.data(), out_big.data()};
  std::vector<size_t> caps = {1024, 1024, 1024};  // "toobig" is 8192 > cap

  auto res = mgr.PrepareReadBatch(keys, dsts, caps);
  ASSERT_EQ(res.size(), 3u);
  EXPECT_EQ(res[0].status, SsdReadStatus::kOk);
  EXPECT_EQ(out_hit, small);
  EXPECT_EQ(res[1].status, SsdReadStatus::kNotFound);
  EXPECT_EQ(res[2].status, SsdReadStatus::kSizeTooLarge);
  EXPECT_EQ(res[2].size, big.size());  // reports the ACTUAL size so the caller can resize
}

// Multi-drive: the manager must fan a batch across every configured directory
// and still resolve each key on read-back.
TEST_F(PeerSsdManagerTest, MultiDriveWriteBatchSpreadsAndReadsBack) {
  PeerSsdConfig cfg = MakeConfig(128ULL * 1024 * 1024);
  const std::string d0 = dir_.string() + "_m0";
  const std::string d1 = dir_.string() + "_m1";
  const std::string d2 = dir_.string() + "_m2";
  cfg.ssd.storage_dir = d0 + "," + d1 + "," + d2;
  PeerSsdManager mgr(cfg);

  constexpr int kN = 60;
  constexpr size_t kSize = 4096;
  std::vector<std::string> keys, payloads;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (int i = 0; i < kN; ++i) {
    keys.push_back("mk-" + std::to_string(i));
    payloads.push_back(std::string(kSize, static_cast<char>('0' + (i % 10))));
  }
  for (int i = 0; i < kN; ++i) {
    srcs.push_back(payloads[i].data());
    sizes.push_back(kSize);
  }

  auto ok = mgr.WriteBatch(keys, srcs, sizes);
  for (int i = 0; i < kN; ++i) EXPECT_TRUE(ok[i]) << "key " << i;

  // Every drive got a share of the batch.
  for (const auto& d : {d0, d1, d2}) {
    uintmax_t bytes = 0;
    for (const auto& entry : fs::directory_iterator(d)) {
      if (entry.is_regular_file()) bytes += entry.file_size();
    }
    EXPECT_GT(bytes, 0u) << "drive " << d << " got nothing";
  }

  std::vector<std::string> outs(kN, std::string(kSize, '\0'));
  std::vector<void*> dsts;
  std::vector<size_t> caps(kN, kSize);
  for (auto& o : outs) dsts.push_back(o.data());
  auto res = mgr.PrepareReadBatch(keys, dsts, caps);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(res[i].status, SsdReadStatus::kOk) << "key " << i;
    EXPECT_EQ(outs[i], payloads[i]);
  }

  for (const auto& d : {d0, d1, d2}) {
    std::error_code ec;
    fs::remove_all(d, ec);
  }
}

}  // namespace
}  // namespace mori::umbp
