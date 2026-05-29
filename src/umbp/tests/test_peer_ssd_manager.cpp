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

// Unique temp dir per fixture instance; backend uses PThread/posix to avoid
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
    cfg.ssd.io.backend = UMBPIoBackend::PThread;  // avoid io_uring container flakiness
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

TEST_F(PeerSsdManagerTest, PrepareReadIsStubInPhase1) {
  PeerSsdManager mgr(MakeConfig());
  std::vector<char> staging(64);
  auto out = mgr.PrepareRead("any", staging.data(), staging.size());
  EXPECT_EQ(out.status, SsdReadStatus::kNotFound);
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

  auto snap = SnapshotAllSources(sources);
  ASSERT_EQ(snap.size(), 3u);
  EXPECT_EQ(snap[2].tier, TierType::SSD);
}

TEST(OwnedLocationSourceAgg, NullSourcesAreSkipped) {
  FakeSource only({{KvEvent::Kind::ADD, "x", TierType::SSD, 1}});
  std::vector<OwnedLocationSource*> sources = {nullptr, &only, nullptr};
  auto drained = DrainAllSources(sources);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].key, "x");
  EXPECT_TRUE(SnapshotAllSources({nullptr}).empty());
}

}  // namespace
}  // namespace mori::umbp
