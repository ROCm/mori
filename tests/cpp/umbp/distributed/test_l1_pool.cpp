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

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "umbp/l1/l1_pool.h"
#include "umbp/l1/l1_service.h"

namespace mori::umbp::l1 {
namespace {

// Host memory that remembers each pool's base, so a test can play the engine
// writing into its slots.
class RecordingApi final : public MemoryApi {
 public:
  bool Allocate(int device, uint64_t bytes, void** base, ExportedMemory* exported,
                std::string* error) override {
    if (!host_->Allocate(device, bytes, base, exported, error)) return false;
    bases[device] = static_cast<uint8_t*>(*base);
    return true;
  }
  void Free(int device, void* base, uint64_t bytes) override { host_->Free(device, base, bytes); }
  bool Copy(int dst_device, void* dst, int src_device, const void* src, uint64_t bytes,
            std::string* error) override {
    ++copies;
    if (fail_copies) {
      *error = "injected";
      return false;
    }
    return host_->Copy(dst_device, dst, src_device, src, bytes, error);
  }

  bool Synchronize(int device, std::string* error) override {
    ++synchronizes;
    return host_->Synchronize(device, error);
  }

  std::map<int, uint8_t*> bases;
  int copies = 0;
  int synchronizes = 0;
  bool fail_copies = false;

 private:
  std::shared_ptr<MemoryApi> host_ = MakeHostMemoryApi();
};

constexpr uint32_t kSlots = 8;
constexpr uint64_t kUnitA = 16;
constexpr uint64_t kUnitB = 4;
constexpr uint64_t kBytes = kSlots * (kUnitA + kUnitB);

Layout TestLayout() {
  Layout layout;
  layout.num_slots = kSlots;
  layout.regions = {Region{0, kUnitA}, Region{kSlots * kUnitA, kUnitB}};
  return layout;
}

class L1PoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api = std::make_shared<RecordingApi>();
    manager = std::make_unique<PoolManager>(api);
    for (int device : {0, 1}) {
      const std::string id = "p" + std::to_string(device);
      ASSERT_TRUE(manager->Attach(id, "g", device, kBytes).status.ok);
      ASSERT_TRUE(manager->BindLayout(id, TestLayout()).ok);
      ASSERT_EQ(manager->Open(id).num_slots, kSlots);
    }
  }

  // Write a recognisable pattern into both regions of a slot.
  void Fill(int device, uint32_t slot, uint8_t value) {
    std::memset(api->bases[device] + slot * kUnitA, value, kUnitA);
    std::memset(api->bases[device] + kSlots * kUnitA + slot * kUnitB, value + 100, kUnitB);
  }
  bool Holds(int device, uint32_t slot, uint8_t value) {
    const uint8_t* a = api->bases[device] + slot * kUnitA;
    const uint8_t* b = api->bases[device] + kSlots * kUnitA + slot * kUnitB;
    for (uint64_t i = 0; i < kUnitA; ++i) {
      if (a[i] != value) return false;
    }
    for (uint64_t i = 0; i < kUnitB; ++i) {
      if (b[i] != static_cast<uint8_t>(value + 100)) return false;
    }
    return true;
  }

  // Allocate, fill, publish and release `keys` in pool p<device>.
  std::vector<uint32_t> Cache(int device, const std::vector<std::string>& keys, uint8_t first) {
    const std::string id = "p" + std::to_string(device);
    AllocateResult got = manager->Allocate(id, static_cast<uint32_t>(keys.size()));
    EXPECT_EQ(got.slots.size(), keys.size());
    std::vector<Op> ops;
    for (size_t i = 0; i < keys.size(); ++i) {
      Fill(device, got.slots[i], static_cast<uint8_t>(first + i));
      ops.push_back(Op{OpKind::kPublish, got.slots[i], keys[i]});
    }
    for (uint32_t slot : got.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
    EXPECT_TRUE(manager->Apply(id, ops).ok);
    return got.slots;
  }

  PoolStats StatsOf(const std::string& id) {
    for (const PoolStats& stats : manager->Stats()) {
      if (stats.pool_id == id) return stats;
    }
    return PoolStats{};
  }

  std::shared_ptr<RecordingApi> api;
  std::unique_ptr<PoolManager> manager;
};

TEST_F(L1PoolTest, AllocatesLowestVacantFirstAndStopsWhenFull) {
  AllocateResult first = manager->Allocate("p0", 3);
  EXPECT_EQ(first.slots, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_TRUE(first.evicted_keys.empty());
  AllocateResult rest = manager->Allocate("p0", 100);
  EXPECT_EQ(rest.slots.size(), kSlots - 3);
  EXPECT_TRUE(manager->Allocate("p0", 1).slots.empty());
  EXPECT_EQ(StatsOf("p0").held, kSlots);
}

TEST_F(L1PoolTest, ReleasedKeylessSlotIsVacantAgain) {
  AllocateResult got = manager->Allocate("p0", 2);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kRelease, got.slots[1], ""}}).ok);
  EXPECT_EQ(manager->Allocate("p0", 1).slots, (std::vector<uint32_t>{1}));
}

TEST_F(L1PoolTest, EvictsCachedContentColdestFirstOnlyAfterVacantRunsOut) {
  Cache(0, {"a", "b"}, 1);  // slots 0, 1 cached; a released before b
  AllocateResult vacant = manager->Allocate("p0", kSlots - 2);
  EXPECT_TRUE(vacant.evicted_keys.empty());
  AllocateResult evicting = manager->Allocate("p0", 1);
  EXPECT_EQ(evicting.slots, (std::vector<uint32_t>{0}));
  EXPECT_EQ(evicting.evicted_keys, (std::vector<std::string>{"a"}));
  EXPECT_EQ(StatsOf("p0").evicted, 1u);
}

TEST_F(L1PoolTest, AskingAheadOfNeedDoesNotEvict) {
  Cache(0, {"a", "b"}, 1);
  manager->Allocate("p0", kSlots - 3);  // one vacant slot left
  // Four wanted, one needed: the vacant slot only.
  AllocateResult ahead = manager->Allocate("p0", 4, 1);
  EXPECT_EQ(ahead.slots.size(), 1u);
  EXPECT_TRUE(ahead.evicted_keys.empty());
  // Four wanted, two needed, none vacant: exactly two evictions.
  AllocateResult needed = manager->Allocate("p0", 4, 2);
  EXPECT_EQ(needed.slots.size(), 2u);
  EXPECT_EQ(needed.evicted_keys, (std::vector<std::string>{"a", "b"}));
}

TEST_F(L1PoolTest, HeldCachedSlotIsNotEvicted) {
  std::vector<uint32_t> slots = Cache(0, {"a", "b"}, 1);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kHold, slots[0], ""}}).ok);
  manager->Allocate("p0", kSlots - 2);
  AllocateResult evicting = manager->Allocate("p0", 2);
  EXPECT_EQ(evicting.slots, (std::vector<uint32_t>{slots[1]}));
  EXPECT_EQ(evicting.evicted_keys, (std::vector<std::string>{"b"}));
}

TEST_F(L1PoolTest, RepublishingAKeyMovesItAndFreesTheOldCopy) {
  std::vector<uint32_t> slots = Cache(0, {"a"}, 1);
  AllocateResult got = manager->Allocate("p0", 1);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kPublish, got.slots[0], "a"}}).ok);
  // The old copy is vacant now, so it is handed out before anything is evicted.
  AllocateResult next = manager->Allocate("p0", 1);
  EXPECT_EQ(next.slots, (std::vector<uint32_t>{slots[0]}));
  EXPECT_TRUE(next.evicted_keys.empty());
  // And a fetch from the other pool reads the new copy.
  Fill(0, got.slots[0], 42);
  FetchResult fetched = manager->Fetch("p1", {"a"});
  ASSERT_EQ(fetched.slots.size(), 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 42));
}

TEST_F(L1PoolTest, RejectsOpsOnSlotsInTheWrongState) {
  EXPECT_FALSE(manager->Apply("p0", {Op{OpKind::kRelease, 0, ""}}).ok);
  EXPECT_FALSE(manager->Apply("p0", {Op{OpKind::kPublish, 0, "a"}}).ok);
  EXPECT_FALSE(manager->Apply("p0", {Op{OpKind::kHold, kSlots, ""}}).ok);
}

TEST_F(L1PoolTest, FetchCopiesEveryRegionAndCachesTheSlots) {
  Cache(0, {"a", "b", "c"}, 10);
  FetchResult fetched = manager->Fetch("p1", {"a", "b", "c"});
  ASSERT_TRUE(fetched.status.ok);
  ASSERT_EQ(fetched.slots.size(), 3u);
  EXPECT_EQ(fetched.copied_slots, 3u);
  EXPECT_EQ(fetched.copied_bytes, 3 * (kUnitA + kUnitB));
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(Holds(1, fetched.slots[i], static_cast<uint8_t>(10 + i)));
  }
  // Consecutive slots on both sides are one copy per region, and the copies
  // are waited for once per device before the slots are handed over.
  EXPECT_EQ(api->copies, 2);
  EXPECT_EQ(api->synchronizes, 2);
  PoolStats p1 = StatsOf("p1");
  EXPECT_EQ(p1.cached, 3u);
  EXPECT_EQ(p1.held, 0u);
  EXPECT_EQ(p1.fetched_in_slots, 3u);
  EXPECT_EQ(StatsOf("p0").served_out_slots, 3u);
}

TEST_F(L1PoolTest, FetchStopsAtTheFirstKeyNobodyHolds) {
  Cache(0, {"a", "c"}, 10);
  FetchResult fetched = manager->Fetch("p1", {"a", "b", "c"});
  EXPECT_EQ(fetched.slots.size(), 1u);
  EXPECT_EQ(StatsOf("p1").cached, 1u);
}

TEST_F(L1PoolTest, FetchOfLocalContentCopiesNothing) {
  Cache(0, {"a"}, 10);
  FetchResult first = manager->Fetch("p1", {"a"});
  const int copies = api->copies;
  FetchResult again = manager->Fetch("p1", {"a"});
  EXPECT_EQ(again.slots, first.slots);
  EXPECT_EQ(again.copied_slots, 0u);
  EXPECT_EQ(api->copies, copies);
}

TEST_F(L1PoolTest, FetchedContentIsServedOnwardAndSurvivesSourceEviction) {
  Cache(0, {"a"}, 10);
  ASSERT_EQ(manager->Fetch("p1", {"a"}).slots.size(), 1u);
  // Evict "a" from p0 by filling the pool.
  AllocateResult all = manager->Allocate("p0", kSlots);
  EXPECT_EQ(all.evicted_keys, (std::vector<std::string>{"a"}));
  std::vector<Op> release;
  for (uint32_t slot : all.slots) release.push_back(Op{OpKind::kRelease, slot, ""});
  ASSERT_TRUE(manager->Apply("p0", release).ok);
  // p0 gets it back from p1.
  FetchResult back = manager->Fetch("p0", {"a"});
  ASSERT_EQ(back.slots.size(), 1u);
  EXPECT_TRUE(Holds(0, back.slots[0], 10));
}

TEST_F(L1PoolTest, FetchEvictsToMakeRoomAndReportsWhatItDestroyed) {
  Cache(0, {"a", "b"}, 10);
  // p1 is full of cached content of its own.
  std::vector<std::string> own;
  for (uint32_t i = 0; i < kSlots; ++i) own.push_back("own" + std::to_string(i));
  Cache(1, own, 50);
  FetchResult fetched = manager->Fetch("p1", {"a", "b"});
  ASSERT_EQ(fetched.slots.size(), 2u);
  EXPECT_EQ(fetched.evicted_keys, (std::vector<std::string>{"own0", "own1"}));
  EXPECT_TRUE(Holds(1, fetched.slots[0], 10));
  EXPECT_TRUE(Holds(1, fetched.slots[1], 11));
}

TEST_F(L1PoolTest, FetchStopsWhenThePoolHasNoFreeSlot) {
  Cache(0, {"a", "b", "c"}, 10);
  manager->Allocate("p1", kSlots - 2);  // two free slots left
  FetchResult fetched = manager->Fetch("p1", {"a", "b", "c"});
  EXPECT_EQ(fetched.slots.size(), 2u);
}

TEST_F(L1PoolTest, FailedCopyLeavesNothingBehind) {
  Cache(0, {"a", "b"}, 10);
  api->fail_copies = true;
  FetchResult fetched = manager->Fetch("p1", {"a", "b"});
  EXPECT_TRUE(fetched.slots.empty());
  PoolStats p1 = StatsOf("p1");
  EXPECT_EQ(p1.cached, 0u);
  EXPECT_EQ(p1.vacant, kSlots);
  // The source is unpinned again: it can be evicted.
  api->fail_copies = false;
  EXPECT_EQ(manager->Allocate("p0", kSlots).slots.size(), kSlots);
}

TEST_F(L1PoolTest, UnsettledPublishIsNotASourceUntilCommitted) {
  AllocateResult got = manager->Allocate("p0", 1);
  Fill(0, got.slots[0], 7);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kPublish, got.slots[0], "a"}}, /*stamp=*/100).ok);
  // The owner's device may still be writing it.
  EXPECT_TRUE(manager->Fetch("p1", {"a"}).slots.empty());
  ASSERT_TRUE(manager->Commit("p0", 99).ok);
  EXPECT_TRUE(manager->Fetch("p1", {"a"}).slots.empty());
  ASSERT_TRUE(manager->Commit("p0", 100).ok);
  FetchResult fetched = manager->Fetch("p1", {"a"});
  ASSERT_EQ(fetched.slots.size(), 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 7));
  // A commit never moves backwards.
  ASSERT_TRUE(manager->Commit("p0", 5).ok);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kRelease, got.slots[0], ""}}, 100).ok);
  EXPECT_EQ(manager->Fetch("p1", {"a"}).copied_slots, 0u);
}

TEST_F(L1PoolTest, ReleasingAPublishedSlotKeepsItReadable) {
  AllocateResult got = manager->Allocate("p0", 1);
  Fill(0, got.slots[0], 7);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kPublish, got.slots[0], "a"}}, 100).ok);
  ASSERT_TRUE(manager->Commit("p0", 100).ok);
  // The request ends: its blocks are released, later than any commit so far.
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kRelease, got.slots[0], ""}}, 200).ok);
  FetchResult fetched = manager->Fetch("p1", {"a"});
  ASSERT_EQ(fetched.slots.size(), 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 7));
}

TEST_F(L1PoolTest, FetchDoesNotFillASlotTheOwnerJustReleased) {
  Cache(0, {"a", "b"}, 10);
  // p1 has exactly one free slot, released a moment ago: a forward of p1's
  // owner may still be writing it.
  AllocateResult all = manager->Allocate("p1", kSlots);
  ASSERT_TRUE(manager->Apply("p1", {Op{OpKind::kRelease, all.slots[3], ""}}, 100).ok);
  EXPECT_TRUE(manager->Fetch("p1", {"a"}).slots.empty());
  // The owner itself may take it at once.
  EXPECT_EQ(manager->Allocate("p1", 1).slots, (std::vector<uint32_t>{all.slots[3]}));
  ASSERT_TRUE(manager->Apply("p1", {Op{OpKind::kRelease, all.slots[3], ""}}, 100).ok);
  ASSERT_TRUE(manager->Commit("p1", 100).ok);
  FetchResult fetched = manager->Fetch("p1", {"a"});
  EXPECT_EQ(fetched.slots, (std::vector<uint32_t>{all.slots[3]}));
}

TEST_F(L1PoolTest, FetchedContentIsSettledForTheNextPool) {
  ASSERT_TRUE(manager->Attach("p2", "g", 2, kBytes).status.ok);
  ASSERT_TRUE(manager->BindLayout("p2", TestLayout()).ok);
  ASSERT_TRUE(manager->Open("p2").status.ok);
  Cache(0, {"a"}, 10);
  ASSERT_EQ(manager->Fetch("p1", {"a"}).slots.size(), 1u);
  ASSERT_TRUE(manager->Detach("p0").ok);
  // p1 never committed anything; the copy UMBP made into it is complete anyway.
  FetchResult onward = manager->Fetch("p2", {"a"});
  ASSERT_EQ(onward.slots.size(), 1u);
  EXPECT_TRUE(Holds(2, onward.slots[0], 10));
}

TEST_F(L1PoolTest, ClearDropsKeysButKeepsHeldSlots) {
  std::vector<uint32_t> slots = Cache(0, {"a", "b"}, 10);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kHold, slots[0], ""}}).ok);
  ASSERT_TRUE(manager->Clear("p0").ok);
  EXPECT_TRUE(manager->Fetch("p1", {"a"}).slots.empty());
  PoolStats p0 = StatsOf("p0");
  EXPECT_EQ(p0.held, 1u);
  EXPECT_EQ(p0.cached, 0u);
  EXPECT_EQ(p0.vacant, kSlots - 1);
}

TEST_F(L1PoolTest, PoolsOfDifferentGroupsDoNotShare) {
  ASSERT_TRUE(manager->Attach("other", "g2", 2, kBytes).status.ok);
  ASSERT_TRUE(manager->BindLayout("other", TestLayout()).ok);
  ASSERT_TRUE(manager->Open("other").status.ok);
  Cache(0, {"a"}, 10);
  EXPECT_TRUE(manager->Fetch("other", {"a"}).slots.empty());
}

TEST_F(L1PoolTest, GroupRejectsADifferentSlotSize) {
  ASSERT_TRUE(manager->Attach("odd", "g", 2, kBytes).status.ok);
  Layout layout;
  layout.num_slots = kSlots;
  layout.regions = {Region{0, kUnitA + 1}};
  EXPECT_FALSE(manager->BindLayout("odd", layout).ok);
}

TEST_F(L1PoolTest, ReattachWithTheSameShapeKeepsThePool) {
  uint8_t* before = api->bases[0];
  AttachResult again = manager->Attach("p0", "g", 0, kBytes);
  EXPECT_TRUE(again.status.ok);
  EXPECT_FALSE(again.created);
  EXPECT_EQ(api->bases[0], before);
}

TEST_F(L1PoolTest, DetachRemovesThePoolAsASource) {
  Cache(0, {"a"}, 10);
  ASSERT_TRUE(manager->Detach("p0").ok);
  EXPECT_TRUE(manager->Fetch("p1", {"a"}).slots.empty());
  EXPECT_FALSE(manager->Allocate("p0", 1).status.ok);
}

TEST(L1ServiceTest, RoundTripsOverAUnixSocket) {
  const std::string path = "/tmp/umbp_l1_test_" + std::to_string(getpid()) + ".sock";
  unlink(path.c_str());
  const std::string address = "unix://" + path;
  {
    L1Server server(address, MakeHostMemoryApi());
    ASSERT_TRUE(server.ok());
    L1Client client(address);
    for (int device : {0, 1}) {
      const std::string id = "p" + std::to_string(device);
      AttachResult attached = client.Attach(id, "g", device, kBytes);
      ASSERT_TRUE(attached.status.ok) << attached.status.error;
      EXPECT_EQ(attached.memory.kind, "host");
      EXPECT_TRUE(attached.created);
      ASSERT_TRUE(client.BindLayout(id, TestLayout()).ok);
      EXPECT_EQ(client.Open(id).num_slots, kSlots);
    }
    AllocateResult got = client.Allocate("p0", 2, 2);
    ASSERT_EQ(got.slots, (std::vector<uint32_t>{0, 1}));
    const std::string binary_key("\x00\xff\x01key", 6);
    ASSERT_TRUE(client
                    .Apply("p0", {Op{OpKind::kPublish, 0, binary_key},
                                  Op{OpKind::kPublish, 1, "k1"}, Op{OpKind::kRelease, 0, ""}})
                    .ok);
    // Stamped ops stay invisible to other pools until the commit arrives.
    AllocateResult late = client.Allocate("p0", 1, 1);
    ASSERT_TRUE(client.Apply("p0", {Op{OpKind::kPublish, late.slots[0], "late"}}, 500).ok);
    EXPECT_TRUE(client.Fetch("p1", {"late"}).slots.empty());
    ASSERT_TRUE(client.Commit("p0", 500).ok);
    EXPECT_EQ(client.Fetch("p1", {"late"}).slots.size(), 1u);
    FetchResult fetched = client.Fetch("p1", {binary_key, "k1", "missing"});
    ASSERT_TRUE(fetched.status.ok) << fetched.status.error;
    EXPECT_EQ(fetched.slots.size(), 2u);
    EXPECT_EQ(fetched.copied_slots, 2u);
    EXPECT_FALSE(client.Apply("p0", {Op{OpKind::kRelease, 5, ""}}).ok);
    std::vector<PoolStats> stats;
    ASSERT_TRUE(client.Stats(&stats).ok);
    ASSERT_EQ(stats.size(), 2u);
    EXPECT_EQ(stats[1].fetched_in_slots, 3u);
    EXPECT_TRUE(client.Detach("p0").ok);
    EXPECT_FALSE(client.Open("p0").status.ok);
  }
  unlink(path.c_str());
}

}  // namespace
}  // namespace mori::umbp::l1
