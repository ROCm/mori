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
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
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
    std::lock_guard<std::mutex> lock(mutex_);
    ++allocations;
    allocated_.insert(*base);
    bases[device] = static_cast<uint8_t*>(*base);
    return true;
  }
  void Free(int device, void* base, uint64_t bytes) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++frees;
      // Memory this api did not allocate is its owner's to free, not ours.
      if (allocated_.erase(base) == 0) {
        ++foreign_frees;
        return;
      }
    }
    host_->Free(device, base, bytes);
  }
  bool Import(int device, const std::string& handle, uint64_t offset, void** base, void** mapped,
              std::string* error) override {
    if (!host_->Import(device, handle, offset, base, mapped, error)) return false;
    bases[device] = static_cast<uint8_t*>(*base);
    return true;
  }
  void Unimport(int device, void* mapped) override {
    ++unimports;
    host_->Unimport(device, mapped);
  }
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
    if (on_synchronize) on_synchronize(device);
    return host_->Synchronize(device, error);
  }

  std::map<int, uint8_t*> bases;
  int copies = 0;
  int synchronizes = 0;
  int frees = 0;
  int unimports = 0;
  int allocations = 0;
  int foreign_frees = 0;  // Free of memory Allocate never returned
  bool fail_copies = false;
  std::function<void(int)> on_synchronize;

 private:
  std::mutex mutex_;
  std::set<void*> allocated_;
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

TEST_F(L1PoolTest, KeepCopiesEvictsWhatAnotherPoolAlsoHoldsBeforeTheColdestSlot) {
  manager->SetEvictionPolicy(/*keep_copies=*/1);
  Cache(0, {"only", "spare"}, 1);                               // "only" is the colder of the two
  ASSERT_EQ(manager->Fetch("p1", {"spare"}).slots.size(), 1u);  // p1 holds "spare" as well
  manager->Allocate("p0", kSlots - 2);
  // The copy p1 also has goes first, although "only" is colder.
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"spare"}));
  // p1's is the last copy of "spare" now, and p1 has nothing else to give up.
  manager->Allocate("p1", kSlots - 1);
  EXPECT_EQ(manager->Allocate("p1", 1).evicted_keys, (std::vector<std::string>{"spare"}));
  // No spare copy left in p0 either: coldest first, as without the policy.
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"only"}));
}

TEST_F(L1PoolTest, KeepCopiesReachesASpareCopyBehindARunOfLastCopies) {
  manager->SetEvictionPolicy(/*keep_copies=*/1);
  Cache(0, {"a", "b", "c", "spare"}, 1);  // a prefix nobody else holds, then one that p1 copies
  ASSERT_EQ(manager->Fetch("p1", {"spare"}).slots.size(), 1u);
  manager->Allocate("p0", kSlots - 4);
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"spare"}));
  // With no spare copy left the last copies go, oldest first.
  EXPECT_EQ(manager->Allocate("p0", 3).evicted_keys, (std::vector<std::string>{"a", "b", "c"}));
}

TEST_F(L1PoolTest, KeepCopiesTakesALastCopyThatAnotherPoolHasSinceCopied) {
  manager->SetEvictionPolicy(/*keep_copies=*/1);
  Cache(0, {"x", "y", "spare"}, 1);
  ASSERT_EQ(manager->Fetch("p1", {"spare"}).slots.size(), 1u);
  manager->Allocate("p0", kSlots - 3);
  // Passes over "x" and "y", the last copies, to take "spare".
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"spare"}));
  // p1 copies "y": p0's copy of it is spare now and goes before the older "x".
  ASSERT_EQ(manager->Fetch("p1", {"y"}).slots.size(), 1u);
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"y"}));
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"x"}));
}

TEST_F(L1PoolTest, KeepTwoCopiesTreatsTheSecondCopyAsWorthKeeping) {
  manager->SetEvictionPolicy(/*keep_copies=*/2);
  Cache(0, {"only", "pair"}, 1);
  ASSERT_EQ(manager->Fetch("p1", {"pair"}).slots.size(), 1u);
  manager->Allocate("p0", kSlots - 2);
  // Two pools hold "pair": neither copy is spare, so the coldest slot goes.
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"only"}));
}

TEST_F(L1PoolTest, KeepCopiesKeepsTheCopyJustFetchedAndGivesUpTheSourcesCopy) {
  manager->SetEvictionPolicy(/*keep_copies=*/1);
  Cache(1, {"own"}, 50);
  Cache(0, {"k"}, 1);
  ASSERT_EQ(manager->Fetch("p1", {"k"}).slots.size(), 1u);
  // p1's copy of "k" was fetched for a request that is about to use it: it is
  // the one to keep, so p1 gives up its older content instead.
  manager->Allocate("p1", kSlots - 2);
  EXPECT_EQ(manager->Allocate("p1", 1).evicted_keys, (std::vector<std::string>{"own"}));
  // p0's copy is the spare one, however recently p0 released it.
  manager->Allocate("p0", kSlots - 1);
  EXPECT_EQ(manager->Allocate("p0", 1).evicted_keys, (std::vector<std::string>{"k"}));
}

TEST_F(L1PoolTest, KeepCopiesTreatsTheCopyAnEngineLastTookAsTheOneToKeep) {
  manager->SetEvictionPolicy(/*keep_copies=*/1);
  const std::vector<uint32_t> source = Cache(0, {"k"}, 1);
  ASSERT_EQ(manager->Fetch("p1", {"k"}).slots.size(), 1u);
  // p0 serves the next request from its own copy: that one is the newest again.
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kHold, source[0], ""}}).ok);
  ASSERT_TRUE(manager->Apply("p0", {Op{OpKind::kRelease, source[0], ""}}).ok);
  manager->Allocate("p1", kSlots - 1);
  EXPECT_EQ(manager->Allocate("p1", 1).evicted_keys, (std::vector<std::string>{"k"}));
  EXPECT_EQ(StatsOf("p0").cached, 1u);
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
  // Consecutive slots on both sides are one copy per region, all in one batch
  // waited for once, on the destination, before the slots are handed over.
  EXPECT_EQ(api->copies, 2);
  EXPECT_EQ(api->synchronizes, 1);
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

TEST_F(L1PoolTest, FetchIntoDefersUntilPreviousOwnersWritesComplete) {
  Cache(0, {"a"}, 10);
  const uint32_t dst = manager->Allocate("p1", 1).slots.at(0);
  Fill(1, dst, 77);
  ASSERT_TRUE(manager->Apply("p1", {Op{OpKind::kRelease, dst, ""}}, 100).ok);
  // Regranting is safe for the owner's own stream. It does not settle that
  // stream for an external background fetch, which is what dst_slots starts.
  ASSERT_EQ(manager->Allocate("p1", 1).slots, (std::vector<uint32_t>{dst}));
  FetchOptions options;
  options.dst_slots = {dst};
  bool old_write_finished = false;
  api->on_synchronize = [&](int device) {
    if (device == 1 && !old_write_finished) {
      Fill(1, dst, 88);  // the previous owner's queued write lands after Copy
      old_write_finished = true;
    }
  };
  FetchResult before = manager->Fetch("p1", {"a"}, options);
  EXPECT_TRUE(before.status.ok);
  EXPECT_TRUE(before.slots.empty());
  EXPECT_EQ(before.copied_slots, 0u);
  EXPECT_EQ(api->copies, 0);
  EXPECT_FALSE(old_write_finished);
  EXPECT_TRUE(Holds(1, dst, 77));
  EXPECT_EQ(StatsOf("p1").held, 1u);

  // Once the actual owner reports completion the caller retries the same
  // held destination. The service never publishes/releases caller-held slots.
  Fill(1, dst, 88);
  old_write_finished = true;
  ASSERT_TRUE(manager->Commit("p1", 100).ok);
  FetchResult after = manager->Fetch("p1", {"a"}, options);
  EXPECT_TRUE(after.status.ok);
  EXPECT_EQ(after.slots, (std::vector<uint32_t>{dst}));
  EXPECT_EQ(after.copied_slots, 1u);
  EXPECT_TRUE(Holds(1, dst, 10));
  EXPECT_EQ(StatsOf("p1").held, 1u);
}

TEST_F(L1PoolTest, FetchIntoWaitsForTheOwnersCommitThenCopiesEverything) {
  Cache(0, {"a", "b"}, 10);
  const auto dst = manager->Allocate("p1", 2).slots;
  ASSERT_EQ(dst.size(), 2u);
  ASSERT_TRUE(manager->Apply("p1", {Op{OpKind::kRelease, dst[1], ""}}, 100).ok);
  ASSERT_EQ(manager->Allocate("p1", 1).slots, (std::vector<uint32_t>{dst[1]}));
  FetchOptions options;
  options.dst_slots = dst;
  // The owner's device catches up while the fetch waits.
  std::thread owner([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(manager->Commit("p1", 100).ok);
  });
  FetchResult fetched = manager->Fetch("p1", {"a", "b"}, options);
  owner.join();
  EXPECT_TRUE(fetched.status.ok);
  EXPECT_EQ(fetched.slots, dst);
  EXPECT_EQ(fetched.copied_slots, 2u);
  EXPECT_TRUE(Holds(1, dst[1], 11));
}

TEST_F(L1PoolTest, FetchIntoReturnsOnlyTheLeadingSettledDestinations) {
  Cache(0, {"a", "b", "c"}, 10);
  const auto dst = manager->Allocate("p1", 3).slots;
  ASSERT_EQ(dst.size(), 3u);
  Fill(1, dst[0], 77);
  Fill(1, dst[1], 88);
  Fill(1, dst[2], 99);
  ASSERT_TRUE(manager->Apply("p1", {Op{OpKind::kRelease, dst[1], ""}}, 100).ok);
  ASSERT_EQ(manager->Allocate("p1", 1).slots, (std::vector<uint32_t>{dst[1]}));
  FetchOptions options;
  options.dst_slots = dst;
  FetchResult partial = manager->Fetch("p1", {"a", "b", "c"}, options);
  EXPECT_TRUE(partial.status.ok);
  EXPECT_EQ(partial.slots, (std::vector<uint32_t>{dst[0]}));
  EXPECT_EQ(partial.copied_slots, 1u);
  EXPECT_TRUE(Holds(1, dst[0], 10));
  EXPECT_TRUE(Holds(1, dst[1], 88));
  // The third destination is settled, but a fetch must not skip the gap.
  EXPECT_TRUE(Holds(1, dst[2], 99));
  EXPECT_EQ(StatsOf("p1").held, 3u);
  // Mirrors ATOM poll_fetches: publish only the landed prefix, release all
  // grants, and leave the unfilled suffix invisible to prefix lookup.
  ASSERT_TRUE(manager
                  ->Apply("p1",
                          {Op{OpKind::kPublish, dst[0], "a"}, Op{OpKind::kRelease, dst[0], ""},
                           Op{OpKind::kRelease, dst[1], ""}, Op{OpKind::kRelease, dst[2], ""}},
                          200)
                  .ok);
  EXPECT_EQ(StatsOf("p1").held, 0u);
  EXPECT_EQ(StatsOf("p1").cached, 1u);
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

TEST_F(L1PoolTest, AdoptedMemoryIsSharedButNeverZeroedOrFreed) {
  // The owner's allocation: a header before the pool, and content already in it.
  std::vector<uint8_t> owned(64 + kBytes, 0xEE);
  const uint64_t address = reinterpret_cast<uint64_t>(owned.data());
  ExternalMemory external{std::string(reinterpret_cast<const char*>(&address), sizeof(address)),
                          64};
  ASSERT_TRUE(manager->Attach("adopted", "g", 2, kBytes, external).status.ok);
  EXPECT_EQ(api->bases[2], owned.data() + 64);
  EXPECT_EQ(owned[64], 0xEE);  // not zeroed
  ASSERT_TRUE(manager->BindLayout("adopted", TestLayout()).ok);
  ASSERT_EQ(manager->Open("adopted").num_slots, kSlots);
  // Its blocks are a source for other pools and a destination like any slot.
  Cache(0, {"k"}, 7);
  ASSERT_TRUE(manager->Commit("p0", UINT64_MAX).ok);
  FetchResult got = manager->Fetch("adopted", {"k"});
  ASSERT_EQ(got.slots.size(), 1u);
  EXPECT_TRUE(Holds(2, got.slots[0], 7));
  const int frees = api->frees;
  ASSERT_TRUE(manager->Detach("adopted").ok);
  EXPECT_EQ(api->frees, frees);  // the owner frees it
  EXPECT_EQ(api->unimports, 1);
}

TEST_F(L1PoolTest, DestroyingTheManagerUnmapsAdoptedMemoryAndNeverFreesIt) {
  std::vector<uint8_t> owned(64 + kBytes, 0xEE);
  const uint64_t address = reinterpret_cast<uint64_t>(owned.data());
  ExternalMemory external{std::string(reinterpret_cast<const char*>(&address), sizeof(address)),
                          64};
  ASSERT_TRUE(manager->Attach("adopted", "g", 2, kBytes, external).status.ok);
  // The server goes away while the engine is still attached.
  manager.reset();
  EXPECT_EQ(api->foreign_frees, 0);
  EXPECT_EQ(api->unimports, 1);
  EXPECT_EQ(api->frees, 2);  // p0 and p1, which the manager allocated
}

TEST_F(L1PoolTest, ReattachingAdoptedMemoryAtAnotherOffsetMovesThePool) {
  std::vector<uint8_t> owned(128 + kBytes, 0xEE);
  const uint64_t address = reinterpret_cast<uint64_t>(owned.data());
  const std::string handle(reinterpret_cast<const char*>(&address), sizeof(address));
  ASSERT_TRUE(manager->Attach("adopted", "g", 2, kBytes, ExternalMemory{handle, 0}).created);
  EXPECT_EQ(api->bases[2], owned.data());
  // The engine re-created its KV tensor elsewhere in the same allocation.
  AttachResult moved = manager->Attach("adopted", "g", 2, kBytes, ExternalMemory{handle, 128});
  ASSERT_TRUE(moved.status.ok);
  EXPECT_TRUE(moved.created);
  EXPECT_EQ(api->bases[2], owned.data() + 128);
  // The same handle and offset again is the same pool.
  EXPECT_FALSE(manager->Attach("adopted", "g", 2, kBytes, ExternalMemory{handle, 128}).created);
  ASSERT_TRUE(manager->Detach("adopted").ok);
}

TEST_F(L1PoolTest, BindLayoutRejectsARegionWhoseExtentOverflows) {
  ASSERT_TRUE(manager->Attach("odd", "g2", 2, kBytes).status.ok);
  Layout wraps;
  wraps.num_slots = 2;
  wraps.regions = {Region{0, uint64_t{1} << 63}};  // 2 * 2^63 wraps to 0
  EXPECT_FALSE(manager->BindLayout("odd", wraps).ok);
  Layout far;
  far.num_slots = kSlots;
  far.regions = {Region{UINT64_MAX - 7, kUnitA}};  // offset + extent wraps past zero
  EXPECT_FALSE(manager->BindLayout("odd", far).ok);
  Layout past;
  past.num_slots = kSlots;
  past.regions = {Region{kBytes + 1, kUnitA}};
  EXPECT_FALSE(manager->BindLayout("odd", past).ok);
}

TEST_F(L1PoolTest, FetchDoesNotEvictAKeyTheSameCallAsksForLater) {
  Cache(1, {"k2"}, 20);  // the coldest slot of p1
  Cache(1, {"x1", "x2", "x3", "x4", "x5", "x6", "x7"}, 30);
  ASSERT_EQ(StatsOf("p1").vacant, 0u);
  Cache(0, {"k1"}, 10);
  FetchResult got = manager->Fetch("p1", {"k1", "k2"});
  // Room for "k1" comes from content this call does not ask for.
  EXPECT_EQ(got.slots.size(), 2u);
  EXPECT_EQ(got.evicted_keys, (std::vector<std::string>{"x1"}));
  ASSERT_FALSE(got.slots.empty());
  EXPECT_TRUE(Holds(1, got.slots[0], 10));
  if (got.slots.size() == 2) EXPECT_TRUE(Holds(1, got.slots[1], 20));
}

TEST_F(L1PoolTest, FetchThatCopiesNothingLeavesNoPinBehindWhenThePoolIsReopened) {
  // Engines read their own copy of "k" while the pool is re-opened. A fetch
  // that copies nothing used to leave the lock with a pin taken and nothing
  // marking the pool busy: Open reset the pin under it, the fetch released it
  // once more, and the slot stayed pinned -- never to be allocated again.
  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!stop.load()) (void)manager->Fetch("p0", {"k"});
    });
  }
  bool stuck = false;
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < until && !stuck) {
    if (!manager->Open("p0").status.ok) break;
    // Looked at before the next Open, which would reset a stuck pin again. The
    // second pass catches a pin released after the first one took its slots.
    for (int pass = 0; pass < 2 && !stuck; ++pass) {
      AllocateResult all = manager->Allocate("p0", kSlots);
      stuck = all.slots.size() != kSlots;
      std::vector<Op> ops;
      if (pass == 1 && !stuck) ops.push_back(Op{OpKind::kPublish, all.slots[0], "k"});
      for (uint32_t slot : all.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
      (void)manager->Apply("p0", ops);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
  stop.store(true);
  for (std::thread& reader : readers) reader.join();
  EXPECT_FALSE(stuck) << "a slot of p0 stayed pinned after Open";
}

TEST_F(L1PoolTest, TwoAttachesReplacingABusyPoolAllocateItOnce) {
  Cache(0, {"a"}, 10);
  std::mutex gate;
  std::condition_variable gate_cv;
  bool copying = false;
  bool release = false;
  api->on_synchronize = [&](int device) {
    if (device != 1) return;
    std::unique_lock<std::mutex> lock(gate);
    copying = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [&] { return release; });
  };
  std::thread fetch([&] { (void)manager->Fetch("p1", {"a"}); });
  {
    std::unique_lock<std::mutex> lock(gate);
    gate_cv.wait(lock, [&] { return copying; });
  }
  // p1 is busy: both attaches wait for the copy before they replace it.
  const int allocations = api->allocations;
  AttachResult first, second;
  std::thread a([&] { first = manager->Attach("p1", "g", 1, 2 * kBytes); });
  std::thread b([&] { second = manager->Attach("p1", "g", 1, 2 * kBytes); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  {
    std::lock_guard<std::mutex> lock(gate);
    release = true;
  }
  gate_cv.notify_all();
  fetch.join();
  a.join();
  b.join();
  api->on_synchronize = nullptr;
  // One of them replaced the pool; the other found that done or under way.
  EXPECT_EQ(api->allocations, allocations + 1);
  EXPECT_EQ(int{first.created} + int{second.created}, 1);
  EXPECT_EQ(StatsOf("p1").bytes, 2 * kBytes);
}

TEST_F(L1PoolTest, AKeyHeldByManyPoolsIsTrackedThroughEveryHolderComingAndGoing) {
  // Six pools: more holders of one key than an index entry keeps in place.
  const int pools = 6;
  for (int device = 2; device < pools; ++device) {
    const std::string id = "p" + std::to_string(device);
    ASSERT_TRUE(manager->Attach(id, "g", device, kBytes).status.ok);
    ASSERT_TRUE(manager->BindLayout(id, TestLayout()).ok);
    ASSERT_EQ(manager->Open(id).num_slots, kSlots);
  }
  auto pool = [](int device) { return "p" + std::to_string(device); };
  FetchOptions probe;
  probe.probe_only = true;
  // p0 publishes; every other pool copies it in turn, each from the last.
  Cache(0, {"shared", "tail"}, 10);
  for (int device = 1; device < pools; ++device) {
    FetchResult got = manager->Fetch(pool(device), {"shared", "tail"});
    ASSERT_EQ(got.slots.size(), 2u) << device;
    EXPECT_EQ(got.copied_slots, 2u);
    EXPECT_TRUE(Holds(device, got.slots[0], 10));
    EXPECT_TRUE(Holds(device, got.slots[1], 11));
  }
  // Holders leave one at a time, oldest first, then from the middle: whoever
  // is left still serves the key, and a pool that dropped it sees it as
  // another pool's again.
  for (int device : {0, 1, 4, 2}) {
    AllocateResult all = manager->Allocate(pool(device), kSlots);
    ASSERT_EQ(all.slots.size(), kSlots);
    EXPECT_EQ(all.evicted_keys.size(), 2u) << device;
    std::vector<Op> back;
    for (uint32_t slot : all.slots) back.push_back(Op{OpKind::kRelease, slot, ""});
    ASSERT_TRUE(manager->Apply(pool(device), back).ok);
    EXPECT_EQ(StatsOf(pool(device)).cached, 0u);
    EXPECT_EQ(manager->Fetch(pool(device), {"shared", "tail"}, probe).available, 2u) << device;
  }
  // Two holders left (p3, p5). One of them republishes the key in another
  // slot: the old slot is free again, and the key still has two holders.
  AllocateResult extra = manager->Allocate("p3", 1);
  ASSERT_EQ(extra.slots.size(), 1u);
  ASSERT_TRUE(manager
                  ->Apply("p3", {Op{OpKind::kPublish, extra.slots[0], "shared"},
                                 Op{OpKind::kRelease, extra.slots[0], ""}})
                  .ok);
  EXPECT_EQ(StatsOf("p3").cached, 2u);
  EXPECT_EQ(StatsOf("p3").vacant, kSlots - 2);
  // The last holders go; then nobody has it.
  for (int device : {3, 5}) {
    ASSERT_TRUE(manager->Clear(pool(device)).ok);
  }
  EXPECT_EQ(manager->Fetch("p0", {"shared"}, probe).available, 0u);
  // And it can come back.
  Cache(1, {"shared"}, 40);
  FetchResult again = manager->Fetch("p0", {"shared"});
  ASSERT_EQ(again.slots.size(), 1u);
  EXPECT_TRUE(Holds(0, again.slots[0], 40));
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
