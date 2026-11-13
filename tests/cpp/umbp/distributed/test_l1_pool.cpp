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

#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
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
  void* AllocateHost(uint64_t bytes) override { return host_->AllocateHost(bytes); }
  void FreeHost(void* base) override { host_->FreeHost(base); }
  bool CopyToHost(int device, const void* src, void* host_dst, uint64_t bytes,
                  std::string* error) override {
    return host_->CopyToHost(device, src, host_dst, bytes, error);
  }
  bool CopyFromHost(int device, void* dst, const void* host_src, uint64_t bytes,
                    std::string* error) override {
    return host_->CopyFromHost(device, dst, host_src, bytes, error);
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

// ---------------------------------------------------------------------------
// Lower tiers. The store is faked: what matters here is what the pool manager
// asks of it, and that the bytes survive the round trip.

// What every node's store shares: the objects any node can read, and the
// directory of which node holds which key in L1.
struct FakeCluster {
  std::mutex mutex;
  std::map<std::string, std::string> objects;
  std::map<std::string, std::set<std::string>> directory;
  std::map<std::string, PoolManager*> nodes;
};

class FakeStore final : public LowerStore {
 public:
  FakeStore(std::string node, std::shared_ptr<FakeCluster> cluster)
      : node_(std::move(node)), cluster_(std::move(cluster)) {}

  bool RegisterHostMemory(void* base, uint64_t bytes) override {
    base_ = static_cast<char*>(base);
    bytes_ = bytes;
    return true;
  }
  void DeregisterHostMemory(void* base) override {
    if (base == base_) base_ = nullptr;
  }
  std::vector<bool> Exists(const std::vector<std::string>& keys) override {
    std::lock_guard<std::mutex> lock(cluster_->mutex);
    std::vector<bool> out;
    for (const std::string& key : keys) out.push_back(cluster_->objects.count(key) != 0);
    return out;
  }
  std::vector<bool> Put(const std::vector<std::string>& keys, const std::vector<const void*>& srcs,
                        uint64_t size) override {
    std::lock_guard<std::mutex> lock(cluster_->mutex);
    std::vector<bool> out;
    for (size_t i = 0; i < keys.size(); ++i) {
      EXPECT_TRUE(Registered(srcs[i], size));
      if (fail_puts) {
        out.push_back(false);
        continue;
      }
      ++puts;
      cluster_->objects[keys[i]].assign(static_cast<const char*>(srcs[i]), size);
      out.push_back(true);
    }
    return out;
  }
  std::vector<bool> Get(const std::vector<std::string>& keys, const std::vector<void*>& dsts,
                        uint64_t size) override {
    std::lock_guard<std::mutex> lock(cluster_->mutex);
    std::vector<bool> out;
    for (size_t i = 0; i < keys.size(); ++i) {
      EXPECT_TRUE(Registered(dsts[i], size));
      if (throw_on_get) throw std::runtime_error("peer closed the connection");
      auto found = cluster_->objects.find(keys[i]);
      if (fail_gets || found == cluster_->objects.end() || found->second.size() != size) {
        out.push_back(false);
        continue;
      }
      ++gets;
      std::memcpy(dsts[i], found->second.data(), size);
      out.push_back(true);
    }
    return out;
  }
  void Publish() override { ++publishes; }
  void Advertise(const std::vector<std::string>& added,
                 const std::vector<std::string>& removed) override {
    std::lock_guard<std::mutex> lock(cluster_->mutex);
    for (const std::string& key : added) cluster_->directory[key].insert(node_);
    for (const std::string& key : removed) {
      auto found = cluster_->directory.find(key);
      if (found == cluster_->directory.end()) continue;
      found->second.erase(node_);
      if (found->second.empty()) cluster_->directory.erase(found);
    }
  }
  std::vector<bool> RequestFromPeers(const std::vector<std::string>& keys) override {
    ++peer_requests;
    std::vector<bool> out(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
      PoolManager* peer = nullptr;
      {
        std::lock_guard<std::mutex> lock(cluster_->mutex);
        auto found = cluster_->directory.find(keys[i]);
        if (found == cluster_->directory.end()) continue;
        for (const std::string& node : found->second) {
          if (node != node_) peer = cluster_->nodes[node];
        }
      }
      if (peer == nullptr) continue;
      std::string group;
      std::string key;
      if (!PoolManager::ParseStoreKey(keys[i], &group, &key)) continue;
      out[i] = peer->Export(group, {key}).stored[0];
    }
    return out;
  }

  int puts = 0;
  int gets = 0;
  int publishes = 0;
  int peer_requests = 0;
  bool fail_puts = false;
  bool fail_gets = false;
  bool throw_on_get = false;

 private:
  bool Registered(const void* ptr, uint64_t size) const {
    const char* at = static_cast<const char*>(ptr);
    return base_ != nullptr && at >= base_ && at + size <= base_ + bytes_;
  }

  std::string node_;
  std::shared_ptr<FakeCluster> cluster_;
  char* base_ = nullptr;
  uint64_t bytes_ = 0;
};

// Poll for something a background thread does.
template <typename Predicate>
bool Eventually(Predicate done) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return done();
}

// Node "a" has pools p0 and p1 (devices 0, 1); node "b" has q0 (device 2).
class L1TierTest : public ::testing::Test {
 protected:
  void Build(const TierOptions& options) {
    cluster = std::make_shared<FakeCluster>();
    api = std::make_shared<RecordingApi>();
    store_a = std::make_shared<FakeStore>("a", cluster);
    store_b = std::make_shared<FakeStore>("b", cluster);
    node_a = std::make_unique<PoolManager>(api);
    node_b = std::make_unique<PoolManager>(api);
    node_a->SetLowerStore(store_a, options);
    node_b->SetLowerStore(store_b, options);
    cluster->nodes["a"] = node_a.get();
    cluster->nodes["b"] = node_b.get();
    Add(node_a.get(), "p0", 0);
    Add(node_a.get(), "p1", 1);
    Add(node_b.get(), "q0", 2);
  }
  void SetUp() override {
    TierOptions options;
    options.staging_bytes = 3 * (kUnitA + kUnitB);  // three slots a round
    Build(options);
  }
  void TearDown() override {
    // The managers go first: their workers still call the stores.
    node_a.reset();
    node_b.reset();
  }

  void Add(PoolManager* node, const std::string& id, int device) {
    ASSERT_TRUE(node->Attach(id, "g", device, kBytes).status.ok);
    ASSERT_TRUE(node->BindLayout(id, TestLayout()).ok);
    ASSERT_EQ(node->Open(id).num_slots, kSlots);
  }
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
  // Allocate, fill, publish and release `keys` in a pool; settled at once.
  std::vector<uint32_t> Cache(PoolManager* node, const std::string& id, int device,
                              const std::vector<std::string>& keys, uint8_t first) {
    AllocateResult got = node->Allocate(id, static_cast<uint32_t>(keys.size()));
    EXPECT_EQ(got.slots.size(), keys.size());
    std::vector<Op> ops;
    for (size_t i = 0; i < keys.size(); ++i) {
      Fill(device, got.slots[i], static_cast<uint8_t>(first + i));
      ops.push_back(Op{OpKind::kPublish, got.slots[i], keys[i]});
    }
    for (uint32_t slot : got.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
    EXPECT_TRUE(node->Apply(id, ops).ok);
    return got.slots;
  }
  std::vector<std::string> Names(const std::string& prefix, uint32_t count) {
    std::vector<std::string> out;
    for (uint32_t i = 0; i < count; ++i) out.push_back(prefix + std::to_string(i));
    return out;
  }
  PoolStats StatsOf(PoolManager* node, const std::string& id) {
    for (const PoolStats& stats : node->Stats()) {
      if (stats.pool_id == id) return stats;
    }
    return PoolStats{};
  }
  bool InStore(const std::string& key) {
    std::lock_guard<std::mutex> lock(cluster->mutex);
    return cluster->objects.count(PoolManager::StoreKey("g", key)) != 0;
  }
  bool Advertised(const std::string& key, const std::string& node) {
    std::lock_guard<std::mutex> lock(cluster->mutex);
    auto found = cluster->directory.find(PoolManager::StoreKey("g", key));
    return found != cluster->directory.end() && found->second.count(node) != 0;
  }

  std::shared_ptr<FakeCluster> cluster;
  std::shared_ptr<RecordingApi> api;
  std::shared_ptr<FakeStore> store_a;
  std::shared_ptr<FakeStore> store_b;
  std::unique_ptr<PoolManager> node_a;
  std::unique_ptr<PoolManager> node_b;
};

TEST_F(L1TierTest, StoreKeyRoundTripsBinaryKeys) {
  const std::string key("\x00\xff/k", 4);
  const std::string name = PoolManager::StoreKey("group/x", key);
  std::string group;
  std::string parsed;
  ASSERT_TRUE(PoolManager::ParseStoreKey(name, &group, &parsed));
  EXPECT_EQ(group, "group/x");
  EXPECT_EQ(parsed, key);
  EXPECT_FALSE(PoolManager::ParseStoreKey("other/g/00", &group, &parsed));
  EXPECT_FALSE(PoolManager::ParseStoreKey("l1/g/0", &group, &parsed));
}

TEST_F(L1TierTest, EvictionOffloadsAndAFetchLoadsItBack) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  // Two more slots: the two coldest go below first. More than one staging
  // round would be needed for four, so take four.
  AllocateResult more = node_a->Allocate("p0", 4);
  ASSERT_EQ(more.slots.size(), 4u);
  EXPECT_EQ(more.evicted_keys, (std::vector<std::string>{"k0", "k1", "k2", "k3"}));
  for (const char* key : {"k0", "k1", "k2", "k3"}) EXPECT_TRUE(InStore(key)) << key;
  EXPECT_FALSE(InStore("k4"));
  EXPECT_EQ(StatsOf(node_a.get(), "p0").offloaded_slots, 4u);
  EXPECT_EQ(StatsOf(node_a.get(), "p0").dropped_slots, 0u);

  // The owner overwrites its new slots; the store's copy is what counts.
  for (uint32_t slot : more.slots) Fill(0, slot, 99);
  FetchResult fetched = node_a->Fetch("p1", {"k0", "k1", "k2", "k3"});
  ASSERT_EQ(fetched.slots.size(), 4u);
  EXPECT_EQ(fetched.loaded_slots, 4u);
  EXPECT_EQ(fetched.copied_slots, 0u);
  for (uint32_t i = 0; i < 4; ++i) EXPECT_TRUE(Holds(1, fetched.slots[i], 10 + i)) << i;
  EXPECT_EQ(StatsOf(node_a.get(), "p1").loaded_slots, 4u);
  EXPECT_GE(api->synchronizes, 1);
}

TEST_F(L1TierTest, UnsettledContentIsDroppedNotOffloaded) {
  AllocateResult got = node_a->Allocate("p0", kSlots);
  std::vector<Op> ops;
  for (uint32_t slot : got.slots) {
    ops.push_back(Op{OpKind::kPublish, slot, "u" + std::to_string(slot)});
    ops.push_back(Op{OpKind::kRelease, slot, ""});
  }
  ASSERT_TRUE(node_a->Apply("p0", ops, /*stamp=*/100).ok);
  // The device has not caught up, so the bytes may not be there yet.
  ASSERT_EQ(node_a->Allocate("p0", 2).slots.size(), 2u);
  EXPECT_EQ(store_a->puts, 0);
  EXPECT_EQ(StatsOf(node_a.get(), "p0").dropped_slots, 2u);
}

TEST_F(L1TierTest, FetchMixesAnotherPoolAndTheStore) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  node_a->Allocate("p0", 1);  // k0 -> store only
  FetchResult fetched = node_a->Fetch("p1", {"k1", "k0", "k2"});
  ASSERT_EQ(fetched.slots.size(), 3u);
  EXPECT_EQ(fetched.copied_slots, 2u);
  EXPECT_EQ(fetched.loaded_slots, 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 11));
  EXPECT_TRUE(Holds(1, fetched.slots[1], 10));
  EXPECT_TRUE(Holds(1, fetched.slots[2], 12));
}

TEST_F(L1TierTest, FetchStopsAtAKeyNoTierHolds) {
  Cache(node_a.get(), "p0", 0, {"a", "b"}, 10);
  FetchResult fetched = node_a->Fetch("p1", {"a", "nowhere", "b"});
  EXPECT_EQ(fetched.slots.size(), 1u);
  EXPECT_EQ(StatsOf(node_a.get(), "p1").cached, 1u);
}

TEST_F(L1TierTest, AStoreThatLosesAKeyEndsTheRunThere) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  node_a->Allocate("p0", 3);  // k0..k2 -> store
  {
    std::lock_guard<std::mutex> lock(cluster->mutex);
    // Present at lookup, gone at read: the store evicts on its own.
    cluster->objects[PoolManager::StoreKey("g", "k1")] = "short";
  }
  FetchResult fetched = node_a->Fetch("p1", {"k0", "k1", "k2"});
  ASSERT_EQ(fetched.slots.size(), 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 10));
  PoolStats p1 = StatsOf(node_a.get(), "p1");
  EXPECT_EQ(p1.cached, 1u);
  EXPECT_EQ(p1.vacant, kSlots - 1);
}

TEST_F(L1TierTest, AStoreThatThrowsCostsAMissAndLeavesThePoolUsable) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  node_a->Allocate("p0", 2);  // k0, k1 -> store
  store_a->throw_on_get = true;
  FetchResult fetched = node_a->Fetch("p1", {"k0", "k1"});
  EXPECT_TRUE(fetched.status.ok);
  EXPECT_TRUE(fetched.slots.empty());
  PoolStats p1 = StatsOf(node_a.get(), "p1");
  EXPECT_EQ(p1.cached, 0u);
  EXPECT_EQ(p1.vacant, kSlots);
  store_a->throw_on_get = false;
  EXPECT_EQ(node_a->Fetch("p1", {"k0", "k1"}).loaded_slots, 2u);
}

TEST_F(L1TierTest, ContentThatCameFromTheStoreIsNotWrittenBack) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  node_a->Allocate("p0", 2);
  ASSERT_EQ(store_a->puts, 2);
  ASSERT_EQ(node_a->Fetch("p1", {"k0", "k1"}).loaded_slots, 2u);
  // p1 evicts them again: the store already has both.
  Cache(node_a.get(), "p1", 1, Names("own", kSlots), 50);
  EXPECT_EQ(store_a->puts, 2);
  EXPECT_EQ(StatsOf(node_a.get(), "p1").dropped_slots, 0u);
}

TEST_F(L1TierTest, AFailedOffloadIsCountedAsDropped) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  store_a->fail_puts = true;
  ASSERT_EQ(node_a->Allocate("p0", 2).slots.size(), 2u);
  EXPECT_FALSE(InStore("k0"));
  EXPECT_EQ(StatsOf(node_a.get(), "p0").dropped_slots, 2u);
}

TEST_F(L1TierTest, FetchOffloadsWhatItsDestinationsHeld) {
  Cache(node_a.get(), "p0", 0, {"a", "b"}, 10);
  Cache(node_a.get(), "p1", 1, Names("own", kSlots), 50);
  FetchResult fetched = node_a->Fetch("p1", {"a", "b"});
  ASSERT_EQ(fetched.slots.size(), 2u);
  EXPECT_TRUE(InStore("own0"));
  EXPECT_TRUE(InStore("own1"));
  // And it comes back with the bytes it had.
  FetchResult back = node_a->Fetch("p0", {"own0"});
  ASSERT_EQ(back.slots.size(), 1u);
  EXPECT_TRUE(Holds(0, back.slots[0], 50));
}

TEST_F(L1TierTest, TheDirectoryFollowsWhatTheNodeHolds) {
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  EXPECT_TRUE(Eventually([&] { return Advertised("k0", "a") && Advertised("k7", "a"); }));
  node_a->Allocate("p0", 1);  // k0 leaves L1
  EXPECT_TRUE(Eventually([&] { return !Advertised("k0", "a"); }));
  EXPECT_TRUE(Advertised("k1", "a"));
  // Held by two pools of the node: gone only when both let go.
  ASSERT_EQ(node_a->Fetch("p1", {"k1"}).slots.size(), 1u);
  node_a->Allocate("p0", 1);  // k1 leaves p0, stays in p1
  ASSERT_TRUE(node_a->Clear("p0").ok);
  EXPECT_TRUE(Eventually([&] { return !Advertised("k2", "a"); }));
  EXPECT_TRUE(Advertised("k1", "a"));
}

TEST_F(L1TierTest, AnotherNodeReadsThisNodesL1ThroughTheStore) {
  Cache(node_a.get(), "p0", 0, {"a", "b"}, 10);
  ASSERT_TRUE(Eventually([&] { return Advertised("b", "a"); }));
  EXPECT_FALSE(InStore("a"));
  FetchResult fetched = node_b->Fetch("q0", {"a", "b"});
  ASSERT_EQ(fetched.slots.size(), 2u);
  EXPECT_EQ(fetched.loaded_slots, 2u);
  EXPECT_TRUE(Holds(2, fetched.slots[0], 10));
  EXPECT_TRUE(Holds(2, fetched.slots[1], 11));
  EXPECT_EQ(store_b->peer_requests, 1);
  EXPECT_GE(store_a->publishes, 1);
  // Node a still holds both, and now knows the store has them too.
  EXPECT_EQ(StatsOf(node_a.get(), "p0").cached, 2u);
  const int puts = store_a->puts;
  node_a->Allocate("p0", kSlots);
  EXPECT_EQ(store_a->puts, puts);
}

TEST_F(L1TierTest, ExportSkipsWhatIsUnsettledOrUnknown) {
  AllocateResult got = node_a->Allocate("p0", 1);
  ASSERT_TRUE(node_a->Apply("p0", {Op{OpKind::kPublish, got.slots[0], "late"}}, 100).ok);
  Cache(node_a.get(), "p0", 0, {"ready"}, 10);
  ExportResult result = node_a->Export("g", {"late", "ready", "nowhere"});
  ASSERT_TRUE(result.status.ok);
  EXPECT_EQ(result.stored, (std::vector<bool>{false, true, false}));
  ASSERT_TRUE(node_a->Commit("p0", 100).ok);
  EXPECT_EQ(node_a->Export("g", {"late"}).stored, (std::vector<bool>{true}));
  EXPECT_TRUE(node_a->Export("other-group", {"ready"}).stored == (std::vector<bool>{false}));
}

TEST_F(L1TierTest, ExportRepeatsACopyTheStoreHasLost) {
  Cache(node_a.get(), "p0", 0, {"a"}, 10);
  ASSERT_EQ(node_a->Export("g", {"a"}).stored, (std::vector<bool>{true}));
  ASSERT_EQ(node_a->Export("g", {"a"}).stored, (std::vector<bool>{true}));
  EXPECT_EQ(store_a->puts, 1);
  {
    std::lock_guard<std::mutex> lock(cluster->mutex);
    cluster->objects.clear();
  }
  ASSERT_EQ(node_a->Export("g", {"a"}).stored, (std::vector<bool>{true}));
  EXPECT_EQ(store_a->puts, 2);
  EXPECT_TRUE(InStore("a"));
}

TEST_F(L1TierTest, OffloadAheadCopiesColdSlotsBeforeTheyAreNeeded) {
  TearDown();
  TierOptions options;
  options.staging_bytes = 3 * (kUnitA + kUnitB);
  options.offload_ahead = 0.5;  // keep four of the eight slots ready
  Build(options);
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  ASSERT_TRUE(Eventually([&] { return InStore("k3"); }));
  for (const char* key : {"k0", "k1", "k2"}) EXPECT_TRUE(InStore(key)) << key;
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_FALSE(InStore("k4"));
  // Nothing left L1, and the eviction that follows has nothing to copy.
  EXPECT_EQ(StatsOf(node_a.get(), "p0").cached, kSlots);
  const int puts = store_a->puts;
  ASSERT_EQ(node_a->Allocate("p0", 4).slots.size(), 4u);
  EXPECT_EQ(store_a->puts, puts);
  EXPECT_EQ(StatsOf(node_a.get(), "p0").dropped_slots, 0u);
}

TEST_F(L1TierTest, WithoutLoadingAFetchStaysInL1) {
  TearDown();
  TierOptions options;
  options.staging_bytes = 3 * (kUnitA + kUnitB);
  options.load = false;
  Build(options);
  Cache(node_a.get(), "p0", 0, Names("k", kSlots), 10);
  node_a->Allocate("p0", 1);
  ASSERT_TRUE(InStore("k0"));
  EXPECT_TRUE(node_a->Fetch("p1", {"k0"}).slots.empty());
  EXPECT_EQ(node_a->Fetch("p1", {"k1"}).copied_slots, 1u);
}

TEST(L1ServiceTest, ExportCrossesTheSocket) {
  const std::string path = "/tmp/umbp_l1_export_" + std::to_string(getpid()) + ".sock";
  unlink(path.c_str());
  const std::string address = "unix://" + path;
  auto cluster = std::make_shared<FakeCluster>();
  auto store = std::make_shared<FakeStore>("a", cluster);
  {
    L1Server server(address, MakeHostMemoryApi());
    ASSERT_TRUE(server.ok());
    TierOptions options;
    options.staging_bytes = 4 * (kUnitA + kUnitB);
    server.manager().SetLowerStore(store, options);
    L1Client client(address);
    ASSERT_TRUE(client.Attach("p0", "g", 0, kBytes).status.ok);
    ASSERT_TRUE(client.BindLayout("p0", TestLayout()).ok);
    ASSERT_EQ(client.Open("p0").num_slots, kSlots);
    AllocateResult got = client.Allocate("p0", 1, 1);
    ASSERT_TRUE(client.Apply("p0", {Op{OpKind::kPublish, got.slots[0], "a"}}).ok);
    ExportResult exported = client.Export("g", {"a", "missing"});
    ASSERT_TRUE(exported.status.ok) << exported.status.error;
    EXPECT_EQ(exported.stored, (std::vector<bool>{true, false}));
    std::vector<PoolStats> stats;
    ASSERT_TRUE(client.Stats(&stats).ok);
    EXPECT_EQ(stats[0].offloaded_slots, 1u);
  }
  unlink(path.c_str());
}

}  // namespace
}  // namespace mori::umbp::l1
