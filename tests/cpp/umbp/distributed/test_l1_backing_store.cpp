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
// The L1 pools over the UMBP store, end to end and without a GPU: a pool
// manager whose pools are host memory, and under it the store an embedded
// client hosts -- the real one, with its DRAM medium, its page allocator and
// its transfer layer. What the pools evict becomes objects of that store and
// comes back out of them; nothing here is a fake.
//
// The pools' memory is registered as host memory, so the store's planner hands
// the copies to LocalCopyEngine where a server's go to HbmCopyEngine. Every
// step before the copy itself is the one a server takes.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/l1/l1_pool.h"
#include "umbp/standalone/l1_backing_store.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {

using l1::AllocateResult;
using l1::FetchOptions;
using l1::FetchResult;
using l1::Layout;
using l1::Op;
using l1::OpKind;
using l1::PartSpec;
using l1::PoolManager;
using l1::PoolStats;
using l1::Region;

constexpr uint32_t kSlots = 6;
constexpr size_t kPage = 4096;
// A slot is three pieces on two allocations, 9000 bytes: more than two of the
// store's pages, with no piece ending on a page boundary.
constexpr uint64_t kUnitA = 3000;
constexpr uint64_t kUnitB = 1000;
constexpr uint64_t kUnitC = 5000;
constexpr uint64_t kSlotBytes = kUnitA + kUnitB + kUnitC;
constexpr uint64_t kBytes0 = kSlots * (kUnitA + kUnitB);
constexpr uint64_t kBytes1 = kSlots * kUnitC;

Layout Layout0() {
  Layout layout;
  layout.num_slots = kSlots;
  layout.regions = {Region{0, kUnitA}, Region{kSlots * kUnitA, kUnitB}};
  return layout;
}

Layout Layout1() {
  Layout layout;
  layout.num_slots = kSlots;
  layout.regions = {Region{0, kUnitC}};
  return layout;
}

class L1BackingStoreTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // The store would otherwise register its DRAM with the GPUs of the machine
    // for its gather kernel. These pools are host memory: no GPU is needed
    // here, and none is opened.
    setenv("UMBP_DRAM_HOST_REGISTER", "0", 1);
  }

  // A store of `pages` pages in this process: no master, no peers.
  void BringStore(size_t pages) {
    UMBPConfig config;
    config.dram.capacity_bytes = pages * kPage;
    config.dram.prefault = false;
    UMBPDistributedConfig distributed;
    distributed.dram_page_size = kPage;
    distributed.io_engine.host.clear();
    distributed.peer_service_port = 0;
    config.distributed = distributed;
    client = CreateUMBPClient(config);
    ASSERT_NE(client, nullptr);
    store = standalone::MakeL1BackingStore(client.get(), mori::io::MemoryLocationType::CPU);
  }

  // Pool `rank`: two parts, each over memory this test owns and the pool adopts.
  void BringPool(int rank) {
    const std::string id = "r" + std::to_string(rank);
    memory[rank][0].assign(kBytes0, 0);
    memory[rank][1].assign(kBytes1, 0);
    for (uint32_t part : {0u, 1u}) {
      const uint64_t address = reinterpret_cast<uint64_t>(memory[rank][part].data());
      l1::ExternalMemory external;
      external.handle.assign(reinterpret_cast<const char*>(&address), sizeof(address));
      ASSERT_TRUE(manager
                      ->Attach(id, "g", 10 * rank + static_cast<int>(part),
                               part == 0 ? kBytes0 : kBytes1, external, PartSpec{part, 2})
                      .status.ok);
      ASSERT_TRUE(manager->BindLayout(id, part == 0 ? Layout0() : Layout1(), part).ok);
    }
    ASSERT_EQ(manager->Open(id).num_slots, kSlots);
  }

  void Bring(size_t pages = 256) {
    BringStore(pages);
    manager = std::make_unique<PoolManager>(l1::MakeHostMemoryApi());
    manager->SetBackingStore(store);
    BringPool(0);
    BringPool(1);
  }

  void TearDown() override {
    manager.reset();  // the pools give their memory back to the store first
    store.reset();
    if (client) client->Close();
  }

  // Every byte of a slot differs from its neighbours and from other slots'.
  static uint8_t ByteOf(uint8_t seed, uint64_t offset) {
    return static_cast<uint8_t>(seed * 31 + offset * 7 + (offset >> 8));
  }
  void Fill(int rank, uint32_t slot, uint8_t seed) {
    uint8_t* a = memory[rank][0].data() + slot * kUnitA;
    uint8_t* b = memory[rank][0].data() + kSlots * kUnitA + slot * kUnitB;
    uint8_t* c = memory[rank][1].data() + slot * kUnitC;
    for (uint64_t i = 0; i < kUnitA; ++i) a[i] = ByteOf(seed, i);
    for (uint64_t i = 0; i < kUnitB; ++i) b[i] = ByteOf(seed, kUnitA + i);
    for (uint64_t i = 0; i < kUnitC; ++i) c[i] = ByteOf(seed, kUnitA + kUnitB + i);
  }
  bool Holds(int rank, uint32_t slot, uint8_t seed) {
    const uint8_t* a = memory[rank][0].data() + slot * kUnitA;
    const uint8_t* b = memory[rank][0].data() + kSlots * kUnitA + slot * kUnitB;
    const uint8_t* c = memory[rank][1].data() + slot * kUnitC;
    for (uint64_t i = 0; i < kUnitA; ++i) {
      if (a[i] != ByteOf(seed, i)) return false;
    }
    for (uint64_t i = 0; i < kUnitB; ++i) {
      if (b[i] != ByteOf(seed, kUnitA + i)) return false;
    }
    for (uint64_t i = 0; i < kUnitC; ++i) {
      if (c[i] != ByteOf(seed, kUnitA + kUnitB + i)) return false;
    }
    return true;
  }

  // Allocate, fill, publish and release `keys` in pool r<rank>.
  void Cache(int rank, const std::vector<std::string>& keys, uint8_t first) {
    const std::string id = "r" + std::to_string(rank);
    AllocateResult got = manager->Allocate(id, static_cast<uint32_t>(keys.size()));
    ASSERT_EQ(got.slots.size(), keys.size());
    std::vector<Op> ops;
    for (size_t i = 0; i < keys.size(); ++i) {
      Fill(rank, got.slots[i], static_cast<uint8_t>(first + i));
      ops.push_back(Op{OpKind::kPublish, got.slots[i], keys[i]});
    }
    for (uint32_t slot : got.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
    ASSERT_TRUE(manager->Apply(id, ops).ok);
  }

  // Take every slot of the pool, which evicts all it caches, scribble over the
  // pool as an engine reusing it would, and give the slots back empty.
  void EvictAll(int rank) {
    const std::string id = "r" + std::to_string(rank);
    AllocateResult got = manager->Allocate(id, kSlots);
    ASSERT_EQ(got.slots.size(), kSlots);
    std::memset(memory[rank][0].data(), 0xEE, kBytes0);
    std::memset(memory[rank][1].data(), 0xEE, kBytes1);
    std::vector<Op> ops;
    for (uint32_t slot : got.slots) ops.push_back(Op{OpKind::kRelease, slot, ""});
    ASSERT_TRUE(manager->Apply(id, ops).ok);
  }

  PoolStats StatsOf(const std::string& id) {
    for (const PoolStats& stats : manager->Stats()) {
      if (stats.pool_id == id) return stats;
    }
    return PoolStats{};
  }

  std::unique_ptr<IUMBPClient> client;
  std::shared_ptr<l1::BackingStore> store;
  std::unique_ptr<PoolManager> manager;
  std::vector<uint8_t> memory[2][2];
};

TEST_F(L1BackingStoreTest, AnEvictedSlotIsStoredAsOneObjectAndReadBackIntoAnotherPool) {
  Bring();
  Cache(0, {"a", "b", "c"}, 1);
  EvictAll(0);
  EXPECT_EQ(StatsOf("r0").written_back_slots, 3u);
  EXPECT_EQ(StatsOf("r0").lost_slots, 0u);
  FetchOptions probe;
  probe.probe_only = true;
  EXPECT_EQ(manager->Fetch("r1", {"a", "b", "c", "never"}, probe).available, 3u);
  FetchResult fetched = manager->Fetch("r1", {"a", "b", "c", "never"});
  ASSERT_TRUE(fetched.status.ok) << fetched.status.error;
  ASSERT_EQ(fetched.slots.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(Holds(1, fetched.slots[i], static_cast<uint8_t>(1 + i))) << "key " << i;
  }
  EXPECT_EQ(fetched.copied_bytes, 3 * kSlotBytes);
  EXPECT_EQ(StatsOf("r1").loaded_slots, 3u);
  // And into slots the caller already holds, in the pool that evicted them.
  AllocateResult held = manager->Allocate("r0", 2);
  FetchOptions into;
  into.dst_slots = held.slots;
  FetchResult filled = manager->Fetch("r0", {"c", "a"}, into);
  ASSERT_EQ(filled.copied_slots, 2u);
  EXPECT_TRUE(Holds(0, held.slots[0], 3));
  EXPECT_TRUE(Holds(0, held.slots[1], 1));
}

TEST_F(L1BackingStoreTest, TheObjectsAreTheStoresOwn) {
  Bring();
  Cache(0, {"a"}, 9);
  EvictAll(0);
  FetchOptions probe;
  probe.probe_only = true;
  ASSERT_EQ(manager->Fetch("r1", {"a"}, probe).available, 1u);
  // Emptying the store empties what the pools can find: they kept no copy and
  // no index of their own.
  ASSERT_TRUE(client->Clear());
  EXPECT_EQ(manager->Fetch("r1", {"a"}, probe).available, 0u);
  EXPECT_TRUE(manager->Fetch("r1", {"a"}).slots.empty());
}

TEST_F(L1BackingStoreTest, KeysOfOpaqueBytesAreStoredUnderTextKeys) {
  Bring();
  const std::string binary("\x00\xff\x80 k\n", 6);
  const std::string plain = "00ff80206b0a";  // the hex of the one above
  Cache(0, {binary, plain}, 20);
  EvictAll(0);
  EXPECT_EQ(StatsOf("r0").written_back_slots, 2u);
  FetchResult fetched = manager->Fetch("r1", {binary, plain});
  ASSERT_EQ(fetched.slots.size(), 2u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 20));
  EXPECT_TRUE(Holds(1, fetched.slots[1], 21));
  EXPECT_EQ(standalone::L1StoreKey("g/1", "abc"), "l1/g/1/s/abc");
  EXPECT_EQ(standalone::L1StoreKey("g/1", binary), "l1/g/1/x/" + plain);
  EXPECT_NE(standalone::L1StoreKey("g/1", binary), standalone::L1StoreKey("g/1", plain));
}

// The pools' share of DRAM is whatever the store's own configuration and
// eviction leave them. A put into a full store is refused -- the store evicts
// in the background, at its watermarks, never inside a put -- so of a burst
// larger than the room it has, the excess is lost and counted; the pools
// neither fail nor serve a slot the store does not hold; and once the store
// has evicted, later evictions are kept again.
TEST_F(L1BackingStoreTest, TheStoresCapacityAndEvictionDecideWhatIsKept) {
  Bring(/*pages=*/12);  // four slots' worth: a slot takes three pages
  const std::vector<std::string> first = {"a0", "a1", "a2", "a3", "a4", "a5"};
  const std::vector<std::string> second = {"b0", "b1", "b2", "b3", "b4", "b5"};
  Cache(0, first, 1);
  EvictAll(0);
  Cache(0, second, 41);
  EvictAll(0);
  PoolStats stats = StatsOf("r0");
  EXPECT_EQ(stats.written_back_slots + stats.lost_slots, 12u);
  EXPECT_GE(stats.written_back_slots, 1u);
  EXPECT_LE(stats.written_back_slots, 4u) << "more slots were taken than the store has pages for";
  size_t found = 0;
  for (size_t i = 0; i < 12; ++i) {
    const std::string& key = i < 6 ? first[i] : second[i - 6];
    FetchResult fetched = manager->Fetch("r1", {key});
    if (fetched.slots.empty()) continue;
    ++found;
    EXPECT_TRUE(Holds(1, fetched.slots[0], static_cast<uint8_t>(i < 6 ? 1 + i : 41 + i - 6)))
        << key;
  }
  EXPECT_GE(found, 1u);
  EXPECT_LE(found, stats.written_back_slots);
  // The store was left over its high watermark: its evictor makes room, and a
  // slot evicted from L1 after that is stored.
  const uint64_t before = stats.written_back_slots;
  for (int round = 0; round < 200 && StatsOf("r0").written_back_slots == before; ++round) {
    Cache(0, {"late" + std::to_string(round)}, static_cast<uint8_t>(90 + round));
    EvictAll(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  EXPECT_GT(StatsOf("r0").written_back_slots, before)
      << "the store never made room for a later eviction";
}

TEST_F(L1BackingStoreTest, ClearingAPoolPutsItsGroupsObjectsOutOfReach) {
  Bring();
  Cache(0, {"a"}, 3);
  EvictAll(0);
  ASSERT_TRUE(manager->Clear("r0").ok);
  EXPECT_TRUE(manager->Fetch("r1", {"a"}).slots.empty());
  // The same key, written after the Clear, is a new object.
  Cache(0, {"a"}, 4);
  EvictAll(0);
  FetchResult fetched = manager->Fetch("r1", {"a"});
  ASSERT_EQ(fetched.slots.size(), 1u);
  EXPECT_TRUE(Holds(1, fetched.slots[0], 4));
}

TEST_F(L1BackingStoreTest, AClosedStoreTakesNothingAndThePoolsCarryOn) {
  Bring();
  Cache(0, {"a"}, 3);
  client->Close();
  EvictAll(0);
  EXPECT_EQ(StatsOf("r0").written_back_slots, 0u);
  EXPECT_EQ(StatsOf("r0").lost_slots, 1u);
  EXPECT_TRUE(manager->Fetch("r1", {"a"}).slots.empty());
}

TEST_F(L1BackingStoreTest, AnAllocationTwoPoolsShareIsRegisteredOnce) {
  BringStore(16);
  std::vector<uint8_t> shared(1 << 16);
  std::string error;
  ASSERT_TRUE(store->AddMemory(shared.data(), shared.size(), 0, &error)) << error;
  ASSERT_TRUE(store->AddMemory(shared.data(), shared.size() / 2, 0, &error)) << error;
  EXPECT_FALSE(store->AddMemory(shared.data(), shared.size() * 2, 0, &error));
  EXPECT_FALSE(error.empty());
  // Still registered for its other user after the first has gone: the object
  // is written from it and read back into it.
  store->RemoveMemory(shared.data());
  for (size_t i = 0; i < 8192; ++i) shared[i] = static_cast<uint8_t>(i * 13);
  l1::BackingStore::Block block;
  block.key = "k";
  block.fragments = {{shared.data(), 5000}, {shared.data() + 5000, 3192}};
  ASSERT_EQ(store->Write("space", {block}), std::vector<bool>({true}));
  EXPECT_EQ(store->Contains("space", {"k", "other"}), std::vector<bool>({true, false}));
  EXPECT_EQ(store->Contains("elsewhere", {"k"}), std::vector<bool>({false}));
  block.fragments = {{shared.data() + 20000, 100}, {shared.data() + 30000, 8092}};
  ASSERT_EQ(store->Read("space", {block}), std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(shared.data() + 20000, shared.data(), 100), 0);
  EXPECT_EQ(std::memcmp(shared.data() + 30000, shared.data() + 100, 8092), 0);
  // A key already stored is not an error: the pools may offer one twice.
  EXPECT_EQ(store->Write("space", {block}), std::vector<bool>({true}));
  store->RemoveMemory(shared.data());
}

}  // namespace
}  // namespace mori::umbp
