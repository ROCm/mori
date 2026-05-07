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

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/peer/peer_dram_allocator.h"

namespace mori::umbp {

namespace {

// 3 buffers x 4 pages of 1 KiB = 12 KiB total DRAM.
constexpr uint64_t kPageSize = 1024;

PeerDramAllocator::TierConfig MakeDramCfg() {
  PeerDramAllocator::TierConfig cfg;
  cfg.buffer_sizes = {kPageSize * 4, kPageSize * 4, kPageSize * 4};
  cfg.buffer_descs = {{0xA0, 0xA1}, {0xB0, 0xB1}, {0xC0, 0xC1}};
  return cfg;
}

PeerDramAllocator::TierConfig EmptyCfg() { return {}; }

std::unique_ptr<PeerDramAllocator> MakeAllocator(
    std::chrono::milliseconds pending_ttl = std::chrono::milliseconds{5000},
    std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500}) {
  return std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), EmptyCfg(), pending_ttl,
                                             read_lease_ttl);
}

}  // namespace

// ---- Allocate / Commit / Resolve happy path ---------------------------------

TEST(PeerDramAllocator, CommitMakesKeyResolvable) {
  auto a = MakeAllocator();
  auto pending = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(pending->size, kPageSize);
  EXPECT_EQ(pending->pages.size(), 1u);

  ASSERT_TRUE(a->Commit(pending->slot_id, "key-1"));
  auto r = a->Resolve("key-1");
  EXPECT_TRUE(r.found);
  EXPECT_EQ(r.size, kPageSize);
  EXPECT_EQ(r.tier, TierType::DRAM);
  EXPECT_EQ(r.pages, pending->pages);

  auto events = a->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].key, "key-1");
  EXPECT_EQ(events[0].size, kPageSize);
  EXPECT_EQ(events[0].tier, TierType::DRAM);
}

// ---- ENOSPC -----------------------------------------------------------------

TEST(PeerDramAllocator, AllocateReturnsNulloptWhenFull) {
  auto a = MakeAllocator();
  std::vector<uint64_t> slot_ids;
  for (int i = 0; i < 12; ++i) {
    auto p = a->Allocate(kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value()) << "i=" << i;
    slot_ids.push_back(p->slot_id);
  }
  EXPECT_FALSE(a->Allocate(kPageSize, TierType::DRAM).has_value());

  EXPECT_TRUE(a->Abort(slot_ids.back()));
  EXPECT_TRUE(a->Allocate(kPageSize, TierType::DRAM).has_value());
}

TEST(PeerDramAllocator, UnconfiguredTierReturnsNullopt) {
  auto a = MakeAllocator();
  EXPECT_FALSE(a->Allocate(kPageSize, TierType::HBM).has_value());
}

// ---- Pending TTL ------------------------------------------------------------

TEST(PeerDramAllocator, PendingSlotExpiresAfterTtl) {
  auto a = std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), EmptyCfg(),
                                               /*pending_ttl=*/std::chrono::milliseconds{1});
  auto pending = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();

  EXPECT_FALSE(a->Commit(pending->slot_id, "key-late"));
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  auto cap = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap[TierType::DRAM].available_bytes, cap[TierType::DRAM].total_bytes);
}

// ---- Abort idempotency ------------------------------------------------------

TEST(PeerDramAllocator, AbortIsIdempotent) {
  auto a = MakeAllocator();
  auto pending = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(999999));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Evict idempotency + REMOVE event ---------------------------------------

TEST(PeerDramAllocator, EvictRemovesKeyAndQueuesEvent) {
  auto a = MakeAllocator();
  auto p = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(a->Commit(p->slot_id, "k"));
  a->DrainPendingEvents();

  auto results = a->Evict({"k", "ghost"});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].key, "k");
  EXPECT_EQ(results[0].bytes_freed, kPageSize);
  EXPECT_EQ(results[1].key, "ghost");
  EXPECT_EQ(results[1].bytes_freed, 0u);

  auto events = a->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::REMOVE);
  EXPECT_EQ(events[0].key, "k");

  EXPECT_FALSE(a->Resolve("k").found);

  results = a->Evict({"k"});
  EXPECT_EQ(results[0].bytes_freed, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Resolve-during-Evict race ---------------------------------------------

TEST(PeerDramAllocator, EvictDefersWhenReadLeaseActive) {
  auto a = std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), EmptyCfg(),
                                               /*pending_ttl=*/std::chrono::milliseconds{5000},
                                               /*read_lease_ttl=*/std::chrono::milliseconds{200});
  auto p = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(a->Commit(p->slot_id, "k"));
  a->DrainPendingEvents();

  auto r = a->Resolve("k");
  ASSERT_TRUE(r.found);

  auto results = a->Evict({"k"});
  EXPECT_EQ(results[0].bytes_freed, 0u);
  EXPECT_TRUE(a->Resolve("k").found);
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  a->RunReaperOnceForTest();
  results = a->Evict({"k"});
  EXPECT_EQ(results[0].bytes_freed, kPageSize);
  auto events = a->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::REMOVE);
}

// ---- Full-sync snapshot -----------------------------------------------------

TEST(PeerDramAllocator, SnapshotOwnedKeysReturnsEveryAdd) {
  auto a = MakeAllocator();
  for (int i = 0; i < 5; ++i) {
    auto p = a->Allocate(kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value());
    ASSERT_TRUE(a->Commit(p->slot_id, "k-" + std::to_string(i)));
  }
  a->DrainPendingEvents();

  auto snap = a->SnapshotOwnedKeys();
  ASSERT_EQ(snap.size(), 5u);
  for (const auto& ev : snap) {
    EXPECT_EQ(ev.kind, KvEvent::Kind::ADD);
    EXPECT_EQ(ev.size, kPageSize);
    EXPECT_EQ(ev.tier, TierType::DRAM);
  }
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Buffer descs filtered to the page set ---------------------------------

TEST(PeerDramAllocator, BufferDescsForPagesDedupAndOrder) {
  auto a = MakeAllocator();
  auto p = a->Allocate(kPageSize * 5, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  ASSERT_EQ(p->pages.size(), 5u);

  auto descs = a->BufferDescsForPages(TierType::DRAM, p->pages);
  ASSERT_EQ(descs.size(), 2u);
  EXPECT_EQ(descs[0].buffer_index, 0u);
  EXPECT_EQ(descs[1].buffer_index, 1u);
  EXPECT_EQ(descs[0].desc_bytes, std::vector<uint8_t>({0xA0, 0xA1}));
  EXPECT_EQ(descs[1].desc_bytes, std::vector<uint8_t>({0xB0, 0xB1}));
}

// ---- Capacities snapshot ----------------------------------------------------

TEST(PeerDramAllocator, TierCapacitiesReflectAllocations) {
  auto a = MakeAllocator();
  auto cap0 = a->TierCapacitiesSnapshot();
  ASSERT_EQ(cap0.count(TierType::DRAM), 1u);
  const uint64_t total = cap0[TierType::DRAM].total_bytes;
  EXPECT_EQ(cap0[TierType::DRAM].available_bytes, total);

  auto p = a->Allocate(kPageSize * 3, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  auto cap1 = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap1[TierType::DRAM].available_bytes, total - 3 * kPageSize);

  ASSERT_TRUE(a->Commit(p->slot_id, "k"));
  auto cap2 = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap2[TierType::DRAM].available_bytes, total - 3 * kPageSize);

  ASSERT_EQ(a->Evict({"k"})[0].bytes_freed, 3 * kPageSize);
  auto cap3 = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap3[TierType::DRAM].available_bytes, total);
}

// ---- Commit after reap ------------------------------------------------------

TEST(PeerDramAllocator, CommitAfterReapReturnsFalse) {
  auto a = std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), EmptyCfg(),
                                               std::chrono::milliseconds{1});
  auto p = a->Allocate(kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();
  EXPECT_FALSE(a->Commit(p->slot_id, "doomed"));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

}  // namespace mori::umbp
