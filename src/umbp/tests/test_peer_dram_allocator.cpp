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
#include <optional>
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

// Strip AllocateResult down to its slot for tests that don't exercise
// the dedup outcome.
std::optional<PeerDramAllocator::PendingSlot> AllocateOk(PeerDramAllocator& a,
                                                         const std::string& key, uint64_t size,
                                                         TierType tier) {
  return a.Allocate(key, size, tier).slot;
}

}  // namespace

// ---- Allocate / Commit / Resolve happy path ---------------------------------

TEST(PeerDramAllocator, CommitMakesKeyResolvable) {
  auto a = MakeAllocator();
  auto pending = AllocateOk(*a, "key-1", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(pending->size, kPageSize);
  EXPECT_EQ(pending->pages.size(), 1u);

  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pending->slot_id, "key-1", committed_bytes));
  EXPECT_EQ(committed_bytes, pending->size);
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

// ---- Allocate-side dedup ----------------------------------------------------
// Defensive layer for master-index lag (primary dedup is at BatchRoutePut).

TEST(PeerDramAllocator, AllocateRejectsAlreadyOwnedKey) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(first.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(first->slot_id, "A", committed_bytes));
  a->DrainPendingEvents();

  const auto cap_after_commit = a->TierCapacitiesSnapshot()[TierType::DRAM];

  auto second = a->Allocate("A", kPageSize, TierType::DRAM);
  EXPECT_EQ(second.outcome, PeerDramAllocator::Outcome::kSuccessAlreadyExists);
  EXPECT_FALSE(second.slot.has_value());

  // No pages reserved -> capacity unchanged.
  const auto cap_after_dedup = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_dedup.available_bytes, cap_after_commit.available_bytes);
}

TEST(PeerDramAllocator, AllocateAllowsDifferentKey) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(first.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(first->slot_id, "A", committed_bytes));

  auto second = a->Allocate("B", kPageSize, TierType::DRAM);
  EXPECT_EQ(second.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
  ASSERT_TRUE(second.slot.has_value());
  EXPECT_TRUE(a->Commit(second.slot->slot_id, "B", committed_bytes));
}

// Lax mode: pending_ not checked.  Two same-key Allocates before any
// Commit both succeed; race absorbed by Commit() (see
// DuplicateCommitIsIdempotentAndKeepsFirst).
TEST(PeerDramAllocator, AllocateDoesNotRejectOnPendingDuplicate) {
  auto a = MakeAllocator();

  auto first = a->Allocate("A", kPageSize, TierType::DRAM);
  EXPECT_EQ(first.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
  ASSERT_TRUE(first.slot.has_value());

  auto second = a->Allocate("A", kPageSize, TierType::DRAM);
  EXPECT_EQ(second.outcome, PeerDramAllocator::Outcome::kSuccessAllocated);
  ASSERT_TRUE(second.slot.has_value());
  ASSERT_NE(second.slot->slot_id, first.slot->slot_id);
}

// ---- Duplicate Commit idempotency -------------------------------------------
// Race-window safety net.  Both Allocates must happen BEFORE either
// Commit — once owned_["dup-key"] is set, the new owned_-check in
// Allocate would reject the second slot before it could reach Commit.

TEST(PeerDramAllocator, DuplicateCommitIsIdempotentAndKeepsFirst) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "dup-key", kPageSize, TierType::DRAM);
  ASSERT_TRUE(first.has_value());
  auto second = AllocateOk(*a, "dup-key", kPageSize, TierType::DRAM);
  ASSERT_TRUE(second.has_value());
  ASSERT_NE(second->slot_id, first->slot_id);

  const auto first_pages = first->pages;

  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(first->slot_id, "dup-key", committed_bytes));
  EXPECT_EQ(committed_bytes, kPageSize);

  auto events = a->DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].key, "dup-key");

  // First owned (1 page) + second still pending (1 page) = 2 occupied.
  const auto cap_after_first_commit = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_first_commit.available_bytes,
            cap_after_first_commit.total_bytes - 2 * kPageSize);

  // Duplicate Commit: idempotent success, consumes the second pending
  // (caller never needs to Abort it), prior owned slot unchanged.
  committed_bytes = 0;
  ASSERT_TRUE(a->Commit(second->slot_id, "dup-key", committed_bytes));
  EXPECT_EQ(committed_bytes, kPageSize);

  // Master's view unchanged: no REMOVE, no second ADD.
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  // Resolve still returns the first commit's pages.
  auto r = a->Resolve("dup-key");
  ASSERT_TRUE(r.found);
  EXPECT_EQ(r.pages, first_pages);
  EXPECT_EQ(r.size, kPageSize);

  // Second slot's pages freed -> only first occupies (1 page).
  const auto cap_after_dup = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_dup.available_bytes, cap_after_dup.total_bytes - kPageSize);
  EXPECT_EQ(cap_after_dup.total_bytes, cap_after_first_commit.total_bytes);

  // Second slot_id no longer pending; idempotent Abort returns true.
  EXPECT_TRUE(a->Abort(second->slot_id));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- ENOSPC -----------------------------------------------------------------

TEST(PeerDramAllocator, AllocateReturnsNulloptWhenFull) {
  auto a = MakeAllocator();
  std::vector<uint64_t> slot_ids;
  for (int i = 0; i < 12; ++i) {
    auto p = AllocateOk(*a, "k-" + std::to_string(i), kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value()) << "i=" << i;
    slot_ids.push_back(p->slot_id);
  }
  EXPECT_FALSE(AllocateOk(*a, "k-overflow", kPageSize, TierType::DRAM).has_value());

  EXPECT_TRUE(a->Abort(slot_ids.back()));
  EXPECT_TRUE(AllocateOk(*a, "k-recovered", kPageSize, TierType::DRAM).has_value());
}

TEST(PeerDramAllocator, UnconfiguredTierReturnsNullopt) {
  auto a = MakeAllocator();
  EXPECT_FALSE(AllocateOk(*a, "k", kPageSize, TierType::HBM).has_value());
}

// ---- Pending TTL ------------------------------------------------------------

TEST(PeerDramAllocator, PendingSlotExpiresAfterTtl) {
  auto a = std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), EmptyCfg(),
                                               /*pending_ttl=*/std::chrono::milliseconds{1});
  auto pending = AllocateOk(*a, "key-late", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pending->slot_id, "key-late", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  auto cap = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap[TierType::DRAM].available_bytes, cap[TierType::DRAM].total_bytes);
}

// ---- Abort idempotency ------------------------------------------------------

TEST(PeerDramAllocator, AbortIsIdempotent) {
  auto a = MakeAllocator();
  auto pending = AllocateOk(*a, "k", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pending.has_value());
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(999999));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Evict idempotency + REMOVE event ---------------------------------------

TEST(PeerDramAllocator, EvictRemovesKeyAndQueuesEvent) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize, TierType::DRAM);
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  EXPECT_EQ(committed_bytes, p->size);
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
  auto p = AllocateOk(*a, "k", kPageSize, TierType::DRAM);
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  EXPECT_EQ(committed_bytes, p->size);
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
    const std::string k = "k-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value());
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, committed_bytes));
    EXPECT_EQ(committed_bytes, p->size);
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

TEST(PeerDramAllocator, SnapshotOwnedKeysTracksExternalSsdEvents) {
  auto a = MakeAllocator();
  a->QueueExternalEvent(KvEvent{KvEvent::Kind::ADD, "ssd-a", TierType::SSD, 123});
  a->QueueExternalEvent(KvEvent{KvEvent::Kind::ADD, "ssd-b", TierType::SSD, 456});

  auto snap = a->SnapshotOwnedKeys();
  ASSERT_EQ(snap.size(), 2u);
  EXPECT_TRUE(std::any_of(snap.begin(), snap.end(), [](const KvEvent& ev) {
    return ev.key == "ssd-a" && ev.tier == TierType::SSD && ev.size == 123;
  }));
  EXPECT_TRUE(std::any_of(snap.begin(), snap.end(), [](const KvEvent& ev) {
    return ev.key == "ssd-b" && ev.tier == TierType::SSD && ev.size == 456;
  }));

  a->QueueExternalEvent(KvEvent{KvEvent::Kind::REMOVE, "ssd-a", TierType::SSD, 0});
  snap = a->SnapshotOwnedKeys();
  ASSERT_EQ(snap.size(), 1u);
  EXPECT_EQ(snap[0].key, "ssd-b");

  a->QueueExternalEvent(KvEvent{KvEvent::Kind::CLEAR_AT_TIER, "", TierType::SSD, 0});
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());
}

// ---- Buffer descs filtered to the page set ---------------------------------

TEST(PeerDramAllocator, BufferDescsForPagesDedupAndOrder) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize * 5, TierType::DRAM);
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

  auto p = AllocateOk(*a, "k", kPageSize * 3, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  auto cap1 = a->TierCapacitiesSnapshot();
  EXPECT_EQ(cap1[TierType::DRAM].available_bytes, total - 3 * kPageSize);

  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  EXPECT_EQ(committed_bytes, p->size);
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
  auto p = AllocateOk(*a, "doomed", kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();
  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(p->slot_id, "doomed", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Distributed Clear ------------------------------------------------------

TEST(PeerDramAllocator, ClearLocalReleasesOwnedAndCancelsPending) {
  auto a = MakeAllocator();

  auto pA = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pA.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pA->slot_id, "A", committed_bytes));

  auto pB = AllocateOk(*a, "B", kPageSize * 2, TierType::DRAM);
  ASSERT_TRUE(pB.has_value());
  a->DrainPendingEvents();  // discard the A ADD

  const auto cap_before = a->TierCapacitiesSnapshot()[TierType::DRAM];
  ASSERT_LT(cap_before.available_bytes, cap_before.total_bytes);

  a->ClearLocal();

  EXPECT_TRUE(a->IsClearFullSyncPending());
  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  // Owned pages (A) returned immediately; pending pages (B) still held.
  auto cap_after_clear = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_clear.available_bytes, cap_before.total_bytes - 2 * kPageSize);

  // Committing the cancelled pending fails AND releases its pages
  // without emitting an ADD.
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
  EXPECT_FALSE(a->Resolve("B").found);

  auto cap_final = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_final.available_bytes, cap_final.total_bytes);
}

TEST(PeerDramAllocator, ClearLocalGatesAllocateUntilAcked) {
  auto a = MakeAllocator();

  a->ClearLocal();
  EXPECT_FALSE(AllocateOk(*a, "blocked", kPageSize, TierType::DRAM).has_value());

  a->ClearFullSyncAcked();
  EXPECT_FALSE(a->IsClearFullSyncPending());
  EXPECT_TRUE(AllocateOk(*a, "ok-after-ack", kPageSize, TierType::DRAM).has_value());
}

TEST(PeerDramAllocator, ClearLocalDropsQueuedAdds) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  // ADD is sitting in the outbox, not yet drained.

  a->ClearLocal();

  EXPECT_TRUE(a->DrainPendingEvents().empty());
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());
}

TEST(PeerDramAllocator, AbortReleasesCancelledPending) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "p1", kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());

  a->ClearLocal();
  // Abort on a cancelled pending is idempotent and frees the pages.
  EXPECT_TRUE(a->Abort(p->slot_id));
  a->ClearFullSyncAcked();

  auto cap = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap.available_bytes, cap.total_bytes);
}

// Pre-clear pending Commit fails; post-ack new Allocate+Commit succeeds.
TEST(PeerDramAllocator, PendingGenerationRejectsPreClearCommit) {
  auto a = MakeAllocator();

  auto pB = AllocateOk(*a, "B", kPageSize * 2, TierType::DRAM);
  ASSERT_TRUE(pB.has_value());
  const auto cap_before = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_before.available_bytes, cap_before.total_bytes - 2 * kPageSize);

  a->ClearLocal();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
  auto cap_after_reject = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_reject.available_bytes, cap_after_reject.total_bytes);

  a->ClearFullSyncAcked();

  auto pC = AllocateOk(*a, "C", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pC.has_value());
  ASSERT_TRUE(a->Commit(pC->slot_id, "C", committed_bytes));
  EXPECT_EQ(committed_bytes, kPageSize);
  EXPECT_TRUE(a->Resolve("C").found);
}

// Repeated Clears still reject the original pre-clear pending Commit.
TEST(PeerDramAllocator, PendingGenerationSurvivesDoubleClear) {
  auto a = MakeAllocator();

  auto pB = AllocateOk(*a, "B", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pB.has_value());

  a->ClearLocal();
  a->ClearLocal();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  auto cap_after_reject = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_reject.available_bytes, cap_after_reject.total_bytes);

  a->ClearFullSyncAcked();
  auto pC = AllocateOk(*a, "C", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pC.has_value());
  EXPECT_TRUE(a->Commit(pC->slot_id, "C", committed_bytes));
}

// Leased owned key: logically gone at Clear, pages freed by reaper after
// the lease expires.
TEST(PeerDramAllocator, ClearLocalDefersLeasedOwnedPages) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{200});

  auto p = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "A", committed_bytes));
  a->DrainPendingEvents();

  const auto cap_committed = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_committed.available_bytes, cap_committed.total_bytes - kPageSize);

  ASSERT_TRUE(a->Resolve("A").found);  // lease.

  a->ClearLocal();

  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());

  auto cap_after_clear = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_clear.available_bytes, cap_committed.total_bytes - kPageSize);

  // Pre-TTL sweep: no-op.
  a->RunReaperOnceForTest();
  auto cap_no_op_sweep = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_no_op_sweep.available_bytes, cap_committed.total_bytes - kPageSize);

  // Past TTL: pages return to bitmap.
  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  a->RunReaperOnceForTest();
  auto cap_swept = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_swept.available_bytes, cap_swept.total_bytes);
}

// Leased owned A defers; pending B rejects via generation.
TEST(PeerDramAllocator, ClearLocalMixedPendingAndLeased) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{200});

  auto pA = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(pA.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pA->slot_id, "A", committed_bytes));
  ASSERT_TRUE(a->Resolve("A").found);  // lease.

  auto pB = AllocateOk(*a, "B", kPageSize * 2, TierType::DRAM);
  ASSERT_TRUE(pB.has_value());
  a->DrainPendingEvents();

  const auto total = a->TierCapacitiesSnapshot()[TierType::DRAM].total_bytes;

  a->ClearLocal();

  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());

  // A deferred + B pending: 3 pages occupied.
  auto cap_after_clear = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_clear.available_bytes, total - 3 * kPageSize);

  // Commit(B) fails on generation mismatch, releases B.
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  auto cap_after_reject = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_after_reject.available_bytes, total - kPageSize);  // only A.

  // Past lease + sweep: A released.
  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  a->RunReaperOnceForTest();
  auto cap_final = a->TierCapacitiesSnapshot()[TierType::DRAM];
  EXPECT_EQ(cap_final.available_bytes, total);
}

// Sweeps are no-ops while the deferred lease is still active.
TEST(PeerDramAllocator, ClearLocalSweepRespectsTtl) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{10000});

  auto p = AllocateOk(*a, "A", kPageSize, TierType::DRAM);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "A", committed_bytes));
  ASSERT_TRUE(a->Resolve("A").found);

  const auto cap_committed = a->TierCapacitiesSnapshot()[TierType::DRAM];

  a->ClearLocal();

  // Lease still live: every sweep is a no-op.
  for (int i = 0; i < 3; ++i) {
    a->RunReaperOnceForTest();
    auto cap = a->TierCapacitiesSnapshot()[TierType::DRAM];
    EXPECT_EQ(cap.available_bytes, cap_committed.available_bytes) << "sweep i=" << i;
  }
}

// ---- OwnedKeyCountByTier ----------------------------------------------------

TEST(PeerDramAllocator, OwnedKeyCountByTierTracksCommitsAndEvicts) {
  auto a = MakeAllocator();

  auto counts0 = a->OwnedKeyCountByTier();
  EXPECT_EQ(counts0[TierType::DRAM], 0u);
  EXPECT_EQ(counts0[TierType::HBM], 0u);
  EXPECT_EQ(counts0[TierType::SSD], 0u);

  for (int i = 0; i < 3; ++i) {
    const std::string k = "key-dram-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value()) << "i=" << i;
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, committed_bytes));
  }
  auto counts1 = a->OwnedKeyCountByTier();
  EXPECT_EQ(counts1[TierType::DRAM], 3u);
  EXPECT_EQ(counts1[TierType::HBM], 0u);
  EXPECT_EQ(counts1[TierType::SSD], 0u);

  a->Evict({"key-dram-0"});
  auto counts2 = a->OwnedKeyCountByTier();
  EXPECT_EQ(counts2[TierType::DRAM], 2u);
  EXPECT_EQ(counts2[TierType::HBM], 0u);
}

TEST(PeerDramAllocator, OwnedKeyCountByTierMultiTier) {
  PeerDramAllocator::TierConfig hbm_cfg;
  hbm_cfg.buffer_sizes = {kPageSize * 4};
  hbm_cfg.buffer_descs = {{0xD0, 0xD1}};
  auto a = std::make_unique<PeerDramAllocator>(kPageSize, MakeDramCfg(), hbm_cfg,
                                               std::chrono::milliseconds{5000});

  for (int i = 0; i < 2; ++i) {
    const std::string k = "d-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize, TierType::DRAM);
    ASSERT_TRUE(p.has_value());
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, committed_bytes));
  }
  {
    auto p = AllocateOk(*a, "h-0", kPageSize, TierType::HBM);
    ASSERT_TRUE(p.has_value());
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, "h-0", committed_bytes));
  }

  auto counts = a->OwnedKeyCountByTier();
  EXPECT_EQ(counts[TierType::DRAM], 2u);
  EXPECT_EQ(counts[TierType::HBM], 1u);
  EXPECT_EQ(counts[TierType::SSD], 0u);
}

}  // namespace mori::umbp
