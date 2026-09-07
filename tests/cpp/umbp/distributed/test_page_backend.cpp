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

#include "umbp/distributed/peer/backend/page_backend.h"

namespace mori::umbp {

namespace {

// 3 buffers x 4 pages of 1 KiB = 12 KiB total DRAM.
constexpr uint64_t kPageSize = 1024;

PageBackend::TierConfig MakeDramCfg() {
  PageBackend::TierConfig cfg;
  cfg.buffer_sizes = {kPageSize * 4, kPageSize * 4, kPageSize * 4};
  cfg.buffer_descs = {{0xA0, 0xA1}, {0xB0, 0xB1}, {0xC0, 0xC1}};
  return cfg;
}

std::unique_ptr<PageBackend> MakeAllocator(
    std::chrono::milliseconds pending_ttl = std::chrono::milliseconds{5000},
    std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500}) {
  return std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(), pending_ttl,
                                       read_lease_ttl);
}

// True iff `r` carries a live allocated slot (mirrors the pre-refactor
// AllocateResult::slot.has_value(), now folded into AllocateResult's own
// fields — see medium_backend.h).
bool HasSlot(const AllocateResult& r) { return r.outcome == AllocateOutcome::kSuccessAllocated; }

// Strip AllocateResult down to itself for tests that don't exercise the
// dedup outcome — nullopt unless the allocation actually landed a slot.
std::optional<AllocateResult> AllocateOk(PageBackend& a, const std::string& key, uint64_t size) {
  auto r = a.Allocate(key, size);
  if (!HasSlot(r)) return std::nullopt;
  return r;
}

}  // namespace

// ---- Allocate / Commit / Resolve happy path ---------------------------------

TEST(PageBackend, CommitMakesKeyResolvable) {
  auto a = MakeAllocator();
  auto pending = AllocateOk(*a, "key-1", kPageSize);
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(pending->size, kPageSize);
  EXPECT_EQ(pending->pages.size(), 1u);

  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pending->slot_id, "key-1", committed_bytes));
  EXPECT_EQ(committed_bytes, pending->size);
  auto r = a->Resolve("key-1");
  EXPECT_TRUE(r.found);
  EXPECT_EQ(r.size, kPageSize);
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

TEST(PageBackend, AllocateRejectsAlreadyOwnedKey) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(first.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(first->slot_id, "A", committed_bytes));
  a->DrainPendingEvents();

  const auto cap_after_commit = a->Capacity();

  auto second = a->Allocate("A", kPageSize);
  EXPECT_EQ(second.outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_FALSE(HasSlot(second));

  // No pages reserved -> capacity unchanged.
  const auto cap_after_dedup = a->Capacity();
  EXPECT_EQ(cap_after_dedup.available_bytes, cap_after_commit.available_bytes);
}

TEST(PageBackend, AllocateAllowsDifferentKey) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(first.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(first->slot_id, "A", committed_bytes));

  auto second = a->Allocate("B", kPageSize);
  EXPECT_EQ(second.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(HasSlot(second));
  EXPECT_TRUE(a->Commit(second.slot_id, "B", committed_bytes));
}

// Lax mode: pending_ not checked.  Two same-key Allocates before any
// Commit both succeed; race absorbed by Commit() (see
// DuplicateCommitIsIdempotentAndKeepsFirst).
TEST(PageBackend, AllocateDoesNotRejectOnPendingDuplicate) {
  auto a = MakeAllocator();

  auto first = a->Allocate("A", kPageSize);
  EXPECT_EQ(first.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(HasSlot(first));

  auto second = a->Allocate("A", kPageSize);
  EXPECT_EQ(second.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(HasSlot(second));
  ASSERT_NE(second.slot_id, first.slot_id);
}

// ---- Duplicate Commit idempotency -------------------------------------------
// Race-window safety net.  Both Allocates must happen BEFORE either
// Commit — once owned_["dup-key"] is set, the new owned_-check in
// Allocate would reject the second slot before it could reach Commit.

TEST(PageBackend, DuplicateCommitIsIdempotentAndKeepsFirst) {
  auto a = MakeAllocator();

  auto first = AllocateOk(*a, "dup-key", kPageSize);
  ASSERT_TRUE(first.has_value());
  auto second = AllocateOk(*a, "dup-key", kPageSize);
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
  const auto cap_after_first_commit = a->Capacity();
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
  const auto cap_after_dup = a->Capacity();
  EXPECT_EQ(cap_after_dup.available_bytes, cap_after_dup.total_bytes - kPageSize);
  EXPECT_EQ(cap_after_dup.total_bytes, cap_after_first_commit.total_bytes);

  // Second slot_id no longer pending; idempotent Abort returns true.
  EXPECT_TRUE(a->Abort(second->slot_id));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- ENOSPC -----------------------------------------------------------------

TEST(PageBackend, AllocateReturnsNulloptWhenFull) {
  auto a = MakeAllocator();
  std::vector<uint64_t> slot_ids;
  for (int i = 0; i < 12; ++i) {
    auto p = AllocateOk(*a, "k-" + std::to_string(i), kPageSize);
    ASSERT_TRUE(p.has_value()) << "i=" << i;
    slot_ids.push_back(p->slot_id);
  }
  EXPECT_FALSE(AllocateOk(*a, "k-overflow", kPageSize).has_value());

  EXPECT_TRUE(a->Abort(slot_ids.back()));
  EXPECT_TRUE(AllocateOk(*a, "k-recovered", kPageSize).has_value());
}

// ---- Pending TTL ------------------------------------------------------------

TEST(PageBackend, PendingSlotExpiresAfterTtl) {
  auto a = std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(),
                                         /*pending_ttl=*/std::chrono::milliseconds{1});
  auto pending = AllocateOk(*a, "key-late", kPageSize);
  ASSERT_TRUE(pending.has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pending->slot_id, "key-late", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  auto cap = a->Capacity();
  EXPECT_EQ(cap.available_bytes, cap.total_bytes);
}

// ---- Abort idempotency ------------------------------------------------------

TEST(PageBackend, AbortIsIdempotent) {
  auto a = MakeAllocator();
  auto pending = AllocateOk(*a, "k", kPageSize);
  ASSERT_TRUE(pending.has_value());
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(pending->slot_id));
  EXPECT_TRUE(a->Abort(999999));
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Evict idempotency + REMOVE event ---------------------------------------

TEST(PageBackend, EvictRemovesKeyAndQueuesEvent) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize);
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

TEST(PageBackend, EvictDefersWhenReadLeaseActive) {
  auto a = std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(),
                                         /*pending_ttl=*/std::chrono::milliseconds{5000},
                                         /*read_lease_ttl=*/std::chrono::milliseconds{200});
  auto p = AllocateOk(*a, "k", kPageSize);
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

TEST(PageBackend, SnapshotOwnedKeysReturnsEveryAdd) {
  auto a = MakeAllocator();
  for (int i = 0; i < 5; ++i) {
    const std::string k = "k-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize);
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

// ---- Buffer descs filtered to the page set ---------------------------------

TEST(PageBackend, BufferDescsForPagesDedupAndOrder) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize * 5);
  ASSERT_TRUE(p.has_value());
  ASSERT_EQ(p->pages.size(), 5u);

  auto descs = a->BufferDescsForPages(p->pages);
  ASSERT_EQ(descs.size(), 2u);
  EXPECT_EQ(descs[0].buffer_index, 0u);
  EXPECT_EQ(descs[1].buffer_index, 1u);
  EXPECT_EQ(descs[0].desc_bytes, std::vector<uint8_t>({0xA0, 0xA1}));
  EXPECT_EQ(descs[1].desc_bytes, std::vector<uint8_t>({0xB0, 0xB1}));
}

// ---- BatchAllocate / BatchCommit / BatchAbort -------------------------------

TEST(PageBackend, BatchAllocateEmptyInputReturnsEmpty) {
  auto a = MakeAllocator();
  EXPECT_TRUE(a->BatchAllocate({}).empty());
}

TEST(PageBackend, BatchAllocateMixedOutcomesAndDescs) {
  auto a = MakeAllocator();
  auto owned = AllocateOk(*a, "owned", kPageSize);
  ASSERT_TRUE(owned.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(owned->slot_id, "owned", committed_bytes));
  a->DrainPendingEvents();

  std::vector<AllocateRequest> requests;
  requests.push_back({"owned", kPageSize});
  requests.push_back({"ok", kPageSize * 5});
  requests.push_back({"zero", 0});
  requests.push_back({"too-big", kPageSize * 20});

  auto results = a->BatchAllocate(requests);
  ASSERT_EQ(results.size(), requests.size());

  EXPECT_EQ(results[0].outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_FALSE(HasSlot(results[0]));
  EXPECT_TRUE(results[0].descs.empty());

  EXPECT_EQ(results[1].outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(HasSlot(results[1]));
  EXPECT_EQ(results[1].size, kPageSize * 5);
  EXPECT_EQ(results[1].pages.size(), 5u);
  ASSERT_EQ(results[1].descs.size(), 2u);
  EXPECT_EQ(results[1].descs[0].buffer_index, 0u);
  EXPECT_EQ(results[1].descs[1].buffer_index, 1u);

  EXPECT_EQ(results[2].outcome, AllocateOutcome::kFailed);
  EXPECT_FALSE(HasSlot(results[2]));
  EXPECT_EQ(results[3].outcome, AllocateOutcome::kFailedNoSpace);
  EXPECT_FALSE(HasSlot(results[3]));
}

TEST(PageBackend, BatchCommitMixedSuccessAndFailure) {
  auto a = MakeAllocator();
  auto allocated = a->BatchAllocate({
      {"dup", kPageSize},
      {"dup", kPageSize * 2},
      {"unique", kPageSize},
  });
  ASSERT_EQ(allocated.size(), 3u);
  ASSERT_TRUE(HasSlot(allocated[0]));
  ASSERT_TRUE(HasSlot(allocated[1]));
  ASSERT_TRUE(HasSlot(allocated[2]));

  auto committed = a->BatchCommit({
      {allocated[0].slot_id, "dup"},
      {999999, "missing"},
      {allocated[1].slot_id, "dup"},
      {allocated[2].slot_id, "unique"},
  });
  ASSERT_EQ(committed.size(), 4u);
  EXPECT_TRUE(committed[0].success);
  EXPECT_EQ(committed[0].bytes_committed, kPageSize);
  EXPECT_FALSE(committed[1].success);
  EXPECT_EQ(committed[1].bytes_committed, 0u);
  EXPECT_TRUE(committed[2].success);
  EXPECT_EQ(committed[2].bytes_committed, kPageSize);
  EXPECT_TRUE(committed[3].success);
  EXPECT_EQ(committed[3].bytes_committed, kPageSize);

  auto dup = a->Resolve("dup");
  ASSERT_TRUE(dup.found);
  EXPECT_EQ(dup.pages, allocated[0].pages);
  EXPECT_EQ(dup.size, kPageSize);
  auto unique = a->Resolve("unique");
  ASSERT_TRUE(unique.found);
  EXPECT_EQ(unique.size, kPageSize);

  auto events = a->DrainPendingEvents();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[0].key, "dup");
  EXPECT_EQ(events[1].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[1].key, "unique");
}

TEST(PageBackend, BatchAbortMixedSlotsIsIdempotent) {
  auto a = MakeAllocator();
  auto allocated = a->BatchAllocate({
      {"drop", kPageSize},
      {"keep", kPageSize},
  });
  ASSERT_EQ(allocated.size(), 2u);
  ASSERT_TRUE(HasSlot(allocated[0]));
  ASSERT_TRUE(HasSlot(allocated[1]));

  auto aborted = a->BatchAbort({allocated[0].slot_id, 999999});
  ASSERT_EQ(aborted.size(), 2u);
  EXPECT_TRUE(aborted[0]);
  EXPECT_TRUE(aborted[1]);

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(allocated[0].slot_id, "drop", committed_bytes));
  EXPECT_TRUE(a->Commit(allocated[1].slot_id, "keep", committed_bytes));
  EXPECT_EQ(committed_bytes, kPageSize);
  EXPECT_TRUE(a->Resolve("keep").found);
}

// ---- BatchResolve ----------------------------------------------------------

TEST(PageBackend, BatchResolveEmptyInputReturnsEmpty) {
  auto a = MakeAllocator();
  EXPECT_TRUE(a->BatchResolve({}, /*include_descs=*/true).empty());
}

TEST(PageBackend, BatchResolveMixedHitsAndMisses) {
  auto a = MakeAllocator();
  // 5 pages over 4-pages-per-buffer config -> exercises dedup'd descs.
  auto p_hit = AllocateOk(*a, "hit", kPageSize * 5);
  ASSERT_TRUE(p_hit.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p_hit->slot_id, "hit", committed_bytes));
  auto p_small = AllocateOk(*a, "small", kPageSize);
  ASSERT_TRUE(p_small.has_value());
  ASSERT_TRUE(a->Commit(p_small->slot_id, "small", committed_bytes));
  a->DrainPendingEvents();

  auto ref_hit = a->Resolve("hit");
  auto ref_descs_hit = a->BufferDescsForPages(ref_hit.pages);
  auto ref_small = a->Resolve("small");
  auto ref_descs_small = a->BufferDescsForPages(ref_small.pages);
  ASSERT_TRUE(ref_hit.found);
  ASSERT_TRUE(ref_small.found);

  auto results = a->BatchResolve({"hit", "ghost-a", "small", "ghost-b"}, /*include_descs=*/true);
  ASSERT_EQ(results.size(), 4u);

  EXPECT_TRUE(results[0].found);
  EXPECT_EQ(results[0].pages, ref_hit.pages);
  EXPECT_EQ(results[0].size, ref_hit.size);
  ASSERT_EQ(results[0].descs.size(), ref_descs_hit.size());
  for (size_t i = 0; i < ref_descs_hit.size(); ++i) {
    EXPECT_EQ(results[0].descs[i].buffer_index, ref_descs_hit[i].buffer_index);
    EXPECT_EQ(results[0].descs[i].desc_bytes, ref_descs_hit[i].desc_bytes);
  }

  EXPECT_FALSE(results[1].found);
  EXPECT_TRUE(results[1].pages.empty());
  EXPECT_EQ(results[1].size, 0u);
  EXPECT_TRUE(results[1].descs.empty());

  EXPECT_TRUE(results[2].found);
  EXPECT_EQ(results[2].pages, ref_small.pages);
  EXPECT_EQ(results[2].size, ref_small.size);
  ASSERT_EQ(results[2].descs.size(), ref_descs_small.size());
  for (size_t i = 0; i < ref_descs_small.size(); ++i) {
    EXPECT_EQ(results[2].descs[i].buffer_index, ref_descs_small[i].buffer_index);
    EXPECT_EQ(results[2].descs[i].desc_bytes, ref_descs_small[i].desc_bytes);
  }

  EXPECT_FALSE(results[3].found);
}

TEST(PageBackend, BatchResolveExtendsLeaseForHitsOnly) {
  auto a = std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(),
                                         /*pending_ttl=*/std::chrono::milliseconds{5000},
                                         /*read_lease_ttl=*/std::chrono::milliseconds{500});
  auto p_x = AllocateOk(*a, "x", kPageSize);
  ASSERT_TRUE(p_x.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p_x->slot_id, "x", committed_bytes));
  auto p_y = AllocateOk(*a, "y", kPageSize);
  ASSERT_TRUE(p_y.has_value());
  ASSERT_TRUE(a->Commit(p_y->slot_id, "y", committed_bytes));
  a->DrainPendingEvents();

  auto results = a->BatchResolve({"x", "missing", "y"}, /*include_descs=*/true);
  ASSERT_EQ(results.size(), 3u);
  ASSERT_TRUE(results[0].found);
  ASSERT_FALSE(results[1].found);
  ASSERT_TRUE(results[2].found);

  auto evict = a->Evict({"x", "y"});
  ASSERT_EQ(evict.size(), 2u);
  EXPECT_EQ(evict[0].bytes_freed, 0u);
  EXPECT_EQ(evict[1].bytes_freed, 0u);
  EXPECT_TRUE(a->Resolve("x").found);
  EXPECT_TRUE(a->Resolve("y").found);
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  // Miss must not poison read_lease_until_: a subsequent
  // Allocate+Commit+Evict on the same key must free as if never touched.
  auto p_miss = AllocateOk(*a, "missing", kPageSize);
  ASSERT_TRUE(p_miss.has_value());
  ASSERT_TRUE(a->Commit(p_miss->slot_id, "missing", committed_bytes));
  a->DrainPendingEvents();
  auto evict_missing = a->Evict({"missing"});
  ASSERT_EQ(evict_missing.size(), 1u);
  EXPECT_EQ(evict_missing[0].bytes_freed, kPageSize);
}

TEST(PageBackend, BatchResolveLeaseExpiresLikeSingleKeyResolve) {
  auto a = std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(),
                                         /*pending_ttl=*/std::chrono::milliseconds{5000},
                                         /*read_lease_ttl=*/std::chrono::milliseconds{50});
  auto p = AllocateOk(*a, "k", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  a->DrainPendingEvents();

  auto results = a->BatchResolve({"k"}, /*include_descs=*/true);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].found);

  EXPECT_EQ(a->Evict({"k"})[0].bytes_freed, 0u);

  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  auto evicted = a->Evict({"k"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].bytes_freed, kPageSize);
}

// ---- Capacity snapshot ----------------------------------------------------

TEST(PageBackend, CapacityReflectsAllocations) {
  auto a = MakeAllocator();
  auto cap0 = a->Capacity();
  const uint64_t total = cap0.total_bytes;
  EXPECT_EQ(cap0.available_bytes, total);

  auto p = AllocateOk(*a, "k", kPageSize * 3);
  ASSERT_TRUE(p.has_value());
  auto cap1 = a->Capacity();
  EXPECT_EQ(cap1.available_bytes, total - 3 * kPageSize);

  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  EXPECT_EQ(committed_bytes, p->size);
  auto cap2 = a->Capacity();
  EXPECT_EQ(cap2.available_bytes, total - 3 * kPageSize);

  ASSERT_EQ(a->Evict({"k"})[0].bytes_freed, 3 * kPageSize);
  auto cap3 = a->Capacity();
  EXPECT_EQ(cap3.available_bytes, total);
}

// ---- Commit after reap ------------------------------------------------------

TEST(PageBackend, CommitAfterReapReturnsFalse) {
  auto a = std::make_unique<PageBackend>(TierType::DRAM, kPageSize, MakeDramCfg(),
                                         std::chrono::milliseconds{1});
  auto p = AllocateOk(*a, "doomed", kPageSize);
  ASSERT_TRUE(p.has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  a->RunReaperOnceForTest();
  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(p->slot_id, "doomed", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
}

// ---- Distributed Clear ------------------------------------------------------

TEST(PageBackend, ClearLocalReleasesOwnedAndCancelsPending) {
  auto a = MakeAllocator();

  auto pA = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(pA.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pA->slot_id, "A", committed_bytes));

  auto pB = AllocateOk(*a, "B", kPageSize * 2);
  ASSERT_TRUE(pB.has_value());
  a->DrainPendingEvents();  // discard the A ADD

  const auto cap_before = a->Capacity();
  ASSERT_LT(cap_before.available_bytes, cap_before.total_bytes);

  a->ClearLocal();

  EXPECT_TRUE(a->IsClearFullSyncPending());
  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  // Owned pages (A) returned immediately; pending pages (B) still held.
  auto cap_after_clear = a->Capacity();
  EXPECT_EQ(cap_after_clear.available_bytes, cap_before.total_bytes - 2 * kPageSize);

  // Committing the cancelled pending fails AND releases its pages
  // without emitting an ADD.
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
  EXPECT_FALSE(a->Resolve("B").found);

  auto cap_final = a->Capacity();
  EXPECT_EQ(cap_final.available_bytes, cap_final.total_bytes);
}

TEST(PageBackend, ClearLocalGatesAllocateUntilAcked) {
  auto a = MakeAllocator();

  a->ClearLocal();
  EXPECT_FALSE(AllocateOk(*a, "blocked", kPageSize).has_value());

  a->ClearFullSyncAcked();
  EXPECT_FALSE(a->IsClearFullSyncPending());
  EXPECT_TRUE(AllocateOk(*a, "ok-after-ack", kPageSize).has_value());
}

TEST(PageBackend, ClearLocalDropsQueuedAdds) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "k", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "k", committed_bytes));
  // ADD is sitting in the outbox, not yet drained.

  a->ClearLocal();

  EXPECT_TRUE(a->DrainPendingEvents().empty());
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());
}

TEST(PageBackend, AbortReleasesCancelledPending) {
  auto a = MakeAllocator();
  auto p = AllocateOk(*a, "p1", kPageSize);
  ASSERT_TRUE(p.has_value());

  a->ClearLocal();
  // Abort on a cancelled pending is idempotent and frees the pages.
  EXPECT_TRUE(a->Abort(p->slot_id));
  a->ClearFullSyncAcked();

  auto cap = a->Capacity();
  EXPECT_EQ(cap.available_bytes, cap.total_bytes);
}

// Pre-clear pending Commit fails; post-ack new Allocate+Commit succeeds.
TEST(PageBackend, PendingGenerationRejectsPreClearCommit) {
  auto a = MakeAllocator();

  auto pB = AllocateOk(*a, "B", kPageSize * 2);
  ASSERT_TRUE(pB.has_value());
  const auto cap_before = a->Capacity();
  EXPECT_EQ(cap_before.available_bytes, cap_before.total_bytes - 2 * kPageSize);

  a->ClearLocal();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  EXPECT_EQ(committed_bytes, 0u);
  EXPECT_TRUE(a->DrainPendingEvents().empty());
  auto cap_after_reject = a->Capacity();
  EXPECT_EQ(cap_after_reject.available_bytes, cap_after_reject.total_bytes);

  a->ClearFullSyncAcked();

  auto pC = AllocateOk(*a, "C", kPageSize);
  ASSERT_TRUE(pC.has_value());
  ASSERT_TRUE(a->Commit(pC->slot_id, "C", committed_bytes));
  EXPECT_EQ(committed_bytes, kPageSize);
  EXPECT_TRUE(a->Resolve("C").found);
}

// Repeated Clears still reject the original pre-clear pending Commit.
TEST(PageBackend, PendingGenerationSurvivesDoubleClear) {
  auto a = MakeAllocator();

  auto pB = AllocateOk(*a, "B", kPageSize);
  ASSERT_TRUE(pB.has_value());

  a->ClearLocal();
  a->ClearLocal();

  uint64_t committed_bytes = 0;
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  auto cap_after_reject = a->Capacity();
  EXPECT_EQ(cap_after_reject.available_bytes, cap_after_reject.total_bytes);

  a->ClearFullSyncAcked();
  auto pC = AllocateOk(*a, "C", kPageSize);
  ASSERT_TRUE(pC.has_value());
  EXPECT_TRUE(a->Commit(pC->slot_id, "C", committed_bytes));
}

// Leased owned key: logically gone at Clear, pages freed by reaper after
// the lease expires.
TEST(PageBackend, ClearLocalDefersLeasedOwnedPages) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{200});

  auto p = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "A", committed_bytes));
  a->DrainPendingEvents();

  const auto cap_committed = a->Capacity();
  EXPECT_EQ(cap_committed.available_bytes, cap_committed.total_bytes - kPageSize);

  ASSERT_TRUE(a->Resolve("A").found);  // lease.

  a->ClearLocal();

  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());

  auto cap_after_clear = a->Capacity();
  EXPECT_EQ(cap_after_clear.available_bytes, cap_committed.total_bytes - kPageSize);

  // Pre-TTL sweep: no-op.
  a->RunReaperOnceForTest();
  auto cap_no_op_sweep = a->Capacity();
  EXPECT_EQ(cap_no_op_sweep.available_bytes, cap_committed.total_bytes - kPageSize);

  // Past TTL: pages return to bitmap.
  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  a->RunReaperOnceForTest();
  auto cap_swept = a->Capacity();
  EXPECT_EQ(cap_swept.available_bytes, cap_swept.total_bytes);
}

// Leased owned A defers; pending B rejects via generation.
TEST(PageBackend, ClearLocalMixedPendingAndLeased) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{200});

  auto pA = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(pA.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(pA->slot_id, "A", committed_bytes));
  ASSERT_TRUE(a->Resolve("A").found);  // lease.

  auto pB = AllocateOk(*a, "B", kPageSize * 2);
  ASSERT_TRUE(pB.has_value());
  a->DrainPendingEvents();

  const auto total = a->Capacity().total_bytes;

  a->ClearLocal();

  EXPECT_FALSE(a->Resolve("A").found);
  EXPECT_TRUE(a->SnapshotOwnedKeys().empty());

  // A deferred + B pending: 3 pages occupied.
  auto cap_after_clear = a->Capacity();
  EXPECT_EQ(cap_after_clear.available_bytes, total - 3 * kPageSize);

  // Commit(B) fails on generation mismatch, releases B.
  EXPECT_FALSE(a->Commit(pB->slot_id, "B", committed_bytes));
  auto cap_after_reject = a->Capacity();
  EXPECT_EQ(cap_after_reject.available_bytes, total - kPageSize);  // only A.

  // Past lease + sweep: A released.
  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  a->RunReaperOnceForTest();
  auto cap_final = a->Capacity();
  EXPECT_EQ(cap_final.available_bytes, total);
}

// Sweeps are no-ops while the deferred lease is still active.
TEST(PageBackend, ClearLocalSweepRespectsTtl) {
  auto a = MakeAllocator(/*pending_ttl=*/std::chrono::milliseconds{5000},
                         /*read_lease_ttl=*/std::chrono::milliseconds{10000});

  auto p = AllocateOk(*a, "A", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t committed_bytes = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "A", committed_bytes));
  ASSERT_TRUE(a->Resolve("A").found);

  const auto cap_committed = a->Capacity();

  a->ClearLocal();

  // Lease still live: every sweep is a no-op.
  for (int i = 0; i < 3; ++i) {
    a->RunReaperOnceForTest();
    auto cap = a->Capacity();
    EXPECT_EQ(cap.available_bytes, cap_committed.available_bytes) << "sweep i=" << i;
  }
}

// ---- OwnedKeyCount -----------------------------------------------------------

TEST(PageBackend, OwnedKeyCountTracksCommitsAndEvicts) {
  auto a = MakeAllocator();

  EXPECT_EQ(a->OwnedKeyCount(), 0u);

  for (int i = 0; i < 3; ++i) {
    const std::string k = "key-dram-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize);
    ASSERT_TRUE(p.has_value()) << "i=" << i;
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, committed_bytes));
  }
  EXPECT_EQ(a->OwnedKeyCount(), 3u);

  a->Evict({"key-dram-0"});
  EXPECT_EQ(a->OwnedKeyCount(), 2u);
}

// A DRAM instance and an HBM instance are fully independent objects — one
// instance == one medium (backend-agnostic refactor Phase 2), so there is no
// shared multi-tier state to cross-check beyond each Tier() staying fixed.
TEST(PageBackend, SeparateInstancesPerMediumAreIndependent) {
  auto dram = MakeAllocator();
  PageBackend::TierConfig hbm_cfg;
  hbm_cfg.buffer_sizes = {kPageSize * 4};
  hbm_cfg.buffer_descs = {{0xD0, 0xD1}};
  auto hbm = std::make_unique<PageBackend>(TierType::HBM, kPageSize, hbm_cfg,
                                           std::chrono::milliseconds{5000});

  EXPECT_EQ(dram->Tier(), TierType::DRAM);
  EXPECT_EQ(hbm->Tier(), TierType::HBM);

  for (int i = 0; i < 2; ++i) {
    const std::string k = "d-" + std::to_string(i);
    auto p = AllocateOk(*dram, k, kPageSize);
    ASSERT_TRUE(p.has_value());
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(dram->Commit(p->slot_id, k, committed_bytes));
  }
  {
    auto p = AllocateOk(*hbm, "h-0", kPageSize);
    ASSERT_TRUE(p.has_value());
    uint64_t committed_bytes = 0;
    ASSERT_TRUE(hbm->Commit(p->slot_id, "h-0", committed_bytes));
  }

  EXPECT_EQ(dram->OwnedKeyCount(), 2u);
  EXPECT_EQ(hbm->OwnedKeyCount(), 1u);
  // Neither instance sees the other's keys.
  EXPECT_FALSE(dram->Resolve("h-0").found);
  EXPECT_FALSE(hbm->Resolve("d-0").found);
}

// ---- Auto-flush: cb fires exactly when the outbox first reaches threshold ----
TEST(PageBackendAutoFlush, FiresAtThreshold) {
  auto a = MakeAllocator();
  int fires = 0;
  a->SetAutoFlushHook(3, [&] { ++fires; });

  for (int i = 0; i < 2; ++i) {
    const std::string k = "af-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize);
    ASSERT_TRUE(p.has_value());
    uint64_t b = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, b));
  }
  EXPECT_EQ(fires, 0);  // below threshold -> no fire

  auto p = AllocateOk(*a, "af-2", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t b = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "af-2", b));
  EXPECT_EQ(fires, 1);  // reached threshold -> exactly one fire
}

// ---- No hook registered -> auto-flush disabled by default (never fires) ----
TEST(PageBackendAutoFlush, NoHookNoFire) {
  auto a = MakeAllocator();
  for (int i = 0; i < 10; ++i) {
    const std::string k = "nh-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize);
    ASSERT_TRUE(p.has_value());
    uint64_t b = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, b));
  }
  SUCCEED();  // default threshold = SIZE_MAX, empty cb: no crash, no fire
}

// ---- Full-sync snapshot returns all owned AND atomically clears the outbox ----
TEST(PageBackendAutoFlush, FullSyncSnapshotClearsOutbox) {
  auto a = MakeAllocator();
  for (int i = 0; i < 3; ++i) {
    const std::string k = "fs-" + std::to_string(i);
    auto p = AllocateOk(*a, k, kPageSize);
    ASSERT_TRUE(p.has_value());
    uint64_t b = 0;
    ASSERT_TRUE(a->Commit(p->slot_id, k, b));
  }
  auto snap = a->SnapshotOwnedKeysForFullSync();
  EXPECT_EQ(snap.size(), 3u);
  for (const auto& ev : snap) EXPECT_EQ(ev.kind, KvEvent::Kind::ADD);

  // The snapshot is authoritative: the queued ADDs must NOT be re-shipped.
  EXPECT_TRUE(a->DrainPendingEvents().empty());

  // A commit AFTER the snapshot is a fresh delta (only genuinely new events).
  auto p = AllocateOk(*a, "fs-new", kPageSize);
  ASSERT_TRUE(p.has_value());
  uint64_t b = 0;
  ASSERT_TRUE(a->Commit(p->slot_id, "fs-new", b));
  auto delta = a->DrainPendingEvents();
  ASSERT_EQ(delta.size(), 1u);
  EXPECT_EQ(delta[0].key, "fs-new");
  EXPECT_EQ(delta[0].kind, KvEvent::Kind::ADD);
}

}  // namespace mori::umbp
