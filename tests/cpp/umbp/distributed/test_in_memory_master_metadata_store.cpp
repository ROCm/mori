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

// Phase 2 behavioral suite for InMemoryMasterMetadataStore (§6a). Written
// against IMasterMetadataStore& so the same cases validate the Redis backend
// later. Tests use injected system_clock times (no real-time sleeps) so they
// are deterministic in CI.
//
// State that the interface does not expose directly — lease_expiry and
// last_accessed_at on block entries — is observed through
// EnumerateEvictionCandidates: a leased entry is filtered out, and LRU
// ordering (EvictionOrder::kLeastRecentlyAccessed) reflects last_accessed_at.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/in_memory_master_metadata_store.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::system_clock;

// Fixed, NTP-plausible base instant so offsets read cleanly.
const Clock::time_point kT0 = Clock::time_point(std::chrono::hours(24 * 365 * 50));

std::map<TierType, TierCapacity> Caps(uint64_t total = 1000, uint64_t available = 1000) {
  return {{TierType::HBM, TierCapacity{total, available}}};
}

ClientRegistration MakeReg(const std::string& node_id) {
  ClientRegistration reg;
  reg.node_id = node_id;
  reg.node_address = "addr:" + node_id;
  reg.peer_address = "peer:" + node_id;
  reg.tier_capacities = Caps();
  reg.tags = {"role=test"};
  return reg;
}

KvEvent Add(const std::string& key, TierType tier, uint64_t size) {
  return KvEvent{KvEvent::Kind::ADD, key, tier, size};
}
KvEvent Remove(const std::string& key, TierType tier) {
  return KvEvent{KvEvent::Kind::REMOVE, key, tier, 0};
}

// Register `node` ALIVE at `now`.
void RegisterAlive(IMasterMetadataStore& store, const std::string& node,
                   Clock::time_point now = kT0) {
  ASSERT_TRUE(store.RegisterClient(MakeReg(node), now, 30s));
}

// Apply a delta heartbeat carrying `events` at sequence `seq`.
HeartbeatResult Beat(IMasterMetadataStore& store, const std::string& node, uint64_t seq,
                     std::vector<KvEvent> events, Clock::time_point now) {
  return store.ApplyHeartbeat(node, seq, now, Caps(), events, /*is_full_sync=*/false);
}

// ---------------------------------------------------------------------------
// RegisterClient
// ---------------------------------------------------------------------------

TEST(InMemoryStore, RegisterNewClient) {
  InMemoryMasterMetadataStore store;
  EXPECT_TRUE(store.RegisterClient(MakeReg("n1"), kT0, 30s));
  EXPECT_TRUE(store.IsClientAlive("n1"));
  EXPECT_EQ(store.AliveClientCount(), 1u);

  auto rec = store.GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->status, ClientStatus::ALIVE);
  EXPECT_EQ(rec->last_applied_seq, 0u);
  EXPECT_EQ(rec->peer_address, "peer:n1");
  EXPECT_EQ(rec->last_heartbeat, kT0);
  EXPECT_EQ(rec->registered_at, kT0);
}

TEST(InMemoryStore, RejectReRegisterNonStaleAlive) {
  InMemoryMasterMetadataStore store;
  ASSERT_TRUE(store.RegisterClient(MakeReg("n1"), kT0, 30s));
  // Still well within stale_after window.
  EXPECT_FALSE(store.RegisterClient(MakeReg("n1"), kT0 + 5s, 30s));
}

TEST(InMemoryStore, AcceptReRegisterStaleAlive) {
  InMemoryMasterMetadataStore store;
  ASSERT_TRUE(store.RegisterClient(MakeReg("n1"), kT0, 30s));
  // last_heartbeat is kT0; now - last_heartbeat > stale_after → re-register OK
  // even though the reaper has not flipped the status yet (hazard #2).
  EXPECT_TRUE(store.RegisterClient(MakeReg("n1"), kT0 + 31s, 30s));
  EXPECT_TRUE(store.IsClientAlive("n1"));
}

TEST(InMemoryStore, AcceptReRegisterExpired) {
  InMemoryMasterMetadataStore store;
  ASSERT_TRUE(store.RegisterClient(MakeReg("n1"), kT0, 30s));
  ASSERT_EQ(store.ExpireStaleClients(kT0 + 1s).size(), 1u);
  EXPECT_FALSE(store.IsClientAlive("n1"));
  // Re-register an EXPIRED record at the same instant: accepted, back to ALIVE.
  EXPECT_TRUE(store.RegisterClient(MakeReg("n1"), kT0 + 2s, 30s));
  EXPECT_TRUE(store.IsClientAlive("n1"));
}

// ---------------------------------------------------------------------------
// UnregisterClient — cascade to block locations AND external KV
// ---------------------------------------------------------------------------

TEST(InMemoryStore, UnregisterClientCascades) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1", "h2"}, TierType::HBM));

  ASSERT_FALSE(store.LookupBlock("k1").empty());
  ASSERT_EQ(store.GetExternalKvCount("n1"), 2u);

  store.UnregisterClient("n1");

  EXPECT_FALSE(store.GetClient("n1").has_value());
  EXPECT_TRUE(store.LookupBlock("k1").empty());
  EXPECT_EQ(store.GetExternalKvCount("n1"), 0u);
  EXPECT_TRUE(store.MatchExternalKv({"h1", "h2"}, false, kT0).empty());
}

TEST(InMemoryStore, UnregisterUnknownIsNoOp) {
  InMemoryMasterMetadataStore store;
  store.UnregisterClient("ghost");  // must not crash
  EXPECT_EQ(store.AliveClientCount(), 0u);
}

// ---------------------------------------------------------------------------
// ApplyHeartbeat
// ---------------------------------------------------------------------------

TEST(InMemoryStore, HeartbeatUnknownNode) {
  InMemoryMasterMetadataStore store;
  auto r = Beat(store, "ghost", 1, {}, kT0);
  EXPECT_EQ(r.status, HeartbeatResult::UNKNOWN);
}

TEST(InMemoryStore, HeartbeatCasSequence) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  EXPECT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  EXPECT_EQ(Beat(store, "n1", 2, {Add("k2", TierType::HBM, 20)}, kT0).status,
            HeartbeatResult::APPLIED);
  // Out-of-order seq → SEQ_GAP, acked echoes last applied (2).
  auto gap = Beat(store, "n1", 4, {Add("k3", TierType::HBM, 30)}, kT0);
  EXPECT_EQ(gap.status, HeartbeatResult::SEQ_GAP);
  EXPECT_EQ(gap.acked_seq, 2u);
  // k3 must not have been applied.
  EXPECT_TRUE(store.LookupBlock("k3").empty());
}

TEST(InMemoryStore, SeqGapKeepsLivenessNotCapsOrSeq) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {}, kT0).status, HeartbeatResult::APPLIED);

  // A gap heartbeat at a later time with different caps.
  std::map<TierType, TierCapacity> new_caps = {{TierType::HBM, TierCapacity{9999, 9999}}};
  auto gap = store.ApplyHeartbeat("n1", 5, kT0 + 10s, new_caps, {}, /*is_full_sync=*/false);
  ASSERT_EQ(gap.status, HeartbeatResult::SEQ_GAP);

  auto rec = store.GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->status, ClientStatus::ALIVE);                           // kept alive
  EXPECT_EQ(rec->last_heartbeat, kT0 + 10s);                             // last_heartbeat bumped
  EXPECT_EQ(rec->last_applied_seq, 1u);                                  // seq NOT advanced
  EXPECT_EQ(rec->tier_capacities.at(TierType::HBM).total_bytes, 1000u);  // caps NOT replaced
}

TEST(InMemoryStore, HeartbeatDeltaAddRemove) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_EQ(store.LookupBlock("k1").size(), 1u);

  ASSERT_EQ(Beat(store, "n1", 2, {Remove("k1", TierType::HBM)}, kT0).status,
            HeartbeatResult::APPLIED);
  EXPECT_TRUE(store.LookupBlock("k1").empty());
}

TEST(InMemoryStore, HeartbeatFullSyncReplaces) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10), Add("k2", TierType::HBM, 20)}, kT0)
                .status,
            HeartbeatResult::APPLIED);

  // full_sync wipes prior locations and installs only the ADDs carried here.
  auto r = store.ApplyHeartbeat("n1", 7, kT0, Caps(), {Add("k3", TierType::HBM, 30)},
                                /*is_full_sync=*/true);
  EXPECT_EQ(r.status, HeartbeatResult::APPLIED);
  EXPECT_EQ(r.acked_seq, 7u);

  EXPECT_TRUE(store.LookupBlock("k1").empty());
  EXPECT_TRUE(store.LookupBlock("k2").empty());
  EXPECT_EQ(store.LookupBlock("k3").size(), 1u);

  auto rec = store.GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->last_applied_seq, 7u);  // full_sync re-baselines the seq
}

// ---------------------------------------------------------------------------
// ExpireStaleClients — flip to EXPIRED, keep row, cascade, idempotent
// ---------------------------------------------------------------------------

TEST(InMemoryStore, ExpireStaleFlipsKeepsRowAndCascades) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1", kT0);
  RegisterAlive(store, "n2", kT0 + 20s);  // fresher
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::HBM));

  // Cutoff after n1's heartbeat but before n2's.
  auto dead = store.ExpireStaleClients(kT0 + 10s);
  ASSERT_EQ(dead.size(), 1u);
  EXPECT_EQ(dead[0], "n1");

  // Row KEPT but EXPIRED (hazard #3).
  auto rec = store.GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->status, ClientStatus::EXPIRED);
  EXPECT_FALSE(store.IsClientAlive("n1"));

  // Cascade dropped its blocks and external KV.
  EXPECT_TRUE(store.LookupBlock("k1").empty());
  EXPECT_EQ(store.GetExternalKvCount("n1"), 0u);

  // n2 untouched.
  EXPECT_TRUE(store.IsClientAlive("n2"));
}

TEST(InMemoryStore, ExpireStaleIsIdempotent) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1", kT0);
  ASSERT_EQ(store.ExpireStaleClients(kT0 + 10s).size(), 1u);
  // Re-tick: already EXPIRED, nothing new to report.
  EXPECT_TRUE(store.ExpireStaleClients(kT0 + 10s).empty());
}

TEST(InMemoryStore, ExpiredRowExcludedFromAliveAccounting) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1", kT0);
  RegisterAlive(store, "n2", kT0);
  ASSERT_EQ(store.AliveClientCount(), 2u);
  ASSERT_EQ(store.ExpireStaleClients(kT0 + 10s).size(), 2u);

  EXPECT_EQ(store.AliveClientCount(), 0u);  // not 2, even though rows remain
  EXPECT_TRUE(store.ListAliveClients().empty());
  EXPECT_TRUE(store.GetClient("n1").has_value());  // row still present
}

// ---------------------------------------------------------------------------
// Block reads — lease/access observed via EnumerateEvictionCandidates
// ---------------------------------------------------------------------------

// Helper: single (node, tier) bucket to enumerate candidates from. The store's
// candidate enumeration is policy-neutral: it takes the buckets to scan, an
// ordering hint, and a per-bucket count cap (0 = no cap). The byte-budget /
// victim decision lives in MasterEvictStrategy, not the store.
std::vector<NodeTierKey> Buckets(const std::string& node, TierType tier) {
  return {NodeTierKey{node, tier}};
}

TEST(InMemoryStore, LookupBlockHasNoLeaseOrAccessSideEffects) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);

  // Plain read twice.
  EXPECT_EQ(store.LookupBlock("k1").size(), 1u);
  EXPECT_EQ(store.LookupBlock("k1").size(), 1u);

  // Not leased → still an eviction candidate at kT0.
  auto cands = store.EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                 EvictionOrder::kLeastRecentlyAccessed, 0, kT0);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands.begin()->second.size(), 1u);
}

TEST(InMemoryStore, LookupBlockForRouteGetGrantsLeaseAndAccess) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);

  auto locs = store.LookupBlockForRouteGet("k1", {}, kT0, 60s);
  ASSERT_EQ(locs.size(), 1u);

  // Leased until kT0+60s → filtered out of eviction at kT0+10s.
  EXPECT_TRUE(store
                  .EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                               EvictionOrder::kLeastRecentlyAccessed, 0, kT0 + 10s)
                  .empty());
  // After lease expiry it is a candidate again.
  EXPECT_FALSE(store
                   .EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                EvictionOrder::kLeastRecentlyAccessed, 0, kT0 + 61s)
                   .empty());
}

TEST(InMemoryStore, RouteGetExcludeNodesNoLeaseWhenFullyExcluded) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);

  std::unordered_set<std::string> exclude = {"n1"};
  auto locs = store.LookupBlockForRouteGet("k1", exclude, kT0, 60s);
  EXPECT_TRUE(locs.empty());  // every location excluded

  // No lease granted (hazard #4) → still an eviction candidate immediately.
  EXPECT_FALSE(store
                   .EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                EvictionOrder::kLeastRecentlyAccessed, 0, kT0)
                   .empty());
}

TEST(InMemoryStore, BatchLookupForRouteGetParallelToKeys) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10), Add("k3", TierType::HBM, 30)}, kT0)
                .status,
            HeartbeatResult::APPLIED);

  auto out = store.BatchLookupBlockForRouteGet({"k1", "missing", "k3"}, {}, kT0, 60s);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].size(), 1u);
  EXPECT_TRUE(out[1].empty());  // missing key
  EXPECT_EQ(out[2].size(), 1u);
}

TEST(InMemoryStore, BatchExistsBlockNoSideEffects) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);

  auto exists = store.BatchExistsBlock({"k1", "missing"});
  ASSERT_EQ(exists.size(), 2u);
  EXPECT_TRUE(exists[0]);
  EXPECT_FALSE(exists[1]);

  // No lease granted by an existence check.
  EXPECT_FALSE(store
                   .EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                EvictionOrder::kLeastRecentlyAccessed, 0, kT0)
                   .empty());
}

// ---------------------------------------------------------------------------
// EnumerateEvictionCandidates
// ---------------------------------------------------------------------------

TEST(InMemoryStore, EvictionLruOrderAndCap) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  // Three keys, each 100 bytes, accessed at increasing times so LRU order is
  // k_old < k_mid < k_new.
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k_old", TierType::HBM, 100)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_EQ(Beat(store, "n1", 2, {Add("k_mid", TierType::HBM, 100)}, kT0 + 1s).status,
            HeartbeatResult::APPLIED);
  ASSERT_EQ(Beat(store, "n1", 3, {Add("k_new", TierType::HBM, 100)}, kT0 + 2s).status,
            HeartbeatResult::APPLIED);

  // max_per_bucket=2 with LRU order → the two oldest, oldest first. (The store
  // caps by count; trimming to a byte budget is MasterEvictStrategy's job.)
  auto cands = store.EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                 EvictionOrder::kLeastRecentlyAccessed,
                                                 /*max_per_bucket=*/2, kT0 + 10s);
  ASSERT_EQ(cands.size(), 1u);
  auto& bucket = cands.at(NodeTierKey{"n1", TierType::HBM});
  ASSERT_EQ(bucket.size(), 2u);
  EXPECT_EQ(bucket[0].key, "k_old");  // oldest first
  EXPECT_EQ(bucket[1].key, "k_mid");
}

TEST(InMemoryStore, EvictionSkipsLeased) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 100)}, kT0).status,
            HeartbeatResult::APPLIED);
  // Lease k1 well past the enumeration time.
  store.LookupBlockForRouteGet("k1", {}, kT0, 1h);
  EXPECT_TRUE(store
                  .EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                               EvictionOrder::kLeastRecentlyAccessed, 0, kT0 + 1s)
                  .empty());
}

TEST(InMemoryStore, EvictionTieTimestampsAllSurvive) {
  // §2d correctness claim: many candidates sharing one identical last_accessed_at
  // (the common case, since a batch RouteGet stamps one `now` across all keys)
  // must all be enumerable — none dropped by tie collisions.
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  std::vector<KvEvent> adds;
  for (int i = 0; i < 50; ++i) {
    adds.push_back(Add("k" + std::to_string(i), TierType::HBM, 10));
  }
  // All keys created (and thus last_accessed) at the identical instant kT0.
  ASSERT_EQ(Beat(store, "n1", 1, adds, kT0).status, HeartbeatResult::APPLIED);

  // No cap → take everything; all 50 tied-timestamp candidates must appear.
  auto cands = store.EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                                 EvictionOrder::kLeastRecentlyAccessed,
                                                 /*max_per_bucket=*/0, kT0 + 10s);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands.at(NodeTierKey{"n1", TierType::HBM}).size(), 50u);
}

TEST(InMemoryStore, EvictionOnlyRequestedBuckets) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("kh", TierType::HBM, 10), Add("kd", TierType::DRAM, 10)}, kT0)
                .status,
            HeartbeatResult::APPLIED);
  // Only ask about the HBM bucket.
  auto cands = store.EnumerateEvictionCandidates(
      Buckets("n1", TierType::HBM), EvictionOrder::kLeastRecentlyAccessed, 0, kT0 + 1s);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands.begin()->first.tier, TierType::HBM);
}

// ---------------------------------------------------------------------------
// External KV
// ---------------------------------------------------------------------------

TEST(InMemoryStore, RegisterExternalKvAliveGate) {
  InMemoryMasterMetadataStore store;
  // Dead/unknown node → rejected, nothing written.
  EXPECT_FALSE(store.RegisterExternalKvIfAlive("ghost", {"h1"}, TierType::HBM));
  EXPECT_TRUE(store.MatchExternalKv({"h1"}, false, kT0).empty());

  RegisterAlive(store, "n1");
  EXPECT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::HBM));
  EXPECT_EQ(store.MatchExternalKv({"h1"}, false, kT0).size(), 1u);
}

TEST(InMemoryStore, UnregisterExternalKvAndByTier) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::HBM));
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::DRAM));

  // Remove only the HBM tier; DRAM remains.
  store.UnregisterExternalKv("n1", {"h1"}, TierType::HBM);
  auto m = store.MatchExternalKv({"h1"}, false, kT0);
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].hashes_by_tier.count(TierType::HBM), 0u);
  EXPECT_EQ(m[0].hashes_by_tier.count(TierType::DRAM), 1u);

  // Whole-tier wipe of DRAM → entry gone.
  store.UnregisterExternalKvByTier("n1", TierType::DRAM);
  EXPECT_TRUE(store.MatchExternalKv({"h1"}, false, kT0).empty());
}

TEST(InMemoryStore, MatchCountsHitsWhenRequested) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1", "h2"}, TierType::HBM));

  // count_as_hit=false: pure read, hit map untouched.
  store.MatchExternalKv({"h1", "h2"}, /*count_as_hit=*/false, kT0);
  EXPECT_TRUE(store.GetExternalKvHitCounts({"h1", "h2"}).empty());

  // count_as_hit=true: increments accumulate across calls.
  store.MatchExternalKv({"h1", "h2"}, /*count_as_hit=*/true, kT0);
  store.MatchExternalKv({"h1"}, /*count_as_hit=*/true, kT0 + 1s);

  auto counts = store.GetExternalKvHitCounts({"h1", "h2"});
  std::map<std::string, uint64_t> by_hash;
  for (const auto& e : counts) by_hash[e.hash] = e.hit_count_total;
  EXPECT_EQ(by_hash["h1"], 2u);
  EXPECT_EQ(by_hash["h2"], 1u);
}

TEST(InMemoryStore, MatchedHashCountAcrossTiers) {
  // Preserves the NodeMatch::MatchedHashCount coverage from
  // test_external_kv_block_index.cpp:57 — one hash mirrored across two tiers
  // counts once.
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::HBM));
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::DRAM));

  auto m = store.MatchExternalKv({"h1"}, false, kT0);
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].hashes_by_tier.size(), 2u);  // appears in two tier buckets
  EXPECT_EQ(m[0].MatchedHashCount(), 1u);     // but is one unique hash
}

TEST(InMemoryStore, GetExternalKvHitCountsDedupesAndSkipsMissing) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::HBM));
  store.MatchExternalKv({"h1"}, true, kT0);

  auto counts = store.GetExternalKvHitCounts({"missing", "h1", "h1"});
  ASSERT_EQ(counts.size(), 1u);
  EXPECT_EQ(counts[0].hash, "h1");
  EXPECT_EQ(counts[0].hit_count_total, 1u);
}

TEST(InMemoryStore, GarbageCollectHitsByLastSeen) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"old", "fresh"}, TierType::HBM));
  store.MatchExternalKv({"old"}, true, kT0);
  store.MatchExternalKv({"fresh"}, true, kT0 + 100s);

  // Drop entries last seen before kT0+50s → only "old" goes.
  EXPECT_EQ(store.GarbageCollectHits(kT0 + 50s), 1u);

  auto counts = store.GetExternalKvHitCounts({"old", "fresh"});
  ASSERT_EQ(counts.size(), 1u);
  EXPECT_EQ(counts[0].hash, "fresh");
}

TEST(InMemoryStore, UnregisterExternalKvByNodeWipesAllTiersOnly) {
  // Whole-node external-KV wipe (backs RevokeAllExternalKvBlocksForNode). Unlike
  // UnregisterClient, it must NOT touch the client record or block locations.
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  ASSERT_EQ(Beat(store, "n1", 1, {Add("k1", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1", "h2"}, TierType::HBM));
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1"}, TierType::DRAM));
  ASSERT_EQ(store.GetExternalKvCount("n1"), 2u);

  store.UnregisterExternalKvByNode("n1");

  // External KV gone across every tier.
  EXPECT_EQ(store.GetExternalKvCount("n1"), 0u);
  EXPECT_TRUE(store.MatchExternalKv({"h1", "h2"}, false, kT0).empty());

  // Client record and block locations untouched (distinguishes from UnregisterClient).
  EXPECT_TRUE(store.IsClientAlive("n1"));
  EXPECT_EQ(store.LookupBlock("k1").size(), 1u);
}

TEST(InMemoryStore, UnregisterExternalKvByNodeUnknownIsNoOp) {
  InMemoryMasterMetadataStore store;
  store.UnregisterExternalKvByNode("ghost");  // must not crash
  EXPECT_EQ(store.GetExternalKvCount("ghost"), 0u);
}

// ---------------------------------------------------------------------------
// Client reads — GetPeerAddress, GetClientTags, ListAliveClients content
// ---------------------------------------------------------------------------

TEST(InMemoryStore, GetPeerAddressAliveExpiredAndUnknown) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");

  // ALIVE → peer surfaced (MakeReg sets peer:<node>).
  auto alive = store.GetPeerAddress("n1");
  ASSERT_TRUE(alive.has_value());
  EXPECT_EQ(*alive, "peer:n1");

  // EXPIRED rows still surface their peer_address (contract: the row is kept).
  ASSERT_EQ(store.ExpireStaleClients(kT0 + 10s).size(), 1u);
  auto expired = store.GetPeerAddress("n1");
  ASSERT_TRUE(expired.has_value());
  EXPECT_EQ(*expired, "peer:n1");

  // Unknown node → nullopt.
  EXPECT_FALSE(store.GetPeerAddress("ghost").has_value());
}

TEST(InMemoryStore, GetClientTagsReturnsRegisteredTagsAndEmptyForUnknown) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");  // MakeReg sets tags = {"role=test"}

  auto tags = store.GetClientTags("n1");
  ASSERT_EQ(tags.size(), 1u);
  EXPECT_EQ(tags[0], "role=test");

  EXPECT_TRUE(store.GetClientTags("ghost").empty());
}

TEST(InMemoryStore, ListAliveClientsReturnsAliveRecordsExcludingExpired) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1", kT0);
  RegisterAlive(store, "n2", kT0 + 20s);  // fresher, survives the cutoff below

  // Expire only n1.
  ASSERT_EQ(store.ExpireStaleClients(kT0 + 10s).size(), 1u);

  auto alive = store.ListAliveClients();
  ASSERT_EQ(alive.size(), 1u);  // n1 excluded even though its row still exists
  EXPECT_EQ(alive[0].node_id, "n2");
  EXPECT_EQ(alive[0].status, ClientStatus::ALIVE);
  EXPECT_EQ(alive[0].peer_address, "peer:n2");
}

// The lightweight peer view maps node->peer (no capacity) and reflects
// membership changes. Ported from the old ClientRegistry peer-view tests.
TEST(InMemoryStore, GetAlivePeerViewTracksMembership) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");

  auto v1 = store.GetAlivePeerView();
  EXPECT_EQ(v1.size(), 1u);
  ASSERT_EQ(v1.count("n1"), 1u);
  EXPECT_EQ(v1.at("n1"), "peer:n1");  // MakeReg sets peer:<node>

  // Membership change → reflected in a freshly built view.
  RegisterAlive(store, "n2");
  auto v2 = store.GetAlivePeerView();
  EXPECT_EQ(v2.size(), 2u);
  EXPECT_EQ(v2.at("n2"), "peer:n2");
}

// EXPIRED rows are excluded from the peer view (unlike GetPeerAddress, which
// still surfaces a single expired row's peer).
TEST(InMemoryStore, GetAlivePeerViewExcludesExpired) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1", kT0);
  RegisterAlive(store, "n2", kT0 + 20s);  // fresher, survives the cutoff below

  ASSERT_EQ(store.ExpireStaleClients(kT0 + 10s).size(), 1u);  // expires n1 only

  auto view = store.GetAlivePeerView();
  EXPECT_EQ(view.size(), 1u);
  EXPECT_EQ(view.count("n1"), 0u);
  ASSERT_EQ(view.count("n2"), 1u);
  EXPECT_EQ(view.at("n2"), "peer:n2");
}

// The peer view carries no capacity, so a capacity-only heartbeat leaves its
// contents (node → peer) unchanged.
TEST(InMemoryStore, GetAlivePeerViewIgnoresCapacity) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");

  auto p1 = store.GetAlivePeerView();
  ASSERT_EQ(Beat(store, "n1", /*seq=*/1, /*events=*/{}, kT0 + 1s).status, HeartbeatResult::APPLIED);
  auto p2 = store.GetAlivePeerView();
  EXPECT_EQ(p1, p2);
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

TEST(InMemoryStore, ConcurrentHeartbeatCasExactlyOneApplied) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");

  std::atomic<int> applied{0};
  std::atomic<int> gap{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      // Both race to apply seq=1 (last_applied starts at 0).
      auto r = store.ApplyHeartbeat("n1", 1, kT0, Caps(), {}, /*is_full_sync=*/false);
      if (r.status == HeartbeatResult::APPLIED) {
        applied.fetch_add(1);
      } else if (r.status == HeartbeatResult::SEQ_GAP) {
        gap.fetch_add(1);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& th : threads) th.join();

  EXPECT_EQ(applied.load(), 1);
  EXPECT_EQ(gap.load(), 1);
  EXPECT_EQ(store.GetClient("n1")->last_applied_seq, 1u);
}

// ThreadSanitizer safety net for collapsing four lock domains into one: a mixed
// read/write workload across the shared/unique split must be race-free.
TEST(InMemoryStore, MixedWorkloadIsRaceFree) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "n1");
  for (int i = 0; i < 100; ++i) {
    store.ApplyHeartbeat("n1", i + 1, kT0, Caps(),
                         {Add("k" + std::to_string(i), TierType::HBM, 10)},
                         /*is_full_sync=*/false);
  }
  ASSERT_TRUE(store.RegisterExternalKvIfAlive("n1", {"h1", "h2", "h3"}, TierType::HBM));

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;

  // RouteGet readers (shared-lock path with atomic lease/access mutation).
  for (int r = 0; r < 4; ++r) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int i = 0; i < 500; ++i) {
        store.BatchLookupBlockForRouteGet({"k1", "k50", "k99"}, {}, kT0 + std::chrono::seconds(i),
                                          30s);
        store.BatchExistsBlock({"k1", "k2"});
      }
    });
  }
  // Hit writers (the formerly-shared path that becomes exclusive).
  threads.emplace_back([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int i = 0; i < 500; ++i) {
      store.MatchExternalKv({"h1", "h2", "h3"}, /*count_as_hit=*/true,
                            kT0 + std::chrono::seconds(i));
    }
  });
  // Eviction-enumeration reader.
  threads.emplace_back([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int i = 0; i < 500; ++i) {
      store.EnumerateEvictionCandidates(Buckets("n1", TierType::HBM),
                                        EvictionOrder::kLeastRecentlyAccessed, 0,
                                        kT0 + std::chrono::seconds(i));
    }
  });

  start.store(true, std::memory_order_release);
  for (auto& th : threads) th.join();

  // After the storm, hit counts reflect exactly the 500 hit-writer iterations.
  auto counts = store.GetExternalKvHitCounts({"h1"});
  ASSERT_EQ(counts.size(), 1u);
  EXPECT_EQ(counts[0].hit_count_total, 500u);
}

// ---------------------------------------------------------------------------
// Cross-domain atomicity conformance (approach A). These make the intermediate-
// state declarations in master_metadata_store.h executable: with the block
// index split into key-hashed shards and cross-domain writes running
// meta→block WITHOUT nesting the locks, a concurrent RouteGet reader must still
// (a) never observe a torn per-key location, and (b) never resolve a peer for a
// node that isn't in the alive-peer view (residual block locations for an
// erased node are simply skipped by the router-style join — router.cpp).
// ---------------------------------------------------------------------------

// A writer repeatedly full-syncs a node's whole key set between two well-formed
// shapes (size 100 vs size 200). full_sync clears+replays per shard; within a
// shard a single key's replacement is atomic, so a reader must always see an
// old-or-new value, never a torn one. Also a ThreadSanitizer net for the
// meta→block split under concurrent full_sync.
TEST(InMemoryStore, ConcurrentFullSyncRouteGetNeverTears) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "w");

  const int kNumKeys = 64;  // spread across shards (default 32)
  std::vector<std::string> keys;
  keys.reserve(kNumKeys);
  std::vector<KvEvent> adds100, adds200;
  for (int i = 0; i < kNumKeys; ++i) {
    keys.push_back("k" + std::to_string(i));
    adds100.push_back(Add(keys.back(), TierType::HBM, 100));
    adds200.push_back(Add(keys.back(), TierType::HBM, 200));
  }
  ASSERT_EQ(store.ApplyHeartbeat("w", 1, kT0, Caps(), adds100, /*is_full_sync=*/false).status,
            HeartbeatResult::APPLIED);

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> torn{false};

  std::thread writer([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    uint64_t seq = 100;  // full_sync re-baselines seq, so any increasing value is fine
    for (int it = 0; it < 2000; ++it) {
      const auto& adds = (it % 2 == 0) ? adds200 : adds100;
      store.ApplyHeartbeat("w", seq++, kT0, Caps(), adds, /*is_full_sync=*/true);
    }
    stop.store(true, std::memory_order_release);
  });

  std::vector<std::thread> readers;
  for (int r = 0; r < 4; ++r) {
    readers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      while (!stop.load(std::memory_order_acquire)) {
        auto locs = store.BatchLookupBlockForRouteGet(keys, {}, kT0, 30s);
        for (const auto& per_key : locs) {
          for (const auto& loc : per_key) {
            const bool well_formed = loc.node_id == "w" && loc.tier == TierType::HBM &&
                                     (loc.size == 100 || loc.size == 200);
            if (!well_formed) torn.store(true, std::memory_order_release);
          }
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  writer.join();
  for (auto& t : readers) t.join();
  EXPECT_FALSE(torn.load());  // every visible location was a valid old-or-new value
}

// A writer churns node "b" (Unregister → Register → re-seed key K) while readers
// do a router-style RouteGet: snapshot GetAlivePeerView, then join it with the
// block locations. The invariant: a location whose node resolves in the alive
// view always yields a non-empty peer for a node that WAS alive in that same
// snapshot — a residual block location for an erased node is absent from the
// view and thus never routed to. Exercises the meta→block non-atomic window.
TEST(InMemoryStore, ConcurrentUnregisterRouteGetNeverResolvesDeadPeer) {
  InMemoryMasterMetadataStore store;
  RegisterAlive(store, "a");
  RegisterAlive(store, "b");
  ASSERT_EQ(Beat(store, "a", 1, {Add("K", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);
  ASSERT_EQ(Beat(store, "b", 1, {Add("K", TierType::HBM, 10)}, kT0).status,
            HeartbeatResult::APPLIED);

  std::atomic<bool> start{false};
  std::atomic<bool> bad_peer{false};

  std::thread writer([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int it = 0; it < 1000; ++it) {
      store.UnregisterClient("b");                   // meta erase + block wipe
      store.RegisterClient(MakeReg("b"), kT0, 30s);  // back to ALIVE
      store.ApplyHeartbeat("b", 1, kT0, Caps(), {Add("K", TierType::HBM, 10)},
                           /*is_full_sync=*/true);  // re-seed K@b
    }
  });

  std::vector<std::thread> readers;
  for (int r = 0; r < 4; ++r) {
    readers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int it = 0; it < 2000; ++it) {
        // Router-style resolution (router.cpp): peer comes from the alive view,
        // NOT from the block location itself.
        auto view = store.GetAlivePeerView();
        auto locs = store.BatchLookupBlockForRouteGet({"K"}, {}, kT0, 30s);
        for (const auto& loc : locs[0]) {
          auto pit = view.find(loc.node_id);
          if (pit != view.end() && pit->second.empty()) {
            bad_peer.store(true, std::memory_order_release);
          }
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  writer.join();
  for (auto& t : readers) t.join();
  EXPECT_FALSE(bad_peer.load());

  // Deterministic post-condition: with b unregistered for good, any location we
  // could route to (i.e. that resolves in the alive view) is a live node, never
  // a stale b; b's residual location (if still present) is absent from the view.
  store.UnregisterClient("b");
  auto view = store.GetAlivePeerView();
  auto locs = store.BatchLookupBlockForRouteGet({"K"}, {}, kT0, 30s);
  EXPECT_EQ(view.count("b"), 0u);
  for (const auto& loc : locs[0]) {
    if (view.count(loc.node_id)) EXPECT_EQ(loc.node_id, "a");
  }
}

}  // namespace
}  // namespace mori::umbp
