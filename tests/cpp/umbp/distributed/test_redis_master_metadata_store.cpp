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

// Phase 1 targeted tests for RedisMasterMetadataStore: the six hot-path methods
// plus the register/heartbeat/unregister/expire semantics that must match the
// in-memory backend. The store tests run against a live RESP store (Redis /
// Dragonfly / Valkey) at UMBP_REDIS_URI (default tcp://127.0.0.1:6379) and skip
// cleanly when none is reachable, so BUILD_TESTS on a host without Redis does
// not fail. Every store test is parameterized over block_shards (1 = legacy
// single-tag layout, 16 = sharded) so both the whole-batch and the per-shard
// fan-out read paths are exercised with identical assertions.
//
// The KeySchema tests are pure (no store) and always run — they guard the shard
// mapping and the "single shard == legacy key strings" invariant even where no
// Redis is available.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "umbp/distributed/master/redis/key_schema.h"
#include "umbp/distributed/master/redis/resp_client.h"
#include "umbp/distributed/master/redis_master_metadata_store.h"

namespace mori::umbp {
namespace {

using namespace std::chrono_literals;

// =====================================================================
// KeySchema — pure unit tests (no live store needed).
// =====================================================================

TEST(KeySchemaTest, SingleShardIsLegacyLayout) {
  redis::KeySchema schema("ns", 1);
  EXPECT_EQ(schema.NumShards(), 1u);
  EXPECT_EQ(schema.Tag(), "{umbp:ns}");
  EXPECT_EQ(schema.ShardOf("abc"), 0u);
  // Byte-identical to the original single-tag schema.
  EXPECT_EQ(schema.Block("abc"), "{umbp:ns}:block:abc");
  EXPECT_EQ(schema.Node("n1"), "{umbp:ns}:node:n1");
  EXPECT_EQ(schema.NodeBlocks("n1"), "{umbp:ns}:node:n1:blocks");
}

TEST(KeySchemaTest, ZeroShardsClampsToOne) {
  redis::KeySchema schema("ns", 0);
  EXPECT_EQ(schema.NumShards(), 1u);
  EXPECT_EQ(schema.Block("x"), "{umbp:ns}:block:x");
}

TEST(KeySchemaTest, MultiShardIsDeterministicAndInRange) {
  constexpr std::size_t kShards = 16;
  redis::KeySchema schema("ns", kShards);
  EXPECT_EQ(schema.NumShards(), kShards);
  for (int i = 0; i < 1000; ++i) {
    const std::string key = "key-" + std::to_string(i);
    const std::size_t shard = schema.ShardOf(key);
    EXPECT_LT(shard, kShards);
    EXPECT_EQ(shard, schema.ShardOf(key)) << "shard mapping must be deterministic";
    EXPECT_EQ(schema.Block(key), "{umbp:ns:b" + std::to_string(shard) + "}:block:" + key);
  }
}

TEST(KeySchemaTest, ShardsAreReasonablySpread) {
  constexpr std::size_t kShards = 16;
  redis::KeySchema schema("ns", kShards);
  std::vector<int> counts(kShards, 0);
  for (int i = 0; i < 4096; ++i) counts[schema.ShardOf("k/" + std::to_string(i))]++;
  for (std::size_t s = 0; s < kShards; ++s) EXPECT_GT(counts[s], 0) << "shard " << s << " is empty";
}

// =====================================================================
// Store tests — parameterized over block_shards, require a live RESP store.
// =====================================================================

std::string RedisUri() {
  const char* v = std::getenv("UMBP_REDIS_URI");
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string("tcp://127.0.0.1:6379");
}

// Unique namespace per process so parallel runs / leftover keys never collide.
std::string UniqueNamespace() {
  return "test_" + std::to_string(::getpid()) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff);
}

// A store mode to run every assertion under: single-endpoint with N block
// shards, or multi-endpoint with E endpoints (one shard each). The multi mode
// points all logical endpoints at the same physical Redis (distinct hash tags
// keep the keyspaces disjoint), so it exercises the split-write / per-endpoint
// fan-out code paths without needing several Redis processes.
struct StoreMode {
  std::size_t block_shards;  // single-endpoint shard count (endpoints == 1)
  std::size_t endpoints;     // >1 => multi-endpoint mode (block_shards ignored)
  const char* name;
};

class RedisStoreTest : public ::testing::TestWithParam<StoreMode> {
 protected:
  void SetUp() override {
    redis::RespClient::Options opts;
    opts.uri = RedisUri();
    opts.pool_size = 2;
    probe_ = std::make_unique<redis::RespClient>(opts);
    if (!probe_->Ping()) {
      GTEST_SKIP() << "no RESP store reachable at " << RedisUri()
                   << " (set UMBP_REDIS_URI); skipping";
    }
    ns_ = UniqueNamespace();
    RedisMasterMetadataStore::Config cfg;
    cfg.uri = RedisUri();
    cfg.namespace_id = ns_;
    const StoreMode& m = GetParam();
    if (m.endpoints > 1) {
      cfg.shard_uris.assign(m.endpoints, RedisUri());
    } else {
      cfg.block_shards = m.block_shards;
    }
    store_ = std::make_unique<RedisMasterMetadataStore>(cfg);
    now_ = std::chrono::system_clock::now();
  }

  // Number of block shards the store built (must match KeySchema for MetaField).
  std::size_t NumShards() const {
    const StoreMode& m = GetParam();
    return m.endpoints > 1 ? m.endpoints : m.block_shards;
  }

  ClientRegistration MakeReg(const std::string& id) {
    ClientRegistration r;
    r.node_id = id;
    r.node_address = id + ".addr";
    r.peer_address = id + ".peer:1234";
    r.tier_capacities[TierType::DRAM] = TierCapacity{1u << 30, 1u << 30};
    r.tags = {"sgl_role=prefill", "zone=a"};
    return r;
  }

  // Raw HGET of a block-hash meta field via the probe client, so tests can
  // assert lease/access bookkeeping the store API does not expose. Returns -1
  // when the field (or the whole block key) is absent.
  long long MetaField(const std::string& user_key, const std::string& field) const {
    redis::KeySchema schema(ns_, NumShards());
    redis::RespValue r = probe_->Command({"HGET", schema.Block(user_key), field});
    if (r.type != redis::RespValue::Type::String) return -1;
    try {
      return std::stoll(r.str);
    } catch (...) {
      return -1;
    }
  }

  std::string ns_;
  std::unique_ptr<redis::RespClient> probe_;
  std::unique_ptr<RedisMasterMetadataStore> store_;
  std::chrono::system_clock::time_point now_;
};

INSTANTIATE_TEST_SUITE_P(
    BlockShards, RedisStoreTest,
    ::testing::Values(StoreMode{1, 1, "shards1"}, StoreMode{16, 1, "shards16"},
                      StoreMode{1, 3, "endpoints3"}),
    [](const ::testing::TestParamInfo<StoreMode>& info) { return std::string(info.param.name); });

TEST_P(RedisStoreTest, RegisterMakesClientAlive) {
  EXPECT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  EXPECT_TRUE(store_->IsClientAlive("n1"));
  EXPECT_EQ(store_->AliveClientCount(), 1u);

  auto peers = store_->GetAlivePeerView();
  ASSERT_EQ(peers.count("n1"), 1u);
  EXPECT_EQ(peers["n1"], "n1.peer:1234");

  auto rec = store_->GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->node_address, "n1.addr");
  EXPECT_EQ(rec->status, ClientStatus::ALIVE);
  ASSERT_EQ(rec->tier_capacities.count(TierType::DRAM), 1u);
  EXPECT_EQ(rec->tier_capacities[TierType::DRAM].total_bytes, 1u << 30);
  EXPECT_EQ(store_->GetClientTags("n1").size(), 2u);
}

TEST_P(RedisStoreTest, RegisterRejectsAliveDuplicateButAllowsStale) {
  EXPECT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Alive and not stale -> rejected.
  EXPECT_FALSE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Stale (stale_after = 0) -> allowed to replace.
  EXPECT_TRUE(store_->RegisterClient(MakeReg("n1"), now_ + 1s, 0s));
}

TEST_P(RedisStoreTest, ListAliveClientsReturnsRecords) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n2"), now_, 30s));
  auto alive = store_->ListAliveClients();
  EXPECT_EQ(alive.size(), 2u);
}

TEST_P(RedisStoreTest, HeartbeatAddThenRouteGet) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));

  std::vector<KvEvent> events = {
      KvEvent{KvEvent::Kind::ADD, "k1", TierType::DRAM, 4096},
      KvEvent{KvEvent::Kind::ADD, "k2", TierType::HBM, 8192},
  };
  auto res = store_->ApplyHeartbeat("n1", 1, now_, MakeReg("n1").tier_capacities, events, false);
  EXPECT_EQ(res.status, HeartbeatResult::APPLIED);
  EXPECT_EQ(res.acked_seq, 1u);

  auto exists = store_->BatchExistsBlock({"k1", "k2", "missing"});
  ASSERT_EQ(exists.size(), 3u);
  EXPECT_TRUE(exists[0]);
  EXPECT_TRUE(exists[1]);
  EXPECT_FALSE(exists[2]);

  auto locs = store_->BatchLookupBlockForRouteGet({"k1", "missing"}, {}, now_, 10s);
  ASSERT_EQ(locs.size(), 2u);
  ASSERT_EQ(locs[0].size(), 1u);
  EXPECT_EQ(locs[0][0].node_id, "n1");
  EXPECT_EQ(locs[0][0].tier, TierType::DRAM);
  EXPECT_EQ(locs[0][0].size, 4096u);
  EXPECT_TRUE(locs[1].empty());
}

TEST_P(RedisStoreTest, RouteGetExcludeFiltersNode) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n2"), now_, 30s));
  ASSERT_EQ(
      store_
          ->ApplyHeartbeat("n1", 1, now_, {}, {{KvEvent::Kind::ADD, "k", TierType::DRAM, 1}}, false)
          .status,
      HeartbeatResult::APPLIED);
  ASSERT_EQ(
      store_
          ->ApplyHeartbeat("n2", 1, now_, {}, {{KvEvent::Kind::ADD, "k", TierType::DRAM, 1}}, false)
          .status,
      HeartbeatResult::APPLIED);

  auto locs = store_->BatchLookupBlockForRouteGet({"k"}, {"n1"}, now_, 10s);
  ASSERT_EQ(locs.size(), 1u);
  ASSERT_EQ(locs[0].size(), 1u);
  EXPECT_EQ(locs[0][0].node_id, "n2");
}

TEST_P(RedisStoreTest, RouteGetBumpsAccessOnlyForTouchedKeys) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_EQ(store_
                ->ApplyHeartbeat("n1", 1, now_, {},
                                 {{KvEvent::Kind::ADD, "hit", TierType::DRAM, 7}}, false)
                .status,
            HeartbeatResult::APPLIED);

  // Freshly added key starts at _acnt = 0 and no lease.
  EXPECT_EQ(MetaField("hit", "_acnt"), 0);
  EXPECT_EQ(MetaField("hit", "_lease"), -1);

  // One RouteGet over a hit + a miss: only the hit is leased / access-bumped;
  // the miss must not be created.
  auto locs = store_->BatchLookupBlockForRouteGet({"hit", "miss"}, {}, now_, 10s);
  ASSERT_EQ(locs.size(), 2u);
  ASSERT_EQ(locs[0].size(), 1u);
  EXPECT_TRUE(locs[1].empty());

  EXPECT_EQ(MetaField("hit", "_acnt"), 1);    // touched -> bumped exactly once
  EXPECT_GT(MetaField("hit", "_lease"), 0);   // touched -> lease granted
  EXPECT_EQ(MetaField("miss", "_acnt"), -1);  // absent -> still absent (no phantom key)

  // A second RouteGet bumps again by exactly one (guards the NOSCRIPT
  // retry-only-failed path from double-applying the write).
  store_->BatchLookupBlockForRouteGet({"hit"}, {}, now_, 10s);
  EXPECT_EQ(MetaField("hit", "_acnt"), 2);
}

TEST_P(RedisStoreTest, HeartbeatSeqGapAndUnknown) {
  EXPECT_EQ(store_->ApplyHeartbeat("ghost", 1, now_, {}, {}, false).status,
            HeartbeatResult::UNKNOWN);

  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Expect seq 1 next; sending 3 is a gap.
  auto gap = store_->ApplyHeartbeat("n1", 3, now_, {}, {}, false);
  EXPECT_EQ(gap.status, HeartbeatResult::SEQ_GAP);
  EXPECT_EQ(gap.acked_seq, 0u);
}

TEST_P(RedisStoreTest, HeartbeatRemoveDropsLocation) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_EQ(
      store_
          ->ApplyHeartbeat("n1", 1, now_, {}, {{KvEvent::Kind::ADD, "k", TierType::DRAM, 1}}, false)
          .status,
      HeartbeatResult::APPLIED);
  ASSERT_TRUE(store_->BatchExistsBlock({"k"})[0]);

  ASSERT_EQ(store_
                ->ApplyHeartbeat("n1", 2, now_, {},
                                 {{KvEvent::Kind::REMOVE, "k", TierType::DRAM, 0}}, false)
                .status,
            HeartbeatResult::APPLIED);
  EXPECT_FALSE(store_->BatchExistsBlock({"k"})[0]);
}

TEST_P(RedisStoreTest, FullSyncReplacesLocations) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_EQ(store_
                ->ApplyHeartbeat("n1", 1, now_, {},
                                 {{KvEvent::Kind::ADD, "old", TierType::DRAM, 1}}, false)
                .status,
            HeartbeatResult::APPLIED);
  // Full sync with a different key set replaces wholesale.
  ASSERT_EQ(store_
                ->ApplyHeartbeat("n1", 5, now_, {},
                                 {{KvEvent::Kind::ADD, "new", TierType::DRAM, 2}}, true)
                .status,
            HeartbeatResult::APPLIED);
  EXPECT_FALSE(store_->BatchExistsBlock({"old"})[0]);
  EXPECT_TRUE(store_->BatchExistsBlock({"new"})[0]);
}

TEST_P(RedisStoreTest, UnregisterWipesClientAndBlocks) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_EQ(
      store_
          ->ApplyHeartbeat("n1", 1, now_, {}, {{KvEvent::Kind::ADD, "k", TierType::DRAM, 1}}, false)
          .status,
      HeartbeatResult::APPLIED);

  store_->UnregisterClient("n1");
  EXPECT_FALSE(store_->IsClientAlive("n1"));
  EXPECT_EQ(store_->AliveClientCount(), 0u);
  EXPECT_FALSE(store_->BatchExistsBlock({"k"})[0]);
}

TEST_P(RedisStoreTest, ExpireStaleFlipsAndWipesBlocks) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_EQ(
      store_
          ->ApplyHeartbeat("n1", 1, now_, {}, {{KvEvent::Kind::ADD, "k", TierType::DRAM, 1}}, false)
          .status,
      HeartbeatResult::APPLIED);

  // cutoff in the future -> n1's last_hb (==now_) is older -> expired.
  auto dead = store_->ExpireStaleClients(now_ + 10s);
  ASSERT_EQ(dead.size(), 1u);
  EXPECT_EQ(dead[0], "n1");
  EXPECT_FALSE(store_->IsClientAlive("n1"));
  EXPECT_FALSE(store_->BatchExistsBlock({"k"})[0]);
  // EXPIRED record is kept (hazard #3).
  auto rec = store_->GetClient("n1");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->status, ClientStatus::EXPIRED);
}

// ---------------------------------------------------------------------
// Cross-shard coverage: with block_shards=16 the keys below spread over
// multiple shards, so these exercise the per-shard fan-out + scatter-merge and
// the multi-shard wipe paths (a no-op difference when block_shards=1).
// ---------------------------------------------------------------------

TEST_P(RedisStoreTest, CrossShardBatchPreservesOrderAndMapping) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));

  constexpr int kKeys = 20;
  std::vector<KvEvent> events;
  events.reserve(kKeys);
  for (int i = 0; i < kKeys; ++i) {
    // size encodes identity (100 + i) so the merge can be verified per key.
    events.push_back({KvEvent::Kind::ADD, "key-" + std::to_string(i), TierType::DRAM,
                      static_cast<uint64_t>(100 + i)});
  }
  ASSERT_EQ(store_->ApplyHeartbeat("n1", 1, now_, {}, events, false).status,
            HeartbeatResult::APPLIED);

  // Scrambled query order, interleaved with a missing key.
  const std::vector<std::string> query = {"key-5", "missing", "key-0", "key-19", "key-12"};
  auto locs = store_->BatchLookupBlockForRouteGet(query, {}, now_, 10s);
  ASSERT_EQ(locs.size(), query.size());
  ASSERT_EQ(locs[0].size(), 1u);
  EXPECT_EQ(locs[0][0].size, 105u);
  EXPECT_TRUE(locs[1].empty());
  ASSERT_EQ(locs[2].size(), 1u);
  EXPECT_EQ(locs[2][0].size, 100u);
  ASSERT_EQ(locs[3].size(), 1u);
  EXPECT_EQ(locs[3][0].size, 119u);
  ASSERT_EQ(locs[4].size(), 1u);
  EXPECT_EQ(locs[4][0].size, 112u);

  // BatchExistsBlock in the same scrambled order maps back correctly too.
  auto exists = store_->BatchExistsBlock(query);
  ASSERT_EQ(exists.size(), query.size());
  EXPECT_TRUE(exists[0]);
  EXPECT_FALSE(exists[1]);
  EXPECT_TRUE(exists[2]);
  EXPECT_TRUE(exists[3]);
  EXPECT_TRUE(exists[4]);
}

TEST_P(RedisStoreTest, FullSyncAcrossShardsWipesAllOldLocations) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));

  constexpr int kKeys = 12;
  std::vector<KvEvent> old_events;
  std::vector<KvEvent> new_events;
  std::vector<std::string> old_keys;
  std::vector<std::string> new_keys;
  for (int i = 0; i < kKeys; ++i) {
    old_keys.push_back("old-" + std::to_string(i));
    new_keys.push_back("new-" + std::to_string(i));
    old_events.push_back({KvEvent::Kind::ADD, old_keys.back(), TierType::DRAM, 1});
    new_events.push_back({KvEvent::Kind::ADD, new_keys.back(), TierType::DRAM, 2});
  }
  ASSERT_EQ(store_->ApplyHeartbeat("n1", 1, now_, {}, old_events, false).status,
            HeartbeatResult::APPLIED);
  ASSERT_EQ(store_->ApplyHeartbeat("n1", 9, now_, {}, new_events, true).status,
            HeartbeatResult::APPLIED);

  for (bool e : store_->BatchExistsBlock(old_keys)) EXPECT_FALSE(e);
  for (bool e : store_->BatchExistsBlock(new_keys)) EXPECT_TRUE(e);
}

TEST_P(RedisStoreTest, UnregisterWipesBlocksAcrossShards) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));

  constexpr int kKeys = 12;
  std::vector<KvEvent> events;
  std::vector<std::string> keys;
  for (int i = 0; i < kKeys; ++i) {
    keys.push_back("u-" + std::to_string(i));
    events.push_back({KvEvent::Kind::ADD, keys.back(), TierType::DRAM, 1});
  }
  ASSERT_EQ(store_->ApplyHeartbeat("n1", 1, now_, {}, events, false).status,
            HeartbeatResult::APPLIED);
  ASSERT_TRUE(store_->BatchExistsBlock({keys.front()})[0]);

  store_->UnregisterClient("n1");
  for (bool e : store_->BatchExistsBlock(keys)) EXPECT_FALSE(e);
}

// Fault tolerance: with one shard instance down, reads for keys on the healthy
// shards still resolve and keys on the dead shard degrade to a miss (no throw).
// Control + shard 0 live; shard 1 points at a closed port.
TEST(RedisFaultToleranceTest, DownShardDegradesToMissNotError) {
  redis::RespClient::Options opts;
  opts.uri = RedisUri();
  opts.pool_size = 2;
  redis::RespClient probe(opts);
  if (!probe.Ping()) {
    GTEST_SKIP() << "no RESP store reachable at " << RedisUri() << "; skipping";
  }

  const std::string ns = UniqueNamespace();
  // Two shards: pick a key on each (shard 0 = live endpoint 0, shard 1 = dead).
  redis::KeySchema schema(ns, 2);
  std::string live_key, dead_key;
  for (int i = 0; (live_key.empty() || dead_key.empty()) && i < 100000; ++i) {
    const std::string k = "ft-" + std::to_string(i);
    if (schema.ShardOf(k) == 0 && live_key.empty()) live_key = k;
    if (schema.ShardOf(k) == 1 && dead_key.empty()) dead_key = k;
  }
  ASSERT_FALSE(live_key.empty());
  ASSERT_FALSE(dead_key.empty());

  RedisMasterMetadataStore::Config cfg;
  cfg.namespace_id = ns;
  cfg.connect_timeout_ms = 300;  // fail fast on the dead endpoint
  cfg.socket_timeout_ms = 300;
  cfg.shard_uris = {RedisUri(), "tcp://127.0.0.1:6399"};  // 6399: nothing listening
  RedisMasterMetadataStore store(cfg);

  const auto now = std::chrono::system_clock::now();
  ClientRegistration reg;
  reg.node_id = "ftn";
  reg.node_address = "a";
  reg.peer_address = "p:1";
  reg.tier_capacities[TierType::DRAM] = TierCapacity{1u << 20, 1u << 20};
  ASSERT_TRUE(store.RegisterClient(reg, now, 30s));  // control endpoint is live

  // Seed only the live shard (a delta touching just shard 0's instance).
  ASSERT_EQ(store.ApplyHeartbeat("ftn", 1, now, {}, {{KvEvent::Kind::ADD, live_key, TierType::DRAM, 7}},
                                 false)
                .status,
            HeartbeatResult::APPLIED);

  // Batch spanning the live and the dead shard must NOT throw; live key resolves,
  // dead-shard key reads as a miss.
  auto exists = store.BatchExistsBlock({live_key, dead_key});
  ASSERT_EQ(exists.size(), 2u);
  EXPECT_TRUE(exists[0]);
  EXPECT_FALSE(exists[1]);

  auto locs = store.BatchLookupBlockForRouteGet({live_key, dead_key}, {}, now, 10s);
  ASSERT_EQ(locs.size(), 2u);
  ASSERT_EQ(locs[0].size(), 1u);
  EXPECT_EQ(locs[0][0].size, 7u);
  EXPECT_TRUE(locs[1].empty());
}

// Write fault tolerance: a heartbeat whose events span a down shard does NOT
// error the RPC — it returns SEQ_GAP (so the peer full_syncs and self-heals) and
// the node stays ALIVE. Events on the live shard are still applied.
TEST(RedisWriteFaultToleranceTest, HeartbeatToDownShardSelfHealsNotError) {
  redis::RespClient::Options opts;
  opts.uri = RedisUri();
  opts.pool_size = 2;
  redis::RespClient probe(opts);
  if (!probe.Ping()) {
    GTEST_SKIP() << "no RESP store reachable at " << RedisUri() << "; skipping";
  }

  const std::string ns = UniqueNamespace();
  redis::KeySchema schema(ns, 2);
  std::string live_key, dead_key;
  for (int i = 0; (live_key.empty() || dead_key.empty()) && i < 100000; ++i) {
    const std::string k = "wft-" + std::to_string(i);
    if (schema.ShardOf(k) == 0 && live_key.empty()) live_key = k;
    if (schema.ShardOf(k) == 1 && dead_key.empty()) dead_key = k;
  }
  ASSERT_FALSE(live_key.empty());
  ASSERT_FALSE(dead_key.empty());

  RedisMasterMetadataStore::Config cfg;
  cfg.namespace_id = ns;
  cfg.connect_timeout_ms = 300;
  cfg.socket_timeout_ms = 300;
  cfg.shard_uris = {RedisUri(), "tcp://127.0.0.1:6399"};  // shard 1 down
  RedisMasterMetadataStore store(cfg);

  const auto now = std::chrono::system_clock::now();
  ClientRegistration reg;
  reg.node_id = "wn";
  reg.node_address = "a";
  reg.peer_address = "p:1";
  reg.tier_capacities[TierType::DRAM] = TierCapacity{1u << 20, 1u << 20};
  ASSERT_TRUE(store.RegisterClient(reg, now, 30s));  // control instance is live

  auto res = store.ApplyHeartbeat("wn", 1, now, {},
                                  {{KvEvent::Kind::ADD, live_key, TierType::DRAM, 1},
                                   {KvEvent::Kind::ADD, dead_key, TierType::DRAM, 1}},
                                  false);
  // Down shard -> self-heal signal, not an exception.
  EXPECT_EQ(res.status, HeartbeatResult::SEQ_GAP);
  EXPECT_TRUE(store.IsClientAlive("wn"));            // control committed -> node alive
  EXPECT_TRUE(store.BatchExistsBlock({live_key})[0]);  // live shard still got its event
}

// A metrics sink attached to the store exports per-op latency as
// mori_umbp_store_op_latency_seconds{op,backend="redis"}. Drives the store
// directly (no gRPC/GPU) and scrapes the Prometheus endpoint over HTTP.
TEST(RedisStoreMetricsTest, EmitsStoreOpLatencyHistogram) {
  redis::RespClient::Options opts;
  opts.uri = RedisUri();
  opts.pool_size = 2;
  redis::RespClient probe(opts);
  if (!probe.Ping()) {
    GTEST_SKIP() << "no RESP store reachable at " << RedisUri() << "; skipping";
  }

  const int port = 19099;
  std::unique_ptr<mori::metrics::MetricsServer> server;
  try {
    server = std::make_unique<mori::metrics::MetricsServer>(port);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "could not start metrics server on " << port << ": " << e.what();
  }

  RedisMasterMetadataStore::Config cfg;
  cfg.namespace_id = UniqueNamespace();
  RedisMasterMetadataStore store(cfg);
  store.SetMetricsSink(server.get());

  const auto now = std::chrono::system_clock::now();
  ClientRegistration reg;
  reg.node_id = "mn";
  reg.node_address = "a";
  reg.peer_address = "p:1";
  reg.tier_capacities[TierType::DRAM] = TierCapacity{1u << 20, 1u << 20};
  ASSERT_TRUE(store.RegisterClient(reg, now, 30s));
  ASSERT_EQ(store.ApplyHeartbeat("mn", 1, now, {}, {{KvEvent::Kind::ADD, "mk", TierType::DRAM, 4}},
                                 false)
                .status,
            HeartbeatResult::APPLIED);
  store.BatchLookupBlockForRouteGet({"mk"}, {}, now, 10s);
  store.BatchExistsBlock({"mk"});

  // Scrape /metrics via curl (present in the build image).
  const std::string cmd = "curl -s http://127.0.0.1:" + std::to_string(port) + "/metrics";
  std::string body;
  FILE* f = popen(cmd.c_str(), "r");
  ASSERT_NE(f, nullptr);
  char buf[8192];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) body.append(buf, n);
  pclose(f);

  EXPECT_NE(body.find("mori_umbp_store_op_latency_seconds"), std::string::npos) << body;
  EXPECT_NE(body.find("backend=\"redis\""), std::string::npos);
  EXPECT_NE(body.find("op=\"BatchLookupBlockForRouteGet\""), std::string::npos);
  EXPECT_NE(body.find("op=\"ApplyHeartbeat\""), std::string::npos);
}

}  // namespace
}  // namespace mori::umbp
