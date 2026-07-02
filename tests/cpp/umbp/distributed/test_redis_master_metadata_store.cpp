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
// in-memory backend. Requires a live RESP store (Redis / Dragonfly / Valkey) at
// UMBP_REDIS_URI (default tcp://127.0.0.1:6379). Skips cleanly when none is
// reachable, so BUILD_TESTS on a host without Redis does not fail.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>

#include "umbp/distributed/master/redis/resp_client.h"
#include "umbp/distributed/master/redis_master_metadata_store.h"

namespace mori::umbp {
namespace {

using namespace std::chrono_literals;

std::string RedisUri() {
  const char* v = std::getenv("UMBP_REDIS_URI");
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string("tcp://127.0.0.1:6379");
}

// Unique namespace per process so parallel runs / leftover keys never collide.
std::string UniqueNamespace() {
  return "test_" + std::to_string(::getpid()) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff);
}

class RedisStoreTest : public ::testing::Test {
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
    RedisMasterMetadataStore::Config cfg;
    cfg.uri = RedisUri();
    cfg.namespace_id = UniqueNamespace();
    store_ = std::make_unique<RedisMasterMetadataStore>(cfg);
    now_ = std::chrono::system_clock::now();
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

  std::unique_ptr<redis::RespClient> probe_;
  std::unique_ptr<RedisMasterMetadataStore> store_;
  std::chrono::system_clock::time_point now_;
};

TEST_F(RedisStoreTest, RegisterMakesClientAlive) {
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

TEST_F(RedisStoreTest, RegisterRejectsAliveDuplicateButAllowsStale) {
  EXPECT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Alive and not stale -> rejected.
  EXPECT_FALSE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Stale (stale_after = 0) -> allowed to replace.
  EXPECT_TRUE(store_->RegisterClient(MakeReg("n1"), now_ + 1s, 0s));
}

TEST_F(RedisStoreTest, ListAliveClientsReturnsRecords) {
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  ASSERT_TRUE(store_->RegisterClient(MakeReg("n2"), now_, 30s));
  auto alive = store_->ListAliveClients();
  EXPECT_EQ(alive.size(), 2u);
}

TEST_F(RedisStoreTest, HeartbeatAddThenRouteGet) {
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

TEST_F(RedisStoreTest, RouteGetExcludeFiltersNode) {
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

TEST_F(RedisStoreTest, HeartbeatSeqGapAndUnknown) {
  EXPECT_EQ(store_->ApplyHeartbeat("ghost", 1, now_, {}, {}, false).status,
            HeartbeatResult::UNKNOWN);

  ASSERT_TRUE(store_->RegisterClient(MakeReg("n1"), now_, 30s));
  // Expect seq 1 next; sending 3 is a gap.
  auto gap = store_->ApplyHeartbeat("n1", 3, now_, {}, {}, false);
  EXPECT_EQ(gap.status, HeartbeatResult::SEQ_GAP);
  EXPECT_EQ(gap.acked_seq, 0u);
}

TEST_F(RedisStoreTest, HeartbeatRemoveDropsLocation) {
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

TEST_F(RedisStoreTest, FullSyncReplacesLocations) {
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

TEST_F(RedisStoreTest, UnregisterWipesClientAndBlocks) {
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

TEST_F(RedisStoreTest, ExpireStaleFlipsAndWipesBlocks) {
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

}  // namespace
}  // namespace mori::umbp
