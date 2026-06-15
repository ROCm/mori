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

// Phase 1 compile/instantiation gate for IMasterMetadataStore.
//
// The interface is abstract with no implementation yet, so there is no runtime
// behavior to exercise. The bar for Phase 1 is that the contract is well-formed
// and instantiable:
//   1. MockMasterMetadataStore overrides every pure-virtual (a missing or
//      ill-typed override makes the mock abstract → fails to instantiate).
//   2. A MockMasterMetadataStore is usable through an IMasterMetadataStore&,
//      proving the override set is complete.
// Behavioral assertions arrive with InMemoryMasterMetadataStore in Phase 2.

#include <gtest/gtest.h>

#include <chrono>

#include "mock_master_metadata_store.h"
#include "umbp/distributed/master/master_metadata_store.h"

namespace mori::umbp {
namespace {

// Instantiation gate: if the interface had an orphaned or ill-typed pure
// virtual, MockMasterMetadataStore would stay abstract and this would not
// compile.
TEST(MasterMetadataStoreInterface, MockIsInstantiableThroughInterface) {
  MockMasterMetadataStore mock;
  IMasterMetadataStore& store = mock;
  (void)store;
  SUCCEED();
}

// Signature-completeness spot check: name every interface method once through
// the base-class pointer with a default ON_CALL, mirroring the §1b delta table
// plus the two added hit-count methods (GetExternalKvHitCounts,
// GarbageCollectHits) and the `now` parameter on MatchExternalKv. This guards
// against silently dropping the live GetExternalKvHitCounts RPC path.
TEST(MasterMetadataStoreInterface, EveryMethodIsCallableThroughInterface) {
  using ::testing::_;
  using ::testing::NiceMock;
  using ::testing::Return;
  using namespace std::chrono_literals;

  // NiceMock: these are default-action calls, not behavior under test, so the
  // "uninteresting call" warnings would just be noise.
  NiceMock<MockMasterMetadataStore> mock;
  const auto now = std::chrono::system_clock::now();

  ON_CALL(mock, RegisterClient(_, _, _)).WillByDefault(Return(true));
  ON_CALL(mock, ApplyHeartbeat(_, _, _, _, _, _))
      .WillByDefault(Return(HeartbeatResult{HeartbeatResult::APPLIED, 0}));
  ON_CALL(mock, ExpireStaleClients(_)).WillByDefault(Return(std::vector<std::string>{}));
  ON_CALL(mock, RegisterExternalKvIfAlive(_, _, _)).WillByDefault(Return(true));
  ON_CALL(mock, GarbageCollectHits(_)).WillByDefault(Return(0));
  ON_CALL(mock, LookupBlock(_)).WillByDefault(Return(std::vector<Location>{}));
  ON_CALL(mock, LookupBlockForRouteGet(_, _, _, _)).WillByDefault(Return(std::vector<Location>{}));
  ON_CALL(mock, BatchLookupBlockForRouteGet(_, _, _, _))
      .WillByDefault(Return(std::vector<std::vector<Location>>{}));
  ON_CALL(mock, BatchExistsBlock(_)).WillByDefault(Return(std::vector<bool>{}));
  ON_CALL(mock, EnumerateLruForEviction(_, _))
      .WillByDefault(Return(std::map<NodeTierKey, std::vector<EvictionCandidate>>{}));
  ON_CALL(mock, GetClient(_)).WillByDefault(Return(std::nullopt));
  ON_CALL(mock, IsClientAlive(_)).WillByDefault(Return(false));
  ON_CALL(mock, GetPeerAddress(_)).WillByDefault(Return(std::nullopt));
  ON_CALL(mock, ListAliveClients()).WillByDefault(Return(std::vector<ClientRecord>{}));
  ON_CALL(mock, AliveClientCount()).WillByDefault(Return(0));
  ON_CALL(mock, GetClientTags(_)).WillByDefault(Return(std::vector<std::string>{}));
  ON_CALL(mock, MatchExternalKv(_, _, _)).WillByDefault(Return(std::vector<NodeMatch>{}));
  ON_CALL(mock, GetExternalKvHitCounts(_))
      .WillByDefault(Return(std::vector<ExternalKvHitCountEntry>{}));
  ON_CALL(mock, GetExternalKvCount(_)).WillByDefault(Return(0));

  IMasterMetadataStore& store = mock;

  // Cross-store writes.
  ClientRegistration reg;
  reg.node_id = "node-a";
  EXPECT_TRUE(store.RegisterClient(reg, now, 30s));
  store.UnregisterClient("node-a");
  EXPECT_EQ(store.ApplyHeartbeat("node-a", 1, now, {}, {}, false).status, HeartbeatResult::APPLIED);
  EXPECT_TRUE(store.ExpireStaleClients(now).empty());

  // External-KV writes.
  EXPECT_TRUE(store.RegisterExternalKvIfAlive("node-a", {"h0"}, TierType::HBM));
  store.UnregisterExternalKv("node-a", {"h0"}, TierType::HBM);
  store.UnregisterExternalKvByTier("node-a", TierType::HBM);
  EXPECT_EQ(store.GarbageCollectHits(now), 0u);

  // Block reads.
  EXPECT_TRUE(store.LookupBlock("k0").empty());
  EXPECT_TRUE(store.LookupBlockForRouteGet("k0", {}, now, 5s).empty());
  EXPECT_TRUE(store.BatchLookupBlockForRouteGet({"k0"}, {}, now, 5s).empty());
  EXPECT_TRUE(store.BatchExistsBlock({"k0"}).empty());
  EXPECT_TRUE(store.EnumerateLruForEviction({}, now).empty());

  // Client reads.
  EXPECT_FALSE(store.GetClient("node-a").has_value());
  EXPECT_FALSE(store.IsClientAlive("node-a"));
  EXPECT_FALSE(store.GetPeerAddress("node-a").has_value());
  EXPECT_TRUE(store.ListAliveClients().empty());
  EXPECT_EQ(store.AliveClientCount(), 0u);
  EXPECT_TRUE(store.GetClientTags("node-a").empty());

  // External-KV reads, incl. the two added hit-count methods + `now` param.
  EXPECT_TRUE(store.MatchExternalKv({"h0"}, /*count_as_hit=*/true, now).empty());
  EXPECT_TRUE(store.GetExternalKvHitCounts({"h0"}).empty());
  EXPECT_EQ(store.GetExternalKvCount("node-a"), 0u);
}

}  // namespace
}  // namespace mori::umbp
