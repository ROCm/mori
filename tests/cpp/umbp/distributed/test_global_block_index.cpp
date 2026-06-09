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

#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {
namespace {

Location MakeLocation(const std::string& node_id, const std::string& location_id, uint64_t size,
                      TierType tier) {
  Location loc;
  loc.node_id = node_id;
  loc.location_id = location_id;
  loc.size = size;
  loc.tier = tier;
  return loc;
}

std::map<TierType, TierCapacity> MakeTierCapacities(uint64_t total_bytes,
                                                    uint64_t available_bytes) {
  return {{TierType::HBM, TierCapacity{total_bytes, available_bytes}}};
}

}  // namespace

TEST(BlockIndexTest, RegisterLookupIdempotent) {
  GlobalBlockIndex index;
  const Location loc = MakeLocation("node-a", "loc-1", 4096, TierType::HBM);

  index.Register("node-a", "key-1", loc);
  index.Register("node-a", "key-1", loc);

  const auto locs = index.Lookup("key-1");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0], loc);
}

TEST(BlockIndexTest, MetricsInitializedAndUpdatedOnAccess) {
  GlobalBlockIndex index;
  const Location loc = MakeLocation("node-a", "loc-1", 4096, TierType::HBM);

  index.Register("node-a", "key-1", loc);

  const auto initial_metrics = index.GetMetrics("key-1");
  ASSERT_TRUE(initial_metrics.has_value());
  EXPECT_EQ(initial_metrics->access_count, 1u);

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  index.RecordAccess("key-1");

  const auto updated_metrics = index.GetMetrics("key-1");
  ASSERT_TRUE(updated_metrics.has_value());
  EXPECT_EQ(updated_metrics->access_count, 2u);
  EXPECT_GE(updated_metrics->last_accessed_at, initial_metrics->last_accessed_at);
  EXPECT_GE(updated_metrics->last_accessed_at, updated_metrics->created_at);
}

TEST(BlockIndexTest, RegisterNewReplicaUpdatesLastAccessedMetrics) {
  GlobalBlockIndex index;
  const Location loc_a = MakeLocation("node-a", "loc-1", 4096, TierType::HBM);
  const Location loc_b = MakeLocation("node-b", "loc-2", 4096, TierType::HBM);

  index.Register("node-a", "key-1", loc_a);
  const auto before = index.GetMetrics("key-1");
  ASSERT_TRUE(before.has_value());

  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  index.Register("node-b", "key-1", loc_b);
  const auto after = index.GetMetrics("key-1");
  ASSERT_TRUE(after.has_value());

  EXPECT_EQ(after->access_count, before->access_count + 1);
  EXPECT_GE(after->last_accessed_at, before->last_accessed_at);
  EXPECT_EQ(index.Lookup("key-1").size(), 2u);
}

TEST(BlockIndexTest, UnregisterRemovesKeyWhenLastReplicaRemoved) {
  GlobalBlockIndex index;
  const Location loc_a = MakeLocation("node-a", "loc-a", 4096, TierType::HBM);
  const Location loc_b = MakeLocation("node-b", "loc-b", 4096, TierType::HBM);

  index.Register("node-a", "key-1", loc_a);
  index.Register("node-b", "key-1", loc_b);

  EXPECT_TRUE(index.Unregister("node-a", "key-1", loc_a));
  auto locs = index.Lookup("key-1");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].node_id, "node-b");

  EXPECT_FALSE(index.Unregister("node-a", "key-1", loc_a));
  EXPECT_TRUE(index.Unregister("node-b", "key-1", loc_b));
  EXPECT_TRUE(index.Lookup("key-1").empty());
  EXPECT_FALSE(index.GetMetrics("key-1").has_value());
}

TEST(BlockIndexTest, UnregisterByNodeRemovesMatchingReplicasOnly) {
  GlobalBlockIndex index;

  index.Register("node-a", "key-1", MakeLocation("node-a", "loc-a1", 1024, TierType::HBM));
  index.Register("node-a", "key-1", MakeLocation("node-a", "loc-a2", 1024, TierType::DRAM));
  index.Register("node-b", "key-1", MakeLocation("node-b", "loc-b1", 1024, TierType::HBM));

  EXPECT_EQ(index.UnregisterByNode("key-1", "node-a"), 2u);

  const auto locs = index.Lookup("key-1");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].node_id, "node-b");
  EXPECT_EQ(index.UnregisterByNode("key-1", "missing"), 0u);
}

TEST(BlockIndexTest, UnregisterByNodeUpdatesRegistryOwnershipTracking) {
  GlobalBlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  index.SetClientRegistry(&registry);

  ASSERT_TRUE(registry.RegisterClient("node-a", "host-a", MakeTierCapacities(80, 64)));
  index.Register("node-a", "key-1", MakeLocation("node-a", "loc-a1", 1024, TierType::HBM));

  EXPECT_EQ(index.UnregisterByNode("key-1", "node-a"), 1u);
  EXPECT_TRUE(index.Lookup("key-1").empty());

  // If UnregisterByNode updates ownership tracking, no stale key should remain.
  EXPECT_EQ(registry.UnregisterClient("node-a"), 0u);
}

TEST(BlockIndexTest, BatchRegisterAndBatchUnregister) {
  GlobalBlockIndex index;

  const Location key1_a = MakeLocation("node-a", "k1-a", 1024, TierType::HBM);
  const Location key1_b = MakeLocation("node-a", "k1-b", 1024, TierType::DRAM);
  const Location key2_a = MakeLocation("node-a", "k2-a", 2048, TierType::HBM);

  std::vector<std::pair<std::string, Location>> add_entries = {
      {"key-1", key1_a},
      {"key-1", key1_a},  // duplicate
      {"key-1", key1_b},
      {"key-2", key2_a},
  };

  EXPECT_EQ(index.BatchRegister("node-a", add_entries), 3u);
  EXPECT_EQ(index.Lookup("key-1").size(), 2u);
  EXPECT_EQ(index.Lookup("key-2").size(), 1u);

  std::vector<std::pair<std::string, Location>> remove_entries = {
      {"key-1", key1_a},
      {"key-1", key1_a},  // duplicate
      {"key-2", key2_a},
      {"missing", key2_a},
  };

  EXPECT_EQ(index.BatchUnregister("node-a", remove_entries), 2u);
  EXPECT_EQ(index.Lookup("key-1").size(), 1u);
  EXPECT_TRUE(index.Lookup("key-2").empty());
}

TEST(BlockIndexTest, BatchLookupExistsMatchesSingleLookup) {
  GlobalBlockIndex index;

  index.Register("node-a", "key-1", MakeLocation("node-a", "loc-1", 1024, TierType::HBM));
  index.Register("node-a", "key-2", MakeLocation("node-a", "loc-2", 2048, TierType::DRAM));
  // key-3 intentionally never registered.

  const std::vector<std::string> keys = {"key-1", "key-2", "key-3", ""};

  const auto batch = index.BatchLookupExists(keys);
  ASSERT_EQ(batch.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    const bool single = !index.Lookup(keys[i]).empty();
    EXPECT_EQ(batch[i], single) << "keys[" << i << "]='" << keys[i] << "'";
  }

  EXPECT_TRUE(index.BatchLookupExists({}).empty());
}

TEST(BlockIndexTest, BatchLookupExistsHasNoMetricsSideEffects) {
  GlobalBlockIndex index;

  index.Register("node-a", "key-1", MakeLocation("node-a", "loc-1", 1024, TierType::HBM));
  index.Register("node-a", "key-2", MakeLocation("node-a", "loc-2", 2048, TierType::DRAM));

  const auto before_k1 = index.GetMetrics("key-1");
  const auto before_k2 = index.GetMetrics("key-2");
  ASSERT_TRUE(before_k1.has_value());
  ASSERT_TRUE(before_k2.has_value());

  // Sleep so that any RecordAccess side-effect would bump last_accessed_at
  // above the captured baseline (RecordAccess uses steady_clock::now()).
  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  const std::vector<std::string> keys = {"key-1", "key-2", "key-missing"};
  for (int i = 0; i < 5; ++i) {
    const auto found = index.BatchLookupExists(keys);
    ASSERT_EQ(found.size(), keys.size());
    EXPECT_TRUE(found[0]);
    EXPECT_TRUE(found[1]);
    EXPECT_FALSE(found[2]);
  }

  const auto after_k1 = index.GetMetrics("key-1");
  const auto after_k2 = index.GetMetrics("key-2");
  ASSERT_TRUE(after_k1.has_value());
  ASSERT_TRUE(after_k2.has_value());

  // Same read-only contract as single Lookup (#9): access_count and
  // last_accessed_at must not move when we merely probe existence.
  EXPECT_EQ(after_k1->access_count, before_k1->access_count);
  EXPECT_EQ(after_k2->access_count, before_k2->access_count);
  EXPECT_EQ(after_k1->last_accessed_at, before_k1->last_accessed_at);
  EXPECT_EQ(after_k2->last_accessed_at, before_k2->last_accessed_at);
}

TEST(BlockIndexTest, ClientRegistryUnregisterCleansIndexEntriesForClient) {
  GlobalBlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  index.SetClientRegistry(&registry);

  ASSERT_TRUE(registry.RegisterClient("node-a", "host-a", MakeTierCapacities(80, 64)));
  ASSERT_TRUE(registry.RegisterClient("node-b", "host-b", MakeTierCapacities(80, 64)));

  index.Register("node-a", "key-1", MakeLocation("node-a", "k1-a", 1024, TierType::HBM));
  index.Register("node-a", "key-2", MakeLocation("node-a", "k2-a", 1024, TierType::HBM));
  index.Register("node-b", "key-2", MakeLocation("node-b", "k2-b", 1024, TierType::HBM));

  EXPECT_EQ(registry.UnregisterClient("node-a"), 2u);
  EXPECT_TRUE(index.Lookup("key-1").empty());

  const auto key2_locs = index.Lookup("key-2");
  ASSERT_EQ(key2_locs.size(), 1u);
  EXPECT_EQ(key2_locs[0].node_id, "node-b");
}

}  // namespace mori::umbp
