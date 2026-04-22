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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/routing/router.h"

namespace mori::umbp {
namespace {

constexpr uint64_t GB = 1024ULL * 1024 * 1024;
// Phase 1 page-bitmap allocator default page_size is 2 MiB; FinalizeAllocation
// now requires the caller to send back the canonical page-string location_id
// returned by AllocateForPut.  The allocator rounds size up to whole pages,
// so any size in (0, 2 MiB] reserves exactly 1 page.
constexpr uint64_t BLOCK_SIZE = 2ULL * 1024 * 1024;  // 1 page

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

class FinalizeIdempotencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    index_.SetClientRegistry(&registry_);
    registry_.RegisterClient("node-a", "addr-a", MakeTierCapacities(80 * GB, 40 * GB), "peer-a", {},
                             {}, {}, {80 * GB});
  }

  GlobalBlockIndex index_;
  ClientRegistry registry_{ClientRegistryConfig{}, index_};
};

TEST_F(FinalizeIdempotencyTest, ConsistentReplay) {
  auto alloc = registry_.AllocateForPut("node-a", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(alloc.has_value());

  Location loc = MakeLocation("node-a", alloc->location_id, BLOCK_SIZE, TierType::HBM);

  EXPECT_TRUE(registry_.FinalizeAllocation("node-a", "key-1", loc, alloc->allocation_id));
  EXPECT_TRUE(registry_.FinalizeAllocation("node-a", "key-1", loc, alloc->allocation_id));
}

TEST_F(FinalizeIdempotencyTest, InconsistentReplay) {
  auto alloc = registry_.AllocateForPut("node-a", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(alloc.has_value());

  Location loc = MakeLocation("node-a", alloc->location_id, BLOCK_SIZE, TierType::HBM);

  EXPECT_TRUE(registry_.FinalizeAllocation("node-a", "key-1", loc, alloc->allocation_id));
  EXPECT_FALSE(registry_.FinalizeAllocation("node-a", "key-DIFFERENT", loc, alloc->allocation_id));
}

TEST_F(FinalizeIdempotencyTest, InconsistentLocation) {
  auto alloc = registry_.AllocateForPut("node-a", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(alloc.has_value());

  Location loc = MakeLocation("node-a", alloc->location_id, BLOCK_SIZE, TierType::HBM);

  EXPECT_TRUE(registry_.FinalizeAllocation("node-a", "key-1", loc, alloc->allocation_id));

  // Different location_id (canonical page format) than the one returned.
  Location different_loc = MakeLocation("node-a", "99:p99", BLOCK_SIZE, TierType::HBM);
  EXPECT_FALSE(
      registry_.FinalizeAllocation("node-a", "key-1", different_loc, alloc->allocation_id));
}

TEST_F(FinalizeIdempotencyTest, LocationValidation) {
  auto alloc = registry_.AllocateForPut("node-a", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(alloc.has_value());

  Location wrong_loc = MakeLocation("node-a", "999:p999", BLOCK_SIZE, TierType::HBM);
  EXPECT_FALSE(registry_.FinalizeAllocation("node-a", "key-1", wrong_loc, alloc->allocation_id));
}

// Regression for Plan `abort_allocation_cleanup` §2: on field-level mismatch
// FinalizeAllocation must auto-release the pending allocation's pages (no
// Abort RPC from the client).  We drive the allocator to full capacity then
// trigger a mismatched finalize and assert the freed slot is immediately
// reusable by the next AllocateForPut.
TEST(FinalizeAutoRollback, MismatchReleasesPagesImmediately) {
  GlobalBlockIndex index;
  ClientRegistry registry{ClientRegistryConfig{}, index};
  index.SetClientRegistry(&registry);

  // Single-page capacity on node-r so we can prove the released page is the
  // only thing keeping the next allocation alive.
  const uint64_t capacity = BLOCK_SIZE;
  registry.RegisterClient("node-r", "addr-r", {{TierType::HBM, TierCapacity{capacity, capacity}}},
                          "peer-r", {}, {}, {}, {capacity});

  auto first = registry.AllocateForPut("node-r", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(first.has_value());

  // Capacity is now exhausted: a fresh AllocateForPut must fail.
  ASSERT_FALSE(registry.AllocateForPut("node-r", TierType::HBM, BLOCK_SIZE).has_value());

  // Drive a size mismatch at finalize -> master auto-rolls back the pending.
  Location wrong_size = MakeLocation("node-r", first->location_id, BLOCK_SIZE / 2, TierType::HBM);
  EXPECT_FALSE(
      registry.FinalizeAllocation("node-r", "key-autoroll", wrong_size, first->allocation_id));

  // Capacity must be fully reclaimed now that the mismatch auto-rolled back.
  auto second = registry.AllocateForPut("node-r", TierType::HBM, BLOCK_SIZE);
  EXPECT_TRUE(second.has_value());
  EXPECT_NE(second->allocation_id, first->allocation_id);

  // The dead allocation_id stays dead: subsequent finalize attempts with the
  // stale id see pending-not-found and return false without touching the
  // newly-allocated pending.
  Location replay = MakeLocation("node-r", first->location_id, BLOCK_SIZE, TierType::HBM);
  EXPECT_FALSE(registry.FinalizeAllocation("node-r", "key-autoroll", replay, first->allocation_id));
}

TEST_F(FinalizeIdempotencyTest, CrossNodeFinalize) {
  registry_.RegisterClient("node-b", "addr-b", MakeTierCapacities(80 * GB, 40 * GB), "peer-b", {},
                           {}, {}, {80 * GB});

  auto router = std::make_unique<Router>(index_, registry_);
  auto result = router->RoutePut("key-cross", "node-a", BLOCK_SIZE);
  ASSERT_TRUE(result.has_value());

  std::string target_node = result->node_id;
  Location loc = MakeLocation(target_node, result->location_id, BLOCK_SIZE, TierType::HBM);

  EXPECT_TRUE(registry_.FinalizeAllocation(target_node, "key-cross", loc, result->allocation_id));

  auto locs = index_.Lookup("key-cross");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].node_id, target_node);
}

TEST_F(FinalizeIdempotencyTest, DepthWrittenToIndex) {
  auto alloc = registry_.AllocateForPut("node-a", TierType::HBM, BLOCK_SIZE);
  ASSERT_TRUE(alloc.has_value());

  Location loc = MakeLocation("node-a", alloc->location_id, BLOCK_SIZE, TierType::HBM);

  ASSERT_TRUE(registry_.FinalizeAllocation("node-a", "depth-key", loc, alloc->allocation_id));

  index_.SetDepth("depth-key", 7);

  std::set<GlobalBlockIndex::NodeTierKey> overloaded;
  overloaded.insert({"node-a", TierType::HBM});
  auto candidates = index_.FindEvictionCandidates(overloaded);

  bool found = false;
  for (const auto& c : candidates) {
    if (c.key == "depth-key") {
      EXPECT_EQ(c.depth, 7);
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace
}  // namespace mori::umbp
