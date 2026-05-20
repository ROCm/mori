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
//
// Master-side dedup for Router::BatchRoutePut: indexed keys come back
// with already_exists=true and bypass node selection.
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/routing/router.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

namespace {

constexpr uint64_t kGB = 1024ULL * 1024 * 1024;

std::map<TierType, TierCapacity> MakeDramCaps(uint64_t total = 8 * kGB) {
  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = {total, total};
  return caps;
}

}  // namespace

// Indexed keys are marked already_exists; unknown keys still routed.
TEST(RouterDedup, BatchRoutePutMarksAlreadyExistsForIndexedKey) {
  GlobalBlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  Router router(index, registry);

  ASSERT_TRUE(registry.RegisterClient("node-a", "node-a:1", MakeDramCaps(),
                                      /*peer_address=*/"node-a:peer"));
  ASSERT_EQ(
      index.ApplyEvents("node-a", {KvEvent{KvEvent::Kind::ADD, "key-X", TierType::DRAM, 4096}}),
      1u);

  std::vector<std::string> keys{"key-X", "key-Y"};
  std::vector<uint64_t> sizes{4096, 4096};
  std::unordered_set<std::string> excludes;

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_TRUE(results[0]->node_id.empty());

  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[1]->node_id, "node-a");
}

// already_exists wins over no-alive-client: caller drops Put even if
// registry is empty (some other node owns the key).
TEST(RouterDedup, BatchRoutePutAlreadyExistsBypassesNoAliveClient) {
  GlobalBlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  Router router(index, registry);

  ASSERT_EQ(
      index.ApplyEvents("node-a", {KvEvent{KvEvent::Kind::ADD, "key-X", TierType::DRAM, 4096}}),
      1u);

  std::vector<std::string> keys{"key-X", "key-Y"};
  std::vector<uint64_t> sizes{4096, 4096};
  std::unordered_set<std::string> excludes;

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_FALSE(results[1].has_value());  // distinct from kAlreadyExists
}

TEST(RouterDedup, ExternalOnlyLocationDoesNotDedupPut) {
  GlobalBlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  Router router(index, registry);

  ASSERT_TRUE(registry.RegisterClient("node-a", "node-a:1", MakeDramCaps(),
                                      /*peer_address=*/"node-a:peer"));
  ASSERT_EQ(index.ApplyEvents("node-a", {KvEvent{KvEvent::Kind::ADD, "key-X", TierType::DRAM, 0,
                                                 LocationOwner::EXTERNAL_HICACHE}}),
            1u);

  std::vector<std::string> keys{"key-X"};
  std::vector<uint64_t> sizes{4096};
  std::unordered_set<std::string> excludes;

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[0]->node_id, "node-a");
}

}  // namespace mori::umbp
