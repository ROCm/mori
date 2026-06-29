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

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/in_memory_master_metadata_store.h"
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

ClientRegistration MakeRegistration(const std::string& node_id, const std::string& node_address,
                                    const std::string& peer_address) {
  ClientRegistration reg;
  reg.node_id = node_id;
  reg.node_address = node_address;
  reg.tier_capacities = MakeDramCaps();
  reg.peer_address = peer_address;
  return reg;
}

// Register `node_id` ALIVE and apply one ADD event for `key` so it has a block
// location in the store. Under the merged store a location can only be created
// through an ApplyHeartbeat from a registered (alive) node — locations no
// longer exist independently of a client record the way the old
// GlobalBlockIndex allowed.
void RegisterWithKey(InMemoryMasterMetadataStore& store, const std::string& node_id,
                     const std::string& key, std::chrono::system_clock::time_point now) {
  ASSERT_TRUE(store.RegisterClient(MakeRegistration(node_id, node_id + ":1", node_id + ":peer"),
                                   now, std::chrono::seconds{30}));
  auto hb = store.ApplyHeartbeat(node_id, /*seq=*/1, now, MakeDramCaps(),
                                 {KvEvent{KvEvent::Kind::ADD, key, TierType::DRAM, 4096}},
                                 /*is_full_sync=*/false);
  ASSERT_EQ(hb.status, HeartbeatResult::APPLIED);
}

}  // namespace

// Indexed keys are marked already_exists; unknown keys still routed.
TEST(RouterDedup, BatchRoutePutMarksAlreadyExistsForIndexedKey) {
  const auto now = std::chrono::system_clock::now();
  InMemoryMasterMetadataStore store;
  Router router(store);

  RegisterWithKey(store, "node-a", "key-X", now);

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

// already_exists wins over an unroutable Put: an existing key is marked
// kAlreadyExists even when no node can accept the write.  In the old design
// "no node" meant an empty registry while a foreign node owned the key; under
// the merged store a location can't outlive its alive owner, so the
// unroutable condition is expressed by excluding the only candidate node.  The
// property under test is unchanged: dedup wins over node selection.
TEST(RouterDedup, BatchRoutePutAlreadyExistsBypassesUnroutablePut) {
  const auto now = std::chrono::system_clock::now();
  InMemoryMasterMetadataStore store;
  Router router(store);

  RegisterWithKey(store, "node-a", "key-X", now);

  std::vector<std::string> keys{"key-X", "key-Y"};
  std::vector<uint64_t> sizes{4096, 4096};
  std::unordered_set<std::string> excludes{"node-a"};  // no routable target left

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_FALSE(results[1].has_value());  // distinct from kAlreadyExists
}

// Single-key RoutePut delegates to the batch path, so master-side dedup applies:
// an indexed key returns kAlreadyExists while an unknown key still routes.  This
// locks in the delegation; without it RoutePut would silently skip dedup.
TEST(RouterDedup, RoutePutMarksAlreadyExistsForIndexedKey) {
  const auto now = std::chrono::system_clock::now();
  InMemoryMasterMetadataStore store;
  Router router(store);

  RegisterWithKey(store, "node-a", "key-X", now);

  std::unordered_set<std::string> excludes;

  auto dedup = router.RoutePut("key-X", "requester", 4096, excludes);
  ASSERT_TRUE(dedup.has_value());
  EXPECT_EQ(dedup->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_TRUE(dedup->node_id.empty());

  auto routed = router.RoutePut("key-Y", "requester", 4096, excludes);
  ASSERT_TRUE(routed.has_value());
  EXPECT_EQ(routed->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(routed->node_id, "node-a");
}

// Dedup hits never enter the planner, so they consume no projected capacity.
// node-a fits exactly one 6GB block on DRAM.  key-X is an indexed dedup hit
// sized 6GB; if dedup had deducted capacity, node-a would have 0 left and the
// new key-Y could not route.  Since dedup does not deduct, key-Y still routes.
TEST(RouterDedup, BatchRoutePutDedupDoesNotConsumeCapacity) {
  const auto now = std::chrono::system_clock::now();
  InMemoryMasterMetadataStore store;
  Router router(store);

  // node-a holds exactly one 6GB slot on DRAM, already occupied by key-X.
  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = {6 * kGB, 6 * kGB};
  ClientRegistration reg = MakeRegistration("node-a", "node-a:1", "node-a:peer");
  reg.tier_capacities = caps;
  ASSERT_TRUE(store.RegisterClient(reg, now, std::chrono::seconds{30}));
  auto hb = store.ApplyHeartbeat("node-a", /*seq=*/1, now, caps,
                                 {KvEvent{KvEvent::Kind::ADD, "key-X", TierType::DRAM, 6 * kGB}},
                                 /*is_full_sync=*/false);
  ASSERT_EQ(hb.status, HeartbeatResult::APPLIED);

  std::vector<std::string> keys{"key-X", "key-Y"};
  std::vector<uint64_t> sizes{6 * kGB, 6 * kGB};
  std::unordered_set<std::string> excludes;

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);

  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[1]->node_id, "node-a");
  EXPECT_EQ(results[1]->tier, TierType::DRAM);
}

// Projected capacity is batch-local: BatchRoutePut must not write the
// deductions back to the store's real client capacity.
TEST(RouterDedup, BatchRoutePutDoesNotWriteBackProjectedCapacity) {
  const auto now = std::chrono::system_clock::now();
  InMemoryMasterMetadataStore store;
  Router router(store);

  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = {80 * kGB, 10 * kGB};
  ClientRegistration reg = MakeRegistration("node-a", "node-a:1", "node-a:peer");
  reg.tier_capacities = caps;
  ASSERT_TRUE(store.RegisterClient(reg, now, std::chrono::seconds{30}));

  std::vector<std::string> keys{"key-A", "key-B"};
  std::vector<uint64_t> sizes{1 * kGB, 2 * kGB};
  std::unordered_set<std::string> excludes;

  auto results = router.BatchRoutePut(keys, "requester", sizes, excludes);
  ASSERT_EQ(results.size(), 2u);
  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kRouted);

  auto rec = store.GetClient("node-a");
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->tier_capacities.at(TierType::DRAM).available_bytes, 10 * kGB);
}

}  // namespace mori::umbp
