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

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "mori/application/utils/cpu_affinity.hpp"
#include "umbp/common/host_registration.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {

inline constexpr size_t kPageSize = 4096;
inline constexpr size_t kObjectSize = 2 * kPageSize;
inline constexpr size_t kCallerCapacity = 16 * kPageSize;
inline constexpr size_t kTargetCapacity = 256 * kPageSize;
inline constexpr size_t kScratchSize = kObjectSize;  // Forces one object per sub-batch.

// The kernel dereferences the tier directly, so it needs the host region
// registered for GPU access; without it the client uses the copy engine.
inline bool GatherPathAvailable() {
  static const bool available = [] {
    std::vector<char> probe(64 * 1024);
    return HostTierRegistration(probe.data(), probe.size()).GatherableOn(0);
  }();
  return available;
}

inline bool WaitForExists(PoolClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (client->Exists(key)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  return client->Exists(key);
}

inline bool WaitForExists(IUMBPClient* client, const std::string& key) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto found = client->BatchExists({key});
    if (found.size() == 1 && found[0]) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  const auto found = client->BatchExists({key});
  return found.size() == 1 && found[0];
}

class PoolClientRangesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    // The registry derives a 2 s client heartbeat from this TTL: short enough
    // for normal publication tests, while the 12 s expiry window keeps a stopped
    // caller alive through the stale-self assertion.
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds(4);
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    master_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 100 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    master_address_ = "localhost:" + std::to_string(master_->GetBoundPort());

    caller_get_scratch_.resize(kScratchSize);
    caller_put_scratch_.resize(kScratchSize);
    target_get_scratch_.resize(kScratchSize);
    target_put_scratch_.resize(kScratchSize);
    caller_ =
        MakeClient("ranges-caller", kCallerCapacity, caller_get_scratch_, caller_put_scratch_);
    target_ =
        MakeClient("ranges-target", kTargetCapacity, target_get_scratch_, target_put_scratch_);
  }

  void TearDown() override {
    if (caller_) caller_->Shutdown();
    if (target_) target_->Shutdown();
    caller_.reset();
    target_.reset();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    master_.reset();
  }

  std::unique_ptr<PoolClient> MakeClient(const std::string& node_id, size_t dram_capacity,
                                         std::vector<char>& get_scratch,
                                         std::vector<char>& put_scratch,
                                         bool locality_prefetch = true, bool direct = false) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_address_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    // A peer service is required for the node to register a peer_address and
    // serve remote AllocateSlot/CommitSlot RPCs; without one, remote access
    // fails with "peer service connection unavailable".  Let gRPC choose the
    // port: probing for a free one and closing the socket before PoolClient
    // binds it races with everything else on a shared CI host.
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPageSize;
    // Smaller than one object: ranged remote I/O can pass only through the
    // registered arena's zero-copy descriptor, never the legacy staging path.
    cfg.staging_buffer_size = 1024;
    // The DRAM backend self-allocates its own pool (refactor Phase 2b), so the
    // test hands it a size rather than a buffer.
    cfg.dram.buffer_sizes = {dram_capacity};
    // Separate GET and PUT arenas so remote ranged get/put run concurrently.
    cfg.ranged_get_scratch = {{get_scratch.data(), get_scratch.size()}};
    cfg.ranged_put_scratch = {{put_scratch.data(), put_scratch.size()}};
    cfg.cache_remote_fetches = false;
    cfg.ranged_locality_prefetch = locality_prefetch;
    // Off unless a test asks: the arena tests read the arena as their evidence.
    cfg.ranged_direct = direct;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(client->Init());
    EXPECT_TRUE(client->RegisterMemory(get_scratch.data(), get_scratch.size()));
    EXPECT_TRUE(client->RegisterMemory(put_scratch.data(), put_scratch.size()));
    return client;
  }

  // MakeClient with its arenas laid out by NUMA node instead: one GET and one PUT
  // buffer per entry of `nodes`, in that order, and no locality prefetch, so every
  // read of a remote key goes through an arena.
  std::unique_ptr<PoolClient> MakeNumaArenaClient(const std::string& node_id,
                                                  const std::vector<int>& nodes,
                                                  std::vector<std::vector<char>>& get_arenas,
                                                  std::vector<std::vector<char>>& put_arenas) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_address_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPageSize;
    cfg.staging_buffer_size = 1024;
    cfg.dram.buffer_sizes = {kCallerCapacity};
    for (size_t i = 0; i < nodes.size(); ++i) {
      cfg.ranged_get_scratch.push_back({get_arenas[i].data(), get_arenas[i].size(), nodes[i]});
      cfg.ranged_put_scratch.push_back({put_arenas[i].data(), put_arenas[i].size(), nodes[i]});
    }
    cfg.ranged_direct = false;
    cfg.cache_remote_fetches = false;
    cfg.ranged_locality_prefetch = false;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(client->Init());
    for (auto* arenas : {&get_arenas, &put_arenas}) {
      for (auto& arena : *arenas) EXPECT_TRUE(client->RegisterMemory(arena.data(), arena.size()));
    }
    return client;
  }

  // A full DistributedClient (not the bare PoolClient the other cases drive),
  // because the opt-in this asserts lives in DistributedClient's constructor
  // and in SupportsRangedIO(), neither of which PoolClient sees.
  UMBPConfig DistributedConfig(const std::string& node_id, size_t ranged_scratch_size) {
    UMBPConfig config;
    config.dram.capacity_bytes = 4 * kTargetCapacity;
    config.dram.use_hugepages = false;
    config.ssd.enabled = false;

    UMBPDistributedConfig distributed;
    distributed.master_config.node_id = node_id;
    distributed.master_config.node_address = "127.0.0.1";
    distributed.master_config.master_address = master_address_;
    distributed.io_engine.host = "0.0.0.0";
    distributed.io_engine.port = 0;
    distributed.auto_peer_service_port = true;
    distributed.dram_page_size = kPageSize;
    distributed.staging_buffer_size = 4 * kObjectSize;
    distributed.ranged_scratch_size = ranged_scratch_size;
    distributed.cache_remote_fetches = false;
    config.distributed = std::move(distributed);
    return config;
  }

  // Publish a replica on one specific node, bypassing RoutePut's de-duplication
  // so two nodes can hold the same key. Writes through the backend's published
  // endpoint rather than a caller-owned buffer, since the medium owns its pool.
  void PutLocalReplica(PoolClient* client, const std::string& key,
                       const std::vector<char>& object) {
    auto* backend = client->Backends().Get(TierType::DRAM);
    ASSERT_NE(backend, nullptr);
    auto allocated = backend->BatchAllocate({AllocateRequest{key, object.size()}}).front();
    ASSERT_EQ(allocated.outcome, AllocateOutcome::kSuccessAllocated);

    size_t copied = 0;
    for (const auto& page : allocated.pages) {
      TransferRef buf = backend->BufferRef(page.buffer_index);
      ASSERT_TRUE(buf.HasHostPtr());
      const size_t bytes = std::min(kPageSize, object.size() - copied);
      const size_t offset = static_cast<size_t>(page.page_index) * kPageSize;
      ASSERT_LE(offset + bytes, buf.size);
      std::memcpy(static_cast<char*>(buf.host_ptr) + offset, object.data() + copied, bytes);
      copied += bytes;
    }
    ASSERT_EQ(copied, object.size());

    auto committed = backend->BatchCommit({CommitRequest{allocated.slot_id, key}}).front();
    ASSERT_TRUE(committed.success);
    ASSERT_EQ(committed.bytes_committed, object.size());
  }

  // Publish `object` on the target only, so the caller's sole route to it is
  // remote.  PutLocalReplica writes the target's medium directly, which is what
  // keeps a copy off the caller.
  void SeedRemoteObject(const std::string& key, const std::vector<char>& object) {
    PutLocalReplica(target_.get(), key, object);
    target_->Master().FlushHeartbeat();
    caller_->Master().FlushHeartbeat();
    ASSERT_TRUE(WaitForExists(caller_.get(), key));
  }

  // Is `key` readable from this node's own medium?  The same question phase 1
  // of BatchGetRanges asks, and therefore the thing that decides whether a
  // later read goes remote.
  static bool LocallyResident(PoolClient* client, const std::string& key) {
    auto* backend = client->Backends().Get(client->Medium());
    return backend != nullptr &&
           backend->BatchResolve({key}, /*include_descs=*/false).front().found;
  }

  static bool WaitLocallyResident(PoolClient* client, const std::string& key,
                                  std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (LocallyResident(client, key)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return LocallyResident(client, key);
  }

  // Two DRAM instances under a tiered policy whose entry tier is the SECOND
  // one registered, so selecting by TierType and asking the policy disagree.
  std::unique_ptr<PoolClient> MakeTieredClient(const std::string& node_id) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_address_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPageSize;
    cfg.staging_buffer_size = 1024;
    tiered_get_scratch_.resize(kScratchSize);
    tiered_put_scratch_.resize(kScratchSize);
    cfg.ranged_get_scratch = {{tiered_get_scratch_.data(), tiered_get_scratch_.size()}};
    cfg.ranged_put_scratch = {{tiered_put_scratch_.data(), tiered_put_scratch_.size()}};
    cfg.cache_remote_fetches = false;

    BackendInstanceConfig cold;
    cold.name = "cold";
    cold.tier = TierType::DRAM;
    cold.dram.buffer_sizes = {kTargetCapacity};
    BackendInstanceConfig hot = cold;
    hot.name = "hot";
    cfg.backends = {cold, hot};

    cfg.placement_policy = PoolPlacementPolicy::TIERED;
    LogicalTierConfig hot_tier;
    hot_tier.name = "hot";
    hot_tier.members = {{"hot", 1}};
    hot_tier.entry = true;
    hot_tier.offload_to = {"cold"};
    LogicalTierConfig cold_tier;
    cold_tier.name = "cold";
    cold_tier.members = {{"cold", 1}};
    cfg.logical_tiers = {hot_tier, cold_tier};

    auto client = std::make_unique<PoolClient>(std::move(cfg));
    EXPECT_TRUE(client->Init());
    EXPECT_TRUE(client->RegisterMemory(tiered_get_scratch_.data(), tiered_get_scratch_.size()));
    EXPECT_TRUE(client->RegisterMemory(tiered_put_scratch_.data(), tiered_put_scratch_.size()));
    return client;
  }

  bool WaitForRoute(PoolClient* client, const std::string& key,
                    const std::unordered_set<std::string>& excludes,
                    const std::string& expected_node) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      std::vector<std::optional<RouteGetResult>> routes;
      const auto status = client->Master().BatchRouteGet({key}, excludes, &routes);
      if (status.ok() && routes.size() == 1 && routes[0].has_value() &&
          routes[0]->node_id == expected_node) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::string master_address_;
  std::unique_ptr<PoolClient> caller_;
  std::unique_ptr<PoolClient> target_;
  std::vector<char> caller_get_scratch_;
  std::vector<char> caller_put_scratch_;
  std::vector<char> target_get_scratch_;
  std::vector<char> target_put_scratch_;
  std::vector<char> tiered_get_scratch_;
  std::vector<char> tiered_put_scratch_;
};

}  // namespace mori::umbp
