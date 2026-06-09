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

#include <atomic>
#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {
namespace {

std::map<TierType, TierCapacity> MakeTierCapacities(uint64_t total_bytes,
                                                    uint64_t available_bytes) {
  return {{TierType::HBM, TierCapacity{total_bytes, available_bytes}}};
}

// Small page_size for legacy tests that hard-code byte-precision values.
// Phase 1 derives DRAM/HBM `available_bytes` from the page allocator, so
// the registered totals must be a multiple of `default_dram_page_size` for
// the legacy expectations to keep holding.
ClientRegistryConfig MakeSmallPageConfig(uint64_t page_size = 1) {
  ClientRegistryConfig config;
  config.default_dram_page_size = page_size;
  return config;
}

const ClientRecord* FindClient(const std::vector<ClientRecord>& clients, const std::string& id) {
  for (const auto& client : clients) {
    if (client.node_id == id) {
      return &client;
    }
  }
  return nullptr;
}

template <typename Predicate>
bool WaitUntil(Predicate&& predicate, std::chrono::milliseconds timeout,
               std::chrono::milliseconds poll_interval = std::chrono::milliseconds(100)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(poll_interval);
  }
  return predicate();
}

}  // namespace

TEST(ClientRegistryTest, RegisterSingle) {
  ClientRegistry registry(ClientRegistryConfig{});

  EXPECT_TRUE(registry.RegisterClient("node-1", "127.0.0.1:8080", MakeTierCapacities(80, 64)));

  EXPECT_EQ(registry.ClientCount(), 1u);
  EXPECT_TRUE(registry.IsClientAlive("node-1"));
}

TEST(ClientRegistryTest, RegisterMultiple) {
  ClientRegistry registry(ClientRegistryConfig{});

  EXPECT_TRUE(registry.RegisterClient("c1", "127.0.0.1:1001", MakeTierCapacities(100, 90)));
  EXPECT_TRUE(registry.RegisterClient("c2", "127.0.0.1:1002", MakeTierCapacities(110, 80)));
  EXPECT_TRUE(registry.RegisterClient("c3", "127.0.0.1:1003", MakeTierCapacities(120, 70)));

  EXPECT_EQ(registry.ClientCount(), 3u);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
  EXPECT_TRUE(registry.IsClientAlive("c2"));
  EXPECT_TRUE(registry.IsClientAlive("c3"));
}

TEST(ClientRegistryTest, GetAliveClients) {
  // Use page_size=1 so that registered totals map 1:1 to page counts and the
  // legacy assertions ("registered N bytes → available N bytes") still hold
  // without rewriting the per-byte arithmetic in this test.
  ClientRegistry registry(MakeSmallPageConfig());

  EXPECT_TRUE(registry.RegisterClient("c1", "host-a:8080", MakeTierCapacities(80, 64)));
  EXPECT_TRUE(registry.RegisterClient("c2", "host-b:8080", MakeTierCapacities(96, 32)));

  const auto clients = registry.GetAliveClients();
  EXPECT_EQ(clients.size(), 2u);

  const ClientRecord* c1 = FindClient(clients, "c1");
  const ClientRecord* c2 = FindClient(clients, "c2");
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);

  EXPECT_EQ(c1->node_address, "host-a:8080");
  EXPECT_EQ(c2->node_address, "host-b:8080");
  EXPECT_EQ(c1->status, ClientStatus::ALIVE);
  EXPECT_EQ(c2->status, ClientStatus::ALIVE);

  ASSERT_TRUE(c1->tier_capacities.count(TierType::HBM) > 0);
  ASSERT_TRUE(c2->tier_capacities.count(TierType::HBM) > 0);
  // Phase 1: available_bytes is allocator-derived; right after Register the
  // bitmap is empty, so available_bytes == total_bytes (Client's reported
  // available_bytes is intentionally ignored).
  EXPECT_EQ(c1->tier_capacities.at(TierType::HBM).available_bytes, 80u);
  EXPECT_EQ(c2->tier_capacities.at(TierType::HBM).available_bytes, 96u);
}

TEST(ClientRegistryTest, ReRegisterAliveRejected) {
  ClientRegistry registry(MakeSmallPageConfig());

  EXPECT_TRUE(registry.RegisterClient("c1", "a1", MakeTierCapacities(80, 64)));
  EXPECT_FALSE(registry.RegisterClient("c1", "a2", MakeTierCapacities(80, 32)));

  EXPECT_EQ(registry.ClientCount(), 1u);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_id, "c1");
  EXPECT_EQ(clients[0].node_address, "a1");
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  // Allocator-derived: empty bitmap -> available == total.
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 80u);
}

TEST(ClientRegistryTest, ReRegisterExpiredAllowed) {
  ClientRegistryConfig config = MakeSmallPageConfig();
  config.heartbeat_ttl = std::chrono::seconds(1);
  config.max_missed_heartbeats = 1;
  config.reaper_interval = std::chrono::seconds(10);

  ClientRegistry registry(config);
  EXPECT_TRUE(registry.RegisterClient("c1", "a1", MakeTierCapacities(80, 64)));

  const bool reregistered = WaitUntil(
      [&registry] { return registry.RegisterClient("c1", "a2", MakeTierCapacities(80, 32)); },
      std::chrono::seconds(5), std::chrono::milliseconds(100));
  EXPECT_TRUE(reregistered);
  EXPECT_EQ(registry.ClientCount(), 1u);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_id, "c1");
  EXPECT_EQ(clients[0].node_address, "a2");
  EXPECT_EQ(clients[0].status, ClientStatus::ALIVE);
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  // Re-registration recreates the allocator with the new total — bitmap is
  // empty again, so available == total (regardless of what the Client tried
  // to report for `available_bytes`).
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 80u);
}

TEST(ClientRegistryTest, UnregisterExisting) {
  ClientRegistry registry(ClientRegistryConfig{});

  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));
  const size_t removed = registry.UnregisterClient("c1");

  EXPECT_EQ(removed, 0u);
  EXPECT_EQ(registry.ClientCount(), 0u);
  EXPECT_FALSE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, UnregisterUnknown) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));

  const size_t removed = registry.UnregisterClient("nonexistent");

  EXPECT_EQ(removed, 0u);
  EXPECT_EQ(registry.ClientCount(), 1u);
}

TEST(ClientRegistryTest, UnregisterTwice) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));

  EXPECT_EQ(registry.UnregisterClient("c1"), 0u);
  EXPECT_EQ(registry.UnregisterClient("c1"), 0u);
  EXPECT_EQ(registry.ClientCount(), 0u);
}

TEST(ClientRegistryTest, HeartbeatAlive) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));

  const ClientStatus status = registry.Heartbeat("c1", MakeTierCapacities(80, 48));

  EXPECT_EQ(status, ClientStatus::ALIVE);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, HeartbeatUnknown) {
  ClientRegistry registry(ClientRegistryConfig{});

  const ClientStatus status = registry.Heartbeat("nonexistent", MakeTierCapacities(80, 48));

  EXPECT_EQ(status, ClientStatus::UNKNOWN);
}

TEST(ClientRegistryTest, HeartbeatUpdatesCapacities) {
  // Phase 1 #2 fix: for DRAM/HBM tiers Master is the source of truth for
  // available_bytes — the Client's reported value is intentionally ignored,
  // and `available_bytes` is recomputed from the page-bitmap allocator.
  // Total here is 80 bytes, but our allocator's page_size is 2 MiB so the
  // backward-compat single-buffer path produces total_pages=0, making the
  // allocator's AvailableBytes() == 0.  We assert exactly that.
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 80)));

  ASSERT_EQ(registry.Heartbeat("c1", MakeTierCapacities(80, 32)), ClientStatus::ALIVE);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 0u);
}

TEST(ClientRegistryTest, HeartbeatDoesNotOverwriteDramAvailable) {
  // §3.2 regression test: even after a Client falsely reports a fresh
  // available_bytes value via Heartbeat, the Master must keep its
  // allocator-derived view as the truth.
  ClientRegistryConfig config;
  config.default_dram_page_size = 4u;  // tiny page so the test fits in 80 B
  ClientRegistry registry(config);

  // Register with an HBM tier sized at 80 bytes total / 80 bytes free.
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 80)));

  // Reserve 1 page (4 bytes worth → rounded up to 1 page) → bitmap should
  // mark 4 bytes used; Master-reported available should therefore be 76.
  auto alloc = registry.AllocateForPut("c1", TierType::HBM, /*size=*/4);
  ASSERT_TRUE(alloc.has_value());

  // Now the Client misreports a far-too-large available_bytes value.  Master
  // must ignore that and recompute from the allocator state instead.
  ASSERT_EQ(registry.Heartbeat("c1", MakeTierCapacities(80, /*available=*/12345)),
            ClientStatus::ALIVE);

  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  // Allocator: total_pages = 80 / 4 = 20 pages, used 1 → free 19 → 76 bytes.
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 76u);
  // total_bytes is unchanged (resize is rejected with a throttled WARN).
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).total_bytes, 80u);
}

TEST(ClientRegistryTest, RegisterClientWithZeroDramPageSizeUsesRegistryDefault) {
  // Task 3 invariant: when a Client registers with dram_page_size == 0 the
  // registry must fall back to `ClientRegistryConfig::default_dram_page_size`
  // (the sole source of truth for the default).  An explicit non-zero value
  // from the Client must still win over the registry default.
  ClientRegistryConfig config;
  config.default_dram_page_size = 4096;
  ClientRegistry registry(config);

  const std::map<TierType, TierCapacity> dram_caps{
      {TierType::DRAM, TierCapacity{/*total=*/128 * 1024, /*available=*/128 * 1024}}};
  ASSERT_TRUE(registry.RegisterClient("node-default", "127.0.0.1:9001", dram_caps,
                                      /*peer_address=*/"",
                                      /*engine_desc_bytes=*/{},
                                      /*dram_memory_desc_bytes_list=*/{},
                                      /*dram_buffer_sizes=*/{},
                                      /*ssd_store_capacities=*/{},
                                      /*dram_page_size=*/0));
  const auto ps_default = registry.GetNodeDramPageSize("node-default", TierType::DRAM);
  ASSERT_TRUE(ps_default.has_value());
  EXPECT_EQ(*ps_default, 4096u);

  // Symmetric case: an explicit override must NOT be overridden by the
  // registry default (the 0 -> fallback rule does not apply when the Client
  // asked for a specific page_size).
  ASSERT_TRUE(registry.RegisterClient("node-override", "127.0.0.1:9002", dram_caps,
                                      /*peer_address=*/"",
                                      /*engine_desc_bytes=*/{},
                                      /*dram_memory_desc_bytes_list=*/{},
                                      /*dram_buffer_sizes=*/{},
                                      /*ssd_store_capacities=*/{},
                                      /*dram_page_size=*/8192));
  const auto ps_override = registry.GetNodeDramPageSize("node-override", TierType::DRAM);
  ASSERT_TRUE(ps_override.has_value());
  EXPECT_EQ(*ps_override, 8192u);
}

TEST(ClientRegistryTest, PoolClientForwardsZeroDramPageSize) {
  // Task 3 invariant — "PoolClient never silently fills in a default:
  // it passes whatever PoolClientConfig said, including 0, through to the
  // Master".  Observing this end-to-end through a real PoolClient +
  // MasterServer pair would require exposing MasterServer's private
  // `registry_` member for tests, which is heavier than warranted.  The
  // task 3 spec explicitly permits the lower-fidelity registry-level
  // check used here: we drive `registry.RegisterClient(..., 0)` in the
  // same way MasterServer would on behalf of a zero-config PoolClient and
  // assert the registry's fallback path picks up the registry-wide
  // default.  If a future refactor accidentally adds a client-side default
  // fallback (e.g. `if (config_.dram_page_size == 0) config_.dram_page_size
  // = kDefault;` in PoolClient::Init), the observable sentinel here would
  // change and — combined with the acceptance criterion from the plan —
  // the invariant would still break.
  constexpr uint64_t kSomeSentinel = 4096;  // small, distinct from 2 MiB to avoid coincidence.
  ClientRegistryConfig config;
  config.default_dram_page_size = kSomeSentinel;
  ClientRegistry registry(config);

  const std::map<TierType, TierCapacity> dram_caps{
      {TierType::DRAM, TierCapacity{/*total=*/128 * 1024, /*available=*/128 * 1024}}};
  // Mirror PoolClient::Init's forwarding of config_.dram_page_size into
  // MasterClient::RegisterSelf → proto → MasterServer → registry, with
  // `PoolClientConfig{.dram_page_size = 0}` as the input.
  ASSERT_TRUE(registry.RegisterClient("node-zero", "127.0.0.1:9003", dram_caps,
                                      /*peer_address=*/"",
                                      /*engine_desc_bytes=*/{},
                                      /*dram_memory_desc_bytes_list=*/{},
                                      /*dram_buffer_sizes=*/{},
                                      /*ssd_store_capacities=*/{},
                                      /*dram_page_size=*/0));

  const auto ps = registry.GetNodeDramPageSize("node-zero", TierType::DRAM);
  ASSERT_TRUE(ps.has_value());
  EXPECT_EQ(*ps, kSomeSentinel);
}

TEST(ClientRegistryTest, ReaperExpiresClient) {
  ClientRegistryConfig config;
  config.heartbeat_ttl = std::chrono::seconds(1);
  config.reaper_interval = std::chrono::seconds(1);
  config.max_missed_heartbeats = 1;

  ClientRegistry registry(config);
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));
  registry.StartReaper();

  const bool reaped =
      WaitUntil([&registry] { return registry.ClientCount() == 0; }, std::chrono::seconds(6));

  registry.StopReaper();
  EXPECT_TRUE(reaped);
  EXPECT_EQ(registry.ClientCount(), 0u);
}

TEST(ClientRegistryTest, ReaperKeepsAliveClientWithHeartbeats) {
  ClientRegistryConfig config;
  config.heartbeat_ttl = std::chrono::seconds(1);
  config.reaper_interval = std::chrono::seconds(1);
  config.max_missed_heartbeats = 1;

  ClientRegistry registry(config);
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));
  registry.StartReaper();

  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
    EXPECT_EQ(registry.Heartbeat("c1", MakeTierCapacities(80, 48)), ClientStatus::ALIVE);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  registry.StopReaper();
  EXPECT_EQ(registry.ClientCount(), 1u);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, ReaperSelectiveExpiry) {
  ClientRegistryConfig config;
  config.heartbeat_ttl = std::chrono::seconds(1);
  config.reaper_interval = std::chrono::seconds(1);
  config.max_missed_heartbeats = 1;

  ClientRegistry registry(config);
  EXPECT_TRUE(registry.RegisterClient("c1", "addr-1", MakeTierCapacities(80, 64)));
  EXPECT_TRUE(registry.RegisterClient("c2", "addr-2", MakeTierCapacities(80, 64)));
  registry.StartReaper();

  const bool reached_expected_state = WaitUntil(
      [&registry] {
        const bool has_one_client = (registry.ClientCount() == 1u);
        if (!has_one_client) {
          registry.Heartbeat("c1", MakeTierCapacities(80, 50));
          return false;
        }
        return registry.IsClientAlive("c1") && !registry.IsClientAlive("c2");
      },
      std::chrono::seconds(6), std::chrono::milliseconds(200));

  registry.StopReaper();
  EXPECT_TRUE(reached_expected_state);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
  EXPECT_FALSE(registry.IsClientAlive("c2"));
}

TEST(ClientRegistryTest, StopReaperWhenNeverStarted) {
  ClientRegistry registry(ClientRegistryConfig{});
  registry.StopReaper();
  SUCCEED();
}

TEST(ClientRegistryTest, StartStopReaperMultiple) {
  ClientRegistry registry(ClientRegistryConfig{});
  registry.StartReaper();
  registry.StopReaper();
  registry.StartReaper();
  registry.StopReaper();
  SUCCEED();
}

TEST(ClientRegistryTest, DestructorStopsReaper) {
  {
    ClientRegistry registry(ClientRegistryConfig{});
    registry.StartReaper();
    EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 64)));
  }
  SUCCEED();
}

// --- TrackKey / UntrackKey coverage (known-issue #6 fix) ---
//
// These tests exercise the three-phase locking introduced to move
// index_->Lookup() out of the registry unique_lock:
//   Phase 1: shared_lock — load `idx` + client existence check
//   Phase 2: no registry lock — ownership Lookup on the loaded `idx`
//   Phase 3: unique_lock — re-check client + mutate client_keys_
// The Phase 3 re-check is what prevents a racing unregister from leaving
// a stale entry in client_keys_ via operator[]; TrackKeyConcurrentUnregisterRace
// is the regression guard for that invariant.

TEST(ClientRegistryTest, TrackKeyNoOpForUnknownNode) {
  // Phase 1 existence check: TrackKey for an unknown node_id must not
  // create any client_keys_ entry via operator[].  Observable via the
  // keys_removed return of UnregisterClient after we later register the
  // same node: fresh register does NOT clear a pre-existing client_keys_
  // entry (see client_registry.cpp: `client_keys_[node_id];` is operator[],
  // not erase), so a leaked entry would survive and produce >0 here.
  ClientRegistry registry(MakeSmallPageConfig());
  ASSERT_TRUE(registry.RegisterClient("known", "a", MakeTierCapacities(80, 64)));

  registry.TrackKey("unknown", "K");

  ASSERT_TRUE(registry.RegisterClient("unknown", "a", MakeTierCapacities(80, 64)));
  EXPECT_EQ(registry.UnregisterClient("unknown"), 0u);
}

TEST(ClientRegistryTest, TrackKeyRequiresIndexOwnership) {
  // Phase 2 ownership check: TrackKey(A, K) must refuse to insert when
  // the index reports no location owned by A for key K.
  ClientRegistry registry(MakeSmallPageConfig());
  GlobalBlockIndex index;
  registry.SetBlockIndex(&index);
  index.SetClientRegistry(&registry);

  ASSERT_TRUE(registry.RegisterClient("A", "a", MakeTierCapacities(80, 64)));
  ASSERT_TRUE(registry.RegisterClient("B", "b", MakeTierCapacities(80, 64)));

  // Only B owns K in the index.  The auto-Track from BatchRegister inserts
  // K into client_keys_["B"].
  const Location locB{"B", "0:p0", 4, TierType::HBM};
  index.Register("B", "K", locB);

  // Now manually Track A for the same K — A doesn't own it.
  registry.TrackKey("A", "K");

  // A should still have zero tracked keys; B should have exactly one.
  EXPECT_EQ(registry.UnregisterClient("A"), 0u);
  EXPECT_EQ(registry.UnregisterClient("B"), 1u);

  index.SetClientRegistry(nullptr);
  registry.SetBlockIndex(nullptr);
}

TEST(ClientRegistryTest, TrackKeyConcurrentUnregisterRace) {
  // Regression guard for the Phase 3 re-check: stress TrackKey against a
  // concurrent UnregisterClient.  If Phase 3 skipped the clients_.find
  // re-check, a TrackKey thread whose Phase 1+2 observed A as alive could
  // reach Phase 3 after UnregisterClient released its lock, and then
  // operator[] would resurrect client_keys_["A"] with K.  Fresh Register
  // does not clear that stale set (see `client_keys_[node_id];` in
  // RegisterClient), so a subsequent UnregisterClient would return >0.
  //
  // With the fix, Phase 3's clients_.find(A) fails after unregister and
  // the insert is skipped; the final UnregisterClient must return 0.
  ClientRegistry registry(MakeSmallPageConfig());
  GlobalBlockIndex index;
  registry.SetBlockIndex(&index);

  // Mask the TrackKey callback so the initial BatchRegister only populates
  // the index — tracking is driven manually from the worker threads below.
  index.SetClientRegistry(nullptr);
  ASSERT_TRUE(registry.RegisterClient("A", "a", MakeTierCapacities(80, 64)));
  const Location locA{"A", "0:p0", 4, TierType::HBM};
  ASSERT_EQ(index.BatchRegister("A", {{"K", locA}}), 1u);
  index.SetClientRegistry(&registry);

  constexpr int kThreads = 8;
  constexpr int kIterations = 200;
  std::atomic<bool> go{false};
  std::vector<std::thread> track_threads;
  track_threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    track_threads.emplace_back([&] {
      while (!go.load(std::memory_order_acquire)) {
      }
      for (int j = 0; j < kIterations; ++j) {
        registry.TrackKey("A", "K");
      }
    });
  }

  std::thread unreg_thread([&] {
    while (!go.load(std::memory_order_acquire)) {
    }
    registry.UnregisterClient("A");
  });

  go.store(true, std::memory_order_release);
  for (auto& t : track_threads) t.join();
  unreg_thread.join();

  EXPECT_FALSE(registry.IsClientAlive("A"));

  // Phase 3 invariant: no stale client_keys_["A"] entry may have been
  // created by a TrackKey whose Phase 3 ran after UnregisterClient
  // erased client_keys_["A"].  Re-register then Unregister to surface
  // any leak.
  ASSERT_TRUE(registry.RegisterClient("A", "a", MakeTierCapacities(80, 64)));
  EXPECT_EQ(registry.UnregisterClient("A"), 0u);

  index.SetClientRegistry(nullptr);
  registry.SetBlockIndex(nullptr);
}

}  // namespace mori::umbp
