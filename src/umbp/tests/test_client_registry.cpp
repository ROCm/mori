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
// Membership-ledger unit tests for ClientRegistry: registration / re-
// registration semantics, capacity round-trip, heartbeat status, and the
// background reaper that expires silent nodes.  These exercise the registry
// in isolation (no GlobalBlockIndex / RPC), complementing the index- and
// external-kv-focused suites.  In the master-as-advisor design the registry
// stores only membership + the capacities a peer last reported, so the
// assertions below check reported values verbatim rather than any allocator-
// derived view.
#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

std::map<TierType, TierCapacity> Caps(uint64_t total, uint64_t available) {
  return {{TierType::HBM, TierCapacity{total, available}}};
}

const ClientRecord* FindClient(const std::vector<ClientRecord>& clients, const std::string& id) {
  for (const auto& c : clients) {
    if (c.node_id == id) return &c;
  }
  return nullptr;
}

// Drive the current 7-arg Heartbeat with no events — the membership-keepalive
// path the reaper cares about.
ClientStatus Beat(ClientRegistry& registry, const std::string& node_id,
                  const std::map<TierType, TierCapacity>& caps) {
  uint64_t acked = 0;
  bool need_full = false;
  return registry.Heartbeat(node_id, caps, /*bundles=*/{}, /*is_full_sync=*/false,
                            /*delta_seq_baseline=*/0, &acked, &need_full);
}

template <typename Predicate>
bool WaitUntil(Predicate&& predicate, std::chrono::milliseconds timeout,
               std::chrono::milliseconds poll = std::chrono::milliseconds(100)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(poll);
  }
  return predicate();
}

// heartbeat_ttl * max_missed_heartbeats == 1s, so a node ages out ~1s after
// its last heartbeat.  reaper_interval keeps the sweep responsive.
ClientRegistryConfig FastExpiryConfig() {
  ClientRegistryConfig config;
  config.heartbeat_ttl = std::chrono::seconds(1);
  config.max_missed_heartbeats = 1;
  config.reaper_interval = std::chrono::seconds(1);
  return config;
}

}  // namespace

// --- Registration / membership ----------------------------------------------

TEST(ClientRegistryTest, RegisterSingle) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("node-1", "127.0.0.1:8080", Caps(80, 64)));
  EXPECT_EQ(registry.ClientCount(), 1u);
  EXPECT_TRUE(registry.IsClientAlive("node-1"));
}

TEST(ClientRegistryTest, RegisterMultiple) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "127.0.0.1:1001", Caps(100, 90)));
  EXPECT_TRUE(registry.RegisterClient("c2", "127.0.0.1:1002", Caps(110, 80)));
  EXPECT_TRUE(registry.RegisterClient("c3", "127.0.0.1:1003", Caps(120, 70)));

  EXPECT_EQ(registry.ClientCount(), 3u);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
  EXPECT_TRUE(registry.IsClientAlive("c2"));
  EXPECT_TRUE(registry.IsClientAlive("c3"));
}

TEST(ClientRegistryTest, GetAliveClientsReportsMembershipAndCapacities) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "host-a:8080", Caps(80, 64)));
  EXPECT_TRUE(registry.RegisterClient("c2", "host-b:8080", Caps(96, 32)));

  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 2u);

  const ClientRecord* c1 = FindClient(clients, "c1");
  const ClientRecord* c2 = FindClient(clients, "c2");
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);

  EXPECT_EQ(c1->node_address, "host-a:8080");
  EXPECT_EQ(c2->node_address, "host-b:8080");
  EXPECT_EQ(c1->status, ClientStatus::ALIVE);
  EXPECT_EQ(c2->status, ClientStatus::ALIVE);

  // Master stores the peer-reported capacities verbatim.
  ASSERT_TRUE(c1->tier_capacities.count(TierType::HBM) > 0);
  ASSERT_TRUE(c2->tier_capacities.count(TierType::HBM) > 0);
  EXPECT_EQ(c1->tier_capacities.at(TierType::HBM).total_bytes, 80u);
  EXPECT_EQ(c1->tier_capacities.at(TierType::HBM).available_bytes, 64u);
  EXPECT_EQ(c2->tier_capacities.at(TierType::HBM).available_bytes, 32u);
}

// The lightweight peer view maps node->peer (no capacity) and reflects
// membership changes.
TEST(ClientRegistryTest, AlivePeerViewTracksMembership) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "host-a:8080", Caps(80, 64),
                                      /*peer_address=*/"peer-a:9000"));

  auto v1 = registry.GetAlivePeerView();
  EXPECT_EQ(v1.size(), 1u);
  ASSERT_EQ(v1.count("c1"), 1u);
  EXPECT_EQ(v1.at("c1"), "peer-a:9000");

  // Membership change -> reflected in a freshly built view.
  EXPECT_TRUE(registry.RegisterClient("c2", "host-b:8080", Caps(96, 32),
                                      /*peer_address=*/"peer-b:9000"));
  auto v2 = registry.GetAlivePeerView();
  EXPECT_EQ(v2.size(), 2u);
  EXPECT_EQ(v2.at("c2"), "peer-b:9000");
}

// The peer view carries no capacity, so a capacity-only heartbeat leaves its
// contents (node -> peer) unchanged.
TEST(ClientRegistryTest, PeerViewIgnoresCapacity) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64), /*peer_address=*/"peer-a"));

  auto p1 = registry.GetAlivePeerView();
  EXPECT_EQ(Beat(registry, "c1", Caps(80, 8)), ClientStatus::ALIVE);
  auto p2 = registry.GetAlivePeerView();
  EXPECT_EQ(p1, p2);
}

// AliveClientCount counts only ALIVE nodes and tracks membership changes.
TEST(ClientRegistryTest, AliveClientCountTracksMembership) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_EQ(registry.AliveClientCount(), 0u);

  EXPECT_TRUE(registry.RegisterClient("c1", "addr-1", Caps(80, 64)));
  EXPECT_TRUE(registry.RegisterClient("c2", "addr-2", Caps(96, 32)));
  EXPECT_EQ(registry.AliveClientCount(), 2u);

  registry.UnregisterClient("c1");
  EXPECT_EQ(registry.AliveClientCount(), 1u);
}

TEST(ClientRegistryTest, ReRegisterAliveRejected) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr-1", Caps(80, 64)));
  // A live node may not silently take over its own id with a new address.
  EXPECT_FALSE(registry.RegisterClient("c1", "addr-2", Caps(80, 32)));

  EXPECT_EQ(registry.ClientCount(), 1u);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_address, "addr-1");  // original record untouched
}

TEST(ClientRegistryTest, ReRegisterExpiredAllowed) {
  // No reaper here: the aged-out branch in RegisterClient (now - last_heartbeat
  // > expiry) must accept the re-registration on its own.
  ClientRegistry registry(FastExpiryConfig());
  EXPECT_TRUE(registry.RegisterClient("c1", "addr-1", Caps(80, 64)));

  const bool reregistered =
      WaitUntil([&registry] { return registry.RegisterClient("c1", "addr-2", Caps(80, 32)); },
                std::chrono::seconds(5));
  EXPECT_TRUE(reregistered);

  EXPECT_EQ(registry.ClientCount(), 1u);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_address, "addr-2");  // new address wins
  EXPECT_EQ(clients[0].status, ClientStatus::ALIVE);
}

// --- Unregister --------------------------------------------------------------

TEST(ClientRegistryTest, UnregisterExisting) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));

  registry.UnregisterClient("c1");
  EXPECT_EQ(registry.ClientCount(), 0u);
  EXPECT_FALSE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, UnregisterUnknownIsNoop) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));

  registry.UnregisterClient("nonexistent");
  EXPECT_EQ(registry.ClientCount(), 1u);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, UnregisterTwiceIsSafe) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));

  registry.UnregisterClient("c1");
  registry.UnregisterClient("c1");
  EXPECT_EQ(registry.ClientCount(), 0u);
}

// --- Heartbeat ---------------------------------------------------------------

TEST(ClientRegistryTest, HeartbeatAliveReplacesCapacities) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));

  EXPECT_EQ(Beat(registry, "c1", Caps(80, 16)), ClientStatus::ALIVE);
  EXPECT_TRUE(registry.IsClientAlive("c1"));

  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  // The most recent heartbeat's capacities replace the stored values.
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 16u);
}

TEST(ClientRegistryTest, HeartbeatUnknownReturnsUnknown) {
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_EQ(Beat(registry, "nonexistent", Caps(80, 48)), ClientStatus::UNKNOWN);
}

// --- Reaper ------------------------------------------------------------------

TEST(ClientRegistryTest, ReaperExpiresIdleClient) {
  ClientRegistry registry(FastExpiryConfig());
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));
  registry.StartReaper();

  const bool reaped =
      WaitUntil([&registry] { return registry.ClientCount() == 0; }, std::chrono::seconds(6));

  registry.StopReaper();
  EXPECT_TRUE(reaped);
  EXPECT_FALSE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, ReaperKeepsClientAliveWithHeartbeats) {
  ClientRegistry registry(FastExpiryConfig());
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));
  registry.StartReaper();

  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
    EXPECT_EQ(Beat(registry, "c1", Caps(80, 48)), ClientStatus::ALIVE);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  registry.StopReaper();
  EXPECT_EQ(registry.ClientCount(), 1u);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
}

TEST(ClientRegistryTest, ReaperSelectiveExpiry) {
  ClientRegistry registry(FastExpiryConfig());
  EXPECT_TRUE(registry.RegisterClient("c1", "addr-1", Caps(80, 64)));
  EXPECT_TRUE(registry.RegisterClient("c2", "addr-2", Caps(80, 64)));
  registry.StartReaper();

  // Keep c1 fed; let c2 go silent.  c2 must be reaped while c1 survives.
  const bool reached = WaitUntil(
      [&registry] {
        Beat(registry, "c1", Caps(80, 48));
        return registry.IsClientAlive("c1") && !registry.IsClientAlive("c2");
      },
      std::chrono::seconds(6), std::chrono::milliseconds(200));

  registry.StopReaper();
  EXPECT_TRUE(reached);
  EXPECT_TRUE(registry.IsClientAlive("c1"));
  EXPECT_FALSE(registry.IsClientAlive("c2"));
}

TEST(ClientRegistryTest, StopReaperWhenNeverStarted) {
  ClientRegistry registry(ClientRegistryConfig{});
  registry.StopReaper();  // must not hang or crash
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

TEST(ClientRegistryTest, DestructorStopsRunningReaper) {
  ClientRegistry registry(ClientRegistryConfig{});
  registry.StartReaper();
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", Caps(80, 64)));
  // Falling out of scope must join the reaper thread cleanly.
}

}  // namespace mori::umbp
