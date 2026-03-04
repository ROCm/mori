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
#include <vector>

#include "umbp/client_registry.h"

namespace mori::umbp {
namespace {

std::map<TierType, TierCapacity> MakeTierCapacities(uint64_t total_bytes,
                                                    uint64_t available_bytes) {
  return {{TierType::HBM, TierCapacity{total_bytes, available_bytes}}};
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
  ClientRegistry registry(ClientRegistryConfig{});

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
  EXPECT_EQ(c1->tier_capacities.at(TierType::HBM).available_bytes, 64u);
  EXPECT_EQ(c2->tier_capacities.at(TierType::HBM).available_bytes, 32u);
}

TEST(ClientRegistryTest, ReRegisterAliveRejected) {
  ClientRegistry registry(ClientRegistryConfig{});

  EXPECT_TRUE(registry.RegisterClient("c1", "a1", MakeTierCapacities(80, 64)));
  EXPECT_FALSE(registry.RegisterClient("c1", "a2", MakeTierCapacities(80, 32)));

  EXPECT_EQ(registry.ClientCount(), 1u);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_id, "c1");
  EXPECT_EQ(clients[0].node_address, "a1");
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 64u);
}

TEST(ClientRegistryTest, ReRegisterExpiredAllowed) {
  ClientRegistryConfig config;
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
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 32u);
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
  ClientRegistry registry(ClientRegistryConfig{});
  EXPECT_TRUE(registry.RegisterClient("c1", "addr", MakeTierCapacities(80, 80)));

  ASSERT_EQ(registry.Heartbeat("c1", MakeTierCapacities(80, 32)), ClientStatus::ALIVE);
  const auto clients = registry.GetAliveClients();
  ASSERT_EQ(clients.size(), 1u);
  ASSERT_TRUE(clients[0].tier_capacities.count(TierType::HBM) > 0);
  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 32u);
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

}  // namespace mori::umbp
