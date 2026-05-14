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

#include <string>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// ---- Helpers ----------------------------------------------------------------

static ClientRegistryConfig MakeConfig() {
  ClientRegistryConfig cfg;
  cfg.heartbeat_ttl = std::chrono::seconds{30};
  cfg.reaper_interval = std::chrono::seconds{60};
  cfg.allocation_ttl = std::chrono::seconds{60};
  cfg.max_missed_heartbeats = 3;
  return cfg;
}

static bool RegisterNode(ClientRegistry& reg, const std::string& node_id) {
  return reg.RegisterClient(node_id, "127.0.0.1:9000", {}, "127.0.0.1:9001");
}

// ---- Tests ------------------------------------------------------------------

TEST(ClientRegistryExternalKv, RegisterRejectedWhenNodeNotAlive) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  ExternalKvBlockIndex ekv;
  reg.SetExternalKvBlockIndex(&ekv);

  // "ghost" node never registered — should be silently rejected
  reg.RegisterExternalKvBlocks("ghost-node", {"h1", "h2"}, TierType::DRAM);

  auto matches = ekv.Match({"h1", "h2"});
  EXPECT_TRUE(matches.empty());
}

TEST(ClientRegistryExternalKv, RegisterAcceptedForAliveNode) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  ExternalKvBlockIndex ekv;
  reg.SetExternalKvBlockIndex(&ekv);

  ASSERT_TRUE(RegisterNode(reg, "node-A"));

  reg.RegisterExternalKvBlocks("node-A", {"h1", "h2"}, TierType::DRAM);

  auto matches = ekv.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-A");
  EXPECT_EQ(matches[0].tier, TierType::DRAM);
  EXPECT_EQ(matches[0].matched_hashes.size(), 2u);
}

TEST(ClientRegistryExternalKv, UnregisterClientClearsExternalKv) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  ExternalKvBlockIndex ekv;
  reg.SetExternalKvBlockIndex(&ekv);

  ASSERT_TRUE(RegisterNode(reg, "node-A"));
  reg.RegisterExternalKvBlocks("node-A", {"h1", "h2"}, TierType::DRAM);

  // Verify entries exist before unregister
  EXPECT_EQ(ekv.Match({"h1", "h2"}).size(), 1u);

  reg.UnregisterClient("node-A");

  // After unregister, external KV entries must be gone
  auto matches = ekv.Match({"h1", "h2"});
  EXPECT_TRUE(matches.empty());
}

TEST(ClientRegistryExternalKv, UnregisterExternalKvBlocksRemovesSpecificHashes) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  ExternalKvBlockIndex ekv;
  reg.SetExternalKvBlockIndex(&ekv);

  ASSERT_TRUE(RegisterNode(reg, "node-A"));
  reg.RegisterExternalKvBlocks("node-A", {"h1", "h2", "h3"}, TierType::DRAM);
  reg.UnregisterExternalKvBlocks("node-A", {"h2"});

  auto matches = ekv.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].matched_hashes.size(), 2u);

  auto h2_match = ekv.Match({"h2"});
  EXPECT_TRUE(h2_match.empty());
}

TEST(ClientRegistryExternalKv, NullIndexDoesNotCrash) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  // No SetExternalKvBlockIndex call — external_kv_index_ remains null
  ASSERT_TRUE(RegisterNode(reg, "node-A"));

  EXPECT_NO_THROW(reg.RegisterExternalKvBlocks("node-A", {"h1"}, TierType::DRAM));
  EXPECT_NO_THROW(reg.UnregisterExternalKvBlocks("node-A", {"h1"}));
  EXPECT_NO_THROW(reg.UnregisterClient("node-A"));
}

TEST(ClientRegistryExternalKv, MultipleNodesSeparateEntries) {
  ClientRegistryConfig cfg = MakeConfig();
  GlobalBlockIndex gbi;
  ClientRegistry reg(cfg, gbi);

  ExternalKvBlockIndex ekv;
  reg.SetExternalKvBlockIndex(&ekv);

  ASSERT_TRUE(RegisterNode(reg, "node-A"));
  ASSERT_TRUE(RegisterNode(reg, "node-B"));

  reg.RegisterExternalKvBlocks("node-A", {"h1", "h2"}, TierType::DRAM);
  reg.RegisterExternalKvBlocks("node-B", {"h2", "h3"}, TierType::SSD);

  reg.UnregisterClient("node-A");

  // node-A's hashes gone, node-B's intact
  auto m_h1 = ekv.Match({"h1"});
  EXPECT_TRUE(m_h1.empty());

  auto m_h2 = ekv.Match({"h2"});
  ASSERT_EQ(m_h2.size(), 1u);
  EXPECT_EQ(m_h2[0].node_id, "node-B");

  auto m_h3 = ekv.Match({"h3"});
  ASSERT_EQ(m_h3.size(), 1u);
  EXPECT_EQ(m_h3[0].node_id, "node-B");
}

}  // namespace mori::umbp
