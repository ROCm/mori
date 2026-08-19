// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>

#include "umbp/distributed/config.h"
#include "umbp/distributed/pool/policy_config.h"

namespace mori::umbp {
namespace {

constexpr const char* kPolicy = R"json(
{
  "schema_version": 1,
  "entry_tier": "hot",
  "backends": {
    "hbm":   { "type": "hbm", "capacity": "80GiB", "devices": [0, 1] },
    "dram":  { "type": "dram", "capacity": "512GiB", "numa_node": 0 },
    "ssd_a": { "type": "ssd", "capacity": "1TiB", "path": "/mnt/kvcache/hot" },
    "ssd_b": { "type": "ssd", "capacity": "3TiB", "path": "/mnt/kvcache/cold" }
  },
  "tiers": [
    {
      "name": "hot",
      "backends": { "hbm": 100 },
      "offload_to": ["warm"],
      "offload_trigger": "on_evict"
    },
    {
      "name": "warm",
      "backends": { "dram": 70, "ssd_a": 30 },
      "offload_to": ["cold"],
      "offload_trigger": "watermark"
    },
    {
      "name": "cold",
      "backends": { "ssd_b": 100 },
      "promote_on_read": true
    }
  ]
}
)json";

TEST(BackendPolicyConfig, ParsesAndLowersDocumentedTopology) {
  auto loaded = LoadBackendPolicyJson(kPolicy);
  ASSERT_TRUE(loaded.ok()) << loaded.error;
  ASSERT_EQ(loaded.config->backends.size(), 4u);
  ASSERT_EQ(loaded.config->logical_tiers.size(), 3u);

  PoolClientConfig output;
  output.dram_page_size = 2ULL << 20;
  std::string error;
  ASSERT_TRUE(ApplyBackendPolicy(*loaded.config, &output, &error, "-node-0")) << error;
  EXPECT_EQ(output.placement_policy, PoolPlacementPolicy::TIERED);
  ASSERT_EQ(output.backends.size(), 5u);

  const auto find_backend = [&](const std::string& name) -> const BackendInstanceConfig* {
    for (const auto& backend : output.backends) {
      if (backend.name == name) return &backend;
    }
    return nullptr;
  };
  ASSERT_NE(find_backend("dram"), nullptr);
  EXPECT_EQ(find_backend("dram")->dram.numa_node, 0);
  ASSERT_NE(find_backend("hbm@0"), nullptr);
  ASSERT_NE(find_backend("hbm@1"), nullptr);
  EXPECT_EQ(find_backend("hbm@0")->hbm.buffer_sizes.front(), 40ULL << 30);
  ASSERT_NE(find_backend("ssd_a"), nullptr);
  EXPECT_EQ(find_backend("ssd_a")->ssd.ssd.storage_dir, "/mnt/kvcache/hot-node-0");

  ASSERT_EQ(output.logical_tiers[0].members.size(), 2u);
  EXPECT_EQ(output.logical_tiers[0].members[0].weight,
            output.logical_tiers[0].members[1].weight);
  EXPECT_EQ(output.logical_tiers[0].offload_to,
            (std::vector<std::string>{"dram", "ssd_a"}));
  EXPECT_EQ(output.logical_tiers[0].trigger, PoolOffloadTrigger::kOnEvict);
  EXPECT_EQ(output.logical_tiers[0].name, "hot");
  EXPECT_TRUE(output.logical_tiers[0].entry);
  EXPECT_EQ(output.logical_tiers[1].trigger, PoolOffloadTrigger::kWatermark);
  EXPECT_TRUE(output.logical_tiers[2].promote_on_read);
  EXPECT_TRUE(output.logical_tiers[2].offload_to.empty());
}

TEST(BackendPolicyConfig, RejectsInvalidSchemaAndBackwardEdges) {
  auto unknown = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB","extra":1}},
     "tiers":[{"backends":{"d":1}}]}
  )json");
  EXPECT_FALSE(unknown.ok());
  EXPECT_NE(unknown.error.find("unknown field"), std::string::npos);

  auto backward = LoadBackendPolicyJson(R"json(
    {"backends":{
       "a":{"type":"dram","capacity":"1GiB"},
       "b":{"type":"ssd","capacity":"1GiB","path":"/tmp/b"}},
     "tiers":[
       {"backends":{"a":1}},
       {"backends":{"b":1},"offload_to":["a"],"offload_trigger":"on_evict"}]}
  )json");
  EXPECT_FALSE(backward.ok());
  EXPECT_NE(backward.error.find("strictly later tier"), std::string::npos);
}

TEST(BackendPolicyConfig, ApplyIsAtomicOnExpansionFailure) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{"h":{"type":"hbm","capacity":"1B","devices":[0,1]}},
     "tiers":[{"backends":{"h":1}}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  output.backends.push_back(BackendInstanceConfig{});
  output.backends.front().name = "unchanged";
  std::string error;
  EXPECT_FALSE(ApplyBackendPolicy(*loaded.config, &output, &error));
  ASSERT_EQ(output.backends.size(), 1u);
  EXPECT_EQ(output.backends.front().name, "unchanged");
}

}  // namespace
}  // namespace mori::umbp
