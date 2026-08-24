// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>

#include "umbp/distributed/config.h"
#include "umbp/distributed/pool/policy_config.h"

namespace mori::umbp {
namespace {

// The three-tier example we ship, read from disk rather than copied here, so
// that editing one without the other fails this test.
TEST(BackendPolicyConfig, ParsesAndLowersShippedExample) {
  auto loaded = LoadBackendPolicyFile(UMBP_EXAMPLE_POLICY_PATH);
  ASSERT_TRUE(loaded.ok()) << UMBP_EXAMPLE_POLICY_PATH << ": " << loaded.error;
  EXPECT_EQ(loaded.config->schema_version, 1u);
  EXPECT_EQ(loaded.config->entry_tier, "hot");
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

  // One backend with two devices lowers to one member per device, splitting the
  // declared capacity evenly, so the tier weights stay equal.
  ASSERT_EQ(output.logical_tiers[0].members.size(), 2u);
  EXPECT_EQ(output.logical_tiers[0].members[0].weight,
            output.logical_tiers[0].members[1].weight);
  EXPECT_EQ(output.logical_tiers[0].offload_to,
            (std::vector<std::string>{"dram", "ssd_a"}));
  EXPECT_EQ(output.logical_tiers[0].trigger, PoolOffloadTrigger::kOnEvict);
  EXPECT_EQ(output.logical_tiers[0].name, "hot");
  EXPECT_TRUE(output.logical_tiers[0].entry);

  ASSERT_EQ(output.logical_tiers[1].members.size(), 2u);
  EXPECT_EQ(output.logical_tiers[1].members[0].backend_name, "dram");
  EXPECT_EQ(output.logical_tiers[1].members[0].weight, 70u);
  EXPECT_EQ(output.logical_tiers[1].members[1].backend_name, "ssd_a");
  EXPECT_EQ(output.logical_tiers[1].members[1].weight, 30u);
  EXPECT_EQ(output.logical_tiers[1].trigger, PoolOffloadTrigger::kWatermark);
  EXPECT_DOUBLE_EQ(output.logical_tiers[1].high_watermark, 0.9);
  EXPECT_DOUBLE_EQ(output.logical_tiers[1].low_watermark, 0.7);

  EXPECT_EQ(output.logical_tiers[2].promote_trigger, PoolPromoteTrigger::kOnRead);
  EXPECT_EQ(output.logical_tiers[2].promotion_mode, PoolTransitionMode::kCopy);
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

  // Tier invariants live in the shared validator rather than in the JSON
  // parser, so these two pin the wiring that runs it on the loading path.
  auto watermarks = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB"}},
     "tiers":[{"backends":{"d":1},"high_watermark":0.2,"low_watermark":0.9}]}
  )json");
  EXPECT_FALSE(watermarks.ok());
  EXPECT_NE(watermarks.error.find("watermarks must satisfy"), std::string::npos);

  auto shared_backend = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB"}},
     "tiers":[{"backends":{"d":1}},{"backends":{"d":1}}]}
  )json");
  EXPECT_FALSE(shared_backend.ok());
  EXPECT_NE(shared_backend.error.find("more than one tier"), std::string::npos);

  // Loading and applying share one validator, so a topology that cannot be
  // expanded is rejected by whichever entry point sees it first.
  auto tiny_hbm = LoadBackendPolicyJson(R"json(
    {"backends":{"h":{"type":"hbm","capacity":"1B","devices":[0,1]}},
     "tiers":[{"backends":{"h":1}}]}
  )json");
  EXPECT_FALSE(tiny_hbm.ok());
  EXPECT_NE(tiny_hbm.error.find("per-device capacity"), std::string::npos);
}

// The shipped example only exercises promotion_mode "copy", so nothing pins the
// other branch of the enum: a policy asking for move would have parsed to copy
// and silently kept the source tier's copy alive.
TEST(BackendPolicyConfig, LowersMovePromotionMode) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{
       "h":{"type":"dram","capacity":"1GiB"},
       "c":{"type":"dram","capacity":"1GiB"}},
     "tiers":[
       {"backends":{"h":1},"offload_to":["cold"],"offload_trigger":"on_evict","name":"hot"},
       {"backends":{"c":1},"name":"cold","promote_trigger":"on_read","promotion_mode":"move"}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  output.dram_page_size = 64ULL << 10;
  std::string error;
  ASSERT_TRUE(ApplyBackendPolicy(*loaded.config, &output, &error)) << error;
  ASSERT_EQ(output.logical_tiers.size(), 2u);
  EXPECT_EQ(output.logical_tiers[1].promote_trigger, PoolPromoteTrigger::kOnRead);
  EXPECT_EQ(output.logical_tiers[1].promotion_mode, PoolTransitionMode::kMove);
  // The entry tier keeps the default, so one tier asking for move does not
  // change how the rest of the graph promotes.
  EXPECT_EQ(output.logical_tiers[0].promotion_mode, PoolTransitionMode::kCopy);

  auto bad = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB"}},
     "tiers":[{"backends":{"d":1},"promotion_mode":"relocate"}]}
  )json");
  EXPECT_FALSE(bad.ok());
}

// promote_trigger is the extension point the promote policy is built on, so the
// rules that keep one policy from having two spellings are worth pinning: a
// threshold only where something reads it, never below 2, and never on the
// entry tier, which has no upstream tier to promote into.
TEST(BackendPolicyConfig, LowersPromoteTriggerAndRejectsIncoherentThresholds) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{
       "h":{"type":"dram","capacity":"1GiB"},
       "c":{"type":"dram","capacity":"1GiB"}},
     "tiers":[
       {"backends":{"h":1},"offload_to":["cold"],"offload_trigger":"watermark","name":"hot"},
       {"backends":{"c":1},"name":"cold","promote_trigger":"on_hits","promote_hits":4}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  output.dram_page_size = 64ULL << 10;
  std::string error;
  ASSERT_TRUE(ApplyBackendPolicy(*loaded.config, &output, &error)) << error;
  ASSERT_EQ(output.logical_tiers.size(), 2u);
  EXPECT_EQ(output.logical_tiers[1].promote_trigger, PoolPromoteTrigger::kOnHits);
  EXPECT_EQ(output.logical_tiers[1].promote_hits, 4u);
  // Absent means never, so a tier says nothing about promotion by saying
  // nothing rather than by carrying a threshold nothing reads.
  EXPECT_EQ(output.logical_tiers[0].promote_trigger, PoolPromoteTrigger::kNever);
  EXPECT_EQ(output.logical_tiers[0].promote_hits, 0u);

  const char* kRejected[] = {
      // on_hits without a threshold: there is no default worth guessing.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot"},
                       {"backends":{"c":1},"name":"cold","promote_trigger":"on_hits"}]})json",
      // A threshold where nothing reads it is a misconfiguration, not a hint.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot"},
                       {"backends":{"c":1},"name":"cold","promote_trigger":"on_read",
                        "promote_hits":3}]})json",
      // 1 hit is on_read spelled a second way.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot"},
                       {"backends":{"c":1},"name":"cold","promote_trigger":"on_hits",
                        "promote_hits":1}]})json",
      // The entry tier has nowhere to promote to; today this is silently dead
      // config because the read path skips it.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "entry_tier":"hot",
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot",
                        "promote_trigger":"on_read"},
                       {"backends":{"c":1},"name":"cold"}]})json",
      // Unknown enumerator, so a typo cannot read as "never".
      R"json({"backends":{"d":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"d":1},"promote_trigger":"on_every_read"}]})json",
      // The old boolean must not keep working, or policies keep using it.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot"},
                       {"backends":{"c":1},"name":"cold","promote_on_read":true}]})json",
  };
  for (const char* json : kRejected) {
    EXPECT_FALSE(LoadBackendPolicyJson(json).ok()) << json;
  }
}

TEST(BackendPolicyConfig, ApplyIsAtomicWhenLoweringFails) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1MiB"}},
     "tiers":[{"backends":{"d":1}}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  // Only Apply can reject this: whether a capacity is usable depends on the
  // page size of the client the policy is being applied to.
  output.dram_page_size = 2ULL << 20;
  output.backends.push_back(BackendInstanceConfig{});
  output.backends.front().name = "unchanged";
  std::string error;
  EXPECT_FALSE(ApplyBackendPolicy(*loaded.config, &output, &error));
  EXPECT_NE(error.find("smaller than page size"), std::string::npos);
  ASSERT_EQ(output.backends.size(), 1u);
  EXPECT_EQ(output.backends.front().name, "unchanged");
}

}  // namespace
}  // namespace mori::umbp
