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
  EXPECT_EQ(output.logical_tiers[0].members[0].weight, output.logical_tiers[0].members[1].weight);
  EXPECT_EQ(output.logical_tiers[0].offload_to, (std::vector<std::string>{"dram", "ssd_a"}));
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
  EXPECT_EQ(output.logical_tiers[2].promote_mode, PoolTransitionMode::kCopy);
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

// The shipped example only exercises promote_mode "copy", so nothing pins the
// other branch of the enum: a policy asking for move would have parsed to copy
// and silently kept the source tier's copy alive.
TEST(BackendPolicyConfig, LowersMovePromotionMode) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{
       "h":{"type":"dram","capacity":"1GiB"},
       "c":{"type":"dram","capacity":"1GiB"}},
     "tiers":[
       {"backends":{"h":1},"offload_to":["cold"],"offload_trigger":"on_evict","name":"hot"},
       {"backends":{"c":1},"name":"cold","promote_trigger":"on_read","promote_mode":"move"}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  output.dram_page_size = 64ULL << 10;
  std::string error;
  ASSERT_TRUE(ApplyBackendPolicy(*loaded.config, &output, &error)) << error;
  ASSERT_EQ(output.logical_tiers.size(), 2u);
  EXPECT_EQ(output.logical_tiers[1].promote_trigger, PoolPromoteTrigger::kOnRead);
  EXPECT_EQ(output.logical_tiers[1].promote_mode, PoolTransitionMode::kMove);
  // The entry tier keeps the default, so one tier asking for move does not
  // change how the rest of the graph promotes.
  EXPECT_EQ(output.logical_tiers[0].promote_mode, PoolTransitionMode::kCopy);

  auto bad = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB"}},
     "tiers":[{"backends":{"d":1},"promote_mode":"relocate"}]}
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
      // Nor the pre-rename spelling of promote_mode, for the same reason.
      R"json({"backends":{"h":{"type":"dram","capacity":"1GiB"},
              "c":{"type":"dram","capacity":"1GiB"}},
              "tiers":[{"backends":{"h":1},"offload_to":["cold"],
                        "offload_trigger":"on_evict","name":"hot"},
                       {"backends":{"c":1},"name":"cold","promote_trigger":"on_read",
                        "promotion_mode":"move"}]})json",
  };
  for (const char* json : kRejected) {
    EXPECT_FALSE(LoadBackendPolicyJson(json).ok()) << json;
  }
}

// An SSD backend's staging arena is staging_slots * page_size, and a read that
// cannot claim a slot is reported as a miss rather than as backpressure, so the
// slot count is a hard read-concurrency ceiling. Left at the default of 16 a
// 64 KiB-page backend has a 1 MiB arena, which under a mixed workload fails the
// majority of its reads and, because the offload target then refuses pages,
// most of its writes too. Omitting the field must keep the previous default so
// existing policies are unaffected.
TEST(BackendPolicyConfig, LowersSsdStagingSlotsAndDefaultsWhenAbsent) {
  auto loaded = LoadBackendPolicyJson(R"json(
    {"backends":{
       "with_slots":{"type":"ssd","capacity":"1GiB","path":"/tmp/a","staging_slots":512},
       "without":{"type":"ssd","capacity":"1GiB","path":"/tmp/b"}},
     "tiers":[{"backends":{"with_slots":1}},{"backends":{"without":1}}]}
  )json");
  ASSERT_TRUE(loaded.ok()) << loaded.error;

  PoolClientConfig output;
  output.dram_page_size = 64ULL << 10;
  std::string error;
  ASSERT_TRUE(ApplyBackendPolicy(*loaded.config, &output, &error)) << error;

  const auto slots_of = [&](const std::string& name) {
    for (const auto& backend : output.backends) {
      if (backend.name == name) return backend.ssd_staging_buffer_slots;
    }
    ADD_FAILURE() << "backend '" << name << "' missing";
    return -1;
  };
  EXPECT_EQ(slots_of("with_slots"), 512);
  EXPECT_EQ(slots_of("without"), BackendInstanceConfig{}.ssd_staging_buffer_slots);

  // A DRAM backend has no staging arena, so the field must not be silently
  // accepted there and then ignored.
  auto on_dram = LoadBackendPolicyJson(R"json(
    {"backends":{"d":{"type":"dram","capacity":"1GiB","staging_slots":512}},
     "tiers":[{"backends":{"d":1}}]}
  )json");
  EXPECT_FALSE(on_dram.ok());
  EXPECT_NE(on_dram.error.find("unknown field"), std::string::npos);

  auto negative = LoadBackendPolicyJson(R"json(
    {"backends":{"s":{"type":"ssd","capacity":"1GiB","path":"/tmp/c","staging_slots":-1}},
     "tiers":[{"backends":{"s":1}}]}
  )json");
  EXPECT_FALSE(negative.ok());
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
