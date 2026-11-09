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
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "umbp/common/config.h"
#include "umbp/distributed/config.h"

namespace mori::umbp {
namespace {

UMBPConfig StandaloneConfig() {
  UMBPConfig config;
  config.ssd.enabled = false;
  UMBPStandaloneProcessConfig standalone;
  standalone.address = "unix:///tmp/umbp-config-test.sock";
  config.standalone_process = standalone;
  return config;
}

TEST(StandaloneConfig, ExistingAttachmentNeedsNoServerPolicy) {
  auto config = StandaloneConfig();
  std::string error;
  EXPECT_TRUE(config.Validate(&error)) << error;
}

TEST(StandaloneConfig, AcceptsPolicyForAutoStartedServer) {
  auto config = StandaloneConfig();
  auto& standalone = *config.standalone_process;
  standalone.auto_start = true;
  config.backend_policy_path = "/tmp/dram-ssd-policy.json";
  config.page_size = 4096;
  std::string error;
  EXPECT_TRUE(config.Validate(&error)) << error;
}

TEST(StandaloneConfig, RejectsPolicyWhenOnlyAttachingToServer) {
  auto config = StandaloneConfig();
  config.backend_policy_path = "/tmp/dram-ssd-policy.json";
  std::string error;
  EXPECT_FALSE(config.Validate(&error));
  EXPECT_NE(error.find("standalone storage settings require auto_start"), std::string::npos);
}

TEST(StandaloneConfig, RejectsZeroPageSizeForServerPolicy) {
  auto config = StandaloneConfig();
  auto& standalone = *config.standalone_process;
  standalone.auto_start = true;
  config.backend_policy_path = "/tmp/dram-ssd-policy.json";
  config.page_size = 0;
  std::string error;
  EXPECT_FALSE(config.Validate(&error));
  EXPECT_EQ(error, "page_size must be > 0 when supplied");
}

TEST(StorageConfig, RejectsConflictingDistributedPolicy) {
  UMBPConfig config;
  config.distributed = UMBPDistributedConfig{};
  config.backend_policy_path = "/tmp/shared.json";
  config.distributed->backend_policy_path = "/tmp/other.json";
  std::string error;
  EXPECT_FALSE(config.ValidateStorageConfig(&error));
  EXPECT_EQ(error, "backend_policy_path conflicts with distributed.backend_policy_path");
}

TEST(StorageConfig, RejectsConflictingDistributedPageSize) {
  UMBPConfig config;
  config.distributed = UMBPDistributedConfig{};
  config.page_size = 4096;
  config.distributed->dram_page_size = 8192;
  std::string error;
  EXPECT_FALSE(config.ValidateStorageConfig(&error));
  EXPECT_EQ(error, "page_size conflicts with distributed.dram_page_size");
}

TEST(StorageConfig, AllowsMatchingSharedAndDistributedSettings) {
  UMBPConfig config;
  config.distributed = UMBPDistributedConfig{};
  config.backend_policy_path = "/tmp/shared.json";
  config.distributed->backend_policy_path = config.backend_policy_path;
  config.page_size = 4096;
  config.distributed->dram_page_size = *config.page_size;
  std::string error;
  EXPECT_TRUE(config.ValidateStorageConfig(&error)) << error;
}

TEST(StorageConfig, DefaultConstructionReadsEnvironment) {
  ASSERT_EQ(setenv("UMBP_SIM_ENABLE", "1", 1), 0);
  ASSERT_EQ(setenv("UMBP_SIM_PROFILE", "/tmp/sim-profile.json", 1), 0);
  ASSERT_EQ(setenv("UMBP_DRAM_CAPACITY", "12345", 1), 0);
  const UMBPConfig config;
  unsetenv("UMBP_SIM_ENABLE");
  unsetenv("UMBP_SIM_PROFILE");
  unsetenv("UMBP_DRAM_CAPACITY");
  EXPECT_TRUE(config.sim.enabled);
  EXPECT_EQ(config.sim.profile_path, "/tmp/sim-profile.json");
  EXPECT_EQ(config.dram.capacity_bytes, 12345u);
  EXPECT_FALSE(UMBPConfig().sim.enabled);
}

class ScopedEnvironment {
 public:
  ScopedEnvironment(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) previous_ = std::string(old);
    if (value)
      setenv(name, value, 1);
    else
      unsetenv(name);
  }
  ~ScopedEnvironment() {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

TEST(FabricStorageConfig, DefaultsPreserveRdmaAndHipAllocation) {
  ScopedEnvironment mode("UMBP_TRANSFER_MODE", nullptr);
  ScopedEnvironment allocation("UMBP_HBM_ALLOCATION", nullptr);
  ScopedEnvironment placement("UMBP_KV_PUT_PLACEMENT", nullptr);
  ScopedEnvironment ranged("UMBP_KV_RANGED_PATH", nullptr);
  UMBPDistributedConfig config;
  EXPECT_EQ(config.io_engine.mode, UMBPTransferMode::kRdma);
  EXPECT_EQ(config.hbm.allocation, UMBPHbmAllocation::kHipMalloc);
  EXPECT_EQ(config.kv_put_placement, UMBPKvPutPlacement::kMaster);
  EXPECT_EQ(config.kv_ranged_path, UMBPKvRangedPath::kLegacy);
}

TEST(FabricStorageConfig, EnvironmentOptionsSurviveLowering) {
  ScopedEnvironment mode("UMBP_TRANSFER_MODE", "fabric_strict");
  ScopedEnvironment module("UMBP_FABRIC_KERNEL_PATH", "/tmp/fabric.hsaco");
  ScopedEnvironment allocation("UMBP_HBM_ALLOCATION", "fabric_vmm");
  ScopedEnvironment placement("UMBP_KV_PUT_PLACEMENT", "owner_local");
  ScopedEnvironment ranged("UMBP_KV_RANGED_PATH", "gpu_direct");
  ScopedEnvironment reads("UMBP_KV_MAX_INFLIGHT_READS", "7");
  ScopedEnvironment bytes("UMBP_KV_MAX_INFLIGHT_BYTES", "1234567");
  UMBPDistributedConfig config;
  config.hbm.device = 3;
  config.hbm.capacity_bytes = 123456;
  const auto lowered = ToPoolClientConfig(config, {});
  EXPECT_EQ(lowered.io_engine.mode, UMBPTransferMode::kFabricStrict);
  EXPECT_EQ(lowered.io_engine.fabric_kernel_path, "/tmp/fabric.hsaco");
  EXPECT_EQ(lowered.hbm.allocation, UMBPHbmAllocation::kFabricVmm);
  EXPECT_EQ(lowered.hbm.device, 3);
  EXPECT_EQ(lowered.hbm.buffer_sizes, (std::vector<uint64_t>{123456}));
  EXPECT_EQ(lowered.kv_put_placement, UMBPKvPutPlacement::kOwnerLocal);
  EXPECT_EQ(lowered.kv_ranged_path, UMBPKvRangedPath::kGpuDirect);
  EXPECT_EQ(lowered.kv_max_inflight_reads, 7u);
  EXPECT_EQ(lowered.kv_max_inflight_bytes, 1234567u);
}

TEST(FabricStorageConfig, RejectsUnknownTransportBeforeHardwareInitialization) {
  ScopedEnvironment mode("UMBP_TRANSFER_MODE", "fabric_typo");
  EXPECT_THROW(UMBPDistributedConfig{}, std::invalid_argument);
}

TEST(FabricStorageConfig, RejectsNonpositiveOrMalformedBudgets) {
  for (const char* value : {"0", "-1", "7garbage"}) {
    ScopedEnvironment reads("UMBP_KV_MAX_INFLIGHT_READS", value);
    EXPECT_THROW(UMBPDistributedConfig{}, std::invalid_argument) << value;
  }
}

TEST(FabricStorageConfig, StrictAllowsAConsumerWithoutAnExportedLocalHbmPool) {
  UMBPConfig config;
  config.ssd.enabled = false;
  config.distributed = UMBPDistributedConfig{};
  auto& distributed = *config.distributed;
  distributed.master_config.node_id = "consumer";
  distributed.master_config.node_address = "127.0.0.1:5000";
  distributed.io_engine.mode = UMBPTransferMode::kFabricStrict;
  distributed.io_engine.fabric_kernel_path = "/tmp/fabric.hsaco";
  std::string error;
  ASSERT_TRUE(config.Validate(&error)) << error;
  distributed.medium = UMBPMedium::HBM;
  distributed.hbm.capacity_bytes = 4096;
  distributed.hbm.allocation = UMBPHbmAllocation::kHipMalloc;
  EXPECT_FALSE(config.Validate(&error));
  EXPECT_NE(error.find("fabric_vmm"), std::string::npos);
  distributed.hbm.allocation = UMBPHbmAllocation::kFabricVmm;
  EXPECT_TRUE(config.Validate(&error)) << error;
  distributed.io_engine.fabric_kernel_path.clear();
  EXPECT_FALSE(config.Validate(&error));
  EXPECT_NE(error.find("fabric_kernel_path"), std::string::npos);
}

}  // namespace
}  // namespace mori::umbp
