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

#include <string>

#include "umbp/common/config.h"

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

}  // namespace
}  // namespace mori::umbp
