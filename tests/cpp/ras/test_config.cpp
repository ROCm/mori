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

#include <map>
#include <string>

#include "src/ras/config.hpp"

using namespace mori::ras;

namespace {

RasConfig Parse(const std::map<std::string, std::string>& env) {
  return ConfigFromEnv([&](const char* name) -> const char* {
    auto it = env.find(name);
    return it == env.end() ? nullptr : it->second.c_str();
  });
}

bool Warned(const RasConfig& c, const std::string& needle) {
  for (const std::string& w : c.warnings) {
    if (w.find(needle) != std::string::npos) return true;
  }
  return false;
}

}  // namespace

TEST(RasConfig, DefaultsLeaveRasOff) {
  RasConfig c = Parse({});
  EXPECT_FALSE(c.enable);
  EXPECT_EQ(c.port, 0);
  EXPECT_EQ(c.ifname, "");
  EXPECT_DOUBLE_EQ(c.timeoutFactor, 1.0);
  EXPECT_EQ(c.heartbeatMs, 1000u);
  EXPECT_EQ(c.maxLinks, 256u);
  EXPECT_TRUE(c.cpuAffinity.empty());
  EXPECT_TRUE(c.warnings.empty());
  EXPECT_EQ(c.HeartbeatNs(), 1000000000ull);
  EXPECT_EQ(c.HandshakeTimeoutNs(), 5000000000ull);
  EXPECT_EQ(c.ReconnectBackoffMaxNs(), 30000000000ull);
}

TEST(RasConfig, ParsesEnable) {
  EXPECT_TRUE(Parse({{"MORI_RAS_ENABLE", "1"}}).enable);
  EXPECT_TRUE(Parse({{"MORI_RAS_ENABLE", "true"}}).enable);
  EXPECT_FALSE(Parse({{"MORI_RAS_ENABLE", "0"}}).enable);
  RasConfig bad = Parse({{"MORI_RAS_ENABLE", "maybe"}});
  EXPECT_FALSE(bad.enable);
  EXPECT_TRUE(Warned(bad, "MORI_RAS_ENABLE"));
}

TEST(RasConfig, NeverUsesRcclRasPort) {
  RasConfig c = Parse({{"MORI_RAS_PORT", "28028"}});
  EXPECT_EQ(c.port, 0);
  EXPECT_TRUE(Warned(c, "28028"));
  EXPECT_EQ(Parse({{"MORI_RAS_PORT", "29001"}}).port, 29001);
  for (const char* bad : {"70000", "-1", "http", "12x"}) {
    RasConfig b = Parse({{"MORI_RAS_PORT", bad}});
    EXPECT_EQ(b.port, 0) << bad;
    EXPECT_TRUE(Warned(b, "MORI_RAS_PORT")) << bad;
  }
}

TEST(RasConfig, TheTimeoutFactorScalesEveryTimeout) {
  RasConfig c = Parse({{"MORI_RAS_TIMEOUT_FACTOR", "2.5"}, {"MORI_RAS_HEARTBEAT_MS", "100"}});
  EXPECT_DOUBLE_EQ(c.timeoutFactor, 2.5);
  EXPECT_EQ(c.HandshakeTimeoutNs(), 12500000000ull);
  EXPECT_EQ(c.ReconnectBackoffInitialNs(), 2500000000ull);
  EXPECT_EQ(c.ReconnectBackoffMaxNs(), 75000000000ull);
  EXPECT_EQ(c.HeartbeatNs(), 100000000ull);  // the period is its own setting
  for (const char* bad : {"0", "-2", "abc", "1e9"}) {
    RasConfig b = Parse({{"MORI_RAS_TIMEOUT_FACTOR", bad}});
    EXPECT_DOUBLE_EQ(b.timeoutFactor, 1.0) << bad;
    EXPECT_TRUE(Warned(b, "MORI_RAS_TIMEOUT_FACTOR")) << bad;
  }
}

TEST(RasConfig, BoundsTheHeartbeatPeriodAndLinkBudget) {
  for (const char* bad : {"0", "5", "999999999", "fast"}) {
    RasConfig b = Parse({{"MORI_RAS_HEARTBEAT_MS", bad}});
    EXPECT_EQ(b.heartbeatMs, 1000u) << bad;
    EXPECT_TRUE(Warned(b, "MORI_RAS_HEARTBEAT_MS")) << bad;
  }
  EXPECT_EQ(Parse({{"MORI_RAS_MAX_LINKS", "64"}}).maxLinks, 64u);
  RasConfig b = Parse({{"MORI_RAS_MAX_LINKS", "0"}});
  EXPECT_EQ(b.maxLinks, 256u);
  EXPECT_TRUE(Warned(b, "MORI_RAS_MAX_LINKS"));
}

TEST(RasConfig, ParsesCpuAffinityLists) {
  RasConfig c = Parse({{"MORI_RAS_CPU_AFFINITY", "0-3,8, 10-11"}});
  EXPECT_EQ(c.cpuAffinity, (std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
  EXPECT_TRUE(c.warnings.empty());
  EXPECT_EQ(Parse({{"MORI_RAS_CPU_AFFINITY", "5"}}).cpuAffinity, (std::vector<int>{5}));
  for (const char* bad : {"3-1", "a", "1,,2", "-4", "0-99999"}) {
    RasConfig b = Parse({{"MORI_RAS_CPU_AFFINITY", bad}});
    EXPECT_TRUE(b.cpuAffinity.empty()) << bad;
    EXPECT_TRUE(Warned(b, "MORI_RAS_CPU_AFFINITY")) << bad;
  }
}

TEST(RasConfig, ReadsTheInterfaceName) {
  EXPECT_EQ(Parse({{"MORI_RAS_IFNAME", "eno1"}}).ifname, "eno1");
}
