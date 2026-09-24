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

// The resolution matrix behind UMBP_RPC_DEADLINES, which decides whether a
// stalled RPC is bounded at all. The interesting cell is "switch off, knob set
// explicitly": that is what lets a deployment disable deadlines everywhere but
// one place, and it is the reason the master switch cannot simply be an `if`
// wrapped around the arming call.
//
// Everything here is resolved uncached on purpose, so one process can walk the
// whole matrix with setenv -- the same property test_env_time.cpp relies on.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>

#include "umbp/common/env_time.h"
#include "umbp/common/rpc_deadline.h"

namespace mori::umbp {
namespace {

constexpr const char* kSwitch = "UMBP_RPC_DEADLINES";
constexpr const char* kKnob = "UMBP_TEST_RPC_DEADLINE_XYZ";
constexpr std::chrono::milliseconds kDefault{5000};

class RpcDeadlineTest : public ::testing::Test {
 protected:
  void SetUp() override { Clear(); }
  void TearDown() override { Clear(); }

 private:
  static void Clear() {
    ::unsetenv(kSwitch);
    ::unsetenv(kKnob);
    // The named knobs below cache on first use, so their value is whatever the
    // environment held then. Clearing them here makes that first use
    // deterministic no matter what the ambient environment carried in.
    ::unsetenv("UMBP_MASTER_RPC_TIMEOUT_MS");
    ::unsetenv("UMBP_PEER_RPC_TIMEOUT_MS");
    ResetEnvWarnStateForTesting();
  }
};

TEST_F(RpcDeadlineTest, SwitchDefaultsToOn) {
  EXPECT_TRUE(RpcDeadlinesEnabled());
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 5000);
}

TEST_F(RpcDeadlineTest, SwitchOffDisablesAnUnsetKnob) {
  ::setenv(kSwitch, "0", 1);
  EXPECT_FALSE(RpcDeadlinesEnabled());
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 0);
}

// The whole reason the switch is not just an `if` around ArmDeadline: "off
// everywhere except here" has to be expressible.
TEST_F(RpcDeadlineTest, AnExplicitKnobWinsOverTheSwitch) {
  ::setenv(kSwitch, "0", 1);
  ::setenv(kKnob, "1234", 1);
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 1234);
}

// And the converse: one knob off while everything else keeps its deadline.
TEST_F(RpcDeadlineTest, AKnobSetToZeroDisablesOnlyItself) {
  ::setenv(kKnob, "0", 1);
  EXPECT_TRUE(RpcDeadlinesEnabled());
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 0);
}

TEST_F(RpcDeadlineTest, SwitchAcceptsTheUsualSpellingsOfOff) {
  for (const char* off : {"0", "off", "false"}) {
    ::setenv(kSwitch, off, 1);
    EXPECT_FALSE(RpcDeadlinesEnabled()) << off;
  }
  // Anything else is on, including "1" and a value nobody meant.
  for (const char* on : {"1", "true", "yes", "banana"}) {
    ::setenv(kSwitch, on, 1);
    EXPECT_TRUE(RpcDeadlinesEnabled()) << on;
  }
}

// An empty value is not "set" -- otherwise `export UMBP_RPC_DEADLINES=` would
// silently disable every deadline in the process.
TEST_F(RpcDeadlineTest, EmptyValuesCountAsUnset) {
  ::setenv(kSwitch, "", 1);
  EXPECT_TRUE(RpcDeadlinesEnabled());
  ::setenv(kKnob, "", 1);
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 5000);
}

// Malformed and out-of-range values fall back to the default rather than to
// "no deadline", so a typo cannot quietly unbound an RPC.
TEST_F(RpcDeadlineTest, BadValuesFallBackToTheDefaultNotToDisabled) {
  for (const char* bad : {"abc", "10abc", "-5"}) {
    ResetEnvWarnStateForTesting();
    ::setenv(kKnob, bad, 1);
    EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 5000) << bad;
  }
}

// A negative value is rejected by min_allowed and must not reach ArmDeadline as
// a "disable", which would make `-1` a second, undocumented spelling of 0.
TEST_F(RpcDeadlineTest, NegativeIsRejectedEvenWithTheSwitchOff) {
  ::setenv(kSwitch, "0", 1);
  ::setenv(kKnob, "-1", 1);
  EXPECT_EQ(ResolveDeadlineMs(kKnob, kDefault), 5000);
}

TEST_F(RpcDeadlineTest, ArmDeadlineLeavesTheContextAloneWhenDisabled) {
  const auto far_future = std::chrono::system_clock::now() + std::chrono::hours(24);
  for (int disabled : {0, -1}) {
    grpc::ClientContext ctx;
    ArmDeadline(ctx, disabled);
    // gRPC reports no deadline as time_point::max(); anything sooner means one
    // was armed.
    EXPECT_GT(ctx.deadline(), far_future) << disabled;
  }
}

TEST_F(RpcDeadlineTest, ArmDeadlineSetsAPositiveTimeout) {
  grpc::ClientContext ctx;
  ArmDeadline(ctx, 1000);
  EXPECT_LT(ctx.deadline(), std::chrono::system_clock::now() + std::chrono::hours(24));
}

// The named knobs are what the call sites actually use, and their relative
// order is a documented invariant: an outbound peer call has to fail before the
// cleanup that follows it, and both have to fit inside the peer's 30s
// pending_ttl or a slow-but-successful allocate comes back to reclaimed slots.
TEST_F(RpcDeadlineTest, NamedKnobDefaultsRespectTheHierarchy) {
  EXPECT_EQ(MasterRpcTimeoutMs(), 30000);
  EXPECT_EQ(PeerRpcTimeoutMs(), 10000);
  EXPECT_EQ(PeerCleanupRpcTimeoutMs(), 2 * PeerRpcTimeoutMs());
  EXPECT_LT(PeerRpcTimeoutMs(), 30000) << "must leave room for transfer + commit in pending_ttl";
  EXPECT_LE(PeerCleanupRpcTimeoutMs(), 30000) << "an abort slower than pending_ttl is moot";
}

}  // namespace
}  // namespace mori::umbp
