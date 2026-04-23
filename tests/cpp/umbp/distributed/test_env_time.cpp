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
// Unit tests for umbp/common/env_time.h.
//
// These only exercise the helper functions, which re-read the environment on
// every call.  Production call sites wrap the helpers in function-local
// static caches; those cannot be reset within a single process and must be
// covered by integration tests that fork a fresh binary per value.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>

#include "umbp/common/env_time.h"

namespace {

using mori::umbp::GetEnvMicroseconds;
using mori::umbp::GetEnvMilliseconds;
using mori::umbp::GetEnvSeconds;
using mori::umbp::GetEnvUint32;
using mori::umbp::ResetEnvWarnStateForTesting;

constexpr const char* kName = "UMBP_TEST_ENV_TIME_XYZ";
constexpr const char* kOther = "UMBP_TEST_ENV_TIME_OTHER";

class EnvTimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::unsetenv(kName);
    ::unsetenv(kOther);
    ResetEnvWarnStateForTesting();
  }
  void TearDown() override {
    ::unsetenv(kName);
    ::unsetenv(kOther);
    ResetEnvWarnStateForTesting();
  }
};

TEST_F(EnvTimeTest, DefaultWhenUnset) {
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(7)).count(), 7);
  EXPECT_EQ(GetEnvMilliseconds(kName, std::chrono::milliseconds(42)).count(), 42);
  EXPECT_EQ(GetEnvMicroseconds(kName, std::chrono::microseconds(99)).count(), 99);
  EXPECT_EQ(GetEnvUint32(kName, 5), 5u);
}

TEST_F(EnvTimeTest, ParsesValid) {
  ::setenv(kName, "12", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(1)).count(), 12);
  EXPECT_EQ(GetEnvMilliseconds(kName, std::chrono::milliseconds(1)).count(), 12);
  EXPECT_EQ(GetEnvMicroseconds(kName, std::chrono::microseconds(1)).count(), 12);
  EXPECT_EQ(GetEnvUint32(kName, 1), 12u);
}

TEST_F(EnvTimeTest, EmptyStringIsUnsetSemantics) {
  ::setenv(kName, "", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(3)).count(), 3);
}

TEST_F(EnvTimeTest, NonNumericFallsBackToDefault) {
  ::setenv(kName, "abc", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(4)).count(), 4);
}

TEST_F(EnvTimeTest, TrailingGarbageFallsBackToDefault) {
  ::setenv(kName, "10abc", 1);
  EXPECT_EQ(GetEnvMilliseconds(kName, std::chrono::milliseconds(9)).count(), 9);
}

TEST_F(EnvTimeTest, NegativeFallsBackToDefault) {
  ::setenv(kName, "-5", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(2)).count(), 2);
  EXPECT_EQ(GetEnvUint32(kName, 11), 11u);
}

TEST_F(EnvTimeTest, BelowMinAllowedFallsBack) {
  ::setenv(kName, "0", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(6), /*min_allowed=*/1).count(), 6);
  EXPECT_EQ(GetEnvUint32(kName, 7, /*min_allowed=*/1), 7u);
}

TEST_F(EnvTimeTest, ZeroIsAllowedWhenMinIsZero) {
  ::setenv(kName, "0", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(6)).count(), 0);
  EXPECT_EQ(GetEnvUint32(kName, 7), 0u);
}

TEST_F(EnvTimeTest, FallsBackWhenAboveUint32Max) {
  // A value that exceeds uint32 range must fall back to default.
  ::setenv(kName, "99999999999", 1);
  EXPECT_EQ(GetEnvUint32(kName, 4), 4u);
}

TEST_F(EnvTimeTest, AcceptsLargeUint32WithoutSignFlip) {
  // UINT32_MAX must round-trip as-is (regression for a past int cast that
  // silently flipped large values negative on the consumer side).
  ::setenv(kName, "4294967295", 1);
  EXPECT_EQ(GetEnvUint32(kName, 1), 4294967295u);
}

TEST_F(EnvTimeTest, DifferentEnvNamesAreIndependent) {
  ::setenv(kName, "bad", 1);
  ::setenv(kOther, "3", 1);
  EXPECT_EQ(GetEnvSeconds(kName, std::chrono::seconds(1)).count(), 1);
  EXPECT_EQ(GetEnvSeconds(kOther, std::chrono::seconds(1)).count(), 3);
}

}  // namespace
