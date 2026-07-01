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
// Hermetic unit tests for the cache_remote_fetches admission gate
// (ShouldAdmitReCache), the pure predicate used by
// PoolClient::MaybeReCacheAfterRemote. No GPU / RDMA / master required.
#include <gtest/gtest.h>

#include "umbp/common/config.h"

namespace mori::umbp {
namespace {

constexpr size_t kCap = 16ull * 1024 * 1024;  // 16 MB default admission cap

// cache_remote_fetches=false disables re-cache regardless of policy/size.
TEST(CacheRemoteAdmission, DisabledFlagNeverAdmits) {
  EXPECT_FALSE(ShouldAdmitReCache(false, CacheRemoteAdmission::ALWAYS, kCap, 1024));
  EXPECT_FALSE(ShouldAdmitReCache(false, CacheRemoteAdmission::SIZE, kCap, 1024));
  EXPECT_FALSE(ShouldAdmitReCache(false, CacheRemoteAdmission::NEVER, kCap, 1024));
}

// NEVER policy rejects even when the flag is on and size is tiny.
TEST(CacheRemoteAdmission, NeverPolicyRejects) {
  EXPECT_FALSE(ShouldAdmitReCache(true, CacheRemoteAdmission::NEVER, kCap, 1));
  EXPECT_FALSE(ShouldAdmitReCache(true, CacheRemoteAdmission::NEVER, 0, 1));
}

// Zero-size blocks are never admitted (nothing to cache).
TEST(CacheRemoteAdmission, ZeroSizeRejected) {
  EXPECT_FALSE(ShouldAdmitReCache(true, CacheRemoteAdmission::ALWAYS, kCap, 0));
  EXPECT_FALSE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, kCap, 0));
}

// ALWAYS admits any non-zero size, ignoring the cap.
TEST(CacheRemoteAdmission, AlwaysAdmitsRegardlessOfSize) {
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::ALWAYS, kCap, 1));
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::ALWAYS, kCap, kCap));
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::ALWAYS, kCap, kCap + 1));
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::ALWAYS, 0, 1ull << 40));
}

// SIZE policy admits at/below the cap and rejects above it.
TEST(CacheRemoteAdmission, SizePolicyRespectsCap) {
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, kCap, 1));
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, kCap, kCap));  // boundary: <=
  EXPECT_FALSE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, kCap, kCap + 1));
}

// SIZE policy with cap==0 means "no size limit" (admit any non-zero size).
TEST(CacheRemoteAdmission, SizePolicyZeroCapIsUnlimited) {
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, 0, 1));
  EXPECT_TRUE(ShouldAdmitReCache(true, CacheRemoteAdmission::SIZE, 0, 1ull << 40));
}

}  // namespace
}  // namespace mori::umbp
