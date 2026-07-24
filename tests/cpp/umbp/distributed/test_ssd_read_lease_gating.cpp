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
// Pure-logic unit tests for the reader-side remote SSD read lease gating
// (umbp/distributed/ssd_read_lease.h).  These cover the decision policy without
// a cluster / RDMA: the full PrepareSsdRead -> RDMA path is exercised at the
// RPC level in test_peer_ssd_read_rpc.cpp.  Retryable outcomes are NO_SLOT and
// a reader-local lease expiry; rpc failures are not-served (RPC-test covered).
#include <gtest/gtest.h>

#include <chrono>

#include "umbp/distributed/ssd_read_lease.h"

namespace mori::umbp::ssd_read_lease {
namespace {

using std::chrono::milliseconds;
using std::chrono::steady_clock;

// ---- LeaseExpired ----

TEST(SsdReadLeaseGating, NotExpiredBeforeDeadline) {
  const auto t_send = steady_clock::now();
  EXPECT_FALSE(LeaseExpired(t_send, /*lease_ttl_ms=*/1000, t_send + milliseconds(500)));
}

TEST(SsdReadLeaseGating, ExpiredAfterDeadline) {
  const auto t_send = steady_clock::now();
  EXPECT_TRUE(LeaseExpired(t_send, /*lease_ttl_ms=*/1000, t_send + milliseconds(1001)));
}

TEST(SsdReadLeaseGating, ExactlyAtDeadlineIsNotExpired) {
  // Boundary: now == t_send + ttl uses '>' so it is still valid.
  const auto t_send = steady_clock::now();
  EXPECT_FALSE(LeaseExpired(t_send, /*lease_ttl_ms=*/1000, t_send + milliseconds(1000)));
}

TEST(SsdReadLeaseGating, ZeroTtlIsBornExpired) {
  const auto t_send = steady_clock::now();
  EXPECT_FALSE(LeaseExpired(t_send, /*lease_ttl_ms=*/0, t_send));  // exactly t_send: valid
  EXPECT_TRUE(LeaseExpired(t_send, /*lease_ttl_ms=*/0, t_send + milliseconds(1)));
}

// ---- DecideSsdReadOutcome ----
// Situation A (not expired): a good RDMA serves + releases; a failed RDMA is a
// hard error but still releases (the lease is still ours).
// Situation B (expired): always a transient retry, and NEVER release (the slot
// is left for the peer's TTL reclaim), regardless of whether the RDMA "worked".

TEST(SsdReadLeaseGating, ValidAndRdmaOk_ServesAndReleases) {
  const auto d = DecideSsdReadOutcome(/*expired=*/false, /*rdma_ok=*/true);
  EXPECT_EQ(d.outcome, GateOutcome::kSuccess);
  EXPECT_TRUE(d.release);
}

TEST(SsdReadLeaseGating, ValidAndRdmaFailed_ErrorButReleases) {
  const auto d = DecideSsdReadOutcome(/*expired=*/false, /*rdma_ok=*/false);
  EXPECT_EQ(d.outcome, GateOutcome::kError);
  EXPECT_TRUE(d.release);
}

TEST(SsdReadLeaseGating, ExpiredWithRdmaOk_RetryNoRelease) {
  // The dangerous case: RDMA "succeeded" but the lease elapsed, so the bytes
  // are untrusted (the peer may have recycled the slot).  Must NOT be success.
  const auto d = DecideSsdReadOutcome(/*expired=*/true, /*rdma_ok=*/true);
  EXPECT_EQ(d.outcome, GateOutcome::kRetry);
  EXPECT_FALSE(d.release);
}

TEST(SsdReadLeaseGating, ExpiredWithRdmaFailed_RetryNoRelease) {
  const auto d = DecideSsdReadOutcome(/*expired=*/true, /*rdma_ok=*/false);
  EXPECT_EQ(d.outcome, GateOutcome::kRetry);
  EXPECT_FALSE(d.release);
}

}  // namespace
}  // namespace mori::umbp::ssd_read_lease
