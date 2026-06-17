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
#pragma once

// Pure decision helpers for the reader-side remote SSD read lease path.
//
// These are intentionally free of PoolClient / RDMA / IO state so the lease
// gating policy can be unit-tested in isolation.  The reader anchors its lease
// deadline at the moment BEFORE sending PrepareSsdRead (t_send) and treats the
// read as valid only while now <= t_send + lease_ttl_ms.  This is strictly
// conservative against the peer: the peer starts the same TTL when it RECEIVES
// the request (received_at > t_send), so its reclaim point (received_at + ttl)
// is always later than the reader's deadline (t_send + ttl) — the reader gives
// up before the peer can reclaim and reassign the slot, so a read is never
// reported successful against a slot the peer has already recycled.
//
// Lease expiry is therefore a reader-local decision, not a wire status: there
// is no SSD_READ_LEASE_EXPIRED.  On expiry the reader returns a transient
// retry (never a miss) and does NOT release — the peer reclaims by TTL.  A
// reader-local lease expiry and a NO_SLOT response are the retryable outcomes;
// a failed/timed-out PrepareSsdRead is a hard failure (retrying a slow peer
// that may already hold a claimed slot just piles up more staging occupation),
// so it does not route through kRetry.

#include <chrono>
#include <cstdint>

namespace mori::umbp::ssd_read_lease {

// Reader-local lease outcome category, independent of PoolClient internals.
// Translated to PoolClient::SsdGetOutcome at the call site.
enum class GateOutcome { kSuccess, kRetry, kError };

struct GateDecision {
  GateOutcome outcome;
  bool release;  // whether to issue a best-effort ReleaseSsdLease
};

// True once now is strictly past the deadline (t_send + lease_ttl_ms); now ==
// deadline is still valid.  ttl 0 expires as soon as now > t_send.
inline bool LeaseExpired(std::chrono::steady_clock::time_point t_send, uint64_t lease_ttl_ms,
                         std::chrono::steady_clock::time_point now) {
  return now > t_send + std::chrono::milliseconds(lease_ttl_ms);
}

// Decide the outcome of a prepared read.  `expired` short-circuits to a
// transient retry with no release (the slot is left for the peer's TTL
// reclaim).  Otherwise a successful RDMA serves the data and frees the slot
// fast via release; a failed RDMA is a hard error but still releases (the
// lease is still ours and matching by lease_id is safe).
inline GateDecision DecideSsdReadOutcome(bool expired, bool rdma_ok) {
  if (expired) return {GateOutcome::kRetry, /*release=*/false};
  if (rdma_ok) return {GateOutcome::kSuccess, /*release=*/true};
  return {GateOutcome::kError, /*release=*/true};
}

}  // namespace mori::umbp::ssd_read_lease
