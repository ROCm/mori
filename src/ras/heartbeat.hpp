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

// Round-trip time and clock offset from HEARTBEAT / HEARTBEAT_ACK, the way an NTP client
// measures a server.
//
//   T1  sender's monotonic clock when it sends the HEARTBEAT (carried in it)
//   H   how long the peer held it: receipt to ACK, on the peer's monotonic clock
//   W3  peer's wall clock when it sends the ACK
//   T4  sender's monotonic clock when the ACK arrives, W4 its wall clock then
//
//   rtt    = (T4 - T1) - H
//   offset = (W3 - H/2) - (W4 - (T4 - T1)/2)     peer wall minus ours, mid-exchange
//   bound  = rtt/2 + H/2 * kMaxRateError
//
// The round-trip time uses only monotonic clocks, so a wall-clock step can't change it. The
// offset compares the peer's wall clock at the middle of its hold with ours at the middle of
// the round trip, projected back from W4 along our monotonic clock, so a step on either side
// moves the estimate by the step and nothing else. The estimate is exact when the two
// directions take equal time. When they don't, the error is half their difference, at most
// rtt/2, plus a term for the two clocks running at different rates while the peer held the
// message: H is measured on the peer's clock, so rtt is off by up to H * kMaxRateError. That
// term is a fraction of a microsecond for any realistic hold.
//
// Every value here is local CLOCK_MONOTONIC or CLOCK_REALTIME in nanoseconds; nothing compares
// two machines' monotonic clocks.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "src/ras/protocol.hpp"

namespace mori {
namespace ras {

// Largest rate difference assumed between two hosts' clocks: NTP's frequency correction
// limit. Quartz is usually within 100 ppm.
constexpr double kMaxRateError = 500e-6;

struct TimingSample {
  int64_t rttNs{0};
  int64_t offsetNs{0};      // peer's wall clock minus ours
  int64_t errorBoundNs{0};  // the true offset lies within offsetNs +- errorBoundNs
  uint64_t atMonoNs{0};     // when the ACK arrived, on our monotonic clock
};

// One sample from the four timestamps. nullopt if they're impossible: the ACK arrived before
// the heartbeat left, or the peer claims to have held it longer than the whole round trip.
std::optional<TimingSample> ComputeTiming(uint64_t sendMonoNs, uint64_t recvMonoNs,
                                          int64_t recvWallNs, uint64_t holdNs,
                                          int64_t peerSendWallNs);

// The reply to `hb`, received at `recvMonoNs` and answered at `ackMonoNs` / `ackWallNs`.
proto::HeartbeatAck MakeAck(const proto::Heartbeat& hb, uint64_t recvMonoNs, uint64_t ackMonoNs,
                            int64_t ackWallNs);

// The sending side of one link: the heartbeats still waiting for an ACK and the samples the
// ACKs produce. Owned by the RAS thread.
class HeartbeatTracker {
 public:
  static constexpr size_t kOutstanding = 8;  // heartbeats remembered while waiting for ACKs
  static constexpr size_t kWindow = 8;       // samples Best() chooses from

  // The next heartbeat to send at `nowMonoNs`. Monotonic clocks can return the same value
  // twice, so the send time is bumped to stay unique on this link.
  proto::Heartbeat Next(uint64_t nowMonoNs, uint32_t flags);

  // The sample for `ack`, if it answers a heartbeat still outstanding. Each heartbeat is
  // matched at most once, so a duplicated or forged ACK yields nothing.
  std::optional<TimingSample> OnAck(const proto::HeartbeatAck& ack, uint64_t recvMonoNs,
                                    int64_t recvWallNs);

  const TimingSample* Last() const;
  // The sample with the smallest error bound among the last kWindow, the best offset estimate.
  const TimingSample* Best() const;

  uint64_t Sent() const { return sent_; }
  uint64_t Acked() const { return acked_; }
  uint64_t Unmatched() const { return unmatched_; }
  uint64_t Impossible() const { return impossible_; }

 private:
  std::array<uint64_t, kOutstanding> outstanding_{};  // send times; 0 is a free entry
  size_t nextOutstanding_{0};
  uint64_t lastSendMonoNs_{0};
  std::array<TimingSample, kWindow> window_{};
  size_t windowCount_{0};
  size_t windowNext_{0};
  uint64_t sent_{0};
  uint64_t acked_{0};
  uint64_t unmatched_{0};
  uint64_t impossible_{0};
};

}  // namespace ras
}  // namespace mori
