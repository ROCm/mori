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
#include "src/ras/heartbeat.hpp"

#include <algorithm>
#include <limits>

namespace mori {
namespace ras {
namespace {

constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
// Integer halving in the offset formula and in the timestamps themselves.
constexpr int64_t kRoundingSlackNs = 4;

// How far a hold of `holdNs`, measured on the peer's clock, can be off on ours.
int64_t RateSlack(int64_t holdNs) {
  constexpr int64_t kPerNs = static_cast<int64_t>(1 / kMaxRateError);  // 2000 for 500 ppm
  return (holdNs + kPerNs - 1) / kPerNs;
}

}  // namespace

std::optional<TimingSample> ComputeTiming(uint64_t sendMonoNs, uint64_t recvMonoNs,
                                          int64_t recvWallNs, uint64_t holdNs,
                                          int64_t peerSendWallNs) {
  if (recvMonoNs < sendMonoNs) return std::nullopt;
  uint64_t elapsedU = recvMonoNs - sendMonoNs;
  if (elapsedU > static_cast<uint64_t>(kInt64Max) || holdNs > static_cast<uint64_t>(kInt64Max)) {
    return std::nullopt;
  }
  int64_t elapsed = static_cast<int64_t>(elapsedU);
  int64_t hold = static_cast<int64_t>(holdNs);
  int64_t slack = RateSlack(hold);
  if (hold > elapsed + slack) return std::nullopt;
  int64_t rtt = std::max<int64_t>(elapsed - hold, 0);

  __int128 offset = static_cast<__int128>(peerSendWallNs) - hold / 2 -
                    (static_cast<__int128>(recvWallNs) - elapsed / 2);
  if (offset > kInt64Max || offset < -kInt64Max) return std::nullopt;

  TimingSample s;
  s.rttNs = rtt;
  s.offsetNs = static_cast<int64_t>(offset);
  s.errorBoundNs = (rtt + 1) / 2 + (slack + 1) / 2 + kRoundingSlackNs;
  s.atMonoNs = recvMonoNs;
  return s;
}

proto::HeartbeatAck MakeAck(const proto::Heartbeat& hb, uint64_t recvMonoNs, uint64_t ackMonoNs,
                            int64_t ackWallNs) {
  proto::HeartbeatAck ack;
  ack.echoSendMonoNs = hb.sendMonoNs;
  ack.holdNs = ackMonoNs >= recvMonoNs ? ackMonoNs - recvMonoNs : 0;
  ack.sendWallNs = ackWallNs;
  return ack;
}

proto::Heartbeat HeartbeatTracker::Next(uint64_t nowMonoNs, uint32_t flags) {
  uint64_t t = std::max<uint64_t>(nowMonoNs, 1);  // 0 marks a free outstanding entry
  if (t <= lastSendMonoNs_) t = lastSendMonoNs_ + 1;
  lastSendMonoNs_ = t;
  outstanding_[nextOutstanding_] = t;
  nextOutstanding_ = (nextOutstanding_ + 1) % kOutstanding;
  ++sent_;
  return proto::Heartbeat{t, flags};
}

std::optional<TimingSample> HeartbeatTracker::OnAck(const proto::HeartbeatAck& ack,
                                                    uint64_t recvMonoNs, int64_t recvWallNs) {
  auto it = ack.echoSendMonoNs == 0
                ? outstanding_.end()
                : std::find(outstanding_.begin(), outstanding_.end(), ack.echoSendMonoNs);
  if (it == outstanding_.end()) {
    ++unmatched_;
    return std::nullopt;
  }
  *it = 0;
  std::optional<TimingSample> s =
      ComputeTiming(ack.echoSendMonoNs, recvMonoNs, recvWallNs, ack.holdNs, ack.sendWallNs);
  if (!s) {
    ++impossible_;
    return std::nullopt;
  }
  ++acked_;
  window_[windowNext_] = *s;
  windowNext_ = (windowNext_ + 1) % kWindow;
  windowCount_ = std::min(windowCount_ + 1, kWindow);
  return s;
}

const TimingSample* HeartbeatTracker::Last() const {
  if (windowCount_ == 0) return nullptr;
  return &window_[(windowNext_ + kWindow - 1) % kWindow];
}

const TimingSample* HeartbeatTracker::Best() const {
  const TimingSample* best = nullptr;
  for (size_t i = 0; i < windowCount_; ++i) {
    if (best == nullptr || window_[i].errorBoundNs < best->errorBoundNs) best = &window_[i];
  }
  return best;
}

}  // namespace ras
}  // namespace mori
