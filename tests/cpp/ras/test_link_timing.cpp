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

#include <cstdlib>
#include <optional>
#include <random>
#include <vector>

#include "src/ras/heartbeat.hpp"

using namespace mori::ras;

namespace {

// A host's clocks as functions of true time t (ns). Both run `ppm` fast; the wall clock also
// carries an offset and any steps applied to it.
struct SimClock {
  int64_t monoBase;
  int64_t wallBase;
  int64_t ppm;
  int64_t wallStep{0};

  int64_t Scaled(int64_t t) const {
    return t + static_cast<int64_t>(static_cast<__int128>(t) * ppm / 1000000);
  }
  uint64_t Mono(int64_t t) const { return static_cast<uint64_t>(monoBase + Scaled(t)); }
  int64_t Wall(int64_t t) const { return wallBase + wallStep + Scaled(t); }
};

constexpr int64_t kWallEpoch = 1700000000000000000ll;
constexpr int64_t kInjectedOffset = 3250000123ll;  // B's wall clock is 3.25 s ahead of A's

SimClock HostA() { return SimClock{5000000000000ll, kWallEpoch, 0}; }
SimClock HostB(int64_t ppm) {
  return SimClock{900000000000000ll, kWallEpoch + kInjectedOffset, ppm};
}

struct Exchange {
  std::optional<TimingSample> sample;
  int64_t trueOffset;  // to's wall minus from's wall, at the middle of to's hold
};

// `from` sends a heartbeat at true time t1. It takes dOut to reach `to`, which holds it for
// `hold` and replies, and the reply takes dBack to come back.
Exchange Simulate(HeartbeatTracker& tracker, const SimClock& from, const SimClock& to, int64_t t1,
                  int64_t dOut, int64_t hold, int64_t dBack) {
  mori::ras::proto::Heartbeat hb = tracker.Next(from.Mono(t1), 0);
  int64_t t2 = t1 + dOut;
  int64_t t3 = t2 + hold;
  int64_t t4 = t3 + dBack;
  mori::ras::proto::HeartbeatAck ack = MakeAck(hb, to.Mono(t2), to.Mono(t3), to.Wall(t3));
  Exchange e;
  e.sample = tracker.OnAck(ack, from.Mono(t4), from.Wall(t4));
  int64_t tm = t2 + hold / 2;
  e.trueOffset = to.Wall(tm) - from.Wall(tm);
  return e;
}

// Rate effects on two RTTs measured by clocks `ppm` apart.
int64_t RateTolerance(int64_t ppm, int64_t span) { return std::llabs(ppm) * span / 1000000 + 4; }

}  // namespace

TEST(RasLinkTiming, BothSidesReportTheSameRoundTripTime) {
  SimClock a = HostA();
  SimClock b = HostB(100);
  HeartbeatTracker ta, tb;
  const int64_t dAB = 40000;   // A to B: 40 us
  const int64_t dBA = 950000;  // B to A: 950 us
  for (int i = 0; i < 50; ++i) {
    int64_t t = 1000000000ll * i;
    int64_t hold = 20000 + 3000 * i;
    Exchange fromA = Simulate(ta, a, b, t, dAB, hold, dBA);
    Exchange fromB = Simulate(tb, b, a, t + 500000000ll, dBA, hold, dAB);
    ASSERT_TRUE(fromA.sample.has_value());
    ASSERT_TRUE(fromB.sample.has_value());
    int64_t tol = RateTolerance(100, dAB + dBA + 2 * hold);
    EXPECT_NEAR(fromA.sample->rttNs, fromB.sample->rttNs, tol) << i;
    EXPECT_NEAR(fromA.sample->rttNs, dAB + dBA, tol) << i;
  }
}

TEST(RasLinkTiming, TheInjectedOffsetAlwaysLiesWithinTheReportedBound) {
  for (int64_t ppm : {100, -100}) {
    SimClock a = HostA();
    SimClock b = HostB(ppm);
    HeartbeatTracker ta, tb;
    std::mt19937_64 rng(42 + ppm);
    std::uniform_int_distribution<int64_t> delay(0, 5000000);  // up to 5 ms each way
    std::uniform_int_distribution<int64_t> holdDist(0, 1000000);
    size_t checked = 0;
    auto check = [&](const Exchange& e, int64_t expectedSign) {
      ASSERT_TRUE(e.sample.has_value());
      EXPECT_LE(std::llabs(e.sample->offsetNs - e.trueOffset), e.sample->errorBoundNs)
          << "rtt " << e.sample->rttNs << " offset " << e.sample->offsetNs << " true "
          << e.trueOffset;
      // The bound stays useful: half the round trip plus well under a microsecond.
      EXPECT_LE(e.sample->errorBoundNs, e.sample->rttNs / 2 + 1000);
      EXPECT_EQ(e.trueOffset > 0 ? 1 : -1, expectedSign);
      ++checked;
    };
    int64_t t = 0;
    for (int i = 0; i < 5000; ++i) {
      t += 1000000000ll;
      int64_t d1 = delay(rng), d2 = delay(rng), hold = holdDist(rng);
      check(Simulate(ta, a, b, t, d1, hold, d2), +1);
      check(Simulate(tb, b, a, t + 1000, d2, hold, d1), -1);
    }
    // The worst cases: one direction takes the whole trip.
    for (int64_t d : {int64_t{0}, int64_t{1}, int64_t{5000000}}) {
      for (int64_t hold : {int64_t{0}, int64_t{1000000}}) {
        t += 1000000000ll;
        check(Simulate(ta, a, b, t, d, hold, 0), +1);
        check(Simulate(ta, a, b, t + 10000000, 0, hold, d), +1);
        check(Simulate(tb, b, a, t + 20000000, d, hold, 0), -1);
        check(Simulate(tb, b, a, t + 30000000, 0, hold, d), -1);
      }
    }
    EXPECT_EQ(checked, 2u * 5000 + 4 * 6);
  }
}

TEST(RasLinkTiming, AWallClockStepMovesTheOffsetButNeverTheRoundTripTime) {
  SimClock a = HostA();
  SimClock b = HostB(100);
  HeartbeatTracker ta;
  const int64_t dOut = 120000, dBack = 380000, hold = 50000;
  auto within = [](const Exchange& e) {
    return std::llabs(e.sample->offsetNs - e.trueOffset) <= e.sample->errorBoundNs;
  };

  Exchange before = Simulate(ta, a, b, 1000000000ll, dOut, hold, dBack);
  ASSERT_TRUE(before.sample && within(before));

  b.wallStep = 2500000000ll;  // the peer's wall clock steps forward 2.5 s
  Exchange peerStepped = Simulate(ta, a, b, 2000000000ll, dOut, hold, dBack);
  ASSERT_TRUE(peerStepped.sample && within(peerStepped));
  EXPECT_EQ(peerStepped.sample->rttNs, before.sample->rttNs);
  // 2.5 s, plus the 100 us the clocks drifted apart in the second between the samples.
  EXPECT_NEAR(peerStepped.sample->offsetNs - before.sample->offsetNs, 2500000000ll + 100000,
              before.sample->errorBoundNs + peerStepped.sample->errorBoundNs);

  a.wallStep = -1000000000ll;  // our own wall clock steps back 1 s
  Exchange selfStepped = Simulate(ta, a, b, 3000000000ll, dOut, hold, dBack);
  ASSERT_TRUE(selfStepped.sample && within(selfStepped));
  EXPECT_EQ(selfStepped.sample->rttNs, before.sample->rttNs);
  EXPECT_NEAR(selfStepped.sample->offsetNs - peerStepped.sample->offsetNs, 1000000000ll + 100000,
              peerStepped.sample->errorBoundNs + selfStepped.sample->errorBoundNs);

  // A step while the heartbeat is in flight: the estimate follows our clock as it reads when
  // the ACK arrives, and the round-trip time still doesn't move.
  int64_t t1 = 4000000000ll;
  mori::ras::proto::Heartbeat hb = ta.Next(a.Mono(t1), 0);
  int64_t t2 = t1 + dOut, t3 = t2 + hold, t4 = t3 + dBack;
  mori::ras::proto::HeartbeatAck ack = MakeAck(hb, b.Mono(t2), b.Mono(t3), b.Wall(t3));
  a.wallStep += 7000000000ll;
  std::optional<TimingSample> s = ta.OnAck(ack, a.Mono(t4), a.Wall(t4));
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->rttNs, before.sample->rttNs);
  int64_t tm = t2 + hold / 2;
  EXPECT_LE(std::llabs(s->offsetNs - (b.Wall(tm) - a.Wall(tm))), s->errorBoundNs);

  // A step on the peer during its hold: its reply carries the stepped clock.
  t1 = 5000000000ll;
  hb = ta.Next(a.Mono(t1), 0);
  t2 = t1 + dOut;
  t3 = t2 + hold;
  t4 = t3 + dBack;
  uint64_t received = b.Mono(t2);
  b.wallStep -= 300000000ll;
  ack = MakeAck(hb, received, b.Mono(t3), b.Wall(t3));
  s = ta.OnAck(ack, a.Mono(t4), a.Wall(t4));
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->rttNs, before.sample->rttNs);
  tm = t2 + hold / 2;
  EXPECT_LE(std::llabs(s->offsetNs - (b.Wall(tm) - a.Wall(tm))), s->errorBoundNs);
}

TEST(RasLinkTiming, AnAckMatchesOnlyAnOutstandingHeartbeatAndOnlyOnce) {
  HeartbeatTracker t;
  mori::ras::proto::Heartbeat hb = t.Next(1000000, 0);
  mori::ras::proto::HeartbeatAck ack = MakeAck(hb, 10, 20, 5000);
  EXPECT_TRUE(t.OnAck(ack, 2000000, 6000).has_value());
  EXPECT_FALSE(t.OnAck(ack, 2000100, 6100).has_value());             // duplicate
  EXPECT_FALSE(t.OnAck({123456, 0, 0}, 2000200, 6200).has_value());  // nothing sent then
  EXPECT_EQ(t.Unmatched(), 2u);
  EXPECT_EQ(t.Acked(), 1u);

  std::vector<mori::ras::proto::Heartbeat> sent;
  for (size_t i = 0; i < HeartbeatTracker::kOutstanding + 2; ++i) {
    sent.push_back(t.Next(3000000 + 1000 * i, 0));
  }
  uint64_t now = 9000000;
  // The two oldest fell out of the outstanding window.
  EXPECT_FALSE(t.OnAck(MakeAck(sent[0], 0, 0, 0), now, 0).has_value());
  EXPECT_FALSE(t.OnAck(MakeAck(sent[1], 0, 0, 0), now, 0).has_value());
  // The rest match, in any order.
  EXPECT_TRUE(t.OnAck(MakeAck(sent.back(), 0, 0, 0), now, 0).has_value());
  EXPECT_TRUE(t.OnAck(MakeAck(sent[4], 0, 0, 0), now, 0).has_value());
  EXPECT_TRUE(t.OnAck(MakeAck(sent[2], 0, 0, 0), now, 0).has_value());
  EXPECT_EQ(t.Sent(), 1 + sent.size());
}

TEST(RasLinkTiming, ImpossibleSamplesAreDiscarded) {
  EXPECT_FALSE(ComputeTiming(1000, 900, 0, 0, 0).has_value());      // ACK before the send
  EXPECT_FALSE(ComputeTiming(1000, 2000, 0, 5000, 0).has_value());  // held longer than the trip
  std::optional<TimingSample> s = ComputeTiming(1000, 2000, 0, 1000, 0);
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->rttNs, 0);
  // A hold just over the round trip, within what a rate difference explains, is a 0 round trip.
  s = ComputeTiming(0, 1000000, 0, 1000100, 0);
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->rttNs, 0);

  HeartbeatTracker t;
  mori::ras::proto::Heartbeat hb = t.Next(1000, 0);
  EXPECT_FALSE(t.OnAck({hb.sendMonoNs, 1000000, 0}, 2000, 0).has_value());
  EXPECT_EQ(t.Impossible(), 1u);
  EXPECT_EQ(t.Last(), nullptr);
  // The heartbeat was used up, so a second ACK for it doesn't match.
  EXPECT_FALSE(t.OnAck({hb.sendMonoNs, 10, 0}, 2000, 0).has_value());
  EXPECT_EQ(t.Unmatched(), 1u);
}

TEST(RasLinkTiming, BestPicksTheSmallestBoundInTheWindow) {
  HeartbeatTracker t;
  EXPECT_EQ(t.Last(), nullptr);
  EXPECT_EQ(t.Best(), nullptr);
  // Round trips in arrival order; the first, the smallest, falls out of the 8-sample window.
  const std::vector<uint64_t> rtts = {100, 900, 700, 800, 650, 990, 720, 680, 810, 300};
  uint64_t now = 1000000;
  for (uint64_t rtt : rtts) {
    mori::ras::proto::Heartbeat hb = t.Next(now, 0);
    ASSERT_TRUE(t.OnAck(MakeAck(hb, 0, 0, 0), now + rtt, 0).has_value());
    now += 1000000;
  }
  ASSERT_NE(t.Last(), nullptr);
  EXPECT_EQ(t.Last()->rttNs, 300);
  ASSERT_NE(t.Best(), nullptr);
  EXPECT_EQ(t.Best()->rttNs, 300);
  mori::ras::proto::Heartbeat hb = t.Next(now, 0);
  ASSERT_TRUE(t.OnAck(MakeAck(hb, 0, 0, 0), now + 5000, 0).has_value());
  EXPECT_EQ(t.Last()->rttNs, 5000);
  EXPECT_EQ(t.Best()->rttNs, 300);
}

TEST(RasLinkTiming, SendTimesStayUniqueWhenTheClockRepeats) {
  HeartbeatTracker t;
  mori::ras::proto::Heartbeat first = t.Next(5000, 0);
  mori::ras::proto::Heartbeat second = t.Next(5000, mori::ras::proto::kHeartbeatFlagDegraded);
  EXPECT_EQ(first.sendMonoNs, 5000u);
  EXPECT_EQ(second.sendMonoNs, 5001u);
  EXPECT_EQ(second.flags, mori::ras::proto::kHeartbeatFlagDegraded);
  EXPECT_TRUE(t.OnAck(MakeAck(first, 0, 0, 0), 6000, 0).has_value());
  EXPECT_TRUE(t.OnAck(MakeAck(second, 0, 0, 0), 6000, 0).has_value());
}
