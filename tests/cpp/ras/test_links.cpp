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
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "src/ras/heartbeat.hpp"
#include "src/ras/runtime.hpp"
#include "tests/cpp/ras/fake_peer.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::ras;
using ras_test::FakePeer;
using ras_test::WaitFor;
namespace proto = mori::ras::proto;

namespace {

// Heartbeats every 50 ms; handshakes time out after 250 ms; reconnects back off from 50 ms.
std::unique_ptr<RasCore> StartCore(const std::string& name) {
  CoreParams p;
  p.config.enable = true;
  p.config.heartbeatMs = 50;
  p.config.timeoutFactor = 0.05;
  p.nodeId = "node-" + name;
  p.hostname = "host-" + name;
  p.host = "127.0.0.1";
  p.pid = getpid();
  auto core = std::make_unique<RasCore>(std::move(p));
  std::string error;
  EXPECT_TRUE(core->Start(&error)) << error;
  return core;
}

PeerAdded Registration(const std::string& local, const std::string& remote,
                       const RasEndpoint& endpoint) {
  PeerAdded e;
  e.localEngine = local;
  e.remoteEngine = remote;
  e.remoteHostname = "remote-host";
  e.remotePid = 1;
  e.endpoint = endpoint;
  return e;
}

void Register(RasCore& from, const std::string& local, const std::string& remote,
              const RasEndpoint& endpoint) {
  ASSERT_TRUE(from.Enqueue(Registration(local, remote, endpoint)));
}

std::optional<LinkInfo> Link(const RasCore& core, const RuntimeUuid& peer) {
  for (const LinkInfo& l : core.Snapshot().links) {
    if (l.peer == peer) return l;
  }
  return std::nullopt;
}

bool WaitLink(const RasCore& core, const RuntimeUuid& peer,
              const std::function<bool(const LinkInfo&)>& pred, int timeoutMs = 3000) {
  return WaitFor(
      [&] {
        std::optional<LinkInfo> l = Link(core, peer);
        return l.has_value() && pred(*l);
      },
      timeoutMs);
}

bool Connected(const LinkInfo& l) { return l.state == LinkState::kConnected; }

RuntimeUuid UuidBeside(const RuntimeUuid& u, bool lower) {
  // Same hi, so the order depends only on lo, which keeps its variant bits.
  return RuntimeUuid{u.hi, lower ? u.lo - 1 : u.lo + 1};
}

}  // namespace

TEST(RasLinks, TwoRuntimesThatRegisterEachOtherShareOneLinkAndExchangeHeartbeats) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  RasEndpoint ea = a->Endpoint(), eb = b->Endpoint();
  Register(*a, "engine-a", "engine-b", eb);
  Register(*b, "engine-b", "engine-a", ea);

  auto flowing = [](const LinkInfo& l) {
    return Connected(l) && l.heartbeatsReceived >= 3 && l.acksReceived >= 3 && l.hasTiming;
  };
  ASSERT_TRUE(WaitLink(*a, eb.uuid, flowing));
  ASSERT_TRUE(WaitLink(*b, ea.uuid, flowing));
  LinkInfo la = *Link(*a, eb.uuid), lb = *Link(*b, ea.uuid);
  // One TCP connection, and both sides agree on who opened it.
  EXPECT_NE(la.outbound, lb.outbound);
  EXPECT_EQ(la.connects, 1u);
  EXPECT_EQ(lb.connects, 1u);
  EXPECT_EQ(la.version, 1);
  EXPECT_EQ(la.relations, 1u);
  EXPECT_EQ(la.peerHostname, "host-b");
  EXPECT_EQ(lb.peerHostname, "host-a");
  EXPECT_FALSE(la.peerDegraded);
  // Same host, same clock: the estimate must contain an offset of zero.
  EXPECT_GT(la.rttNs, 0);
  EXPECT_LT(la.rttNs, 100000000);
  EXPECT_LE(std::llabs(la.offsetNs), la.offsetBoundNs);
  EXPECT_LE(std::llabs(lb.offsetNs), lb.offsetBoundNs);
  EXPECT_GE(la.lastRxAgeMs, 0);
  EXPECT_LT(la.lastRxAgeMs, 1000);
}

TEST(RasLinks, AHigherUuidThatAloneRegisteredStillOpensTheLink) {
  // MORI-IO's initiator must register its target, but the target needn't register back, so
  // "the lower UUID connects" alone would leave this link unopened whenever the target is lower.
  auto x = StartCore("x");
  auto y = StartCore("y");
  RasCore& higher = x->Endpoint().uuid < y->Endpoint().uuid ? *y : *x;
  RasCore& lower = &higher == x.get() ? *y : *x;
  Register(higher, "initiator", "target", lower.Endpoint());

  ASSERT_TRUE(WaitLink(higher, lower.Endpoint().uuid, [](const LinkInfo& l) {
    return Connected(l) && l.outbound && l.acksReceived >= 2;
  }));
  ASSERT_TRUE(WaitLink(lower, higher.Endpoint().uuid, [](const LinkInfo& l) {
    return Connected(l) && !l.outbound && l.relations == 0 && l.acksReceived >= 2;
  }));
}

TEST(RasLinks, CrossingConnectionsKeepTheOneTheLowerUuidOpened) {
  for (bool fakeIsLower : {true, false}) {
    SCOPED_TRACE(fakeIsLower ? "the fake peer has the lower UUID" : "the core has the lower UUID");
    auto core = StartCore("c");
    RasEndpoint ec = core->Endpoint();
    FakePeer fake(UuidBeside(ec.uuid, fakeIsLower), "fake-token");
    // The core connects to the fake, which accepts but holds back its HELLO reply...
    Register(*core, "engine-c", "engine-f", fake.Endpoint());
    int fromCore = fake.Accept(2000);
    ASSERT_GE(fromCore, 0);
    std::optional<proto::Frame> request = fake.Recv(fromCore, 2000);
    ASSERT_TRUE(request.has_value());
    // ...and meanwhile opens its own connection to the core: the two cross.
    int toCore = fake.Connect(ec.port);
    ASSERT_GE(toCore, 0);
    ASSERT_TRUE(fake.Send(toCore, 1, fake.RequestTo(ec)));
    std::optional<proto::Frame> reply = fake.Recv(toCore, 2000);
    ASSERT_TRUE(reply.has_value());
    const proto::Hello& hello = std::get<proto::Hello>(reply->message);
    EXPECT_EQ(hello.role, proto::HelloRole::kReply);
    if (fakeIsLower) {
      // The fake's connection wins; the core drops its own.
      EXPECT_EQ(hello.status, proto::HelloStatus::kOk);
      EXPECT_TRUE(FakePeer::WaitClosed(fromCore, 2000));
      EXPECT_TRUE(WaitLink(*core, fake.identity.uuid,
                           [](const LinkInfo& l) { return Connected(l) && !l.outbound; }));
    } else {
      // The core keeps its own connection and refuses the fake's.
      EXPECT_EQ(hello.status, proto::HelloStatus::kRejectedDuplicate);
      EXPECT_TRUE(FakePeer::WaitClosed(toCore, 2000));
      ASSERT_TRUE(fake.Send(fromCore, 1, fake.Reply(proto::HelloStatus::kOk)));
      EXPECT_TRUE(WaitLink(*core, fake.identity.uuid,
                           [](const LinkInfo& l) { return Connected(l) && l.outbound; }));
    }
  }
}

TEST(RasLinks, ATokenMismatchIsRejected) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  RasEndpoint wrong = b->Endpoint();
  wrong.token = "00000000000000000000000000000000";
  Register(*a, "engine-a", "engine-b", wrong);
  ASSERT_TRUE(WaitLink(*a, wrong.uuid, [](const LinkInfo& l) {
    return l.lastError == LinkError::kTokenMismatch && l.connectAttempts >= 2;
  }));
  LinkInfo l = *Link(*a, wrong.uuid);
  EXPECT_NE(l.state, LinkState::kConnected);
  EXPECT_EQ(l.heartbeatsReceived, 0u);
  EXPECT_EQ(l.connects, 0u);
  EXPECT_TRUE(WaitFor([&] { return b->Snapshot().inboundRejected >= 2; }, 2000));
  EXPECT_FALSE(Link(*b, a->Endpoint().uuid).has_value());
}

TEST(RasLinks, AnEndpointNowHeldByAnotherRuntimeIsReported) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  // A stale EngineDesc: the port and token are B's, the UUID a runtime that is gone.
  RasEndpoint stale = b->Endpoint();
  stale.uuid = UuidBeside(stale.uuid, true);
  Register(*a, "engine-a", "engine-old", stale);
  ASSERT_TRUE(WaitLink(
      *a, stale.uuid, [](const LinkInfo& l) { return l.lastError == LinkError::kEndpointReused; }));
  EXPECT_NE(Link(*a, stale.uuid)->state, LinkState::kConnected);
}

TEST(RasLinks, ANewerOnlyPeerIsRejectedForItsVersion) {
  auto core = StartCore("v");
  RasEndpoint ec = core->Endpoint();
  FakePeer fake(UuidBeside(ec.uuid, true), "fake-token");

  // Inbound: a HELLO from a peer that speaks only versions 2 and 3.
  int fd = fake.Connect(ec.port);
  proto::Hello request = fake.RequestTo(ec);
  request.minVersion = 2;
  request.maxVersion = 3;
  ASSERT_TRUE(fake.Send(fd, 1, request, /*version=*/2));
  std::optional<proto::Frame> reply = fake.Recv(fd, 2000);
  ASSERT_TRUE(reply.has_value());
  const proto::Hello& hello = std::get<proto::Hello>(reply->message);
  EXPECT_EQ(hello.status, proto::HelloStatus::kRejectedVersion);
  EXPECT_EQ(hello.minVersion, 1);
  EXPECT_EQ(hello.maxVersion, 1);
  EXPECT_TRUE(FakePeer::WaitClosed(fd, 2000));

  // Outbound: an acceptor that answers with the same rejection, in its own version.
  Register(*core, "engine-v", "engine-f", fake.Endpoint());
  int fromCore = fake.Accept(2000);
  ASSERT_GE(fromCore, 0);
  ASSERT_TRUE(fake.Recv(fromCore, 2000).has_value());
  proto::Hello rejection = fake.Reply(proto::HelloStatus::kRejectedVersion);
  rejection.minVersion = 2;
  rejection.maxVersion = 3;
  ASSERT_TRUE(fake.Send(fromCore, 1, rejection, /*version=*/2));
  EXPECT_TRUE(WaitLink(*core, fake.identity.uuid, [](const LinkInfo& l) {
    return l.lastError == LinkError::kVersionMismatch;
  }));
}

TEST(RasLinks, GarbageAndSilenceAreDroppedWithoutHurtingTheRuntime) {
  auto a = StartCore("a");
  RasEndpoint ea = a->Endpoint();
  FakePeer fake(UuidBeside(ea.uuid, true), "fake-token");

  // An HTTP probe, long enough to fill a header: rejected as soon as it is read.
  int http = fake.Connect(ea.port);
  std::string get = "GET / HTTP/1.1\r\nHost: x\r\nUser-Agent: probe/1.0\r\n\r\n";
  ASSERT_GT(get.size(), proto::kHeaderSize);
  ASSERT_TRUE(FakePeer::SendRaw(http, std::vector<uint8_t>(get.begin(), get.end())));
  EXPECT_TRUE(FakePeer::WaitClosed(http, 200));

  // Half a header, then nothing: closed at the 250 ms handshake timeout.
  int silent = fake.Connect(ea.port);
  ASSERT_TRUE(FakePeer::SendRaw(silent, {0x01, 0x00, 0x01}));
  auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(FakePeer::WaitClosed(silent, 3000));
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(2000));

  // A well-formed HELLO that is out of sequence.
  int outOfSequence = fake.Connect(ea.port);
  ASSERT_TRUE(fake.Send(outOfSequence, 5, fake.RequestTo(ea)));
  EXPECT_TRUE(FakePeer::WaitClosed(outOfSequence, 2000));

  EXPECT_TRUE(WaitFor([&] { return a->Snapshot().protocolErrors >= 2; }, 2000));
  // The runtime still links normally.
  auto b = StartCore("b");
  Register(*b, "engine-b", "engine-a", ea);
  EXPECT_TRUE(WaitLink(*b, ea.uuid, [](const LinkInfo& l) { return l.acksReceived >= 1; }));
}

TEST(RasLinks, ARefusedConnectIsRasEndpointUnavailableAndRetriedWithBackoff) {
  auto a = StartCore("a");
  int held = -1;
  uint16_t port = ras_test::HoldPort("127.0.0.1", &held);
  close(held);  // nothing listens there now
  RasEndpoint dead{"127.0.0.1", port, "0123456789abcdef0123456789abcdef",
                   UuidBeside(a->Endpoint().uuid, true)};
  Register(*a, "engine-a", "engine-dead", dead);
  ASSERT_TRUE(WaitLink(*a, dead.uuid, [](const LinkInfo& l) {
    return l.lastError == LinkError::kRasEndpointUnavailable && l.connectAttempts >= 3;
  }));
  // Backoff doubles from 50 ms toward 1.5 s, so a second holds a handful of attempts.
  uint64_t before = Link(*a, dead.uuid)->connectAttempts;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  uint64_t after = Link(*a, dead.uuid)->connectAttempts;
  EXPECT_LE(after - before, 4u);
  EXPECT_NE(Link(*a, dead.uuid)->state, LinkState::kConnected);
}

TEST(RasLinks, LeaveAtStopRetiresThePeer) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  RasEndpoint eb = b->Endpoint();
  Register(*a, "engine-a", "engine-b", eb);
  Register(*b, "engine-b", "engine-a", a->Endpoint());
  ASSERT_TRUE(WaitLink(*a, eb.uuid, Connected));
  b->Stop(/*sendLeave=*/true);
  ASSERT_TRUE(
      WaitLink(*a, eb.uuid, [](const LinkInfo& l) { return l.state == LinkState::kPeerLeft; }));
  // A runtime that left is gone for good: no reconnects.
  uint64_t attempts = Link(*a, eb.uuid)->connectAttempts;
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_EQ(Link(*a, eb.uuid)->connectAttempts, attempts);
  EXPECT_EQ(Link(*a, eb.uuid)->state, LinkState::kPeerLeft);
}

TEST(RasLinks, AnAbruptCloseIsRetriedAtOnceAndThenRefused) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  RasEndpoint eb = b->Endpoint();
  Register(*a, "engine-a", "engine-b", eb);
  ASSERT_TRUE(WaitLink(*a, eb.uuid, Connected));
  b->Stop(/*sendLeave=*/false);  // sockets close with no LEAVE, as when a process is killed
  ASSERT_TRUE(WaitLink(*a, eb.uuid, [](const LinkInfo& l) {
    return l.lastError == LinkError::kRasEndpointUnavailable;
  }));
  LinkInfo l = *Link(*a, eb.uuid);
  EXPECT_NE(l.state, LinkState::kConnected);  // between retries, or in one
  EXPECT_EQ(l.disconnects, 1u);
  // The history shows the close and then the refused reconnect.
  ASSERT_GE(l.transitions.size(), 2u);
  bool closedThenRefused = false;
  for (size_t i = 1; i < l.transitions.size(); ++i) {
    if (l.transitions[i - 1].error == LinkError::kLinkClosed &&
        l.transitions[i].error == LinkError::kRasEndpointUnavailable) {
      closedThenRefused = true;
    }
  }
  EXPECT_TRUE(closedThenRefused);
}

TEST(RasLinks, ReleasingTheLastRelationSendsLeaveAndARegisteredPeerReconnects) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  RasEndpoint ea = a->Endpoint(), eb = b->Endpoint();
  // Only A registers: releasing it removes the link on both sides.
  Register(*a, "engine-a", "engine-b", eb);
  ASSERT_TRUE(WaitLink(*b, ea.uuid, Connected));
  ASSERT_TRUE(a->Enqueue(PeerRemoved{"engine-a", "engine-b"}));
  EXPECT_TRUE(WaitFor([&] { return !Link(*a, eb.uuid) && !Link(*b, ea.uuid); }, 2000));

  // Both register, then A's engine goes away. B still needs the link: if A had opened it, A
  // releases it and B reopens it; if B had, A keeps it. Either way B ends up the opener.
  Register(*a, "engine-a", "engine-b", eb);
  Register(*b, "engine-b", "engine-a", ea);
  ASSERT_TRUE(WaitLink(*a, eb.uuid, Connected));
  ASSERT_TRUE(WaitLink(*b, ea.uuid, Connected));
  ASSERT_TRUE(a->Enqueue(EngineUnregistered{"engine-a"}));
  ASSERT_TRUE(WaitLink(*b, ea.uuid, [&](const LinkInfo& l) {
    return Connected(l) && l.outbound && l.acksReceived >= 1;
  }));
  EXPECT_TRUE(WaitLink(*a, eb.uuid, [](const LinkInfo& l) {
    return Connected(l) && !l.outbound && l.relations == 0;
  }));
}

TEST(RasLinks, ADegradedRuntimeSaysSoInHelloAndInHeartbeats) {
  auto a = StartCore("a");
  auto b = StartCore("b");
  auto c = StartCore("c");
  b->AddDegraded(kDegradedForTest);
  Register(*a, "engine-a", "engine-b", b->Endpoint());
  ASSERT_TRUE(WaitLink(*a, b->Endpoint().uuid,
                       [](const LinkInfo& l) { return Connected(l) && l.peerDegraded; }));
  // A runtime that degrades after the handshake reports it in its next heartbeat.
  Register(*a, "engine-a", "engine-c", c->Endpoint());
  ASSERT_TRUE(WaitLink(*a, c->Endpoint().uuid,
                       [](const LinkInfo& l) { return Connected(l) && !l.peerDegraded; }));
  c->AddDegraded(kDegradedForTest);
  EXPECT_TRUE(WaitLink(*a, c->Endpoint().uuid, [](const LinkInfo& l) { return l.peerDegraded; }));
}

TEST(RasLinks, TheNewerOfTwoConnectionsAPeerOpenedWins) {
  auto core = StartCore("n");
  RasEndpoint ec = core->Endpoint();
  FakePeer fake(UuidBeside(ec.uuid, false), "fake-token");
  int first = fake.Connect(ec.port);
  ASSERT_TRUE(fake.Send(first, 1, fake.RequestTo(ec)));
  ASSERT_TRUE(fake.Recv(first, 2000).has_value());
  // The fake gave up on its first connection without closing it, and opens another.
  int second = fake.Connect(ec.port);
  ASSERT_TRUE(fake.Send(second, 1, fake.RequestTo(ec)));
  std::optional<proto::Frame> reply = fake.Recv(second, 2000);
  ASSERT_TRUE(reply.has_value());
  EXPECT_EQ(std::get<proto::Hello>(reply->message).status, proto::HelloStatus::kOk);
  EXPECT_TRUE(FakePeer::WaitClosed(first, 2000));
}

TEST(RasLinks, HeartbeatsAreAnsweredAtOnceAndSentOnSchedule) {
  auto core = StartCore("h");
  RasEndpoint ec = core->Endpoint();
  FakePeer fake(UuidBeside(ec.uuid, true), "fake-token");
  int fd = fake.Connect(ec.port);
  ASSERT_TRUE(fake.Send(fd, 1, fake.RequestTo(ec)));
  std::optional<proto::Frame> reply = fake.Recv(fd, 2000);
  ASSERT_TRUE(reply.has_value());
  ASSERT_EQ(std::get<proto::Hello>(reply->message).status, proto::HelloStatus::kOk);

  uint64_t seq = 2;
  int heartbeatsFromCore = 0;
  for (int i = 0; i < 5; ++i) {
    uint64_t sent = MonoNowNs();
    ASSERT_TRUE(fake.Send(fd, seq++, proto::Heartbeat{sent, 0}));
    for (;;) {
      std::optional<proto::Frame> f = fake.Recv(fd, 1000);
      ASSERT_TRUE(f.has_value());
      if (std::holds_alternative<proto::Heartbeat>(f->message)) {
        ++heartbeatsFromCore;
        // Answer the core's own heartbeats, so it can measure the link too.
        ASSERT_TRUE(fake.Send(fd, seq++,
                              MakeAck(std::get<proto::Heartbeat>(f->message), MonoNowNs(),
                                      MonoNowNs(), WallNowNs())));
        continue;
      }
      ASSERT_TRUE(std::holds_alternative<proto::HeartbeatAck>(f->message));
      const auto& ack = std::get<proto::HeartbeatAck>(f->message);
      EXPECT_EQ(ack.echoSendMonoNs, sent);
      EXPECT_LT(ack.holdNs, 20000000u);           // held for well under 20 ms
      EXPECT_LT(MonoNowNs() - sent, 100000000u);  // answered within 100 ms
      EXPECT_LE(std::llabs(ack.sendWallNs - WallNowNs()), 1000000000ll);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
  }
  EXPECT_GE(heartbeatsFromCore, 3);  // every 50 ms
  EXPECT_TRUE(WaitLink(*core, fake.identity.uuid,
                       [](const LinkInfo& l) { return l.hasTiming && l.heartbeatsReceived >= 5; }));
}
