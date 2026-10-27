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

#include <cstring>
#include <msgpack.hpp>
#include <random>
#include <string>
#include <vector>

#include "src/ras/protocol.hpp"
#include "tests/cpp/ras/protocol_fuzz_corpus.hpp"

using namespace mori::ras;
using namespace mori::ras::proto;

namespace {

constexpr VersionRange kV1{1, 1};
constexpr uint64_t kSender = 0x8877665544332211ull;

RasProcessId SampleIdentity() {
  RasProcessId id;
  id.uuid = {0x0123456789abcdefull, 0x8edcba9876543210ull};
  id.nodeId = "node-7";
  id.hostname = "host-7.example";
  id.host = "10.1.2.3";
  id.pid = 4242;
  id.pidNamespace = 4026531836ull;
  id.startTimeTicks = 123456789ull;
  id.bootId = "1b6c9a52-3f0e-4c8e-9a31-5d2c7e8f9a01";
  id.topology.rack = "r12";
  id.topology.leaf = "leaf-3";
  id.topology.source = "sysfs,file";
  id.topology.nics.push_back(
      {"ionic_3", 1, "enp9s0", "10.3.0.7", "10.3.0.0/24", "sw-r3", "Eth1/7"});
  id.topology.nics.push_back({"ionic_4", 1, "enp10s0", "fd00:4::7", "fd00:4::/64", "", ""});
  id.labels = {{"role", "prefill"}, {"instance", "3"}};
  return id;
}

Hello SampleHelloRequest() {
  Hello h;
  h.role = HelloRole::kRequest;
  h.status = HelloStatus::kOk;
  h.token = "00112233445566778899aabbccddeeff";
  h.expectedPeer = {0x1111ull, 0x8000000000002222ull};
  h.identity = SampleIdentity();
  return h;
}

Frame RoundTrip(const RasHeader& h, const Message& m) {
  std::vector<uint8_t> bytes = EncodeFrame(h, m);
  Frame f;
  EXPECT_EQ(DecodeFrame(bytes.data(), bytes.size(), kV1, &f), DecodeStatus::kOk);
  return f;
}

std::vector<uint8_t> HeartbeatFrame(uint64_t seq = 1) {
  return EncodeFrame(LinkHeader(1, kSender, seq), Heartbeat{0x0102030405060708ull, 0});
}

// A frame whose payload is exactly `payload`, with a consistent header.
std::vector<uint8_t> RawFrame(MessageType type, const std::vector<uint8_t>& payload) {
  RasHeader h = LinkHeader(1, kSender, 1);
  h.type = static_cast<uint16_t>(type);
  h.length = static_cast<uint32_t>(payload.size());
  std::vector<uint8_t> out(kHeaderSize);
  EncodeHeader(h, out.data());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

template <typename T>
std::vector<uint8_t> Pack(const T& v) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, v);
  return std::vector<uint8_t>(buf.data(), buf.data() + buf.size());
}

DecodeStatus DecodeBytes(const std::vector<uint8_t>& b) {
  Frame f;
  return DecodeFrame(b.data(), b.size(), kV1, &f);
}

}  // namespace

TEST(RasProtocol, HeaderGoldenBytesPinEveryFieldOffsetAndByteOrder) {
  static_assert(kHeaderSize == 36, "v1 header is 36 bytes");
  RasHeader h;
  h.version = 0x0102;
  h.type = 0x0304;
  h.length = 0x05060708;
  h.sender = 0x1112131415161718ull;
  h.sequence = 0x2122232425262728ull;
  h.target = 0x3132333435363738ull;
  h.scope = 0x41;
  h.hops = 0x42;
  h.reserved = 0x4344;
  const uint8_t golden[kHeaderSize] = {
      0x02, 0x01,                                      // version
      0x04, 0x03,                                      // type
      0x08, 0x07, 0x06, 0x05,                          // length
      0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,  // sender
      0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,  // sequence
      0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31,  // target
      0x41,                                            // scope
      0x42,                                            // hops
      0x44, 0x43,                                      // reserved
  };
  uint8_t out[kHeaderSize];
  std::memset(out, 0xee, sizeof(out));
  EncodeHeader(h, out);
  EXPECT_EQ(std::memcmp(out, golden, kHeaderSize), 0);

  RasHeader back = ReadHeaderFields(golden);
  EXPECT_EQ(back.version, h.version);
  EXPECT_EQ(back.type, h.type);
  EXPECT_EQ(back.length, h.length);
  EXPECT_EQ(back.sender, h.sender);
  EXPECT_EQ(back.sequence, h.sequence);
  EXPECT_EQ(back.target, h.target);
  EXPECT_EQ(back.scope, h.scope);
  EXPECT_EQ(back.hops, h.hops);
  EXPECT_EQ(back.reserved, h.reserved);
}

TEST(RasProtocol, HeartbeatFrameGoldenBytesPinThePayloadLayout) {
  const std::vector<uint8_t> golden = {
      0x01, 0x00,                                      // version 1
      0x04, 0x00,                                      // HEARTBEAT
      0x0b, 0x00, 0x00, 0x00,                          // 11 payload bytes
      0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,  // sender
      0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // sequence 7
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // target 0: the link peer
      0x00,                                            // scope LINK
      0x01,                                            // hops 1
      0x00, 0x00,                                      // reserved
      0x92,                                            // [sendMonoNs, flags]
      0xcf, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00,
  };
  EXPECT_EQ(HeartbeatFrame(7), golden);
}

TEST(RasProtocol, AckAndLeaveGoldenPayloads) {
  std::vector<uint8_t> ack =
      EncodeFrame(LinkHeader(1, kSender, 2), HeartbeatAck{0x0102030405060708ull, 1500, -2});
  const std::vector<uint8_t> ackPayload = {0x93, 0xcf, 0x01, 0x02, 0x03, 0x04, 0x05,
                                           0x06, 0x07, 0x08, 0xcd, 0x05, 0xdc, 0xfe};
  ASSERT_EQ(ack.size(), kHeaderSize + ackPayload.size());
  EXPECT_EQ(std::vector<uint8_t>(ack.begin() + kHeaderSize, ack.end()), ackPayload);
  EXPECT_EQ(ack[2], static_cast<uint8_t>(MessageType::kHeartbeatAck));

  std::vector<uint8_t> leave =
      EncodeFrame(LinkHeader(1, kSender, 3), Leave{LeaveReason::kLinkReleased});
  EXPECT_EQ(std::vector<uint8_t>(leave.begin() + kHeaderSize, leave.end()),
            (std::vector<uint8_t>{0x91, 0x01}));
  EXPECT_EQ(leave[2], static_cast<uint8_t>(MessageType::kLeave));
}

TEST(RasProtocol, RoundTripsEveryV1Message) {
  Hello request = SampleHelloRequest();
  Frame f = RoundTrip(LinkHeader(1, kSender, 1), request);
  EXPECT_EQ(f.header.type, static_cast<uint16_t>(MessageType::kHello));
  EXPECT_EQ(f.header.sender, kSender);
  ASSERT_TRUE(std::holds_alternative<Hello>(f.message));
  EXPECT_EQ(std::get<Hello>(f.message), request);

  Hello reply = request;
  reply.role = HelloRole::kReply;
  reply.status = HelloStatus::kDegraded;
  reply.token.clear();
  reply.expectedPeer = {};
  reply.identity.labels.clear();
  reply.identity.topology = {};
  f = RoundTrip(LinkHeader(1, kSender, 1), reply);
  EXPECT_EQ(std::get<Hello>(f.message), reply);

  for (LeaveReason r : {LeaveReason::kProcessExit, LeaveReason::kLinkReleased}) {
    f = RoundTrip(LinkHeader(1, kSender, 9), Leave{r});
    EXPECT_EQ(std::get<Leave>(f.message), Leave{r});
  }

  Heartbeat hb{0xfedcba9876543210ull, kHeartbeatFlagDegraded};
  f = RoundTrip(LinkHeader(1, kSender, 2), hb);
  EXPECT_EQ(f.header.sequence, 2u);
  EXPECT_EQ(std::get<Heartbeat>(f.message), hb);

  HeartbeatAck ack{123, 456, -1700000000000000000ll};
  f = RoundTrip(LinkHeader(1, kSender, 3), ack);
  EXPECT_EQ(std::get<HeartbeatAck>(f.message), ack);
  ack.sendWallNs = 1700000000123456789ll;
  f = RoundTrip(LinkHeader(1, kSender, 4), ack);
  EXPECT_EQ(std::get<HeartbeatAck>(f.message), ack);
}

TEST(RasProtocol, EncodeFrameTakesTypeAndLengthFromTheMessage) {
  RasHeader h = LinkHeader(1, kSender, 5);
  h.type = 0x7777;
  h.length = 99999;
  std::vector<uint8_t> bytes = EncodeFrame(h, Heartbeat{1, 0});
  RasHeader back = ReadHeaderFields(bytes.data());
  EXPECT_EQ(back.type, static_cast<uint16_t>(MessageType::kHeartbeat));
  EXPECT_EQ(back.length, bytes.size() - kHeaderSize);
}

TEST(RasProtocol, RejectsTruncatedMessages) {
  std::vector<uint8_t> frame = HeartbeatFrame();
  RasHeader h;
  for (size_t n = 0; n < kHeaderSize; ++n) {
    EXPECT_EQ(DecodeHeader(frame.data(), n, kV1, &h), DecodeStatus::kTruncated) << n;
  }
  for (size_t n = 0; n < frame.size(); ++n) {
    Frame f;
    EXPECT_EQ(DecodeFrame(frame.data(), n, kV1, &f), DecodeStatus::kTruncated) << n;
  }
  EXPECT_EQ(DecodeHeader(frame.data(), kHeaderSize, kV1, &h), DecodeStatus::kOk);
}

TEST(RasProtocol, RejectsOversizedLengthFromTheHeaderAlone) {
  RasHeader h = LinkHeader(1, kSender, 1);
  h.type = static_cast<uint16_t>(MessageType::kHeartbeat);
  h.length = kMaxPayload + 1;
  uint8_t buf[kHeaderSize];
  EncodeHeader(h, buf);
  RasHeader out;
  EXPECT_EQ(DecodeHeader(buf, kHeaderSize, kV1, &out), DecodeStatus::kOversized);
  h.length = 0xffffffffu;
  EncodeHeader(h, buf);
  EXPECT_EQ(DecodeHeader(buf, kHeaderSize, kV1, &out), DecodeStatus::kOversized);
  h.length = kMaxPayload;
  EncodeHeader(h, buf);
  EXPECT_EQ(DecodeHeader(buf, kHeaderSize, kV1, &out), DecodeStatus::kOk);
}

TEST(RasProtocol, RejectsUnknownAndReservedTypes) {
  std::vector<uint8_t> frame = HeartbeatFrame();
  auto withType = [&](uint16_t type) {
    std::vector<uint8_t> b = frame;
    b[2] = static_cast<uint8_t>(type);
    b[3] = static_cast<uint8_t>(type >> 8);
    return DecodeBytes(b);
  };
  for (uint16_t t : {uint16_t{0}, uint16_t{16}, uint16_t{0x0100}, uint16_t{0xffff}}) {
    EXPECT_EQ(withType(t), DecodeStatus::kUnknownType) << t;
  }
  for (MessageType t :
       {MessageType::kJoin, MessageType::kStateUpdate, MessageType::kProgressUpdate,
        MessageType::kPeerSuspect, MessageType::kPeerUnreachable, MessageType::kPeerExited,
        MessageType::kQuery, MessageType::kQueryResponse, MessageType::kControl,
        MessageType::kDiagnosticsRequest, MessageType::kDiagnosticsResponse}) {
    EXPECT_EQ(withType(static_cast<uint16_t>(t)), DecodeStatus::kReservedType)
        << static_cast<int>(t);
  }
}

TEST(RasProtocol, RejectsALengthThatDisagreesWithTheBytesReceived) {
  std::vector<uint8_t> frame = HeartbeatFrame();
  std::vector<uint8_t> longer = frame;
  longer.push_back(0x00);
  EXPECT_EQ(DecodeBytes(longer), DecodeStatus::kLengthMismatch);

  // The length covers a trailing byte that the payload's msgpack object doesn't use.
  std::vector<uint8_t> payload(frame.begin() + kHeaderSize, frame.end());
  payload.push_back(0xc0);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, payload)), DecodeStatus::kLengthMismatch);

  // DecodePayload must be handed exactly header.length bytes.
  RasHeader h = ReadHeaderFields(frame.data());
  Message m;
  EXPECT_EQ(DecodePayload(h, frame.data() + kHeaderSize, h.length - 1, &m),
            DecodeStatus::kLengthMismatch);
  EXPECT_EQ(DecodePayload(h, frame.data() + kHeaderSize, h.length, &m), DecodeStatus::kOk);
}

TEST(RasProtocol, RejectsVersionsOutsideTheNegotiatedRange) {
  std::vector<uint8_t> frame = HeartbeatFrame();
  auto withVersion = [&](uint16_t v, VersionRange accept) {
    std::vector<uint8_t> b = frame;
    b[0] = static_cast<uint8_t>(v);
    b[1] = static_cast<uint8_t>(v >> 8);
    Frame f;
    return DecodeFrame(b.data(), b.size(), accept, &f);
  };
  EXPECT_EQ(withVersion(0, kV1), DecodeStatus::kBadVersion);
  EXPECT_EQ(withVersion(2, kV1), DecodeStatus::kBadVersion);
  EXPECT_EQ(withVersion(0xffff, kV1), DecodeStatus::kBadVersion);
  EXPECT_EQ(withVersion(1, VersionRange{2, 3}), DecodeStatus::kBadVersion);
  EXPECT_EQ(withVersion(1, kV1), DecodeStatus::kOk);
}

TEST(RasProtocol, RejectsBadRoutingReservedAndIdentityFields) {
  std::vector<uint8_t> frame = HeartbeatFrame();
  auto mutateHeader = [&](auto&& fn) {
    RasHeader h = ReadHeaderFields(frame.data());
    fn(h);
    std::vector<uint8_t> b = frame;
    EncodeHeader(h, b.data());
    return DecodeBytes(b);
  };
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.target = 5; }), DecodeStatus::kBadRouting);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.scope = static_cast<uint8_t>(Scope::kJob); }),
            DecodeStatus::kBadRouting);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.scope = 9; }), DecodeStatus::kBadRouting);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.hops = 0; }), DecodeStatus::kBadRouting);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.hops = 2; }), DecodeStatus::kBadRouting);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.reserved = 1; }), DecodeStatus::kBadReserved);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.sender = 0; }), DecodeStatus::kBadField);
  EXPECT_EQ(mutateHeader([](RasHeader& h) { h.sequence = 0; }), DecodeStatus::kBadField);
}

TEST(RasProtocol, RejectsMalformedPayloads) {
  using P = std::vector<uint8_t>;
  // Not an array.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, Pack(uint64_t{5}))),
            DecodeStatus::kBadPayload);
  // Wrong field counts.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, Pack(std::vector<uint64_t>{1}))),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, Pack(std::vector<uint64_t>{1, 0, 0}))),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kLeave, Pack(std::vector<uint64_t>{}))),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeatAck, Pack(std::vector<uint64_t>{1, 2}))),
            DecodeStatus::kBadPayload);
  // Wrong types: a string where an integer belongs, a negative unsigned, an int too wide.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{0x92, 0xa1, 0x78, 0x00})),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{0x92, 0xff, 0x00})),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kLeave, P{0x91, 0xcd, 0x01, 0x00})),
            DecodeStatus::kBadPayload);
  // Out-of-range values.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kLeave, P{0x91, 0x09})), DecodeStatus::kBadField);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{0x92, 0x01, 0x02})),
            DecodeStatus::kBadField);
  // An empty payload, and msgpack that declares more bytes than the payload holds.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{})), DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{0x92, 0xcf, 0x01})),
            DecodeStatus::kBadPayload);
  // A declared array of 2^32-1 elements in a 5-byte payload: refused by the unpack limits
  // instead of allocating for it.
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, P{0xdd, 0xff, 0xff, 0xff, 0xff})),
            DecodeStatus::kBadPayload);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHello, P{0xdf, 0xff, 0xff, 0xff, 0xff})),
            DecodeStatus::kBadPayload);
}

TEST(RasProtocol, RejectsHelloFieldsOutOfRange) {
  auto encodeHello = [](const Hello& h) { return EncodeFrame(LinkHeader(1, kSender, 1), h); };
  Hello h = SampleHelloRequest();
  EXPECT_EQ(DecodeBytes(encodeHello(h)), DecodeStatus::kOk);

  Hello bad = h;
  bad.minVersion = 3;
  bad.maxVersion = 2;
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.minVersion = 0;
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.role = static_cast<HelloRole>(2);
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.status = static_cast<HelloStatus>(kMaxHelloStatus + 1);
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.token.assign(kMaxTokenBytes + 1, 'a');
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.identity.hostname.assign(kMaxIdentityString + 1, 'h');
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.identity.uuid = {};
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  for (size_t i = 0; i <= kMaxLabels; ++i) bad.identity.labels["k" + std::to_string(i)] = "v";
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.identity.labels = {{std::string(kMaxLabelKey + 1, 'k'), "v"}};
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
  bad = h;
  bad.identity.topology.nics.assign(kMaxNics + 1, h.identity.topology.nics[0]);
  EXPECT_EQ(DecodeBytes(encodeHello(bad)), DecodeStatus::kBadField);
}

TEST(RasProtocol, HelloToleratesTrailingFieldsFromANewerPeer) {
  Hello h = SampleHelloRequest();
  std::vector<uint8_t> frame = EncodeFrame(LinkHeader(1, kSender, 1), h);
  std::vector<uint8_t> payload(frame.begin() + kHeaderSize, frame.end());
  ASSERT_EQ(payload[0], 0x98);  // fixarray of the 8 v1 fields
  payload[0] = 0x99;            // a 9th field appended by a newer version
  payload.push_back(0xa3);
  payload.insert(payload.end(), {'n', 'e', 'w'});
  std::vector<uint8_t> extended = RawFrame(MessageType::kHello, payload);
  Frame f;
  ASSERT_EQ(DecodeFrame(extended.data(), extended.size(), kV1, &f), DecodeStatus::kOk);
  EXPECT_EQ(std::get<Hello>(f.message), h);

  // Other payloads have an exact field count.
  std::vector<uint8_t> hb = HeartbeatFrame();
  std::vector<uint8_t> hbPayload(hb.begin() + kHeaderSize, hb.end());
  hbPayload[0] = 0x93;
  hbPayload.push_back(0x00);
  EXPECT_EQ(DecodeBytes(RawFrame(MessageType::kHeartbeat, hbPayload)), DecodeStatus::kBadPayload);
}

TEST(RasProtocol, NegotiatesTheHighestCommonVersion) {
  EXPECT_EQ(NegotiateVersion({1, 1}, {1, 1}), std::optional<uint16_t>(1));
  EXPECT_EQ(NegotiateVersion({1, 3}, {2, 5}), std::optional<uint16_t>(3));
  EXPECT_EQ(NegotiateVersion({2, 5}, {1, 3}), std::optional<uint16_t>(3));
  EXPECT_EQ(NegotiateVersion({1, 4}, {4, 4}), std::optional<uint16_t>(4));
  EXPECT_EQ(NegotiateVersion({1, 1}, {2, 3}), std::nullopt);
  EXPECT_EQ(NegotiateVersion({0, 1}, {1, 1}), std::nullopt);
  EXPECT_EQ(NegotiateVersion({3, 2}, {1, 5}), std::nullopt);
}

TEST(RasProtocol, FuzzCorpusIsRejectedOrDecodedConsistently) {
  std::vector<std::vector<uint8_t>> seeds = FuzzCorpusSeeds();
  seeds.push_back(EncodeFrame(LinkHeader(1, kSender, 1), SampleHelloRequest()));
  seeds.push_back(HeartbeatFrame(3));
  seeds.push_back(EncodeFrame(LinkHeader(1, kSender, 4), HeartbeatAck{1, 2, 3}));
  seeds.push_back(EncodeFrame(LinkHeader(1, kSender, 5), Leave{LeaveReason::kProcessExit}));

  std::mt19937_64 rng(0x5eedull);
  size_t accepted = 0;
  size_t total = 0;
  auto check = [&](const std::vector<uint8_t>& input) {
    ++total;
    Frame f;
    DecodeStatus st = DecodeFrame(input.data(), input.size(), kV1, &f);
    if (st != DecodeStatus::kOk) return;
    ++accepted;
    // Anything accepted re-encodes to a frame that decodes to the same message.
    std::vector<uint8_t> again = EncodeFrame(f.header, f.message);
    Frame g;
    ASSERT_EQ(DecodeFrame(again.data(), again.size(), kV1, &g), DecodeStatus::kOk);
    EXPECT_EQ(g.header.sender, f.header.sender);
    EXPECT_EQ(g.header.sequence, f.header.sequence);
    EXPECT_EQ(g.message.index(), f.message.index());
    EXPECT_TRUE(g.message == f.message);
    EXPECT_EQ(f.header.target, 0u);
    EXPECT_EQ(f.header.hops, 1u);
  };

  for (const auto& seed : seeds) {
    check(seed);
    for (int i = 0; i < 4000; ++i) {
      std::vector<uint8_t> m = seed;
      switch (rng() % 6) {
        case 0:  // flip one bit
          if (!m.empty()) m[rng() % m.size()] ^= static_cast<uint8_t>(1u << (rng() % 8));
          break;
        case 1:  // overwrite one byte
          if (!m.empty()) m[rng() % m.size()] = static_cast<uint8_t>(rng());
          break;
        case 2:  // truncate
          m.resize(m.empty() ? 0 : rng() % m.size());
          break;
        case 3:  // append junk
          for (int k = 0, n = 1 + rng() % 16; k < n; ++k) m.push_back(static_cast<uint8_t>(rng()));
          break;
        case 4: {  // rewrite the length field, keeping everything else
          if (m.size() >= kHeaderSize) {
            uint32_t len = static_cast<uint32_t>(rng() % (m.size() + 8));
            std::memcpy(&m[4], &len, 4);  // tests run little-endian
          }
          break;
        }
        default: {  // corrupt a payload byte and fix up nothing else
          if (m.size() > kHeaderSize) {
            m[kHeaderSize + rng() % (m.size() - kHeaderSize)] = static_cast<uint8_t>(rng());
          }
          break;
        }
      }
      check(m);
    }
  }
  EXPECT_EQ(total, seeds.size() * 4001);
  EXPECT_GT(accepted, seeds.size());  // the valid seeds, plus mutations that stayed valid
  std::printf("fuzz: %zu inputs, %zu accepted\n", total, accepted);
}
