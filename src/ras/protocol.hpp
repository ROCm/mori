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

// MORI RAS link protocol, version 1.
//
// A message is a 36-byte header followed by `length` bytes of msgpack payload. The header is
// encoded and decoded field by field, little-endian. RasHeader is never copied to the wire:
// its padding and the host byte order can differ from this layout.
//
//   offset  size  field
//        0     2  version    protocol version of this message
//        2     2  type       MessageType
//        4     4  length     payload bytes, at most kMaxPayload
//        8     8  sender     low 64 bits of the sender's runtime UUID; never 0
//       16     8  sequence   per connection and direction: 1 for HELLO, then +1 per message
//       24     8  target     destination object ID; 0 means the link peer itself
//       32     1  scope      Scope: how far a QUERY or CONTROL travels
//       33     1  hops       hop limit, decremented on each forward; dropped at 0
//       34     2  reserved   0 in v1
//
// v1 implements HELLO, LEAVE, HEARTBEAT, and HEARTBEAT_ACK. All four are link messages: target
// 0, scope LINK, hops 1. Every other type is reserved and rejected, so a v1 peer can't misread
// a message that a later version adds. Target, scope, and hops are reserved now so that P7 can
// forward queries through node and job aggregators without a header change.
//
// Version negotiation. HELLO goes out with the sender's lowest version in the header and its
// [min, max] range in the payload. The link then uses the highest version in both ranges, and
// no overlap rejects the link. Every later message must carry exactly the negotiated version.
// HELLO is the one payload that tolerates trailing fields, so a newer peer can extend it and
// still be read; every other v1 payload has an exact field count.
//
// Strict decoding. A frame is rejected when it is truncated, when its length is over
// kMaxPayload (checked from the header, before any payload is read), when its type is unknown
// or reserved, when its version is outside the accepted range, when the payload's msgpack
// object doesn't fill exactly `length` bytes, or when any field has the wrong type, an
// out-of-range value, or an oversized string or list. msgpack is unpacked under fixed limits,
// so a payload that declares a huge array can't force a large allocation.
//
// Trust model (P0). RAS links run on a private job network, the same one MORI-IO's TCP control
// plane already trusts. The token that EngineDesc advertises guards only against accidental or
// uninformed connections: a stale EngineDesc whose port now belongs to another process, a
// misrouted job, or a port scanner. It isn't authentication. EngineDesc travels in clear text
// over framework side channels, and links aren't encrypted. A v1 runtime advertises the
// capabilities kCapToken | kCapReadOnly: it accepts no commands from peers, so a connection that
// passes the token check can read heartbeats and identity, nothing more. Remote control (P5)
// needs authenticated control, which v1 doesn't offer.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "mori/ras/types.hpp"

namespace mori {
namespace ras {
namespace proto {

constexpr size_t kHeaderSize = 36;
constexpr uint32_t kMaxPayload = 64 * 1024;
constexpr uint16_t kVersionMin = 1;
constexpr uint16_t kVersionMax = 1;

enum class MessageType : uint16_t {
  kHello = 1,
  kJoin = 2,
  kLeave = 3,
  kHeartbeat = 4,
  kHeartbeatAck = 5,
  kStateUpdate = 6,
  kProgressUpdate = 7,
  kPeerSuspect = 8,
  kPeerUnreachable = 9,
  kPeerExited = 10,
  kQuery = 11,
  kQueryResponse = 12,
  kControl = 13,
  kDiagnosticsRequest = 14,
  kDiagnosticsResponse = 15,
};
constexpr uint16_t kMaxKnownType = 15;

enum class Scope : uint8_t { kLink = 0, kProcess = 1, kNode = 2, kJob = 3 };

// In-memory header. See the layout above for the wire form.
struct RasHeader {
  uint16_t version{0};
  uint16_t type{0};
  uint32_t length{0};
  uint64_t sender{0};
  uint64_t sequence{0};
  uint64_t target{0};
  uint8_t scope{0};
  uint8_t hops{0};
  uint16_t reserved{0};
};

struct VersionRange {
  uint16_t min{0};
  uint16_t max{0};
};
constexpr VersionRange kSupportedVersions{kVersionMin, kVersionMax};

enum class DecodeStatus : uint8_t {
  kOk,
  kTruncated,       // fewer bytes than the header, or than the header plus its length
  kOversized,       // length over kMaxPayload
  kUnknownType,     // type 0 or above kMaxKnownType
  kReservedType,    // a known type that v1 doesn't implement
  kLengthMismatch,  // bytes received, or the payload's msgpack object, disagree with length
  kBadVersion,      // version outside the accepted range
  kBadReserved,     // reserved header bits set
  kBadRouting,      // target, scope, or hops invalid for the type
  kBadPayload,      // payload isn't the expected msgpack shape or types
  kBadField,        // a field value is out of range or too large
};
const char* ToString(DecodeStatus s);

// Security capabilities a runtime offers. v1 offers only these two.
constexpr uint64_t kCapToken = 1ull << 0;     // the connecting side presents the acceptor's token
constexpr uint64_t kCapReadOnly = 1ull << 1;  // the runtime accepts no commands from peers

enum class HelloRole : uint8_t { kRequest = 0, kReply = 1 };
enum class HelloStatus : uint8_t {
  kOk = 0,
  kDegraded = 1,            // accepted; the replying runtime is RAS_DEGRADED
  kRejectedVersion = 2,     // no common protocol version
  kRejectedToken = 3,       // the token doesn't match the acceptor's
  kRejectedNotMe = 4,       // the acceptor isn't the runtime the initiator expected
  kRejectedDuplicate = 5,   // the acceptor keeps another connection to this peer
  kRejectedSelf = 6,        // the initiator is the acceptor's own runtime
  kRejectedCapability = 7,  // no common security capability
};
constexpr uint8_t kMaxHelloStatus = 7;

enum class LeaveReason : uint8_t {
  kProcessExit = 0,   // the sender's runtime is shutting down; its UUID is gone for good
  kLinkReleased = 1,  // the sender no longer has an engine relationship with the receiver
};
constexpr uint8_t kMaxLeaveReason = 1;

constexpr uint32_t kHeartbeatFlagDegraded = 1u << 0;  // the sender is RAS_DEGRADED
constexpr uint32_t kHeartbeatKnownFlags = kHeartbeatFlagDegraded;

struct Hello {
  uint16_t minVersion{kVersionMin};
  uint16_t maxVersion{kVersionMax};
  HelloRole role{HelloRole::kRequest};
  HelloStatus status{HelloStatus::kOk};
  uint64_t capabilities{kCapToken | kCapReadOnly};
  std::string token;         // request: the acceptor's token from its EngineDesc; reply: empty
  RuntimeUuid expectedPeer;  // request: the runtime the initiator expects; reply: zero
  RasProcessId identity;     // the sender

  bool operator==(const Hello& o) const {
    return minVersion == o.minVersion && maxVersion == o.maxVersion && role == o.role &&
           status == o.status && capabilities == o.capabilities && token == o.token &&
           expectedPeer == o.expectedPeer && identity == o.identity;
  }
};

struct Leave {
  LeaveReason reason{LeaveReason::kProcessExit};
  bool operator==(const Leave& o) const { return reason == o.reason; }
};

struct Heartbeat {
  uint64_t sendMonoNs{0};  // sender's CLOCK_MONOTONIC when it sent the heartbeat
  uint32_t flags{0};       // kHeartbeatFlag*
  bool operator==(const Heartbeat& o) const {
    return sendMonoNs == o.sendMonoNs && flags == o.flags;
  }
};

struct HeartbeatAck {
  uint64_t echoSendMonoNs{0};  // the heartbeat's sendMonoNs, echoed
  uint64_t holdNs{0};          // receipt to reply, on the acknowledging side's monotonic clock
  int64_t sendWallNs{0};       // acknowledging side's CLOCK_REALTIME when it sent the reply
  bool operator==(const HeartbeatAck& o) const {
    return echoSendMonoNs == o.echoSendMonoNs && holdNs == o.holdNs && sendWallNs == o.sendWallNs;
  }
};

using Message = std::variant<Hello, Leave, Heartbeat, HeartbeatAck>;
MessageType TypeOf(const Message& m);

struct Frame {
  RasHeader header;
  Message message;
};

// Field limits enforced on decode.
constexpr size_t kMaxTokenBytes = 128;
constexpr size_t kMaxIdentityString = 255;
constexpr size_t kMaxNics = 64;
constexpr size_t kMaxLabels = 32;
constexpr size_t kMaxLabelKey = 64;
constexpr size_t kMaxLabelValue = 256;

// Writes the kHeaderSize-byte wire form.
void EncodeHeader(const RasHeader& h, uint8_t* out);
// Reads the wire form without validating it.
RasHeader ReadHeaderFields(const uint8_t* in);
// Reads and validates a header. `size` is the bytes available, which may include payload.
DecodeStatus DecodeHeader(const uint8_t* data, size_t size, VersionRange accept, RasHeader* out);
// Decodes the payload of a validated header. `size` must equal header.length.
DecodeStatus DecodePayload(const RasHeader& header, const uint8_t* payload, size_t size,
                           Message* out);
// Decodes one complete frame of exactly `size` bytes.
DecodeStatus DecodeFrame(const uint8_t* data, size_t size, VersionRange accept, Frame* out);

// Encodes `header` and `message` as one frame. The type and length fields are taken from the
// message, whatever `header` holds.
std::vector<uint8_t> EncodeFrame(RasHeader header, const Message& message);

// The highest version both ranges contain, or nullopt if they don't overlap or one is invalid.
std::optional<uint16_t> NegotiateVersion(VersionRange a, VersionRange b);

// A link-message header from `sender`, with routing fields set for v1.
RasHeader LinkHeader(uint16_t version, uint64_t sender, uint64_t sequence);

}  // namespace proto
}  // namespace ras
}  // namespace mori
