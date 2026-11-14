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
#include "src/ras/protocol.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <msgpack.hpp>

namespace mori {
namespace ras {
namespace proto {
namespace {

void PutU16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}
void PutU32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
void PutU64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
uint16_t GetU16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t GetU32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}
uint64_t GetU64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

bool IsImplemented(uint16_t type) {
  switch (static_cast<MessageType>(type)) {
    case MessageType::kHello:
    case MessageType::kLeave:
    case MessageType::kHeartbeat:
    case MessageType::kHeartbeatAck:
      return true;
    default:
      return false;
  }
}

// No v1 payload comes near these, so anything that does is malformed. They also stop a payload
// from declaring a container far larger than its own bytes.
msgpack::unpack_limit PayloadLimits() {
  return msgpack::unpack_limit(/*array=*/128, /*map=*/64, /*str=*/1024, /*bin=*/0, /*ext=*/0,
                               /*depth=*/8);
}

// Thrown for a well-formed value outside its allowed range; msgpack::type_error covers shape.
struct BadField : std::exception {};

using Packer = msgpack::packer<msgpack::sbuffer>;

constexpr uint32_t kHelloFields = 8;
constexpr uint32_t kIdentityFields = 10;
constexpr uint32_t kTopologyFields = 4;
constexpr uint32_t kNicFields = 7;

void PackUuid(Packer& pk, const RuntimeUuid& u) {
  pk.pack_array(2);
  pk.pack(u.hi);
  pk.pack(u.lo);
}

void PackNic(Packer& pk, const NicAttachment& n) {
  pk.pack_array(kNicFields);
  pk.pack(n.device);
  pk.pack(n.port);
  pk.pack(n.netdev);
  pk.pack(n.address);
  pk.pack(n.rail);
  pk.pack(n.switchName);
  pk.pack(n.switchPort);
}

void PackTopology(Packer& pk, const Topology& t) {
  pk.pack_array(kTopologyFields);
  pk.pack(t.rack);
  pk.pack(t.leaf);
  pk.pack(t.source);
  pk.pack_array(static_cast<uint32_t>(t.nics.size()));
  for (const NicAttachment& n : t.nics) PackNic(pk, n);
}

void PackIdentity(Packer& pk, const RasProcessId& id) {
  pk.pack_array(kIdentityFields);
  PackUuid(pk, id.uuid);
  pk.pack(id.nodeId);
  pk.pack(id.hostname);
  pk.pack(id.host);
  pk.pack(id.pid);
  pk.pack(id.pidNamespace);
  pk.pack(id.startTimeTicks);
  pk.pack(id.bootId);
  PackTopology(pk, id.topology);
  pk.pack(id.labels);
}

void PackPayload(Packer& pk, const Hello& h) {
  pk.pack_array(kHelloFields);
  pk.pack(h.minVersion);
  pk.pack(h.maxVersion);
  pk.pack(static_cast<uint8_t>(h.role));
  pk.pack(static_cast<uint8_t>(h.status));
  pk.pack(h.capabilities);
  pk.pack(h.token);
  PackUuid(pk, h.expectedPeer);
  PackIdentity(pk, h.identity);
}

void PackPayload(Packer& pk, const Leave& l) {
  pk.pack_array(1);
  pk.pack(static_cast<uint8_t>(l.reason));
}

void PackPayload(Packer& pk, const Heartbeat& h) {
  pk.pack_array(2);
  pk.pack(h.sendMonoNs);
  pk.pack(h.flags);
}

void PackPayload(Packer& pk, const HeartbeatAck& a) {
  pk.pack_array(3);
  pk.pack(a.echoSendMonoNs);
  pk.pack(a.holdNs);
  pk.pack(a.sendWallNs);
}

// An array of exactly `size` elements, or of at least `size` when `open` (HELLO and the
// structures inside it, which later versions may extend).
const msgpack::object* Array(const msgpack::object& o, uint32_t size, bool open = false) {
  if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
  uint32_t n = o.via.array.size;
  if (n < size || (!open && n != size)) throw msgpack::type_error();
  return o.via.array.ptr;
}

std::string String(const msgpack::object& o, size_t maxBytes) {
  if (o.type != msgpack::type::STR) throw msgpack::type_error();
  if (o.via.str.size > maxBytes) throw BadField();
  return std::string(o.via.str.ptr, o.via.str.size);
}

RuntimeUuid Uuid(const msgpack::object& o) {
  const msgpack::object* a = Array(o, 2);
  return RuntimeUuid{a[0].as<uint64_t>(), a[1].as<uint64_t>()};
}

NicAttachment Nic(const msgpack::object& o) {
  const msgpack::object* a = Array(o, kNicFields, /*open=*/true);
  NicAttachment n;
  n.device = String(a[0], kMaxIdentityString);
  n.port = a[1].as<uint32_t>();
  n.netdev = String(a[2], kMaxIdentityString);
  n.address = String(a[3], kMaxIdentityString);
  n.rail = String(a[4], kMaxIdentityString);
  n.switchName = String(a[5], kMaxIdentityString);
  n.switchPort = String(a[6], kMaxIdentityString);
  return n;
}

Topology TopologyOf(const msgpack::object& o) {
  const msgpack::object* a = Array(o, kTopologyFields, /*open=*/true);
  Topology t;
  t.rack = String(a[0], kMaxIdentityString);
  t.leaf = String(a[1], kMaxIdentityString);
  t.source = String(a[2], kMaxIdentityString);
  if (a[3].type != msgpack::type::ARRAY) throw msgpack::type_error();
  if (a[3].via.array.size > kMaxNics) throw BadField();
  for (uint32_t i = 0; i < a[3].via.array.size; ++i) t.nics.push_back(Nic(a[3].via.array.ptr[i]));
  return t;
}

Labels LabelsOf(const msgpack::object& o) {
  if (o.type != msgpack::type::MAP) throw msgpack::type_error();
  if (o.via.map.size > kMaxLabels) throw BadField();
  Labels labels;
  for (uint32_t i = 0; i < o.via.map.size; ++i) {
    std::string key = String(o.via.map.ptr[i].key, kMaxLabelKey);
    std::string value = String(o.via.map.ptr[i].val, kMaxLabelValue);
    if (!labels.emplace(std::move(key), std::move(value)).second) throw BadField();
  }
  return labels;
}

RasProcessId Identity(const msgpack::object& o) {
  const msgpack::object* a = Array(o, kIdentityFields, /*open=*/true);
  RasProcessId id;
  id.uuid = Uuid(a[0]);
  if (id.uuid.IsZero()) throw BadField();
  id.nodeId = String(a[1], kMaxIdentityString);
  id.hostname = String(a[2], kMaxIdentityString);
  id.host = String(a[3], kMaxIdentityString);
  id.pid = a[4].as<int32_t>();
  id.pidNamespace = a[5].as<uint64_t>();
  id.startTimeTicks = a[6].as<uint64_t>();
  id.bootId = String(a[7], kMaxIdentityString);
  id.topology = TopologyOf(a[8]);
  id.labels = LabelsOf(a[9]);
  return id;
}

Hello HelloOf(const msgpack::object& o) {
  const msgpack::object* a = Array(o, kHelloFields, /*open=*/true);
  Hello h;
  h.minVersion = a[0].as<uint16_t>();
  h.maxVersion = a[1].as<uint16_t>();
  if (h.minVersion == 0 || h.minVersion > h.maxVersion) throw BadField();
  uint8_t role = a[2].as<uint8_t>();
  if (role > static_cast<uint8_t>(HelloRole::kReply)) throw BadField();
  h.role = static_cast<HelloRole>(role);
  uint8_t status = a[3].as<uint8_t>();
  if (status > kMaxHelloStatus) throw BadField();
  h.status = static_cast<HelloStatus>(status);
  h.capabilities = a[4].as<uint64_t>();
  h.token = String(a[5], kMaxTokenBytes);
  h.expectedPeer = Uuid(a[6]);
  h.identity = Identity(a[7]);
  return h;
}

Leave LeaveOf(const msgpack::object& o) {
  const msgpack::object* a = Array(o, 1);
  uint8_t reason = a[0].as<uint8_t>();
  if (reason > kMaxLeaveReason) throw BadField();
  return Leave{static_cast<LeaveReason>(reason)};
}

Heartbeat HeartbeatOf(const msgpack::object& o) {
  const msgpack::object* a = Array(o, 2);
  Heartbeat h{a[0].as<uint64_t>(), a[1].as<uint32_t>()};
  if (h.flags & ~kHeartbeatKnownFlags) throw BadField();
  return h;
}

HeartbeatAck AckOf(const msgpack::object& o) {
  const msgpack::object* a = Array(o, 3);
  return HeartbeatAck{a[0].as<uint64_t>(), a[1].as<uint64_t>(), a[2].as<int64_t>()};
}

}  // namespace

const char* ToString(DecodeStatus s) {
  switch (s) {
    case DecodeStatus::kOk:
      return "ok";
    case DecodeStatus::kTruncated:
      return "truncated";
    case DecodeStatus::kOversized:
      return "oversized";
    case DecodeStatus::kUnknownType:
      return "unknown type";
    case DecodeStatus::kReservedType:
      return "reserved type";
    case DecodeStatus::kLengthMismatch:
      return "length mismatch";
    case DecodeStatus::kBadVersion:
      return "bad version";
    case DecodeStatus::kBadReserved:
      return "reserved bits set";
    case DecodeStatus::kBadRouting:
      return "bad routing fields";
    case DecodeStatus::kBadPayload:
      return "malformed payload";
    case DecodeStatus::kBadField:
      return "field out of range";
  }
  return "unknown";
}

MessageType TypeOf(const Message& m) {
  switch (m.index()) {
    case 0:
      return MessageType::kHello;
    case 1:
      return MessageType::kLeave;
    case 2:
      return MessageType::kHeartbeat;
    default:
      return MessageType::kHeartbeatAck;
  }
}

void EncodeHeader(const RasHeader& h, uint8_t* out) {
  PutU16(out + 0, h.version);
  PutU16(out + 2, h.type);
  PutU32(out + 4, h.length);
  PutU64(out + 8, h.sender);
  PutU64(out + 16, h.sequence);
  PutU64(out + 24, h.target);
  out[32] = h.scope;
  out[33] = h.hops;
  PutU16(out + 34, h.reserved);
}

RasHeader ReadHeaderFields(const uint8_t* in) {
  RasHeader h;
  h.version = GetU16(in + 0);
  h.type = GetU16(in + 2);
  h.length = GetU32(in + 4);
  h.sender = GetU64(in + 8);
  h.sequence = GetU64(in + 16);
  h.target = GetU64(in + 24);
  h.scope = in[32];
  h.hops = in[33];
  h.reserved = GetU16(in + 34);
  return h;
}

DecodeStatus DecodeHeader(const uint8_t* data, size_t size, VersionRange accept, RasHeader* out) {
  if (size < kHeaderSize) return DecodeStatus::kTruncated;
  RasHeader h = ReadHeaderFields(data);
  if (h.version == 0 || h.version < accept.min || h.version > accept.max) {
    return DecodeStatus::kBadVersion;
  }
  if (h.type == 0 || h.type > kMaxKnownType) return DecodeStatus::kUnknownType;
  if (!IsImplemented(h.type)) return DecodeStatus::kReservedType;
  if (h.length > kMaxPayload) return DecodeStatus::kOversized;
  if (h.reserved != 0) return DecodeStatus::kBadReserved;
  // Every v1 type is a link message.
  if (h.target != 0 || h.scope != static_cast<uint8_t>(Scope::kLink) || h.hops != 1) {
    return DecodeStatus::kBadRouting;
  }
  if (h.sender == 0 || h.sequence == 0) return DecodeStatus::kBadField;
  *out = h;
  return DecodeStatus::kOk;
}

DecodeStatus DecodePayload(const RasHeader& header, const uint8_t* payload, size_t size,
                           Message* out) {
  if (size != header.length) return DecodeStatus::kLengthMismatch;
  try {
    size_t offset = 0;
    msgpack::object_handle oh = msgpack::unpack(reinterpret_cast<const char*>(payload), size,
                                                offset, nullptr, nullptr, PayloadLimits());
    if (offset != size) return DecodeStatus::kLengthMismatch;
    const msgpack::object& o = oh.get();
    switch (static_cast<MessageType>(header.type)) {
      case MessageType::kHello:
        *out = HelloOf(o);
        break;
      case MessageType::kLeave:
        *out = LeaveOf(o);
        break;
      case MessageType::kHeartbeat:
        *out = HeartbeatOf(o);
        break;
      case MessageType::kHeartbeatAck:
        *out = AckOf(o);
        break;
      default:
        return DecodeStatus::kReservedType;
    }
  } catch (const BadField&) {
    return DecodeStatus::kBadField;
  } catch (const std::exception&) {
    // msgpack's parse, size-limit, and type errors, plus anything else unpacking throws.
    return DecodeStatus::kBadPayload;
  }
  return DecodeStatus::kOk;
}

DecodeStatus DecodeFrame(const uint8_t* data, size_t size, VersionRange accept, Frame* out) {
  RasHeader h;
  DecodeStatus st = DecodeHeader(data, size, accept, &h);
  if (st != DecodeStatus::kOk) return st;
  if (size < kHeaderSize + h.length) return DecodeStatus::kTruncated;
  if (size > kHeaderSize + h.length) return DecodeStatus::kLengthMismatch;
  Message m;
  st = DecodePayload(h, data + kHeaderSize, h.length, &m);
  if (st != DecodeStatus::kOk) return st;
  out->header = h;
  out->message = std::move(m);
  return DecodeStatus::kOk;
}

std::vector<uint8_t> EncodeFrame(RasHeader header, const Message& message) {
  msgpack::sbuffer buf;
  Packer pk(buf);
  std::visit([&](const auto& m) { PackPayload(pk, m); }, message);
  header.type = static_cast<uint16_t>(TypeOf(message));
  header.length = static_cast<uint32_t>(buf.size());
  std::vector<uint8_t> out(kHeaderSize + buf.size());
  EncodeHeader(header, out.data());
  std::memcpy(out.data() + kHeaderSize, buf.data(), buf.size());
  return out;
}

std::optional<uint16_t> NegotiateVersion(VersionRange a, VersionRange b) {
  if (a.min == 0 || b.min == 0 || a.min > a.max || b.min > b.max) return std::nullopt;
  uint16_t lo = std::max(a.min, b.min);
  uint16_t hi = std::min(a.max, b.max);
  if (lo > hi) return std::nullopt;
  return hi;
}

RasHeader LinkHeader(uint16_t version, uint64_t sender, uint64_t sequence) {
  RasHeader h;
  h.version = version;
  h.sender = sender;
  h.sequence = sequence;
  h.target = 0;
  h.scope = static_cast<uint8_t>(Scope::kLink);
  h.hops = 1;
  return h;
}

}  // namespace proto
}  // namespace ras
}  // namespace mori
