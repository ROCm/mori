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

#include <arpa/inet.h>
#include <endian.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mori {
namespace io {
namespace tcp {

constexpr uint32_t kCtrlMagic = 0x4D544330;  // "MTC0"
constexpr uint32_t kDataMagic = 0x4D544430;  // "MTD0"
constexpr uint16_t kProtoVersion = 1;

enum class Channel : uint8_t { CTRL = 1, DATA = 2 };

enum class CtrlMsgType : uint8_t {
  HELLO = 1,
  WRITE_REQ = 2,
  READ_REQ = 3,
  BATCH_WRITE_REQ = 4,
  BATCH_READ_REQ = 5,
  COMPLETION = 6,
};

// Fixed framing:
//   CtrlHeader (12B) + body
//     magic(4) ver(2) type(1) reserved(1) body_len(4)
//   DataHeader (24B) + payload
//     magic(4) ver(2) flags(2) op_id(8) payload_len(8)
constexpr size_t kCtrlHeaderSize = 12;
constexpr size_t kDataHeaderSize = 24;

struct CtrlHeaderView {
  CtrlMsgType type{CtrlMsgType::HELLO};
  uint32_t bodyLen{0};
};

struct DataHeaderView {
  uint16_t flags{0};
  uint64_t opId{0};
  uint64_t payloadLen{0};
};

inline uint64_t HostToBe64(uint64_t v) { return htobe64(v); }
inline uint64_t BeToHost64(uint64_t v) { return be64toh(v); }

inline void AppendU8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

inline void AppendU16BE(std::vector<uint8_t>& out, uint16_t v) {
  uint16_t be = htons(v);
  uint8_t* p = reinterpret_cast<uint8_t*>(&be);
  out.insert(out.end(), p, p + sizeof(be));
}

inline void AppendU32BE(std::vector<uint8_t>& out, uint32_t v) {
  uint32_t be = htonl(v);
  uint8_t* p = reinterpret_cast<uint8_t*>(&be);
  out.insert(out.end(), p, p + sizeof(be));
}

inline void AppendU64BE(std::vector<uint8_t>& out, uint64_t v) {
  uint64_t be = HostToBe64(v);
  uint8_t* p = reinterpret_cast<uint8_t*>(&be);
  out.insert(out.end(), p, p + sizeof(be));
}

inline bool ReadU16BE(const uint8_t* p, size_t len, size_t* off, uint16_t* out) {
  if (*off + sizeof(uint16_t) > len) return false;
  uint16_t be;
  std::memcpy(&be, p + *off, sizeof(be));
  *out = ntohs(be);
  *off += sizeof(be);
  return true;
}

inline bool ReadU32BE(const uint8_t* p, size_t len, size_t* off, uint32_t* out) {
  if (*off + sizeof(uint32_t) > len) return false;
  uint32_t be;
  std::memcpy(&be, p + *off, sizeof(be));
  *out = ntohl(be);
  *off += sizeof(be);
  return true;
}

inline bool ReadU64BE(const uint8_t* p, size_t len, size_t* off, uint64_t* out) {
  if (*off + sizeof(uint64_t) > len) return false;
  uint64_t be;
  std::memcpy(&be, p + *off, sizeof(be));
  *out = BeToHost64(be);
  *off += sizeof(be);
  return true;
}

inline bool TryParseCtrlHeader(const uint8_t* buf, size_t len, CtrlHeaderView* out) {
  if (len < kCtrlHeaderSize) return false;
  size_t off = 0;

  uint32_t magic = 0;
  uint16_t ver = 0;
  uint8_t type = 0;
  uint8_t reserved = 0;
  uint32_t bodyLen = 0;

  if (!ReadU32BE(buf, len, &off, &magic)) return false;
  if (!ReadU16BE(buf, len, &off, &ver)) return false;
  if (off + 2 > len) return false;
  type = buf[off++];
  reserved = buf[off++];
  (void)reserved;
  if (!ReadU32BE(buf, len, &off, &bodyLen)) return false;

  if (magic != kCtrlMagic || ver != kProtoVersion) return false;
  out->type = static_cast<CtrlMsgType>(type);
  out->bodyLen = bodyLen;
  return true;
}

inline bool TryParseDataHeader(const uint8_t* buf, size_t len, DataHeaderView* out) {
  if (len < kDataHeaderSize) return false;
  size_t off = 0;

  uint32_t magic = 0;
  uint16_t ver = 0;
  uint16_t flags = 0;
  uint64_t opId = 0;
  uint64_t payloadLen = 0;

  if (!ReadU32BE(buf, len, &off, &magic)) return false;
  if (!ReadU16BE(buf, len, &off, &ver)) return false;
  if (!ReadU16BE(buf, len, &off, &flags)) return false;
  if (!ReadU64BE(buf, len, &off, &opId)) return false;
  if (!ReadU64BE(buf, len, &off, &payloadLen)) return false;

  if (magic != kDataMagic || ver != kProtoVersion) return false;
  out->flags = flags;
  out->opId = opId;
  out->payloadLen = payloadLen;
  return true;
}

inline std::vector<uint8_t> BuildCtrlFrame(CtrlMsgType type, const std::vector<uint8_t>& body) {
  std::vector<uint8_t> out;
  out.reserve(kCtrlHeaderSize + body.size());
  AppendU32BE(out, kCtrlMagic);
  AppendU16BE(out, kProtoVersion);
  AppendU8(out, static_cast<uint8_t>(type));
  AppendU8(out, 0 /*reserved*/);
  AppendU32BE(out, static_cast<uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

inline std::vector<uint8_t> BuildHello(Channel ch, const std::string& engineKey) {
  std::vector<uint8_t> body;
  body.reserve(1 + 4 + engineKey.size());
  AppendU8(body, static_cast<uint8_t>(ch));
  AppendU32BE(body, static_cast<uint32_t>(engineKey.size()));
  body.insert(body.end(), engineKey.begin(), engineKey.end());
  return BuildCtrlFrame(CtrlMsgType::HELLO, body);
}

inline std::vector<uint8_t> BuildCompletion(uint64_t opId, uint32_t statusCode,
                                            const std::string& msg) {
  std::vector<uint8_t> body;
  body.reserve(8 + 4 + 4 + msg.size());
  AppendU64BE(body, opId);
  AppendU32BE(body, statusCode);
  AppendU32BE(body, static_cast<uint32_t>(msg.size()));
  body.insert(body.end(), msg.begin(), msg.end());
  return BuildCtrlFrame(CtrlMsgType::COMPLETION, body);
}

inline std::vector<uint8_t> BuildWriteReq(uint64_t opId, uint32_t remoteMemId, uint64_t remoteOff,
                                          uint64_t size) {
  std::vector<uint8_t> body;
  body.reserve(8 + 4 + 8 + 8);
  AppendU64BE(body, opId);
  AppendU32BE(body, remoteMemId);
  AppendU64BE(body, remoteOff);
  AppendU64BE(body, size);
  return BuildCtrlFrame(CtrlMsgType::WRITE_REQ, body);
}

inline std::vector<uint8_t> BuildReadReq(uint64_t opId, uint32_t srcMemId, uint64_t srcOff,
                                         uint64_t size) {
  std::vector<uint8_t> body;
  body.reserve(8 + 4 + 8 + 8);
  AppendU64BE(body, opId);
  AppendU32BE(body, srcMemId);
  AppendU64BE(body, srcOff);
  AppendU64BE(body, size);
  return BuildCtrlFrame(CtrlMsgType::READ_REQ, body);
}

inline std::vector<uint8_t> BuildBatchWriteReq(uint64_t opId, uint32_t remoteMemId,
                                               const std::vector<uint64_t>& remoteOffs,
                                               const std::vector<uint64_t>& sizes) {
  std::vector<uint8_t> body;
  const uint32_t n = static_cast<uint32_t>(sizes.size());
  body.reserve(8 + 4 + 4 + n * (8 + 8));
  AppendU64BE(body, opId);
  AppendU32BE(body, remoteMemId);
  AppendU32BE(body, n);
  for (uint32_t i = 0; i < n; ++i) {
    AppendU64BE(body, remoteOffs[i]);
    AppendU64BE(body, sizes[i]);
  }
  return BuildCtrlFrame(CtrlMsgType::BATCH_WRITE_REQ, body);
}

inline std::vector<uint8_t> BuildBatchReadReq(uint64_t opId, uint32_t srcMemId,
                                              const std::vector<uint64_t>& srcOffs,
                                              const std::vector<uint64_t>& sizes) {
  std::vector<uint8_t> body;
  const uint32_t n = static_cast<uint32_t>(sizes.size());
  body.reserve(8 + 4 + 4 + n * (8 + 8));
  AppendU64BE(body, opId);
  AppendU32BE(body, srcMemId);
  AppendU32BE(body, n);
  for (uint32_t i = 0; i < n; ++i) {
    AppendU64BE(body, srcOffs[i]);
    AppendU64BE(body, sizes[i]);
  }
  return BuildCtrlFrame(CtrlMsgType::BATCH_READ_REQ, body);
}

inline std::vector<uint8_t> BuildDataHeader(uint64_t opId, uint64_t payloadLen, uint16_t flags) {
  std::vector<uint8_t> out;
  out.reserve(kDataHeaderSize);
  AppendU32BE(out, kDataMagic);
  AppendU16BE(out, kProtoVersion);
  AppendU16BE(out, flags);
  AppendU64BE(out, opId);
  AppendU64BE(out, payloadLen);
  return out;
}

}  // namespace tcp
}  // namespace io
}  // namespace mori

