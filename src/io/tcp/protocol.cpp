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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include "src/io/tcp/protocol.hpp"

#include <algorithm>

namespace mori {
namespace io {
namespace tcp {

ParseError TryParseCtrlHeader(const uint8_t* buf, size_t len, CtrlHeaderView* out) {
  if (!buf || !out || len < kCtrlHeaderSize) return ParseError::Truncated;
  WireReader r{buf, len};
  uint32_t magic;
  uint16_t version;
  uint8_t type;
  uint8_t reserved;
  if (!r.u32(&magic) || !r.u16(&version) || !r.u8(&type) || !r.u8(&reserved) ||
      !r.u32(&out->bodyLen))
    return ParseError::Truncated;
  if (magic != kCtrlMagic) return ParseError::BadMagic;
  if (version != kProtoVersion) return ParseError::BadVersion;
  if (out->bodyLen > kMaxControlMessageBytes) return ParseError::BodyTooLarge;
  if (type < static_cast<uint8_t>(CtrlMsgType::HELLO) ||
      type > static_cast<uint8_t>(CtrlMsgType::COMPLETION))
    return ParseError::BadKind;
  out->type = static_cast<CtrlMsgType>(type);
  return ParseError::Ok;
}

ParseError TryParseDataHeader(const uint8_t* buf, size_t len, ParsedDataHeader* out) {
  if (!buf || !out || len < kDataHeaderSize) return ParseError::Truncated;
  WireReader r{buf, len};
  uint32_t magic;
  uint16_t version;
  uint8_t kind;
  uint8_t lane;
  if (!r.u32(&magic) || !r.u16(&version) || !r.u8(&kind) || !r.u8(&lane) || !r.u64(&out->opId) ||
      !r.u64(&out->payloadLen))
    return ParseError::Truncated;
  if (magic != kDataMagic) return ParseError::BadMagic;
  if (version != kProtoVersion) return ParseError::BadVersion;
  if (kind != static_cast<uint8_t>(DataKind::WRITE_PAYLOAD) &&
      kind != static_cast<uint8_t>(DataKind::READ_RESPONSE))
    return ParseError::BadKind;
  if (lane >= kMaxLanes) return ParseError::BadLane;
  out->kind = static_cast<DataKind>(kind);
  out->lane = lane;
  return ParseError::Ok;
}

std::vector<uint8_t> BuildCtrlFrame(CtrlMsgType type,
                                    const std::function<void(WireWriter&)>& writeBody) {
  WireWriter body;
  writeBody(body);
  WireWriter frame;
  frame.reserve(kCtrlHeaderSize + body.buf.size());
  frame.u32(kCtrlMagic);
  frame.u16(kProtoVersion);
  frame.u8(static_cast<uint8_t>(type));
  frame.u8(0);
  frame.u32(static_cast<uint32_t>(body.buf.size()));
  frame.bytes(body.buf.data(), body.buf.size());
  return std::move(frame.buf);
}

std::vector<uint8_t> BuildHello(Channel ch, const EngineKey& key) {
  return BuildCtrlFrame(CtrlMsgType::HELLO, [&](WireWriter& w) {
    w.u8(static_cast<uint8_t>(ch));
    w.u32(static_cast<uint32_t>(key.size()));
    w.bytes(key.data(), key.size());
  });
}

std::vector<uint8_t> BuildLinearReq(CtrlMsgType type, uint64_t opId, uint32_t memId, uint64_t off,
                                    uint64_t size, uint8_t lanes) {
  return BuildCtrlFrame(type, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(memId);
    w.u64(off);
    w.u64(size);
    w.u8(lanes);
  });
}

std::vector<uint8_t> BuildBatchReq(CtrlMsgType type, uint64_t opId, uint32_t memId,
                                   const std::vector<uint64_t>& offs,
                                   const std::vector<uint64_t>& sizes, uint8_t lanes) {
  return BuildCtrlFrame(type, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(memId);
    w.u32(static_cast<uint32_t>(offs.size()));
    for (size_t i = 0; i < offs.size(); ++i) {
      w.u64(offs[i]);
      w.u64(sizes[i]);
    }
    w.u8(lanes);
  });
}

std::vector<uint8_t> BuildCompletion(uint64_t opId, uint32_t code, const std::string& msg) {
  return BuildCtrlFrame(CtrlMsgType::COMPLETION, [&](WireWriter& w) {
    w.u64(opId);
    w.u32(code);
    w.u32(static_cast<uint32_t>(msg.size()));
    w.bytes(msg.data(), msg.size());
  });
}

std::array<uint8_t, kDataHeaderSize> BuildDataHeader(DataKind kind, uint8_t lane, uint64_t opId,
                                                     uint64_t payloadLen) {
  WireWriter w;
  w.reserve(kDataHeaderSize);
  w.u32(kDataMagic);
  w.u16(kProtoVersion);
  w.u8(static_cast<uint8_t>(kind));
  w.u8(lane);
  w.u64(opId);
  w.u64(payloadLen);
  std::array<uint8_t, kDataHeaderSize> result{};
  std::copy(w.buf.begin(), w.buf.end(), result.begin());
  return result;
}

}  // namespace tcp
}  // namespace io
}  // namespace mori
