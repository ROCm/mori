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

#pragma once

#include <arpa/inet.h>
#include <endian.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mori/io/common.hpp"

namespace mori {
namespace io {
namespace tcp {

constexpr uint32_t kCtrlMagic = 0x4D544330;  // "MTC0"
constexpr uint32_t kDataMagic = 0x4D544430;  // "MTD0"
constexpr uint16_t kProtoVersion = 3;
constexpr size_t kCtrlHeaderSize = 12;
constexpr size_t kDataHeaderSize = 24;
constexpr uint8_t kMaxLanes = 16;
constexpr size_t kMaxControlMessageBytes = 1U << 20;

enum class Channel : uint8_t { CTRL = 1, DATA = 2 };

enum class CtrlMsgType : uint8_t {
  HELLO = 1,
  WRITE_REQ = 2,
  READ_REQ = 3,
  BATCH_WRITE_REQ = 4,
  BATCH_READ_REQ = 5,
  COMPLETION = 6,
};

enum class DataKind : uint8_t { WRITE_PAYLOAD = 1, READ_RESPONSE = 2 };

enum class ParseError : uint8_t {
  Ok = 0,
  Truncated,
  BadMagic,
  BadVersion,
  BadKind,
  BadLane,
  BodyTooLarge,
  Malformed,
};

struct CtrlHeaderView {
  CtrlMsgType type{CtrlMsgType::HELLO};
  uint32_t bodyLen{0};
};

struct ParsedDataHeader {
  DataKind kind{DataKind::WRITE_PAYLOAD};
  uint8_t lane{0};
  uint64_t opId{0};
  uint64_t payloadLen{0};
};

struct WireWriter {
  std::vector<uint8_t> buf;
  void reserve(size_t n) { buf.reserve(n); }
  void u8(uint8_t v) { buf.push_back(v); }
  void u16(uint16_t v) {
    v = htons(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(v));
  }
  void u32(uint32_t v) {
    v = htonl(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(v));
  }
  void u64(uint64_t v) {
    v = htobe64(v);
    auto* p = reinterpret_cast<uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(v));
  }
  void bytes(const void* data, size_t n) {
    auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + n);
  }
};

struct WireReader {
  const uint8_t* data;
  size_t len;
  size_t off{0};
  bool u8(uint8_t* out) {
    if (off + sizeof(*out) > len) return false;
    *out = data[off++];
    return true;
  }
  bool u16(uint16_t* out) {
    if (off + sizeof(*out) > len) return false;
    uint16_t value;
    std::memcpy(&value, data + off, sizeof(value));
    *out = ntohs(value);
    off += sizeof(value);
    return true;
  }
  bool u32(uint32_t* out) {
    if (off + sizeof(*out) > len) return false;
    uint32_t value;
    std::memcpy(&value, data + off, sizeof(value));
    *out = ntohl(value);
    off += sizeof(value);
    return true;
  }
  bool u64(uint64_t* out) {
    if (off + sizeof(*out) > len) return false;
    uint64_t value;
    std::memcpy(&value, data + off, sizeof(value));
    *out = be64toh(value);
    off += sizeof(value);
    return true;
  }
};

ParseError TryParseCtrlHeader(const uint8_t* buf, size_t len, CtrlHeaderView* out);
ParseError TryParseDataHeader(const uint8_t* buf, size_t len, ParsedDataHeader* out);

std::vector<uint8_t> BuildCtrlFrame(CtrlMsgType type,
                                    const std::function<void(WireWriter&)>& writeBody);
std::vector<uint8_t> BuildHello(Channel ch, const EngineKey& key);
std::vector<uint8_t> BuildLinearReq(CtrlMsgType type, uint64_t opId, uint32_t memId, uint64_t off,
                                    uint64_t size, uint8_t lanes);
std::vector<uint8_t> BuildBatchReq(CtrlMsgType type, uint64_t opId, uint32_t memId,
                                   const std::vector<uint64_t>& offs,
                                   const std::vector<uint64_t>& sizes, uint8_t lanes);
std::vector<uint8_t> BuildCompletion(uint64_t opId, uint32_t code, const std::string& msg);
std::array<uint8_t, kDataHeaderSize> BuildDataHeader(DataKind kind, uint8_t lane, uint64_t opId,
                                                     uint64_t payloadLen);

inline uint16_t LanesAllMask(uint8_t n) {
  return n >= kMaxLanes ? uint16_t{0xFFFF} : uint16_t((uint32_t{1} << n) - 1);
}

}  // namespace tcp
}  // namespace io
}  // namespace mori
