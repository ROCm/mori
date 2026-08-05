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

#include <gtest/gtest.h>

#include "src/io/tcp/protocol.hpp"

namespace mori::io::tcp {
namespace {

TEST(TcpProtocol, DataHeaderRoundTrip) {
  constexpr uint64_t kOpId = 0xfedcba9876543210ULL;
  constexpr uint64_t kPayloadLen = 0x123456789ULL;
  auto bytes = BuildDataHeader(DataKind::READ_RESPONSE, 15, kOpId, kPayloadLen);

  ParsedDataHeader parsed;
  EXPECT_EQ(TryParseDataHeader(bytes.data(), bytes.size(), &parsed), ParseError::Ok);
  EXPECT_EQ(parsed.kind, DataKind::READ_RESPONSE);
  EXPECT_EQ(parsed.lane, 15);
  EXPECT_EQ(parsed.opId, kOpId);
  EXPECT_EQ(parsed.payloadLen, kPayloadLen);
}

TEST(TcpProtocol, RejectsWrongVersionAndLane) {
  auto bytes = BuildDataHeader(DataKind::WRITE_PAYLOAD, 0, 7, 11);
  bytes[5] = static_cast<uint8_t>(kProtoVersion - 1);
  ParsedDataHeader parsed;
  EXPECT_EQ(TryParseDataHeader(bytes.data(), bytes.size(), &parsed), ParseError::BadVersion);

  bytes = BuildDataHeader(DataKind::WRITE_PAYLOAD, 0, 7, 11);
  bytes[7] = kMaxLanes;
  EXPECT_EQ(TryParseDataHeader(bytes.data(), bytes.size(), &parsed), ParseError::BadLane);
}

TEST(TcpProtocol, RejectsOversizedControlBody) {
  WireWriter header;
  header.u32(kCtrlMagic);
  header.u16(kProtoVersion);
  header.u8(static_cast<uint8_t>(CtrlMsgType::WRITE_REQ));
  header.u8(0);
  header.u32(static_cast<uint32_t>(kMaxControlMessageBytes + 1));
  CtrlHeaderView parsed;
  EXPECT_EQ(TryParseCtrlHeader(header.buf.data(), header.buf.size(), &parsed),
            ParseError::BodyTooLarge);
}

TEST(TcpProtocol, LaneMaskSupportsAllSixteenLanes) {
  EXPECT_EQ(LanesAllMask(1), 0x0001);
  EXPECT_EQ(LanesAllMask(15), 0x7fff);
  EXPECT_EQ(LanesAllMask(16), 0xffff);
}

}  // namespace
}  // namespace mori::io::tcp
