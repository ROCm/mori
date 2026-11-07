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

// Seed inputs for the RAS protocol fuzz test. Headers are written here byte by byte, not with
// the encoder under test, so a seed can't inherit an encoder bug.

#include <cstdint>
#include <string>
#include <vector>

namespace fuzz_corpus {

inline void AppendLe(std::vector<uint8_t>& out, uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

// A v1 link header from sender 0x8877665544332211 with sequence 1.
inline std::vector<uint8_t> Header(uint16_t type, uint32_t length, uint16_t version = 1) {
  std::vector<uint8_t> h;
  AppendLe(h, version, 2);
  AppendLe(h, type, 2);
  AppendLe(h, length, 4);
  AppendLe(h, 0x8877665544332211ull, 8);  // sender
  AppendLe(h, 1, 8);                      // sequence
  AppendLe(h, 0, 8);                      // target: the link peer
  AppendLe(h, 0, 1);                      // scope LINK
  AppendLe(h, 1, 1);                      // hops
  AppendLe(h, 0, 2);                      // reserved
  return h;
}

inline std::vector<uint8_t> Frame(uint16_t type, const std::vector<uint8_t>& payload,
                                  uint16_t version = 1) {
  std::vector<uint8_t> f = Header(type, static_cast<uint32_t>(payload.size()), version);
  f.insert(f.end(), payload.begin(), payload.end());
  return f;
}

}  // namespace fuzz_corpus

inline std::vector<std::vector<uint8_t>> FuzzCorpusSeeds() {
  using fuzz_corpus::Frame;
  using fuzz_corpus::Header;
  std::vector<std::vector<uint8_t>> seeds;
  // Valid: LEAVE(kProcessExit), HEARTBEAT, HEARTBEAT_ACK with a negative wall time.
  seeds.push_back(Frame(3, {0x91, 0x00}));
  seeds.push_back(Frame(4, {0x92, 0xcf, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00}));
  seeds.push_back(Frame(
      5, {0x93, 0x01, 0xcd, 0x05, 0xdc, 0xd3, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00}));
  // Header-only HEARTBEAT: length 0, no payload.
  seeds.push_back(Header(4, 0));
  // Reserved type (JOIN) with a plausible payload.
  seeds.push_back(Frame(2, {0x91, 0x00}));
  // Version 2, which a v1 decoder must refuse.
  seeds.push_back(Frame(4, {0x92, 0x01, 0x00}, 2));
  // HELLO whose payload declares a 2^32-1 entry map, and one with a 4 GB string.
  seeds.push_back(Frame(1, {0xdf, 0xff, 0xff, 0xff, 0xff}));
  seeds.push_back(Frame(1, {0x98, 0x01, 0x01, 0x00, 0x00, 0x03, 0xdb, 0xff, 0xff, 0xff, 0xff}));
  // Nesting 100 arrays deep.
  seeds.push_back(Frame(1, std::vector<uint8_t>(100, 0x91)));
  // An HTTP probe and a TLS ClientHello prefix, as a port scanner might send.
  std::string http = "GET / HTTP/1.1\r\nHost: x\r\n\r\n";
  seeds.push_back(std::vector<uint8_t>(http.begin(), http.end()));
  std::vector<uint8_t> tls = {0x16, 0x03, 0x01, 0x02, 0x00, 0x01, 0x00, 0x01, 0xfc, 0x03, 0x03};
  tls.resize(36, 0x00);
  seeds.push_back(tls);
  return seeds;
}
