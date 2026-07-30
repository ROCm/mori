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

// Executed coverage for `19b718f3` -- the `hdr.len` bound and the message-type
// validation in `Protocol::ReadMessageHeader`.
//
// Why a NEW file rather than extending `tests/cpp/io/test_protocol.cpp`: that
// file is STALE and cannot compile. It uses `MessageRegEngine` /
// `WriteMessageRegEngine` / `ReadMessageRegEngine` and `msg.engineDesc`, none
// of which exist in `src/io/rdma/protocol.hpp` any more (the type is
// `MessageRegEndpoint` with an `ekey`/`topo`/`devId`/`eph`). It is also not
// registered in `tests/cpp/CMakeLists.txt`, which is how it rotted unnoticed.
// Fixing it is a separate change from giving `19b718f3` its first test.
//
// Every check is an unconditional CHECK, never `assert()`: mori builds Release
// with `-DNDEBUG`, under which asserts compile to nothing and the binary exits
// 0 having verified NOTHING. That vacuity is exactly what `f2f02821` caught in
// `test_transport_tcp.cpp` and what `19b718f3` fixed in the product code (the
// only type guard was an `assert` at the call sites).

#include <arpa/inet.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "src/io/rdma/protocol.hpp"

using namespace mori::application;
using namespace mori::io;

#define CHECK(cond, msg)                                                 \
  do {                                                                   \
    if (!(cond)) {                                                       \
      fprintf(stderr, "[%s:%d] CHECK FAILED: %s\n    %s\n", __FILE__,    \
              __LINE__, #cond, (msg));                                   \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

namespace {

// A connected loopback pair. The writer side is a raw TCPEndpoint rather than a
// Protocol, deliberately: the whole point is to put bytes on the wire that a
// well-behaved `WriteMessageHeader` would never produce -- a peer that flipped
// role mid-message, or a byte-swapped/truncated header.
struct Pair {
  TCPContext server{"127.0.0.1", 0};
  TCPContext client{"127.0.0.1", 0};
  TCPEndpointHandle srvHandle{};
  TCPEndpointHandle cliHandle{};

  Pair() {
    server.Listen();
    cliHandle = client.Connect("127.0.0.1", server.GetPort());
    srvHandle = server.Accept()[0];
  }

  ~Pair() {
    server.Close();
    client.Close();
  }
};

// Put a header on the wire the way the product code does: 1 byte of type, then
// 4 bytes of big-endian length.
void SendRawHeader(TCPEndpoint& ep, uint8_t type, uint32_t len) {
  uint32_t netLen = htonl(len);
  CHECK(ep.Send(&type, sizeof(type)) == 0, "sending the raw type byte failed");
  CHECK(ep.Send(&netLen, sizeof(netLen)) == 0, "sending the raw length failed");
}

// THE bug. `hdr.len` is a uint32_t straight off the wire; before `19b718f3` it
// went into `std::vector<char> buf(len)` (protocol.cpp:89,104) with no check,
// so four bytes from a peer request up to 4 GiB. 0xFFFFFFFF is the worst case.
//
// Note what a FAILURE of this test looks like, because it is not a clean
// assertion failure: without the bound, `ReadMessageHeader` RETURNS this header
// happily and the allocation happens in the caller. So the check has to be "it
// threw", and the test must not itself perform the 4 GiB allocation while
// proving that -- which is why it stops at the header and never calls
// ReadMessageRegEndpoint.
void TestOversizedLengthIsRefused() {
  Pair p;
  TCPEndpoint cli(p.cliHandle);
  Protocol srv(p.srvHandle);

  SendRawHeader(cli, static_cast<uint8_t>(MessageType::RegEndpoint), 0xFFFFFFFFu);

  bool threw = false;
  std::string what;
  try {
    srv.ReadMessageHeader();
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK(threw,
        "a 4 GiB hdr.len must be refused BEFORE the allocation. Returning "
        "here is the unbounded-allocation bug 19b718f3 fixed: the caller then "
        "does std::vector<char> buf(len).");
  CHECK(what.find("exceeds the maximum") != std::string::npos,
        "the throw must name the length bound, so an operator can tell it "
        "apart from a socket error on the same channel");
  printf("oversized length refused: %s\n", what.c_str());
}

// One byte over the bound, to prove the boundary is where it is claimed to be
// and not merely somewhere. 1 MiB exactly must still be ACCEPTED -- a bound
// that rejects the legal maximum is a different bug.
void TestBoundaryIsExact() {
  {
    Pair p;
    TCPEndpoint cli(p.cliHandle);
    Protocol srv(p.srvHandle);
    SendRawHeader(cli, static_cast<uint8_t>(MessageType::AskMemoryRegion), (1u << 20));
    bool threw = false;
    try {
      MessageHeader hdr = srv.ReadMessageHeader();
      CHECK(hdr.len == (1u << 20), "the accepted header's len was mangled");
      CHECK(hdr.type == MessageType::AskMemoryRegion, "the accepted header's type was mangled");
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(!threw, "exactly kMaxControlMessageBytes (1 MiB) must be ACCEPTED");
  }
  {
    Pair p;
    TCPEndpoint cli(p.cliHandle);
    Protocol srv(p.srvHandle);
    SendRawHeader(cli, static_cast<uint8_t>(MessageType::AskMemoryRegion), (1u << 20) + 1);
    bool threw = false;
    try {
      srv.ReadMessageHeader();
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "one byte over the bound must be REFUSED");
  }
  printf("bound is exact at 1 MiB\n");
}

// The type guard. Before `19b718f3` the only check was
// `assert(hdr.type == MessageType::RegEndpoint)` at backend_impl.cpp:935/:975,
// compiled out under `-DNDEBUG`, so in the configuration that actually ships
// an unknown type was dispatched as if it were a known one.
void TestUnknownTypeIsRefused() {
  Pair p;
  TCPEndpoint cli(p.cliHandle);
  Protocol srv(p.srvHandle);

  // 0 and 1 are the only defined values of MessageType; 0x7F is not.
  SendRawHeader(cli, 0x7F, 16);

  bool threw = false;
  std::string what;
  try {
    srv.ReadMessageHeader();
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK(threw, "an unknown message type must be refused");
  CHECK(what.find("unknown message type") != std::string::npos,
        "the throw must name the type, not the length");
  printf("unknown type refused: %s\n", what.c_str());
}

// A well-formed header must still round-trip. Without this, a bound that
// rejected EVERYTHING would pass all three tests above -- the mirror image of
// the vacuity problem, and the reason the boundary test above checks the
// accept side too.
void TestWellFormedHeaderRoundTrips() {
  Pair p;
  Protocol cli(p.cliHandle);
  Protocol srv(p.srvHandle);

  MessageHeader out{};
  out.type = MessageType::RegEndpoint;
  out.len = 123;
  cli.WriteMessageHeader(out);

  MessageHeader in = srv.ReadMessageHeader();
  CHECK(in.type == MessageType::RegEndpoint, "type did not round-trip");
  CHECK(in.len == 123, "len did not round-trip");
  printf("well-formed header round-tripped (type=%u len=%u)\n",
         static_cast<unsigned>(in.type), in.len);
}

}  // namespace

int main() {
  TestWellFormedHeaderRoundTrips();
  TestOversizedLengthIsRefused();
  TestBoundaryIsExact();
  TestUnknownTypeIsRefused();
  printf("test_protocol_header: ALL PASSED\n");
  return 0;
}
