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

// The control-plane PAYLOAD path: the two `ReadMessage*`/`WriteMessage*` pairs,
// end to end over a real loopback socket.
//
// This file was DEAD before this commit -- it did not compile. It still called
// `MessageRegEngine` / `WriteMessageRegEngine` / `ReadMessageRegEngine` and
// `msg.engineDesc`, none of which exist in `src/io/rdma/protocol.hpp` any more;
// the message is `MessageRegEndpoint` with `ekey`/`topo`/`devId`/`eph`/`nicRank`,
// and there is a second message type (`MessageAskMemoryRegion`) it never knew
// about at all. It rotted unnoticed for exactly one reason: it was never listed
// in `tests/cpp/CMakeLists.txt`, so no build ever compiled it. A test that
// cannot compile is the same class of defect as a vacuous one -- both report
// nothing while looking like coverage -- so it is ported and REGISTERED here.
//
// It is not redundant with `test_protocol_header.cpp` (43c39f3c). That file
// deliberately stops at the header, because proving a 4 GiB length is refused
// must not itself allocate 4 GiB. Nothing therefore covers the msgpack payload
// round-trip, and nothing covers what a SHORT payload does -- which matters
// because `99768347` is precisely the commit that made a mid-message EOF
// reportable, and `3c6bc1a1` turned that report from `exit(-1)` into a throw.
// Those two commits' interaction on the payload path is tested below.
//
// Every check is an unconditional CHECK, never `assert()`: mori builds Release
// with `-DNDEBUG`, under which asserts vanish and the binary exits 0 having
// verified nothing. The original file used `assert()` throughout -- so even if
// it HAD been registered, it would have been vacuous in the shipping config.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "src/io/rdma/protocol.hpp"

using namespace mori::application;
using namespace mori::io;

#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      fprintf(stderr, "[%s:%d] CHECK FAILED: %s\n    %s\n", __FILE__, \
              __LINE__, #cond, (msg));                                \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {

struct Pair {
  TCPContext server{"127.0.0.1", 0};
  TCPContext client{"127.0.0.1", 0};
  TCPEndpointHandle srvHandle{};
  TCPEndpointHandle cliHandle{};

  Pair() {
    server.Listen();
    CHECK(server.GetPort() > 0, "the listener did not get a port");
    CHECK(server.GetListenFd() >= 0, "the listener did not get an fd");
    cliHandle = client.Connect("127.0.0.1", server.GetPort());
    srvHandle = server.Accept()[0];
  }

  ~Pair() {
    server.Close();
    client.Close();
  }
};

// A handle with every field set to a DISTINCT non-default value. Defaults are
// the trap here: `RdmaEndpointHandle` value-initialises psn/qpn/portId to 0 and
// `gidIdx` to -1, so a convert() that dropped a field entirely would still
// compare equal against a mostly-default fixture.
RdmaEndpointHandle MakeHandle() {
  RdmaEndpointHandle eph{};
  eph.psn = 22;
  eph.qpn = 35;
  eph.portId = 9999;
  eph.maxSge = 4;
  eph.ib.lid = 678;
  for (size_t i = 0; i < sizeof(eph.eth.gid); i++) eph.eth.gid[i] = static_cast<uint8_t>(i);
  for (size_t i = 0; i < sizeof(eph.eth.mac); i++) eph.eth.mac[i] = static_cast<uint8_t>(0xA0 + i);
  eph.eth.gidIdx = 3;
  return eph;
}

void TestRegEndpointRoundTrip() {
  Pair p;
  Protocol initiator(p.cliHandle);
  Protocol target(p.srvHandle);

  MessageRegEndpoint msg;
  msg.ekey = "initiator";
  msg.topo.local = TopoKey{2, MemoryLocationType::GPU, 1};
  msg.topo.remote = TopoKey{5, MemoryLocationType::CPU, 0};
  msg.devId = 7;
  msg.eph = MakeHandle();
  msg.nicRank = 3;

  initiator.WriteMessageRegEndpoint(msg);

  MessageHeader hdr = target.ReadMessageHeader();
  CHECK(hdr.type == MessageType::RegEndpoint, "the writer tagged the wrong message type");
  CHECK(hdr.len > 0 && hdr.len <= (1u << 20),
        "a real RegEndpoint must fit well inside kMaxControlMessageBytes -- if this "
        "fires, 19b718f3's bound is too tight for legitimate traffic");
  MessageRegEndpoint recv = target.ReadMessageRegEndpoint(hdr.len);

  CHECK(recv.ekey == msg.ekey, "ekey did not round-trip");
  CHECK(recv.topo == msg.topo, "topo did not round-trip");
  CHECK(recv.devId == msg.devId, "devId did not round-trip");
  CHECK(recv.nicRank == msg.nicRank, "nicRank did not round-trip");
  CHECK(recv.eph == msg.eph, "the endpoint handle did not round-trip");

  // RdmaEndpointHandle::operator== compares psn/qpn/portId/ib/eth and NOT
  // maxSge, and EthernetEndpointHandle::operator== does compare gid/mac/gidIdx
  // -- but check the byte arrays field by field anyway. A `==` that silently
  // omits a member is exactly the kind of thing this test should not inherit
  // its verdict from.
  for (size_t i = 0; i < sizeof(msg.eph.eth.gid); i++)
    CHECK(recv.eph.eth.gid[i] == static_cast<uint8_t>(i), "a gid byte did not round-trip");
  for (size_t i = 0; i < sizeof(msg.eph.eth.mac); i++)
    CHECK(recv.eph.eth.mac[i] == static_cast<uint8_t>(0xA0 + i), "a mac byte did not round-trip");
  CHECK(recv.eph.eth.gidIdx == 3, "gidIdx did not round-trip");
  printf("RegEndpoint round-tripped (%u payload bytes)\n", hdr.len);
}

void TestAskMemoryRegionRoundTrip() {
  Pair p;
  Protocol initiator(p.cliHandle);
  Protocol target(p.srvHandle);

  MessageAskMemoryRegion msg;
  msg.ekey = "asker";
  msg.devId = 4;
  msg.id = 0xDEADBEEFu;
  msg.mr.addr = 0x7F0011223344ull;
  msg.mr.lkey = 0x11223344u;
  msg.mr.rkey = 0x55667788u;
  msg.mr.length = 1ull << 33;  // > 4 GiB: catches a size_t narrowed to uint32_t

  initiator.WriteMessageAskMemoryRegion(msg);

  MessageHeader hdr = target.ReadMessageHeader();
  CHECK(hdr.type == MessageType::AskMemoryRegion, "the writer tagged the wrong message type");
  MessageAskMemoryRegion recv = target.ReadMessageAskMemoryRegion(hdr.len);

  CHECK(recv.ekey == msg.ekey, "ekey did not round-trip");
  CHECK(recv.devId == msg.devId, "devId did not round-trip");
  CHECK(recv.id == msg.id, "MemoryUniqueId did not round-trip");
  CHECK(recv.mr.addr == msg.mr.addr, "mr.addr did not round-trip");
  CHECK(recv.mr.lkey == msg.mr.lkey, "mr.lkey did not round-trip");
  CHECK(recv.mr.rkey == msg.mr.rkey, "mr.rkey did not round-trip");
  CHECK(recv.mr.length == msg.mr.length,
        "mr.length did not round-trip -- a >4 GiB region narrowed somewhere");
  printf("AskMemoryRegion round-tripped (%u payload bytes, length=%zu)\n", hdr.len,
         recv.mr.length);
}

// Two messages back to back on ONE connection, which is how the control-plane
// server loop actually uses it. If `ReadMessage*` ever over- or under-reads its
// payload, the FIRST message still passes and the second desynchronises -- so a
// single-message test cannot see it.
void TestBackToBackMessagesStayFramed() {
  Pair p;
  Protocol initiator(p.cliHandle);
  Protocol target(p.srvHandle);

  MessageRegEndpoint a;
  a.ekey = "first";
  a.topo.local = TopoKey{1, MemoryLocationType::GPU, 0};
  a.topo.remote = TopoKey{2, MemoryLocationType::GPU, 0};
  a.devId = 1;
  a.eph = MakeHandle();
  a.nicRank = 1;

  MessageAskMemoryRegion b;
  b.ekey = "second";
  b.devId = 2;
  b.id = 99;
  b.mr.addr = 0x1000;
  b.mr.lkey = 1;
  b.mr.rkey = 2;
  b.mr.length = 4096;

  initiator.WriteMessageRegEndpoint(a);
  initiator.WriteMessageAskMemoryRegion(b);

  MessageHeader h1 = target.ReadMessageHeader();
  CHECK(h1.type == MessageType::RegEndpoint, "first message's type is wrong");
  MessageRegEndpoint ra = target.ReadMessageRegEndpoint(h1.len);
  CHECK(ra.ekey == "first", "first message's payload is wrong");

  MessageHeader h2 = target.ReadMessageHeader();
  CHECK(h2.type == MessageType::AskMemoryRegion,
        "the SECOND header is wrong -- the first read left the stream "
        "desynchronised, i.e. it consumed the wrong number of bytes");
  MessageAskMemoryRegion rb = target.ReadMessageAskMemoryRegion(h2.len);
  CHECK(rb.ekey == "second", "second message's payload is wrong");
  CHECK(rb.id == 99, "second message's id is wrong");
  printf("two messages stayed framed on one connection\n");
}

// A peer that dies mid-payload -- the case `99768347` and `3c6bc1a1` are about,
// on the half of the protocol `test_protocol_header.cpp` cannot reach.
//
// Before `99768347`, `TCPEndpoint::Recv` stored `recv()`'s return in a `size_t`,
// so a short read/EOF could not be reported and this path returned "success"
// with a partially-filled buffer, which msgpack then parsed as garbage. Before
// `3c6bc1a1` the report was `exit(-1)`, uncatchable. The contract now is: it
// THROWS, and the caller drops that one connection.
//
// This is the ordinary case on the path this campaign exists to exercise: a PD
// role flip tears the control plane down and re-establishes it, so a peer
// vanishing between the header and its payload is expected traffic, not an
// attack.
void TestTruncatedPayloadThrows() {
  Pair p;
  Protocol target(p.srvHandle);

  {
    TCPEndpoint cli(p.cliHandle);
    // Announce a payload, send only part of it, then close.
    MessageHeader hdr{MessageType::AskMemoryRegion, 0};
    hdr.len = 64;
    Protocol writer(p.cliHandle);
    writer.WriteMessageHeader(hdr);
    std::vector<char> partial(8, 'x');
    CHECK(cli.Send(partial.data(), partial.size()) == 0, "sending the partial payload failed");
  }
  p.client.Close();  // EOF mid-payload

  MessageHeader hdr = target.ReadMessageHeader();
  CHECK(hdr.len == 64, "the header itself should have arrived intact");

  bool threw = false;
  std::string what;
  try {
    target.ReadMessageAskMemoryRegion(hdr.len);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK(threw,
        "a payload that ends early must be REPORTED. Returning here means the "
        "short read was treated as success and msgpack parsed a half-filled "
        "buffer -- the 99768347 bug. And it must throw, not exit(-1): the "
        "3c6bc1a1 contract is that one bad connection costs that connection.");
  printf("truncated payload reported: %s\n", what.c_str());
}

}  // namespace

int main() {
  TestRegEndpointRoundTrip();
  TestAskMemoryRegionRoundTrip();
  TestBackToBackMessagesStayFramed();
  TestTruncatedPayloadThrows();
  printf("test_protocol: ALL PASSED\n");
  return 0;
}
