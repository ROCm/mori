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
#include <csignal>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"

using namespace mori::application;

// NOT assert(): mori builds Release with -DNDEBUG, under which every assert in
// this file compiles to nothing and the binary exits 0 having checked NOTHING.
// A test that passes vacuously in the configuration it actually ships in is
// worse than no test -- the same vacuity trap as the heap-stats guards
// (5597e2fe). CHECK is unconditional and prints what failed.
#define CHECK(cond, msg)                                                      \
  do {                                                                        \
    if (!(cond)) {                                                            \
      fprintf(stderr, "[%s:%d] CHECK FAILED: %s\n    %s\n", __FILE__,        \
              __LINE__, #cond, (msg));                                        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

void TestTcpContext() {
  std::string host = "127.0.0.1";

  TCPContext context1(host, 0);
  TCPContext context2(host, 0);

  context1.Listen();
  context2.Listen();
  printf("port 1 %d port 2 %d\n", context1.GetPort(), context2.GetPort());
  CHECK((context1.GetPort() > 0) && (context2.GetPort() > 0), "Listen() gave no port");
  CHECK((context1.GetListenFd() >= 0) && (context2.GetListenFd() >= 0),
        "Listen() gave no listen fd");

  TCPEndpointHandle eph1 = context1.Connect(host, context2.GetPort());
  TCPEndpointHandle eph2 = context2.Accept()[0];

  TCPEndpoint ep1(eph1);
  TCPEndpoint ep2(eph2);

  std::string sendBuf("Hello Mori!");
  std::vector<char> recvBuf(sendBuf.size());

  CHECK(ep1.Send(sendBuf.c_str(), sendBuf.size()) == 0, "happy-path Send failed");
  CHECK(ep2.Recv(recvBuf.data(), sendBuf.size()) == 0, "happy-path Recv failed");
  CHECK(std::string(recvBuf.data()) == sendBuf, "payload did not round-trip");

  context1.Close();
  context2.Close();
}

// The bug `99768347` fixed, as a test. `TCPEndpoint::Recv` stored recv()'s
// return in a `size_t`, so the `n < 0` branch was DEAD and a peer that closed
// mid-message came back as 0 == "success" -- the caller then consumed a
// partially-filled struct. For the control plane that struct is a
// MessageHeader whose `hdr.len` goes straight into `std::vector<char> buf(len)`
// (protocol.cpp), which is the `std::length_error` in Team E's stack.
//
// This is also the ONLY executed coverage of that commit: `mori_ops` does not
// link `mori_io`, so the 227/608 python suite cannot reach one line of it
// (RESULTS_M T24b, struck). Everything else standing under 99768347 is an
// objdump of the emitted `js`.
void TestRecvReportsMidMessageEof() {
  std::string host = "127.0.0.1";

  TCPContext server(host, 0);
  server.Listen();

  TCPContext client(host, 0);
  TCPEndpointHandle cliHandle = client.Connect(host, server.GetPort());
  TCPEndpointHandle srvHandle = server.Accept()[0];

  TCPEndpoint cli(cliHandle);
  TCPEndpoint srv(srvHandle);

  // Send a HEADER-SIZED PREFIX and then close: the receiver asks for more than
  // will ever arrive, so recv() returns a short count and then 0. That is the
  // teardown/reconnect timing a PD role switch actually produces, since a flip
  // tears down and re-establishes the peer's endpoints.
  const char partial[4] = {'m', 'o', 'r', 'i'};
  CHECK(cli.Send(partial, sizeof(partial)) == 0, "sending the partial prefix failed");
  client.Close();

  // 16 requested, 4 will arrive, then EOF.
  std::vector<char> buf(16, '\0');
  int rc = srv.Recv(buf.data(), buf.size());
  CHECK(rc == -1,
        "mid-message EOF must be reported as -1, not as success. A 0 here IS "
        "the size_t bug 99768347 fixed: the caller goes on to use a "
        "partially-filled MessageHeader whose hdr.len then sizes a vector.");

  printf("mid-message EOF correctly reported as %d\n", rc);
  server.Close();
}

// A closed peer must be reported on SEND too, not silently swallowed.
void TestSendReportsClosedPeer() {
  std::string host = "127.0.0.1";

  TCPContext server(host, 0);
  server.Listen();
  TCPContext client(host, 0);
  TCPEndpointHandle cliHandle = client.Connect(host, server.GetPort());
  TCPEndpointHandle srvHandle = server.Accept()[0];

  TCPEndpoint cli(cliHandle);
  server.Close();
  ::close(srvHandle.fd);

  // SIGPIPE would kill the process before Send could return; the control-plane
  // thread runs with it ignored, so match that here rather than testing a
  // configuration nothing runs in.
  ::signal(SIGPIPE, SIG_IGN);

  // The first write may land in the socket buffer and succeed; the second sees
  // the RST. Either way at least one must report failure rather than 0.
  std::vector<char> payload(4096, 'x');
  int first = cli.Send(payload.data(), payload.size());
  int second = cli.Send(payload.data(), payload.size());
  CHECK(first != 0 || second != 0,
        "writing to a closed peer must eventually report an error");
  printf("closed-peer send reported %d then %d\n", first, second);

  client.Close();
}

int main() {
  TestTcpContext();
  TestRecvReportsMidMessageEof();
  TestSendReportsClosedPeer();
  printf("test_transport_tcp: ALL PASSED\n");
  return 0;
}
