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
#include <cassert>
#include <system_error>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"

using namespace mori::application;

void TestTcpContext() {
  std::string host = "127.0.0.1";

  TCPContext context1(host, 0);
  TCPContext context2(host, 0);

  context1.Listen();
  context2.Listen();
  printf("port 1 %d port 2 %d\n", context1.GetPort(), context2.GetPort());
  assert((context1.GetPort() > 0) && (context2.GetPort() > 0));
  assert((context1.GetListenFd() >= 0) && (context2.GetListenFd() >= 0));

  TCPEndpointHandle eph1 = context1.Connect(host, context2.GetPort());
  TCPEndpointHandle eph2 = context2.Accept()[0];

  TCPEndpoint ep1(eph1);
  TCPEndpoint ep2(eph2);

  std::string sendBuf("Hello Mori!");
  std::vector<char> recvBuf(sendBuf.size());

  assert(ep1.Send(sendBuf.c_str(), sendBuf.size()) == 0);
  assert(ep2.Recv(recvBuf.data(), sendBuf.size()) == 0);
  assert(std::string(recvBuf.begin(), recvBuf.end()) == sendBuf);

  context1.Close();
  context2.Close();
}

void TestTcpRecvReportsEof() {
  std::string host = "127.0.0.1";

  TCPContext context1(host, 0);
  TCPContext context2(host, 0);

  context1.Listen();
  context2.Listen();

  TCPEndpointHandle eph1 = context1.Connect(host, context2.GetPort());
  TCPEndpointHandle eph2 = context2.Accept()[0];
  context1.CloseEndpoint(eph1);

  TCPEndpoint ep2(eph2);
  char buf[8]{};
  size_t consumed = 99;
  assert(ep2.Recv(buf, sizeof(buf), &consumed) == 1);
  assert(consumed == 0);

  context2.Close();
}

void TestTcpConnectFailureThrows() {
  std::string host = "127.0.0.1";

  TCPContext listener(host, 0);
  listener.Listen();
  uint16_t closedPort = listener.GetPort();
  listener.Close();

  TCPContext client(host, 0);
  bool threw = false;
  try {
    (void)client.Connect(host, closedPort);
  } catch (const std::system_error& e) {
    threw = true;
    assert(e.code().value() != 0);
  }
  assert(threw);
}

int main() {
  TestTcpContext();
  TestTcpRecvReportsEof();
  TestTcpConnectFailureThrows();
}
