
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
#include "mori/application/transport/tcp/tcp.hpp"

#include <string.h>

#include <cassert>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace application {

#define DEFAULT_LISTEN_BACKLOG 128

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPEndpoint                                          */
/* ---------------------------------------------------------------------------------------------- */

int TCPEndpoint::Send(const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len > 0) {
    size_t n = send(handle.fd, p, len, 0);
    if (n < 0) return n;
    p += n;
    len -= n;
  }
  return 0;
}

int TCPEndpoint::Recv(void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  while (len > 0) {
    size_t n = ::recv(handle.fd, p, len, 0);
    if (n <= 0) return n;
    p += n;
    len -= n;
  }
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPContext                                           */
/* ---------------------------------------------------------------------------------------------- */
TCPContext::TCPContext(std::string host, uint16_t port) {
  handle.host = host;
  handle.port = port;
}

TCPContext::~TCPContext() { Close(); }

void TCPContext::Listen() {
  listenFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  assert(listenFd >= 0);

  int opt = 1;
  setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(handle.port);
  addr.sin_addr.s_addr = inet_addr(handle.host.c_str());

  SYSCALL_RETURN_ZERO(bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)));

  socklen_t len = sizeof(addr);
  getsockname(listenFd, reinterpret_cast<sockaddr*>(&addr), &len);
  handle.port = ntohs(addr.sin_port);

  SYSCALL_RETURN_ZERO(listen(listenFd, DEFAULT_LISTEN_BACKLOG));
}

void TCPContext::Close() {
  if (listenFd >= 0) {
    SYSCALL_RETURN_ZERO(close(listenFd));
    listenFd = -1;
  }
  while (!endpoints.empty()) {
    auto it = endpoints.begin();
    CloseEndpoint(it->second);
  }
}

TCPEndpointHandle TCPContext::Connect(std::string remote, uint16_t port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  assert(sock >= 0);

  sockaddr_in peer{};
  peer.sin_family = AF_INET;
  peer.sin_port = htons(port);
  peer.sin_addr.s_addr = inet_addr(remote.c_str());

  SYSCALL_RETURN_ZERO(connect(sock, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)));

  TCPEndpointHandle ep{sock, peer};
  endpoints.insert({sock, ep});
  return ep;
}

TCPEndpointHandleVec TCPContext::Accept() {
  sockaddr_in peer{};
  socklen_t len = sizeof(peer);

  TCPEndpointHandleVec newEps;

  while (true) {
    int sock = accept(listenFd, reinterpret_cast<sockaddr*>(&peer), &len);
    if (sock >= 0) {
      TCPEndpointHandle ep{sock, peer};
      newEps.push_back(ep);
      endpoints.insert({sock, ep});
    }
    if ((sock == -1) && ((errno == EAGAIN) || (errno == EWOULDBLOCK))) {
      break;
    }
  }

  return newEps;
}

void TCPContext::CloseEndpoint(TCPEndpointHandle ep) {
  if (endpoints.find(ep.fd) == endpoints.end()) return;
  SYSCALL_RETURN_ZERO_IGNORE_ERROR(shutdown(ep.fd, SHUT_WR), ENOTCONN);
  SYSCALL_RETURN_ZERO(close(ep.fd));
  endpoints.erase(ep.fd);
}

}  // namespace application
}  // namespace mori
