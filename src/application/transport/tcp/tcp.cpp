
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
#include <sys/socket.h>

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <system_error>

#include "mori/application/utils/check.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

#define DEFAULT_LISTEN_BACKLOG 128

namespace {

int ReadControlPlaneTimeoutMs() {
  static const int timeoutMs = [] {
    const char* raw = std::getenv("MORI_IO_CP_TIMEOUT_MS");
    if (raw == nullptr || raw[0] == '\0') return 5000;

    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed < 0 ||
        parsed > 24L * 60L * 60L * 1000L) {
      MORI_APP_WARN("Ignoring invalid MORI_IO_CP_TIMEOUT_MS={} (expected [0, 86400000])", raw);
      return 5000;
    }
    return static_cast<int>(parsed);
  }();
  return timeoutMs;
}

void SetControlPlaneSocketTimeoutNoThrow(int fd) noexcept {
  int timeoutMs = ReadControlPlaneTimeoutMs();
  if (fd < 0 || timeoutMs == 0) return;

  timeval tv{};
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;

  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
    MORI_APP_WARN("Failed to set SO_RCVTIMEO on control-plane fd {}: errno={} ({})", fd, errno,
                  strerror(errno));
  }
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
    MORI_APP_WARN("Failed to set SO_SNDTIMEO on control-plane fd {}: errno={} ({})", fd, errno,
                  strerror(errno));
  }
}

bool ShutdownCloseNoThrow(int fd) noexcept {
  bool ok = true;
  if (shutdown(fd, SHUT_WR) != 0) {
    int err = errno;
    if (err != ENOTCONN && err != EINVAL) {
      MORI_APP_WARN("shutdown(fd={}) failed during endpoint close: errno={} ({})", fd, err,
                    strerror(err));
      ok = false;
    }
  }
  if (close(fd) != 0) {
    int err = errno;
    MORI_APP_WARN("close(fd={}) failed during endpoint close: errno={} ({})", fd, err,
                  strerror(err));
    ok = false;
  }
  return ok;
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPEndpoint                                          */
/* ---------------------------------------------------------------------------------------------- */

int TCPEndpoint::Send(const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len > 0) {
    ssize_t n = send(handle.fd, p, len, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (n == 0) {
      errno = EPIPE;
      return -1;
    }
    p += n;
    len -= n;
  }
  return 0;
}

int TCPEndpoint::Recv(void* buf, size_t len, size_t* consumed) {
  char* p = static_cast<char*>(buf);
  size_t got = 0;
  while (len > 0) {
    ssize_t n = ::recv(handle.fd, p, len, 0);
    if (n == 0) {
      if (consumed) *consumed = got;
      return 1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (consumed) *consumed = got;
      return -1;
    }
    p += n;
    len -= n;
    got += static_cast<size_t>(n);
  }
  if (consumed) *consumed = got;
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
    if (close(listenFd) != 0) {
      MORI_APP_WARN("close(listenFd={}) failed: errno={} ({})", listenFd, errno, strerror(errno));
    }
    listenFd = -1;
  }

  while (true) {
    TCPEndpointHandle ep;
    {
      std::lock_guard<std::mutex> lock(endpointsMu);
      if (endpoints.empty()) break;
      ep = endpoints.begin()->second;
    }
    CloseEndpointNoThrow(ep);
  }
}

TCPEndpointHandle TCPContext::Connect(std::string remote, uint16_t port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::system_error(errno, std::generic_category(), "socket(AF_INET, SOCK_STREAM)");
  }
  SetControlPlaneSocketTimeoutNoThrow(sock);

  sockaddr_in peer{};
  peer.sin_family = AF_INET;
  peer.sin_port = htons(port);
  peer.sin_addr.s_addr = inet_addr(remote.c_str());

  if (connect(sock, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)) != 0) {
    int err = errno;
    close(sock);
    throw std::system_error(err, std::generic_category(),
                            "connect(" + remote + ":" + std::to_string(port) + ")");
  }

  TCPEndpointHandle ep{sock, peer};
  {
    std::lock_guard<std::mutex> lock(endpointsMu);
    endpoints.insert({sock, ep});
  }
  return ep;
}

TCPEndpointHandleVec TCPContext::Accept() {
  sockaddr_in peer{};
  socklen_t len = sizeof(peer);

  TCPEndpointHandleVec newEps;

  while (true) {
    len = sizeof(peer);
    int sock = accept(listenFd, reinterpret_cast<sockaddr*>(&peer), &len);
    if (sock >= 0) {
      SetControlPlaneSocketTimeoutNoThrow(sock);
      TCPEndpointHandle ep{sock, peer};
      {
        std::lock_guard<std::mutex> lock(endpointsMu);
        endpoints.insert({sock, ep});
      }
      newEps.push_back(ep);
      continue;
    }
    if ((sock == -1) && errno == EINTR) continue;
    if ((sock == -1) && ((errno == EAGAIN) || (errno == EWOULDBLOCK))) {
      break;
    }
    if (sock == -1) {
      MORI_APP_WARN("accept(listenFd={}) failed: errno={} ({})", listenFd, errno, strerror(errno));
      break;
    }
  }

  return newEps;
}

void TCPContext::CloseEndpoint(TCPEndpointHandle ep) { CloseEndpointNoThrow(ep); }

bool TCPContext::CloseEndpointNoThrow(int fd) noexcept {
  if (fd < 0) return true;

  {
    std::lock_guard<std::mutex> lock(endpointsMu);
    auto it = endpoints.find(fd);
    if (it == endpoints.end()) return true;
    endpoints.erase(it);
  }
  return ShutdownCloseNoThrow(fd);
}

bool TCPContext::CloseEndpointNoThrow(TCPEndpointHandle ep) noexcept {
  if (ep.fd < 0) return true;

  {
    std::lock_guard<std::mutex> lock(endpointsMu);
    auto it = endpoints.find(ep.fd);
    if (it == endpoints.end()) return true;
    endpoints.erase(it);
  }
  return ShutdownCloseNoThrow(ep.fd);
}

}  // namespace application
}  // namespace mori
