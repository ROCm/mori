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
#include "http.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/sockios.h>
#include <netdb.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace xgmi {
namespace {

constexpr size_t kMaxResponse = 1 << 20;

int64_t NowMs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Waits for `events` on fd until the deadline; false on timeout or error.
bool WaitFd(int fd, short events, int64_t deadline) {
  for (;;) {
    const int64_t left = deadline - NowMs();
    if (left <= 0) return false;
    pollfd p{fd, events, 0};
    const int r = poll(&p, 1, static_cast<int>(left));
    if (r > 0) return true;
    if (r == 0) return false;
    if (errno != EINTR) return false;
  }
}

// WaitFd, but a timeout while the kernel is still sending queued bytes (SIOCOUTQ went down) is
// progress: on a slow link POLLOUT only comes back once half the send buffer is free.
bool WaitWithProgress(int fd, short events, int64_t* deadline, int idle_ms, int64_t hard) {
  for (;;) {
    int before = 0, after = 0;
    ioctl(fd, SIOCOUTQ, &before);
    if (WaitFd(fd, events, *deadline)) return true;
    if (ioctl(fd, SIOCOUTQ, &after) != 0 || after >= before || NowMs() >= hard) return false;
    *deadline = std::min(NowMs() + idle_ms, hard);
  }
}

int Connect(const Url& url, int64_t deadline, std::string* err) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const int rc = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &res);
  if (rc != 0) {
    *err = "resolve " + url.host + ": " + gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  for (addrinfo* a = res; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, a->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    if (errno == EINPROGRESS && WaitFd(fd, POLLOUT, deadline)) {
      int so = 0;
      socklen_t len = sizeof(so);
      if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &len) == 0 && so == 0) break;
      errno = so;
    }
    *err =
        "connect " + url.host + ":" + url.port + ": " + (errno ? std::strerror(errno) : "timeout");
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  return fd;
}

}  // namespace

bool ParseUrl(const std::string& s, Url* out, std::string* err) {
  const std::string scheme = "http://";
  if (s.compare(0, scheme.size(), scheme) != 0) {
    *err = "URL must start with http:// (got \"" + s + "\")";
    return false;
  }
  std::string rest = s.substr(scheme.size());
  const size_t slash = rest.find('/');
  std::string hostport = rest.substr(0, slash);
  out->prefix = slash == std::string::npos ? "" : rest.substr(slash);
  while (!out->prefix.empty() && out->prefix.back() == '/') out->prefix.pop_back();
  out->port = "80";
  if (!hostport.empty() && hostport[0] == '[') {  // [v6addr]:port
    const size_t close = hostport.find(']');
    if (close == std::string::npos) {
      *err = "cannot parse URL \"" + s + "\"";
      return false;
    }
    out->host = hostport.substr(1, close - 1);
    if (close + 1 < hostport.size() && hostport[close + 1] == ':')
      out->port = hostport.substr(close + 2);
  } else {
    const size_t colon = hostport.rfind(':');
    out->host = hostport.substr(0, colon);
    if (colon != std::string::npos) out->port = hostport.substr(colon + 1);
  }
  if (out->host.empty() || out->port.empty()) {
    *err = "cannot parse URL \"" + s + "\"";
    return false;
  }
  return true;
}

std::string UrlEncode(const std::string& s) {
  std::string out;
  char buf[4];
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out += static_cast<char>(c);
    } else {
      std::snprintf(buf, sizeof(buf), "%%%02X", c);
      out += buf;
    }
  }
  return out;
}

int HttpRequest(const Url& url, const std::string& method, const std::string& path_and_query,
                const std::string& body, int idle_ms, int total_ms, std::string* response,
                std::string* err) {
  // Inactivity timeout: every byte sent or received moves the deadline, so a slow but working
  // link (e.g. an ssh tunnel) is not cut off; a stalled one is.
  const int64_t hard = total_ms > 0 ? NowMs() + total_ms : INT64_MAX;
  int64_t deadline = std::min(NowMs() + idle_ms, hard);
  const int fd = Connect(url, deadline, err);
  if (fd < 0) return -1;
  std::string req = method + " " + url.prefix + path_and_query + " HTTP/1.0\r\nHost: " + url.host +
                    ":" + url.port + "\r\nContent-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n";
  req += body;
  size_t sent = 0;
  while (sent < req.size()) {
    const ssize_t w = send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
    if (w > 0) {
      sent += static_cast<size_t>(w);
      deadline = std::min(NowMs() + idle_ms, hard);
    } else if (w < 0 && errno == EINTR) {
      continue;
    } else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
               WaitWithProgress(fd, POLLOUT, &deadline, idle_ms, hard)) {
      continue;
    } else {
      *err = std::string("send: ") + (w < 0 && errno != EAGAIN ? std::strerror(errno) : "timeout");
      close(fd);
      return -1;
    }
  }
  std::string resp;
  char buf[65536];
  for (;;) {
    const ssize_t r = recv(fd, buf, sizeof(buf), 0);
    if (r > 0) {
      deadline = std::min(NowMs() + idle_ms, hard);
      if (resp.size() < kMaxResponse)
        resp.append(buf, static_cast<size_t>(std::min<ssize_t>(r, kMaxResponse)));
      continue;
    }
    if (r == 0) break;
    if (errno == EINTR) continue;
    if ((errno == EAGAIN || errno == EWOULDBLOCK) &&
        WaitWithProgress(fd, POLLIN, &deadline, idle_ms, hard))
      continue;
    *err = std::string("receive: ") +
           (errno == EAGAIN || errno == EWOULDBLOCK ? "timeout" : std::strerror(errno));
    close(fd);
    return -1;
  }
  close(fd);
  int status = 0;
  if (std::sscanf(resp.c_str(), "HTTP/%*d.%*d %d", &status) != 1) {
    *err = "malformed HTTP response";
    return -1;
  }
  const size_t sep = resp.find("\r\n\r\n");
  if (response) *response = sep == std::string::npos ? "" : resp.substr(sep + 4);
  return status;
}

}  // namespace xgmi
