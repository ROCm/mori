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
#include "http_client.h"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "util.h"

namespace netmon {

bool ParseHttpUrl(const std::string& url, HttpUrl* out, std::string* error) {
  const std::string scheme = "http://";
  if (url.compare(0, scheme.size(), scheme) != 0) {
    *error = "only http:// URLs are supported: " + url;
    return false;
  }
  std::string rest = url.substr(scheme.size());
  const size_t slash = rest.find('/');
  std::string hostport = rest.substr(0, slash);
  std::string prefix = slash == std::string::npos ? "" : rest.substr(slash);
  while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
  HttpUrl u;
  std::string port;
  if (!hostport.empty() && hostport[0] == '[') {
    const size_t close = hostport.find(']');
    if (close == std::string::npos) {
      *error = "bad IPv6 address in " + url;
      return false;
    }
    u.host = hostport.substr(1, close - 1);
    if (close + 1 < hostport.size()) {
      if (hostport[close + 1] != ':') {
        *error = "bad host in " + url;
        return false;
      }
      port = hostport.substr(close + 2);
    }
  } else {
    const size_t colon = hostport.rfind(':');
    u.host = hostport.substr(0, colon);
    if (colon != std::string::npos) port = hostport.substr(colon + 1);
  }
  if (u.host.empty()) {
    *error = "no host in " + url;
    return false;
  }
  if (!port.empty()) {
    char* end = nullptr;
    const unsigned long p = strtoul(port.c_str(), &end, 10);
    if (*end != '\0' || p == 0 || p > 65535) {
      *error = "bad port in " + url;
      return false;
    }
    u.port = static_cast<uint16_t>(p);
  }
  if (prefix.find_first_of("?# ") != std::string::npos) {
    *error = "the URL must not have a query or fragment: " + url;
    return false;
  }
  u.prefix = prefix;
  *out = u;
  return true;
}

std::string UrlEncode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      o += static_cast<char>(c);
    } else {
      o += '%';
      o += hex[c >> 4];
      o += hex[c & 15];
    }
  }
  return o;
}

bool HttpClient::Resolve(const HttpUrl& url, std::string* error) {
  url_ = url;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port = std::to_string(url.port);
  const int rc = getaddrinfo(url.host.c_str(), port.c_str(), &hints, &res);
  if (rc != 0 || !res) {
    *error = "cannot resolve " + url.host + ": " + gai_strerror(rc);
    return false;
  }
  addrs_.clear();
  for (addrinfo* a = res; a; a = a->ai_next) {  // e.g. localhost: ::1 first, then 127.0.0.1
    Addr x;
    memcpy(&x.sa, a->ai_addr, a->ai_addrlen);
    x.len = a->ai_addrlen;
    addrs_.push_back(x);
  }
  freeaddrinfo(res);
  const bool v6 = url.host.find(':') != std::string::npos;
  host_header_ = (v6 ? "[" + url.host + "]" : url.host) + ":" + std::to_string(url.port);
  return true;
}

namespace {

// Waits for `events` on fd until the deadline; false on timeout or poll error.
bool WaitFd(int fd, short events, uint64_t deadline) {
  for (;;) {
    const uint64_t now = MonoNs();
    if (now >= deadline) return false;
    pollfd p{fd, events, 0};
    const int ms = static_cast<int>(std::min<uint64_t>((deadline - now) / 1000000ull + 1, 1000));
    const int rc = poll(&p, 1, ms);
    if (rc < 0 && errno == EINTR) continue;
    if (rc < 0) return false;
    if (rc > 0) return true;
  }
}

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  ~Fd() {
    if (fd_ >= 0) close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

// Decodes a chunked body in place (sizes in hex, optional extensions, trailers ignored).
bool Dechunk(std::string* body) {
  std::string out;
  size_t pos = 0;
  for (;;) {
    const size_t eol = body->find("\r\n", pos);
    if (eol == std::string::npos) return false;
    char* end = nullptr;
    const unsigned long long n = strtoull(body->c_str() + pos, &end, 16);
    if (end == body->c_str() + pos) return false;
    pos = eol + 2;
    if (n == 0) break;
    if (n > body->size() - pos || body->compare(pos + n, 2, "\r\n") != 0) return false;
    out.append(*body, pos, n);
    pos += n + 2;
  }
  body->swap(out);
  return true;
}

}  // namespace

int HttpClient::Connect(uint64_t deadline, std::string* error) {
  *error = "connect: no address";
  for (const Addr& a : addrs_) {
    if (MonoNs() >= deadline) {
      *error = "connect: timed out";
      break;
    }
    const int fd = socket(a.sa.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
      *error = std::string("socket: ") + strerror(errno);
      continue;
    }
    int soerr = 0;
    if (connect(fd, reinterpret_cast<const sockaddr*>(&a.sa), a.len) != 0) {
      soerr = errno;
      if (soerr == EINPROGRESS) {
        if (!WaitFd(fd, POLLOUT, deadline)) {
          soerr = ETIMEDOUT;
        } else {
          socklen_t len = sizeof(soerr);
          getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len);
        }
      }
    }
    if (soerr == 0) return fd;
    *error = std::string("connect: ") + strerror(soerr);
    close(fd);  // next address: still nothing was sent
  }
  return -1;
}

HttpResult HttpClient::Request(const char* method, const std::string& path, const std::string& body,
                               const char* content_type, uint64_t deadline) {
  HttpResult r;
  // ---- connect (own budget within the request's deadline): nothing sent yet ----
  Fd fd(Connect(std::min(deadline, MonoNs() + kConnectBudgetNs), &r.error));
  if (fd.get() < 0) return r;  // kNotSent
  r.error.clear();
  // ---- send ----
  std::string head =
      std::string(method) + " " + url_.prefix + path + " HTTP/1.1\r\nHost: " + host_header_ +
      "\r\nConnection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
  if (content_type) head += std::string("Content-Type: ") + content_type + "\r\n";
  head += "\r\n";
  const std::string* parts[2] = {&head, &body};
  bool any_sent = false;
  for (const std::string* part : parts) {
    size_t off = 0;
    while (off < part->size()) {
      if (MonoNs() >= deadline) {  // also when every call makes progress or gets EINTR
        r.kind = any_sent ? HttpResult::Kind::kUncertain : HttpResult::Kind::kNotSent;
        r.error = "send: timed out";
        return r;
      }
      const ssize_t w = send(fd.get(), part->data() + off, part->size() - off, MSG_NOSIGNAL);
      if (w > 0) {
        off += static_cast<size_t>(w);
        any_sent = true;
        continue;
      }
      if (w < 0 && errno == EINTR) continue;
      if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
          WaitFd(fd.get(), POLLOUT, deadline)) {
        continue;
      }
      r.kind = any_sent ? HttpResult::Kind::kUncertain : HttpResult::Kind::kNotSent;
      r.error = w < 0 && errno != EAGAIN ? std::string("send: ") + strerror(errno)
                                         : std::string("send: timed out");
      return r;
    }
  }
  r.kind = HttpResult::Kind::kUncertain;  // from here on the server may have acted on it
  // ---- receive until EOF ----
  std::string resp;
  char buf[4096];
  for (;;) {
    if (MonoNs() >= deadline) {
      r.error = "response: timed out";
      return r;
    }
    const ssize_t n = recv(fd.get(), buf, sizeof(buf), 0);
    if (n > 0) {
      resp.append(buf, static_cast<size_t>(n));
      if (resp.size() > kMaxResponseBytes + 4096) {
        r.error = "response too large";
        return r;
      }
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    if ((errno == EAGAIN || errno == EWOULDBLOCK) && WaitFd(fd.get(), POLLIN, deadline)) continue;
    r.error = errno == EAGAIN ? "response: timed out" : std::string("recv: ") + strerror(errno);
    return r;
  }
  // ---- parse: status line, headers, body ----
  const size_t eoh = resp.find("\r\n\r\n");
  if (resp.compare(0, 9, "HTTP/1.1 ") != 0 && resp.compare(0, 9, "HTTP/1.0 ") != 0) {
    r.error = "malformed response";
    return r;
  }
  if (eoh == std::string::npos || resp.size() < 12) {
    r.error = "truncated response";
    return r;
  }
  r.status = atoi(resp.c_str() + 9);
  std::string headers = resp.substr(0, eoh);
  for (auto& ch : headers) ch = static_cast<char>(tolower(static_cast<unsigned char>(ch)));
  const size_t te = headers.find("\r\ntransfer-encoding:");
  const bool chunked = te != std::string::npos;
  if (chunked && headers.find("chunked", te) == std::string::npos) {
    r.error = "unsupported transfer encoding";
    return r;
  }
  r.body = resp.substr(eoh + 4);
  if (chunked && !Dechunk(&r.body)) {
    r.error = "malformed chunked response";
    return r;
  }
  const size_t cl = headers.find("\r\ncontent-length:");
  if (!chunked && cl != std::string::npos) {
    const size_t want = strtoull(headers.c_str() + cl + 17, nullptr, 10);
    if (r.body.size() != want) {
      r.error = "response body length mismatch";
      return r;
    }
  }
  if (r.body.size() > kMaxResponseBytes) {
    r.error = "response too large";
    return r;
  }
  r.kind = HttpResult::Kind::kOk;
  return r;
}

}  // namespace netmon
