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

// Minimal HTTP/1.1 client for the live push (NETMON_LIVE_PUSH_DESIGN §2): plain http only,
// one request per connection (Connection: close), non-blocking socket with an absolute
// deadline, bounded response. libc only.

#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdint>
#include <string>
#include <vector>

namespace netmon {

struct HttpUrl {
  std::string host;
  uint16_t port = 80;
  std::string prefix;  // path prefix without a trailing slash ("" or "/vm")
};

// Accepts http://HOST[:PORT][/prefix]. IPv6 literals in brackets are accepted.
bool ParseHttpUrl(const std::string& url, HttpUrl* out, std::string* error);

struct HttpResult {
  enum class Kind {
    kOk,         // a complete response was received (any status code)
    kNotSent,    // failed before any byte of the request was sent: safe to retry
    kUncertain,  // failed after sending started: the server may have acted on it
  };
  Kind kind = Kind::kNotSent;
  int status = 0;
  std::string body;
  std::string error;
};

class HttpClient {
 public:
  // Resolves the host (getaddrinfo; may block, call it on the exporter thread only).
  bool Resolve(const HttpUrl& url, std::string* error);

  // One request on a fresh connection. `path` is appended to the URL prefix. The whole
  // request (connect, send, receive) ends by `deadline_mono_ns`.
  HttpResult Request(const char* method, const std::string& path, const std::string& body,
                     const char* content_type, uint64_t deadline_mono_ns);

  static constexpr size_t kMaxResponseBytes = 64 * 1024;
  static constexpr uint64_t kConnectBudgetNs = 1000000000ull;  // all addresses together

 private:
  struct Addr {
    sockaddr_storage sa{};
    socklen_t len = 0;
  };
  // Connects to the first reachable address (nothing is sent before this returns). Returns
  // the socket or -1 with *error set.
  int Connect(uint64_t deadline, std::string* error);

  HttpUrl url_;
  std::vector<Addr> addrs_;
  std::string host_header_;
};

// Percent-encodes a query value.
std::string UrlEncode(const std::string& s);

}  // namespace netmon
