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

// A tiny fake VictoriaMetrics for the push tests: serves /flags and /api/v1/series and
// records /api/v1/import bodies; the import behaviour is configurable.

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nt {

class FakeVm {
 public:
  enum class Import { kOk, kFail500, kBlackHole, kCloseNoReply };
  std::atomic<int> import_mode{static_cast<int>(Import::kOk)};
  std::atomic<int> fail_first{0};  // first N imports answer 500 (then import_mode)
  std::atomic<bool> series_conflict{false};
  std::string flags = "-retentionPeriod=\"100y\"\n-search.latencyOffset=\"0s\"\n";
  std::atomic<int> raw_mode{0};  // 1: answer with a chunked body, 2: oversized body

  FakeVm() {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
    port_ = ntohs(a.sin_port);
    listen(fd_, 64);
    std::thread([this] { Loop(); }).detach();  // lives until the process ends
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
  std::vector<std::string> bodies() {
    std::lock_guard<std::mutex> lk(mu_);
    return bodies_;
  }
  int imports() {
    std::lock_guard<std::mutex> lk(mu_);
    return imports_;
  }

 private:
  void Loop() {
    for (;;) {
      int c = accept(fd_, nullptr, nullptr);
      if (c < 0) continue;
      std::thread([this, c] { Serve(c); }).detach();
    }
  }
  static void Reply(int c, int status, const std::string& body) {
    std::string r = "HTTP/1.1 " + std::to_string(status) +
                    " X\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nConnection: close\r\n\r\n" + body;
    send(c, r.data(), r.size(), MSG_NOSIGNAL);
  }
  void Serve(int c) {
    std::string req;
    char buf[65536];
    size_t need = std::string::npos;
    for (;;) {
      ssize_t n = recv(c, buf, sizeof(buf), 0);
      if (n <= 0) break;
      req.append(buf, static_cast<size_t>(n));
      size_t eoh = req.find("\r\n\r\n");
      if (eoh != std::string::npos && need == std::string::npos) {
        size_t cl = req.find("Content-Length: ");
        need =
            eoh + 4 + (cl != std::string::npos ? strtoull(req.c_str() + cl + 16, nullptr, 10) : 0);
      }
      if (need != std::string::npos && req.size() >= need) break;
    }
    const size_t eoh = req.find("\r\n\r\n");
    const std::string body = eoh == std::string::npos ? "" : req.substr(eoh + 4);
    if (req.compare(0, 11, "GET /flags ") == 0) {
      if (raw_mode == 1) {
        std::string r = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" +
                        Hex(flags.size()) + "\r\n" + flags + "\r\n0\r\n\r\n";
        send(c, r.data(), r.size(), MSG_NOSIGNAL);
      } else if (raw_mode == 2) {
        Reply(c, 200, std::string(200000, 'x'));
      } else {
        Reply(c, 200, flags);
      }
    } else if (req.compare(0, 19, "GET /api/v1/series?") == 0) {
      Reply(c, 200,
            series_conflict ? "{\"status\":\"success\",\"data\":[{\"host\":\"h\"}]}"
                            : "{\"status\":\"success\",\"data\":[]}");
    } else if (req.compare(0, 20, "POST /api/v1/import ") == 0) {
      int k;
      {
        std::lock_guard<std::mutex> lk(mu_);
        k = imports_++;
      }
      if (k < fail_first) {
        Reply(c, 500, "fail");
      } else {
        switch (static_cast<Import>(import_mode.load())) {
          case Import::kOk: {
            std::lock_guard<std::mutex> lk(mu_);
            bodies_.push_back(body);
          }
            Reply(c, 204, "");
            break;
          case Import::kFail500:
            Reply(c, 500, "fail");
            break;
          case Import::kBlackHole:
            for (;;) sleep(1000);  // never answers (holds the connection)
          case Import::kCloseNoReply:
            break;
        }
      }
    } else {
      Reply(c, 404, "");
    }
    close(c);
  }
  static std::string Hex(size_t n) {
    char b[32];
    snprintf(b, sizeof(b), "%zx", n);
    return b;
  }
  int fd_ = -1;
  int port_ = 0;
  std::mutex mu_;
  std::vector<std::string> bodies_;
  int imports_ = 0;
};

}  // namespace nt
