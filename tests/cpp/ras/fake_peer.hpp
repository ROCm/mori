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

// A scripted RAS peer on raw sockets, for the link cases two real runtimes can't produce on
// demand: crossing connections, a peer that never answers, a newer protocol version, garbage.

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <vector>

#include "src/ras/protocol.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

namespace ras_test {

class FakePeer {
 public:
  FakePeer(mori::ras::RuntimeUuid uuid, std::string token) : token_(std::move(token)) {
    identity.uuid = uuid;
    identity.nodeId = "fake-node";
    identity.hostname = "fake-host";
    identity.host = "127.0.0.1";
    identity.pid = 4242;
    port_ = HoldPort("127.0.0.1", &listenFd_);
  }
  ~FakePeer() {
    for (int fd : fds_) close(fd);
    close(listenFd_);
  }

  mori::ras::RasEndpoint Endpoint() const {
    return mori::ras::RasEndpoint{"127.0.0.1", port_, token_, identity.uuid};
  }
  const std::string& Token() const { return token_; }

  // An fd for the next connection to the fake's listener, or -1 after `timeoutMs`.
  int Accept(int timeoutMs) {
    pollfd p{listenFd_, POLLIN, 0};
    if (poll(&p, 1, timeoutMs) != 1) return -1;
    return Track(accept(listenFd_, nullptr, nullptr));
  }
  int Connect(uint16_t port) { return Track(ConnectTcp("127.0.0.1", port, nullptr)); }

  // The HELLO request a correct initiator would send to `target`.
  mori::ras::proto::Hello RequestTo(const mori::ras::RasEndpoint& target) const {
    mori::ras::proto::Hello h;
    h.role = mori::ras::proto::HelloRole::kRequest;
    h.token = target.token;
    h.expectedPeer = target.uuid;
    h.identity = identity;
    return h;
  }
  mori::ras::proto::Hello Reply(mori::ras::proto::HelloStatus status) const {
    mori::ras::proto::Hello h;
    h.role = mori::ras::proto::HelloRole::kReply;
    h.status = status;
    h.identity = identity;
    return h;
  }

  bool Send(int fd, uint64_t seq, const mori::ras::proto::Message& m, uint16_t version = 1) {
    return SendRaw(fd, mori::ras::proto::EncodeFrame(
                           mori::ras::proto::LinkHeader(version, identity.uuid.lo, seq), m));
  }
  static bool SendRaw(int fd, const std::vector<uint8_t>& bytes) {
    return send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(bytes.size());
  }

  // The next frame on `fd`, decoded with the given header-version range.
  std::optional<mori::ras::proto::Frame> Recv(int fd, int timeoutMs,
                                              mori::ras::proto::VersionRange accept = {1, 1}) {
    std::vector<uint8_t> buf;
    if (!ReadExactly(fd, mori::ras::proto::kHeaderSize, timeoutMs, &buf)) return std::nullopt;
    mori::ras::proto::RasHeader h = mori::ras::proto::ReadHeaderFields(buf.data());
    if (h.length > mori::ras::proto::kMaxPayload) return std::nullopt;
    if (!ReadExactly(fd, h.length, timeoutMs, &buf)) return std::nullopt;
    mori::ras::proto::Frame f;
    if (mori::ras::proto::DecodeFrame(buf.data(), buf.size(), accept, &f) !=
        mori::ras::proto::DecodeStatus::kOk) {
      return std::nullopt;
    }
    return f;
  }

  // True if the other side closes `fd` within `timeoutMs`, discarding anything it sends first.
  static bool WaitClosed(int fd, int timeoutMs) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
      pollfd p{fd, POLLIN, 0};
      if (poll(&p, 1, 20) != 1) continue;
      char c[256];
      ssize_t n = recv(fd, c, sizeof(c), 0);
      if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR)) return true;
    }
    return false;
  }

  mori::ras::RasProcessId identity;

 private:
  int Track(int fd) {
    if (fd >= 0) fds_.push_back(fd);
    return fd;
  }
  static bool ReadExactly(int fd, size_t n, int timeoutMs, std::vector<uint8_t>* out) {
    size_t start = out->size();
    out->resize(start + n);
    size_t got = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (got < n) {
      auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                      deadline - std::chrono::steady_clock::now())
                      .count();
      pollfd p{fd, POLLIN, 0};
      if (left <= 0 || poll(&p, 1, static_cast<int>(left)) != 1) return false;
      ssize_t r = recv(fd, out->data() + start + got, n - got, 0);
      if (r <= 0) return false;
      got += static_cast<size_t>(r);
    }
    return true;
  }

  std::string token_;
  int listenFd_{-1};
  uint16_t port_{0};
  std::vector<int> fds_;
};

}  // namespace ras_test
