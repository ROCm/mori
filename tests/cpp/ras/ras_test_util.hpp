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

// Helpers for RAS tests that run code in forked children. The RAS runtime is a per-process
// singleton, so a test that needs a fresh one runs in a child, which starts with none.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>

// In a child body: report the failed condition and return its line as the exit status.
#define CHILD_CHECK(cond)                                                                 \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "%s:%d: CHILD_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
      return __LINE__ % 250 + 1;                                                          \
    }                                                                                     \
  } while (0)

namespace ras_test {

// Exit status of `body` run in a forked child, or 128 + signal if it died of one.
inline int RunInChild(const std::function<int()>& body) {
  std::fflush(nullptr);
  pid_t pid = fork();
  if (pid == 0) {
    int rc = body();
    std::fflush(nullptr);
    _exit(rc);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return WEXITSTATUS(status);
}

inline bool WaitFor(const std::function<bool()>& pred, int timeoutMs) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return pred();
}

// A blocking IPv4 connect. Returns the fd, or -1 with the connect errno in *err.
inline int ConnectTcp(const std::string& host, uint16_t port, int* err) {
  int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    if (err) *err = errno;
    close(fd);
    return -1;
  }
  if (err) *err = 0;
  return fd;
}

inline bool CanConnect(const std::string& host, uint16_t port) {
  int fd = ConnectTcp(host, port, nullptr);
  if (fd < 0) return false;
  close(fd);
  return true;
}

// A listening socket on `host` with an ephemeral port, standing in for another process that
// holds the port. Returns the port; *fd keeps it held.
inline uint16_t HoldPort(const std::string& host, int* fd) {
  *fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  bind(*fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  listen(*fd, 4);
  socklen_t len = sizeof(addr);
  getsockname(*fd, reinterpret_cast<sockaddr*>(&addr), &len);
  return ntohs(addr.sin_port);
}

}  // namespace ras_test
