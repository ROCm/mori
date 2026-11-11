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

#include <arpa/inet.h>
#include <unistd.h>

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPEndpoint                                          */
/* ---------------------------------------------------------------------------------------------- */
struct TCPEndpointHandle {
  int fd{-1};
  sockaddr_in peer{};
};

using TCPEndpointHandleVec = std::vector<TCPEndpointHandle>;

using TCPReadDeadline = std::optional<std::chrono::steady_clock::time_point>;

// One budget shared by all reads of a control message. A zero configured
// timeout deliberately returns no deadline, preserving the existing opt-out.
TCPReadDeadline MakeControlPlaneReadDeadline();

class TCPEndpoint {
 public:
  TCPEndpoint(TCPEndpointHandle handle) : handle(handle) {}
  ~TCPEndpoint() = default;

  int Send(const void* buf, size_t len);
  int Recv(void* buf, size_t len, size_t* consumed = nullptr);
  // Deadline-aware reads preserve partial byte counts and report ETIMEDOUT.
  // Passing no deadline preserves the legacy per-syscall socket timeout.
  int Recv(void* buf, size_t len, size_t* consumed, const TCPReadDeadline& deadline);

 public:
  TCPEndpointHandle handle;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPContext                                           */
/* ---------------------------------------------------------------------------------------------- */
struct TCPContextHandle {
  std::string host{};
  uint16_t port{0};

  constexpr bool operator==(const TCPContextHandle& rhs) const noexcept {
    return (host == rhs.host) && (port == rhs.port);
  }
};

class TCPContext {
 public:
  // TODO: delete copy ctor
  TCPContext(std::string ip, uint16_t port = 0);
  ~TCPContext();

  std::string GetHost() const { return handle.host; }
  uint16_t GetPort() const { return handle.port; }
  int GetListenFd() const { return listenFd; }

  void Listen();
  void Close();

  TCPEndpointHandle Connect(std::string remote, uint16_t port);
  TCPEndpointHandleVec Accept();
  void CloseEndpoint(TCPEndpointHandle);
  bool CloseEndpointNoThrow(TCPEndpointHandle) noexcept;
  bool CloseEndpointNoThrow(int fd) noexcept;

 public:
  TCPContextHandle handle;

 private:
  int listenFd{-1};
  mutable std::mutex endpointsMu;
  std::unordered_map<int, TCPEndpointHandle> endpoints;
};

}  // namespace application
}  // namespace mori
