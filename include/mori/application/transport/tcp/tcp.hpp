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

#include <string>
#include <unordered_map>
#include <vector>

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPEndpoint                                          */
/* ---------------------------------------------------------------------------------------------- */
struct TCPEndpointHandle {
  int fd;
  sockaddr_in peer;
};

using TCPEndpointHandleVec = std::vector<TCPEndpointHandle>;

class TCPEndpoint {
 public:
  TCPEndpoint(TCPEndpointHandle handle) : handle(handle) {}
  ~TCPEndpoint() = default;

  int Send(const void* buf, size_t len);
  int Recv(void* buf, size_t len);
  int RecvSomeExact(void* buf, size_t len);

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
  TCPContext() = default;
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
  void CloseFd(int fd) {
    if (fd < 0) return;
    auto it = endpoints.find(fd);
    if (it != endpoints.end()) {
      CloseEndpoint(it->second);
    } else {
      // best effort close of unknown fd
      ::close(fd);
    }
  }

 public:
  TCPContextHandle handle;

 private:
  int listenFd{-1};
  std::unordered_map<int, TCPEndpointHandle> endpoints;
};

}  // namespace application
}  // namespace mori
