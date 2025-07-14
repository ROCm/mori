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

 public:
  TCPEndpointHandle handle;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           TCPContext                                           */
/* ---------------------------------------------------------------------------------------------- */
struct TCPContextHandle {
  std::string host{};
  uint16_t port{0};
  int listenFd{-1};
};

class TCPContext {
 public:
  TCPContext(std::string ip, uint16_t port = 0);
  ~TCPContext();

  std::string GetHost() const { return handle.host; }
  uint16_t GetPort() const { return handle.port; }
  int GetListenFd() const { return handle.listenFd; }

  void Listen();
  void Close();

  TCPEndpointHandle Connect(std::string remote, uint16_t port);
  TCPEndpointHandleVec Accept();
  void CloseEndpoint(TCPEndpointHandle);

 public:
  TCPContextHandle handle;

 private:
  std::unordered_map<int, TCPEndpointHandle> endpoints;
};

}  // namespace application
}  // namespace mori