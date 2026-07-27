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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/standalone/ipc.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

namespace mori::umbp::standalone {
namespace {

bool StartsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

void SetError(std::string* error, const std::string& message) {
  if (error) *error = message;
}

std::string ErrnoMessage(const std::string& op) { return op + ": " + std::strerror(errno); }

bool FillSockaddr(const std::string& path, sockaddr_un* addr, socklen_t* addr_len,
                  std::string* error) {
  if (!addr || !addr_len) return false;
  if (path.empty()) {
    SetError(error, "empty UDS path");
    return false;
  }
  if (path.size() >= sizeof(addr->sun_path)) {
    SetError(error, "UDS path too long: " + path);
    return false;
  }
  std::memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;
  std::strncpy(addr->sun_path, path.c_str(), sizeof(addr->sun_path) - 1);
  *addr_len = static_cast<socklen_t>(sizeof(sa_family_t) + path.size() + 1);
  return true;
}

bool SendAll(int fd, const void* data, size_t len, std::string* error) {
  const char* pos = static_cast<const char*>(data);
  size_t left = len;
  while (left > 0) {
    ssize_t n = send(fd, pos, left, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      SetError(error, ErrnoMessage("send"));
      return false;
    }
    pos += n;
    left -= static_cast<size_t>(n);
  }
  return true;
}

bool RecvAll(int fd, void* data, size_t len, std::string* error) {
  char* pos = static_cast<char*>(data);
  size_t left = len;
  while (left > 0) {
    ssize_t n = recv(fd, pos, left, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      SetError(error, ErrnoMessage("recv"));
      return false;
    }
    pos += n;
    left -= static_cast<size_t>(n);
  }
  return true;
}

bool SetSocketTimeouts(int fd, int timeout_ms, std::string* error) {
  if (timeout_ms <= 0) return true;
  timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
    SetError(error, ErrnoMessage("setsockopt(SO_SNDTIMEO)"));
    return false;
  }
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
    SetError(error, ErrnoMessage("setsockopt(SO_RCVTIMEO)"));
    return false;
  }
  return true;
}

}  // namespace

std::string UnixPathFromGrpcAddress(const std::string& address) {
  if (StartsWith(address, "unix://")) return address.substr(7);
  if (StartsWith(address, "unix:")) return address.substr(5);
  return address;
}

std::string DeriveFdSocketPath(const std::string& grpc_address) {
  std::string path = UnixPathFromGrpcAddress(grpc_address);
  const std::string suffix = ".grpc.sock";
  if (path.size() >= suffix.size() &&
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    path.replace(path.size() - suffix.size(), suffix.size(), ".fd.sock");
  } else {
    path += ".fd.sock";
  }
  return path;
}

std::string DefaultStandaloneAddress() {
  const char* node = std::getenv("UMBP_NODE_ID");
  std::string node_id = (node && node[0] != '\0') ? node : "node0";
  return "unix:///run/umbp/standalone/" + node_id + ".grpc.sock";
}

bool EnsureParentDirectory(const std::string& path, std::string* error) {
  size_t slash = path.rfind('/');
  if (slash == std::string::npos || slash == 0) return true;

  std::string parent = path.substr(0, slash);
  size_t pos = 1;
  while (pos <= parent.size()) {
    size_t next = parent.find('/', pos);
    std::string part = parent.substr(0, next == std::string::npos ? parent.size() : next);
    if (!part.empty() && mkdir(part.c_str(), 0700) != 0 && errno != EEXIST) {
      SetError(error, ErrnoMessage("mkdir(" + part + ")"));
      return false;
    }
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  return true;
}

int SendFdRegistration(const std::string& socket_path, int fd, const std::string& client_id,
                       uintptr_t worker_base, size_t size, int timeout_ms, std::string* error) {
  if (client_id.empty() || client_id.size() >= kClientIdBytes) {
    SetError(error, "client_id is empty or too long");
    return -1;
  }

  int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0) {
    SetError(error, ErrnoMessage("socket"));
    return -1;
  }
  if (!SetSocketTimeouts(sock, timeout_ms, error)) {
    close(sock);
    return -1;
  }

  sockaddr_un addr;
  socklen_t addr_len = 0;
  if (!FillSockaddr(socket_path, &addr, &addr_len, error)) {
    close(sock);
    return -1;
  }
  if (connect(sock, reinterpret_cast<sockaddr*>(&addr), addr_len) != 0) {
    SetError(error, ErrnoMessage("connect(" + socket_path + ")"));
    close(sock);
    return -1;
  }

  FdRegistrationMessage msg;
  std::strncpy(msg.client_id, client_id.c_str(), sizeof(msg.client_id) - 1);
  msg.worker_base = static_cast<uint64_t>(worker_base);
  msg.size = static_cast<uint64_t>(size);

  msghdr hdr;
  std::memset(&hdr, 0, sizeof(hdr));
  iovec iov;
  iov.iov_base = &msg;
  iov.iov_len = sizeof(msg);
  hdr.msg_iov = &iov;
  hdr.msg_iovlen = 1;

  char control[CMSG_SPACE(sizeof(int))];
  std::memset(control, 0, sizeof(control));
  hdr.msg_control = control;
  hdr.msg_controllen = sizeof(control);
  cmsghdr* cmsg = CMSG_FIRSTHDR(&hdr);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

  while (true) {
    ssize_t sent = sendmsg(sock, &hdr, 0);
    if (sent < 0 && errno == EINTR) continue;
    if (sent != static_cast<ssize_t>(sizeof(msg))) {
      SetError(error, sent < 0 ? ErrnoMessage("sendmsg") : "short sendmsg");
      close(sock);
      return -1;
    }
    break;
  }

  int32_t status = -1;
  if (!RecvStatus(sock, &status, error)) {
    close(sock);
    return -1;
  }
  close(sock);
  return status;
}

int RecvFdRegistration(int socket_fd, FdRegistrationMessage* message, std::string* error) {
  if (!message) return -1;
  msghdr hdr;
  std::memset(&hdr, 0, sizeof(hdr));
  iovec iov;
  iov.iov_base = message;
  iov.iov_len = sizeof(*message);
  hdr.msg_iov = &iov;
  hdr.msg_iovlen = 1;

  char control[CMSG_SPACE(sizeof(int))];
  std::memset(control, 0, sizeof(control));
  hdr.msg_control = control;
  hdr.msg_controllen = sizeof(control);

  ssize_t received = 0;
  while (true) {
    received = recvmsg(socket_fd, &hdr, 0);
    if (received < 0 && errno == EINTR) continue;
    break;
  }
  auto close_delivered_fd = [&]() {
    cmsghdr* delivered = CMSG_FIRSTHDR(&hdr);
    if (delivered && delivered->cmsg_level == SOL_SOCKET && delivered->cmsg_type == SCM_RIGHTS) {
      int delivered_fd = -1;
      std::memcpy(&delivered_fd, CMSG_DATA(delivered), sizeof(int));
      if (delivered_fd >= 0) close(delivered_fd);
    }
  };
  if (received != static_cast<ssize_t>(sizeof(*message)) ||
      (hdr.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
    close_delivered_fd();
    SetError(error, received < 0 ? ErrnoMessage("recvmsg") : "malformed fd registration message");
    return -1;
  }

  cmsghdr* cmsg = CMSG_FIRSTHDR(&hdr);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
    close_delivered_fd();
    SetError(error, "fd registration message missing SCM_RIGHTS fd");
    return -1;
  }
  int received_fd = -1;
  std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
  if (message->magic != kFdRegistrationMagic || message->version != kFdRegistrationVersion ||
      message->client_id[kClientIdBytes - 1] != '\0' || message->size == 0) {
    close(received_fd);
    SetError(error, "invalid fd registration payload");
    return -1;
  }
  return received_fd;
}

bool SendStatus(int socket_fd, int32_t status, std::string* error) {
  return SendAll(socket_fd, &status, sizeof(status), error);
}

bool RecvStatus(int socket_fd, int32_t* status, std::string* error) {
  if (!status) return false;
  return RecvAll(socket_fd, status, sizeof(*status), error);
}

}  // namespace mori::umbp::standalone
