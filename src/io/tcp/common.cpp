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
#include "src/io/tcp/common.hpp"

namespace mori {
namespace io {

int FullSend(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::send(fd, p, remaining, 0);
    if (n == 0) {
      errno = ECONNRESET;
      return -1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;  // spin; could epoll later
      return -1;
    }
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return 0;
}

int FullWritev(int fd, struct iovec* iov, int iovcnt) {
  int idx = 0;
  while (idx < iovcnt) {
    ssize_t n = ::writev(fd, &iov[idx], iovcnt - idx);
    if (n == 0) {
      errno = ECONNRESET;
      return -1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return -1;
    }
    ssize_t consumed = n;
    while (consumed > 0 && idx < iovcnt) {
      if (consumed >= static_cast<ssize_t>(iov[idx].iov_len)) {
        consumed -= static_cast<ssize_t>(iov[idx].iov_len);
        ++idx;
      } else {
        iov[idx].iov_base = static_cast<char*>(iov[idx].iov_base) + consumed;
        iov[idx].iov_len -= consumed;
        consumed = 0;
      }
    }
  }
  return 0;
}

int FullRecv(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::recv(fd, p, remaining, 0);
    if (n == 0) {  // peer closed
      errno = ECONNRESET;
      return -1;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return -1;
    }
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return 0;
}

}  // namespace io
}  // namespace mori
