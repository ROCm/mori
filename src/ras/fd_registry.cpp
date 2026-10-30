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
#include "src/ras/fd_registry.hpp"

#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <mutex>

namespace mori {
namespace ras {
namespace fdreg {
namespace {

// The link budget, the listeners, the eventfd, the epoll fd, and pending handshakes all fit.
constexpr size_t kMaxFds = 4096;

// A plain pthread mutex and array: the child handler below may touch only these.
pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
int g_fds[kMaxFds];
size_t g_count = 0;
std::once_flag g_installed;

void RemoveLocked(int fd) {
  for (size_t i = 0; i < g_count; ++i) {
    if (g_fds[i] == fd) {
      g_fds[i] = g_fds[--g_count];
      return;
    }
  }
}

template <typename OpenFn>
int OpenRegistered(OpenFn open) {
  InstallForkHandlers();
  pthread_mutex_lock(&g_mu);
  int fd = open();
  if (fd >= 0) {
    if (g_count < kMaxFds) {
      g_fds[g_count++] = fd;
    } else {
      close(fd);
      fd = -1;
      errno = EMFILE;
    }
  }
  int saved = errno;
  pthread_mutex_unlock(&g_mu);
  errno = saved;
  return fd;
}

void Prepare() { pthread_mutex_lock(&g_mu); }
void Parent() { pthread_mutex_unlock(&g_mu); }
// Single-threaded and async-signal-safe: close() only, no allocation, no logging.
void Child() {
  for (size_t i = 0; i < g_count; ++i) close(g_fds[i]);
  g_count = 0;
  pthread_mutex_unlock(&g_mu);
}

}  // namespace

int Socket(int domain, int type, int protocol) {
  return OpenRegistered([&] { return socket(domain, type | SOCK_CLOEXEC, protocol); });
}

int Accept(int listenFd, int flags) {
  return OpenRegistered([&] { return accept4(listenFd, nullptr, nullptr, flags | SOCK_CLOEXEC); });
}

int EpollCreate() {
  return OpenRegistered([] { return epoll_create1(EPOLL_CLOEXEC); });
}

int EventFd(unsigned int initval, int flags) {
  return OpenRegistered([&] { return eventfd(initval, flags | EFD_CLOEXEC); });
}

void Close(int fd) {
  if (fd < 0) return;
  pthread_mutex_lock(&g_mu);
  RemoveLocked(fd);
  close(fd);
  pthread_mutex_unlock(&g_mu);
}

size_t Count() {
  pthread_mutex_lock(&g_mu);
  size_t n = g_count;
  pthread_mutex_unlock(&g_mu);
  return n;
}

std::vector<int> Snapshot() {
  pthread_mutex_lock(&g_mu);
  std::vector<int> fds(g_fds, g_fds + g_count);
  pthread_mutex_unlock(&g_mu);
  return fds;
}

void InstallForkHandlers() {
  std::call_once(g_installed, [] { pthread_atfork(Prepare, Parent, Child); });
}

}  // namespace fdreg
}  // namespace ras
}  // namespace mori
