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

// Every fd RAS owns, so that a forked child can close them all before anything else runs. A
// child that kept them would hold a killed parent's listener and links open: the parent's
// peers would see neither the link close nor their reconnects refused.
//
// Opens and closes happen under one lock, which fork's prepare handler also takes. A fork
// therefore can't land between an open and its registration, which would leave an unregistered
// RAS fd in the child, or between a close and its unregistration. A close clears the entry
// before closing the fd, so the fd number is never registered while something else could
// reuse it, and the child never closes an fd that isn't RAS's.

#include <cstddef>
#include <vector>

namespace mori {
namespace ras {
namespace fdreg {

// The usual calls, opened close-on-exec and registered. -1 with errno on failure, including
// EMFILE when the registry is full.
int Socket(int domain, int type, int protocol);
int Accept(int listenFd, int flags);
int EpollCreate();
int EventFd(unsigned int initval, int flags);

// Unregisters, then closes. Ignores -1.
void Close(int fd);

size_t Count();
std::vector<int> Snapshot();

// Installs the pthread_atfork handlers once. Every open above calls it too. Handlers that
// runtime code registers afterwards run after this child handler, with the fds already closed.
void InstallForkHandlers();

}  // namespace fdreg
}  // namespace ras
}  // namespace mori
