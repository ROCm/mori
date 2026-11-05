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
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <set>

#include "mori/ras/ras.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::ras;
using ras_test::RunInChild;

namespace {

StartParams Params() { return StartParams{"node-f", "host-f", "127.0.0.1", getpid()}; }

}  // namespace

TEST(RasFork, AForkedChildHasNoRuntimeAndLeavesTheParentsAlone) {
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              RasRuntime* parent = RasRuntime::EnsureStarted(Params());
              CHILD_CHECK(parent != nullptr);
              RasEndpoint before = parent->Endpoint();

              auto forked = std::chrono::steady_clock::now();
              int rc = RunInChild([] {
                CHILD_CHECK(RasRuntime::Get() == nullptr);
                CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == nullptr);
                CHILD_CHECK(RasRuntime::Status().state == RunState::kDisabled);
                // Runs the atexit handler RAS registered. It must not wait up to 2 s for the
                // parent's RAS thread, which isn't in this process.
                std::exit(0);
              });
              CHILD_CHECK(rc == 0);
              auto took = std::chrono::steady_clock::now() - forked;
              // Under the 2 s Stop() waits. TSan alone adds 1 s to the exit.
              CHILD_CHECK(took < std::chrono::milliseconds(1500));

              CHILD_CHECK(RasRuntime::Get() == parent);
              CHILD_CHECK(RasRuntime::Status().state == RunState::kRunning);
              CHILD_CHECK(parent->Endpoint().uuid == before.uuid);
              CHILD_CHECK(ras_test::CanConnect("127.0.0.1", before.port));
              return 0;
            }),
            0);
}

TEST(RasFork, EveryRasFdIsCloseOnExec) {
  EXPECT_EQ(RunInChild([] {
              std::set<int> before = ras_test::SocketLikeFds();
              setenv("MORI_RAS_ENABLE", "1", 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) != nullptr);
              int checked = 0;
              for (int fd : ras_test::SocketLikeFds()) {
                if (before.count(fd) != 0) continue;
                int flags = fcntl(fd, F_GETFD);
                if (flags < 0) continue;  // a short-lived one, closed since
                CHILD_CHECK(flags & FD_CLOEXEC);
                ++checked;
              }
              CHILD_CHECK(checked >= 3);  // the listener, the epoll fd and the eventfd
              return 0;
            }),
            0);
}
