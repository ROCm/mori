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
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

#include "mori/ras/ras.hpp"
#include "src/ras/fd_registry.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::ras;
using ras_test::RunInChild;

namespace {

StartParams Params() { return StartParams{"node-f", "host-f", "127.0.0.1", getpid()}; }

bool IsOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }

bool ReadFull(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  while (n > 0) {
    ssize_t r = read(fd, p, n);
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

}  // namespace

TEST(RasFork, AChildStartsAFreshRuntime) {
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              RasRuntime* parent = RasRuntime::EnsureStarted(Params());
              CHILD_CHECK(parent != nullptr);
              RasEndpoint before = parent->Endpoint();
              std::vector<int> rasFds = fdreg::Snapshot();
              CHILD_CHECK(rasFds.size() >= 3);  // listener, epoll, eventfd

              int rc = RunInChild([&] {
                // Before anything else: no runtime, and none of the parent's RAS fds.
                CHILD_CHECK(RasRuntime::Get() == nullptr);
                CHILD_CHECK(RasRuntime::Status().state == RunState::kNotStarted);
                CHILD_CHECK(fdreg::Count() == 0);
                for (int fd : rasFds) CHILD_CHECK(!IsOpen(fd));

                RasRuntime* child = RasRuntime::EnsureStarted(
                    StartParams{"node-f", "host-f", "127.0.0.1", static_cast<int32_t>(getpid())});
                CHILD_CHECK(child != nullptr);
                RasEndpoint after = child->Endpoint();
                CHILD_CHECK(after.Valid());
                CHILD_CHECK(after.uuid != before.uuid);
                CHILD_CHECK(after.token != before.token);
                CHILD_CHECK(after.port != before.port);
                CHILD_CHECK(ras_test::CanConnect("127.0.0.1", after.port));
                return 0;
              });
              CHILD_CHECK(rc == 0);

              // The parent's runtime is untouched by the fork.
              CHILD_CHECK(RasRuntime::Get() == parent);
              CHILD_CHECK(parent->Endpoint().uuid == before.uuid);
              CHILD_CHECK(ras_test::CanConnect("127.0.0.1", before.port));
              CHILD_CHECK(fdreg::Count() == rasFds.size());
              return 0;
            }),
            0);
}

TEST(RasFork, AKilledParentsListenerRefusesConnectionsWhileItsChildLives) {
  // The forked grandchild is reparented here when its parent dies, so it can be reaped.
  ASSERT_EQ(prctl(PR_SET_CHILD_SUBREAPER, 1), 0);
  int pipeFds[2];
  ASSERT_EQ(pipe(pipeFds), 0);
  pid_t parent = fork();
  if (parent == 0) {
    setpgid(0, 0);
    close(pipeFds[0]);
    setenv("MORI_RAS_ENABLE", "1", 1);
    RasRuntime* r = RasRuntime::EnsureStarted(Params());
    uint16_t port = r != nullptr ? r->Endpoint().port : 0;
    pid_t child = fork();
    if (child == 0) {
      // Outlives its parent, which is SIGKILLed without any chance to close its sockets.
      pause();
      _exit(0);
    }
    int32_t msg[2] = {port, child};
    if (write(pipeFds[1], msg, sizeof(msg)) != sizeof(msg)) _exit(1);
    pause();
    _exit(0);
  }
  setpgid(parent, parent);
  // However the test ends, nothing it forked outlives it.
  struct Reaper {
    pid_t group;
    ~Reaper() {
      kill(-group, SIGKILL);
      while (waitpid(-group, nullptr, 0) > 0 || errno == EINTR) {
      }
      prctl(PR_SET_CHILD_SUBREAPER, 0);
    }
  } reaper{parent};

  close(pipeFds[1]);
  int32_t msg[2] = {0, 0};
  ASSERT_TRUE(ReadFull(pipeFds[0], msg, sizeof(msg)));
  close(pipeFds[0]);
  uint16_t port = static_cast<uint16_t>(msg[0]);
  pid_t child = msg[1];
  ASSERT_NE(port, 0);
  ASSERT_TRUE(ras_test::CanConnect("127.0.0.1", port));

  kill(parent, SIGKILL);
  waitpid(parent, nullptr, 0);
  int err = 0;
  EXPECT_LT(ras_test::ConnectTcp("127.0.0.1", port, &err), 0);
  EXPECT_EQ(err, ECONNREFUSED) << "the child kept the killed parent's RAS listener";
  EXPECT_EQ(kill(child, 0), 0) << "the child should still be alive";
}

TEST(RasFork, AForkNeverClosesAnFdThatIsNotRas) {
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) != nullptr);
              // Churn RAS fds while the main thread forks, so forks land between RAS opens
              // and closes and the fd numbers RAS frees get reused by the pipes below.
              std::atomic<bool> stop{false};
              std::thread churn([&] {
                while (!stop.load()) {
                  int fd = fdreg::Socket(AF_INET, SOCK_STREAM, 0);
                  if (fd >= 0) fdreg::Close(fd);
                }
              });
              int failures = 0;
              for (int i = 0; i < 200; ++i) {
                int p[2];
                CHILD_CHECK(pipe(p) == 0);
                int rc = RunInChild([&] {
                  CHILD_CHECK(IsOpen(p[0]) && IsOpen(p[1]));
                  CHILD_CHECK(fdreg::Count() == 0);
                  return 0;
                });
                if (rc != 0) ++failures;
                close(p[0]);
                close(p[1]);
              }
              stop.store(true);
              churn.join();
              CHILD_CHECK(failures == 0);
              return 0;
            }),
            0);
}
