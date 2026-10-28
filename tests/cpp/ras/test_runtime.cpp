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
#include <dirent.h>
#include <gtest/gtest.h>
#include <sched.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "mori/ras/ras.hpp"
#include "src/ras/runtime.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::ras;
using ras_test::RunInChild;
using ras_test::WaitFor;

namespace {

StartParams Params() { return StartParams{"node-t", "host-t", "127.0.0.1", getpid()}; }

std::string ReadAll(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// The TID of this process's thread named `name`, or -1.
int FindThread(const std::string& name) {
  DIR* d = opendir("/proc/self/task");
  if (d == nullptr) return -1;
  int found = -1;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string comm = ReadAll(std::string("/proc/self/task/") + e->d_name + "/comm");
    if (!comm.empty() && comm.back() == '\n') comm.pop_back();
    if (comm == name) found = std::atoi(e->d_name);
  }
  closedir(d);
  return found;
}

CoreParams LocalCore() {
  CoreParams p;
  p.config.enable = true;
  p.nodeId = "node-c";
  p.hostname = "host-c";
  p.host = "127.0.0.1";
  p.pid = getpid();
  return p;
}

}  // namespace

TEST(RasRuntime, IsOffUnlessEnabled) {
  EXPECT_EQ(RunInChild([] {
              unsetenv("MORI_RAS_ENABLE");
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == nullptr);
              CHILD_CHECK(RasRuntime::Get() == nullptr);
              CHILD_CHECK(RasRuntime::Status().state == RunState::kDisabled);
              setenv("MORI_RAS_ENABLE", "0", 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == nullptr);
              return 0;
            }),
            0);
}

TEST(RasRuntime, StartsOnceAndAdvertisesItsEndpoint) {
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              CHILD_CHECK(RasRuntime::Status().state == RunState::kNotStarted);
              RasRuntime* r = RasRuntime::EnsureStarted(Params());
              CHILD_CHECK(r != nullptr);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == r);
              CHILD_CHECK(RasRuntime::Get() == r);
              CHILD_CHECK(RasRuntime::Status().state == RunState::kRunning);
              CHILD_CHECK(RasRuntime::Status().degraded == 0);

              RasEndpoint ep = r->Endpoint();
              CHILD_CHECK(ep.Valid());
              CHILD_CHECK(ep.host == "127.0.0.1");
              CHILD_CHECK(ep.port != 0 && ep.port != 28028);
              CHILD_CHECK(ep.token.size() == 32);
              CHILD_CHECK(((ep.uuid.hi >> 12) & 0xf) == 4);
              CHILD_CHECK(ras_test::CanConnect("127.0.0.1", ep.port));

              CHILD_CHECK(WaitFor([&] { return r->Snapshot().self.uuid == ep.uuid; }, 2000));
              RuntimeSnapshot s = r->Snapshot();
              CHILD_CHECK(s.self.pid == getpid());
              CHILD_CHECK(s.self.nodeId == "node-t");
              CHILD_CHECK(s.self.hostname == "host-t");
              CHILD_CHECK(s.self.host == "127.0.0.1");
              CHILD_CHECK(s.self.startTimeTicks != 0);
              CHILD_CHECK(s.listenPort == ep.port);
              return 0;
            }),
            0);
}

TEST(RasRuntime, ABlockedPortLeavesRasOffAndDegraded) {
  EXPECT_EQ(RunInChild([] {
              int holder = -1;
              uint16_t port = ras_test::HoldPort("0.0.0.0", &holder);
              setenv("MORI_RAS_ENABLE", "1", 1);
              setenv("MORI_RAS_PORT", std::to_string(port).c_str(), 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == nullptr);
              CHILD_CHECK(RasRuntime::Get() == nullptr);
              RuntimeStatus st = RasRuntime::Status();
              CHILD_CHECK(st.state == RunState::kFailed);
              CHILD_CHECK((st.degraded & kDegradedInitFailed) != 0);
              CHILD_CHECK(st.detail.find("in use") != std::string::npos);
              // The failure sticks: a later engine doesn't retry.
              close(holder);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) == nullptr);
              return 0;
            }),
            0);
}

TEST(RasRuntime, TheRasThreadRunsAtLowPriorityWhereItIsPinned) {
  EXPECT_EQ(RunInChild([] {
              cpu_set_t allowed;
              CHILD_CHECK(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
              int cpu = 0;
              while (!CPU_ISSET(cpu, &allowed)) ++cpu;
              setenv("MORI_RAS_ENABLE", "1", 1);
              setenv("MORI_RAS_CPU_AFFINITY", std::to_string(cpu).c_str(), 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) != nullptr);
              // The thread names itself after setting its priority and affinity.
              int tid = -1;
              CHILD_CHECK(WaitFor([&] { return (tid = FindThread("mori-ras")) > 0; }, 2000));
              std::string base = "/proc/self/task/" + std::to_string(tid);
              // nice is field 19 of stat; read after the ')' that ends the thread name.
              std::string stat = ReadAll(base + "/stat");
              std::istringstream fields(stat.substr(stat.rfind(')') + 1));
              std::string f;
              int nice = 0;
              for (int i = 3; fields >> f; ++i) {
                if (i == 19) {
                  nice = std::atoi(f.c_str());
                  break;
                }
              }
              CHILD_CHECK(nice == 19);
              std::string status = ReadAll(base + "/status");
              std::string want = "Cpus_allowed_list:\t" + std::to_string(cpu) + "\n";
              CHILD_CHECK(status.find(want) != std::string::npos);
              return 0;
            }),
            0);
}

TEST(RasRuntime, LabelsAreValidatedAndJoinTheIdentity) {
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              RasRuntime* r = RasRuntime::EnsureStarted(Params());
              CHILD_CHECK(r != nullptr);
              CHILD_CHECK(r->SetLabels({{"role", "decode"}, {"instance", "2"}}));
              CHILD_CHECK(WaitFor([&] { return r->Snapshot().self.labels.size() == 2; }, 2000));
              CHILD_CHECK(r->Snapshot().self.labels.at("role") == "decode");
              CHILD_CHECK(!r->SetLabels({{"", "empty key"}}));
              CHILD_CHECK(r->Snapshot().self.labels.size() == 2);
              return 0;
            }),
            0);
}

TEST(RasRuntime, EventsReachTheRasThread) {
  RasCore core(LocalCore());
  ASSERT_TRUE(core.Start(nullptr));
  PeerAdded noRas;
  noRas.localEngine = "local";
  noRas.remoteEngine = "remote-without-ras";
  EXPECT_TRUE(core.Enqueue(std::move(noRas)));
  PeerAdded self;
  self.localEngine = "local";
  self.remoteEngine = "local-twin";
  self.endpoint = core.Endpoint();
  EXPECT_TRUE(core.Enqueue(std::move(self)));
  EXPECT_TRUE(WaitFor(
      [&] {
        RuntimeSnapshot s = core.Snapshot();
        return s.peersWithoutRas == 1 && s.sameProcessPeers == 1;
      },
      2000));
  RuntimeSnapshot s = core.Snapshot();
  EXPECT_EQ(s.events.enqueued, 2u);
  EXPECT_EQ(s.events.dropped, 0u);
  EXPECT_TRUE(s.links.empty());
}

TEST(RasRuntime, StoppingClosesTheListenerAndRefusesLaterEvents) {
  RasCore core(LocalCore());
  ASSERT_TRUE(core.Start(nullptr));
  uint16_t port = core.Endpoint().port;
  ASSERT_TRUE(ras_test::CanConnect("127.0.0.1", port));
  auto start = std::chrono::steady_clock::now();
  core.Stop(true);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
  int err = 0;
  EXPECT_LT(ras_test::ConnectTcp("127.0.0.1", port, &err), 0);
  EXPECT_EQ(err, ECONNREFUSED);
  EXPECT_FALSE(core.Enqueue(EngineRegistered{"late"}));
  core.Stop(true);  // a second stop is harmless
}

TEST(RasRuntime, ExitStopsTheRuntimeWithoutHangingOrLogging) {
  auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(RunInChild([] {
              setenv("MORI_RAS_ENABLE", "1", 1);
              setenv("MORI_RAS_LOG_LEVEL", "trace", 1);
              CHILD_CHECK(RasRuntime::EnsureStarted(Params()) != nullptr);
              std::exit(0);  // runs the atexit handler and static destructors
              return 1;
            }),
            0);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}
