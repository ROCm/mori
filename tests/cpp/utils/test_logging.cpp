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
//
// Multi-threaded stress tests for the ModuleLogger wrapper in
// include/mori/utils/mori_log.hpp.
//
// These guard against the previously-present data race on ModuleLogger's
// internal std::unordered_maps (loggers_/envOverrides_), which under the
// multithreaded RDMA workload could hand back an empty shared_ptr that callers
// then cached forever. The functional asserts here catch null/crash
// regressions; running this binary under ThreadSanitizer is what actually
// proves the map access is synchronized (build with -fsanitize=thread; if the
// container aborts with "unexpected memory mapping", run via
// `setarch $(uname -m) -R ./test_logging`).

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "mori/utils/mori_log.hpp"

namespace {

// Spin barrier so all threads hit the racy window together.
struct StartGate {
  std::atomic<bool> go{false};
  void wait() const {
    while (!go.load(std::memory_order_acquire)) {
    }
  }
  void open() { go.store(true, std::memory_order_release); }
};

const char* kKnown[] = {mori::modules::APPLICATION, mori::modules::IO,  mori::modules::SHMEM,
                        mori::modules::CORE,        mori::modules::OPS, mori::modules::UMBP,
                        mori::modules::METRICS};

}  // namespace

// Concurrent resolution of known + many distinct unknown modules. Unknown names
// force on-demand InitModuleLocked() inserts, i.e. writes racing reads.
TEST(ModuleLoggerConcurrency, ResolveNeverReturnsNull) {
  constexpr int kThreads = 32;
  constexpr int kIters = 5000;
  StartGate gate;
  std::atomic<int> nulls{0};

  std::vector<std::thread> ts;
  ts.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      gate.wait();
      for (int i = 0; i < kIters; ++i) {
        const char* known = kKnown[(t + i) % 7];
        if (!mori::ModuleLogger::GetInstance().GetLogger(known)) nulls.fetch_add(1);
        // Distinct (but bounded) unknown module -> exercises the insert path.
        std::string unknown = "utmod_" + std::to_string((t * kIters + i) % 128);
        if (!mori::ModuleLogger::GetInstance().GetLogger(unknown)) nulls.fetch_add(1);
      }
    });
  }
  gate.open();
  for (auto& th : ts) th.join();
  EXPECT_EQ(nulls.load(), 0);
}

// Many threads resolve the SAME brand-new module at once: the create-once path
// (spdlog registry race handling) must yield one stable, non-null logger.
TEST(ModuleLoggerConcurrency, ConcurrentFirstTouchSameModule) {
  constexpr int kThreads = 64;
  StartGate gate;
  std::vector<spdlog::logger*> got(kThreads, nullptr);

  std::vector<std::thread> ts;
  ts.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      gate.wait();
      got[t] = mori::ModuleLogger::GetInstance().GetLogger("ut_first_touch").get();
    });
  }
  gate.open();
  for (auto& th : ts) th.join();

  ASSERT_NE(got[0], nullptr);
  for (int t = 1; t < kThreads; ++t) {
    EXPECT_NE(got[t], nullptr);
    EXPECT_EQ(got[t], got[0]);  // all observe the same registered instance
  }
}

// Logging while levels are reconfigured. Primarily a TSAN target; the functional
// assert is just "no crash/hang". Kept quiet at 'critical' to avoid stdout spam.
TEST(ModuleLoggerConcurrency, LogWhileReconfiguring) {
  constexpr int kLoggers = 24;
  constexpr int kIters = 5000;
  StartGate gate;

  const std::string savedGlobal = mori::GetGlobalLogLevel();
  mori::SetGlobalLogLevel("critical");

  std::vector<std::thread> ts;
  ts.reserve(kLoggers + 4);
  for (int t = 0; t < kLoggers; ++t) {
    ts.emplace_back([&, t] {
      gate.wait();
      for (int i = 0; i < kIters; ++i) {
        MORI_IO_TRACE("io {} {}", t, i);
        MORI_APP_DEBUG("app {} {}", t, i);
        MORI_CORE_INFO("core {} {}", t, i);
      }
    });
  }
  // Reconfigurer threads racing the loggers above.
  for (int r = 0; r < 4; ++r) {
    ts.emplace_back([&] {
      gate.wait();
      for (int i = 0; i < kIters; ++i) {
        mori::SetGlobalLogLevel((i & 1) ? "trace" : "critical");
        mori::SetModuleLogLevel(mori::modules::IO, (i & 2) ? "debug" : "warn");
        mori::ForceSetModuleLogLevel(mori::modules::CORE, "info");
        (void)mori::GetGlobalLogLevel();
      }
    });
  }
  gate.open();
  for (auto& th : ts) th.join();

  mori::SetGlobalLogLevel(savedGlobal);
  SUCCEED();
}
