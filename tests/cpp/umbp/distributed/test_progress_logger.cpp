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

// ScopedProgressLog is what makes a multi-minute blocking call distinguishable
// from a hang, and both of its users configure it for minutes -- far too coarse
// for the call sites themselves to be testable. These drive it at millisecond
// intervals instead, which is the only way the timing behaviour gets covered at
// all.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "umbp/common/progress_logger.h"

namespace mori::umbp {
namespace {

using namespace std::chrono_literals;

TEST(ScopedProgressLogTest, ReportsWhileTheAwaitedCallIsStillRunning) {
  std::atomic<int> reports{0};
  {
    ScopedProgressLog log(
        10ms, [&](std::chrono::milliseconds) { reports.fetch_add(1, std::memory_order_relaxed); });
    std::this_thread::sleep_for(150ms);
  }
  // Deliberately a floor, not a count: the scheduler decides how many ticks
  // actually land, and asserting an exact number is how a test like this turns
  // flaky on a loaded CI host.
  EXPECT_GE(reports.load(std::memory_order_relaxed), 2);
}

// The whole point is to be silent for a call that returns promptly -- otherwise
// it would be noise on every fast path that happens to share the helper.
TEST(ScopedProgressLogTest, StaysSilentWhenTheCallFinishesFirst) {
  std::atomic<int> reports{0};
  {
    ScopedProgressLog log(
        10s, [&](std::chrono::milliseconds) { reports.fetch_add(1, std::memory_order_relaxed); });
  }
  EXPECT_EQ(reports.load(std::memory_order_relaxed), 0);
}

// Destruction has to join the reporting thread, not race it: a callback that
// outlived the scope would touch captures that are already gone. If this is
// broken the process dies rather than failing the assertion, which is the
// loudest available signal.
TEST(ScopedProgressLogTest, DestructorJoinsTheReporterBeforeReturning) {
  for (int i = 0; i < 50; ++i) {
    std::atomic<bool> inside{false};
    std::atomic<int> reports{0};
    {
      ScopedProgressLog log(1ms, [&](std::chrono::milliseconds) {
        inside.store(true, std::memory_order_release);
        reports.fetch_add(1, std::memory_order_relaxed);
        inside.store(false, std::memory_order_release);
      });
      std::this_thread::sleep_for(5ms);
    }
    // The captures above are still alive here only because the destructor
    // joined; reading them after a detached reporter would be the bug.
    EXPECT_FALSE(inside.load(std::memory_order_acquire));
  }
}

// Both call sites report the elapsed time in their final message too, so it has
// to keep running while the reporter is idle.
TEST(ScopedProgressLogTest, ElapsedAdvancesIndependentlyOfReporting) {
  ScopedProgressLog log(10s, [](std::chrono::milliseconds) {});
  const auto first = log.Elapsed();
  std::this_thread::sleep_for(30ms);
  const auto second = log.Elapsed();
  EXPECT_GE(second.count(), first.count() + 20);
}

// A disabled logger is how a call site opts out without branching around the
// object, so neither form may start a thread or ever call back.
TEST(ScopedProgressLogTest, NonPositiveIntervalOrAbsentCallbackDisablesIt) {
  std::atomic<int> reports{0};
  {
    ScopedProgressLog zero(
        0ms, [&](std::chrono::milliseconds) { reports.fetch_add(1, std::memory_order_relaxed); });
    ScopedProgressLog negative(
        -5ms, [&](std::chrono::milliseconds) { reports.fetch_add(1, std::memory_order_relaxed); });
    ScopedProgressLog no_callback(1ms, nullptr);
    std::this_thread::sleep_for(30ms);
  }
  EXPECT_EQ(reports.load(std::memory_order_relaxed), 0);
}

// The real users construct one per registration, and a node brings up many at
// once. Overlapping instances must not share anything.
TEST(ScopedProgressLogTest, ConcurrentInstancesAreIndependent) {
  constexpr int kInstances = 8;
  std::vector<std::atomic<int>> reports(kInstances);
  for (auto& count : reports) count.store(0, std::memory_order_relaxed);

  std::vector<std::thread> threads;
  threads.reserve(kInstances);
  for (int i = 0; i < kInstances; ++i) {
    threads.emplace_back([&, i]() {
      ScopedProgressLog log(5ms, [&reports, i](std::chrono::milliseconds) {
        reports[i].fetch_add(1, std::memory_order_relaxed);
      });
      std::this_thread::sleep_for(80ms);
    });
  }
  for (auto& thread : threads) thread.join();

  for (int i = 0; i < kInstances; ++i) {
    EXPECT_GE(reports[i].load(std::memory_order_relaxed), 2) << "instance " << i;
  }
}

}  // namespace
}  // namespace mori::umbp
