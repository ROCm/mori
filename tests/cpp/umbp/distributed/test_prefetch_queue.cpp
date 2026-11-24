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
#include <gtest/gtest.h>

#include <atomic>
#include <future>

#include "umbp/common/prefetch_queue.h"

namespace mori::umbp {
namespace {
using namespace std::chrono_literals;

struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, released = false;
  void Block() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    EXPECT_TRUE(cv.wait_for(lock, 3s, [&] { return released; }));
  }
  bool Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, 2s, [&] { return entered; });
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
};

bool WaitTask(PrefetchQueue& queue, const std::string& id) {
  const auto deadline = PrefetchQueue::Clock::now() + 2s;
  while (!PrefetchTerminal(queue.Status(id).state) && PrefetchQueue::Clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  return PrefetchTerminal(queue.Status(id).state);
}

TEST(PrefetchQueue, ExpiredQueuedRequestDoesNotStartIo) {
  Gate io;
  std::atomic<int> calls{0};
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    ++calls;
    io.Block();
    return std::vector<bool>(keys.size(), true);
  });
  ASSERT_TRUE(queue.Submit({"active"}, {}, "active").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(queue.Submit({"expired"}, {100ms, 10ms}, "expired").accepted);
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(queue.Status("expired").state, PrefetchState::kExpired);
  io.Release();
  ASSERT_TRUE(WaitTask(queue, "active"));
  EXPECT_TRUE(queue.WaitForKeys({"expired"}, 0ms));
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(queue.Status("active").state, PrefetchState::kReady);
  EXPECT_EQ(queue.Status("expired").state, PrefetchState::kExpired);
}

TEST(PrefetchQueue, ClearDrainsRealIoAndResumesAdmission) {
  Gate io;
  std::atomic<int> calls{0};
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    ++calls;
    io.Block();
    return std::vector<bool>(keys.size(), true);
  });
  ASSERT_TRUE(queue.Submit({"active"}, {}, "active").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(queue.Submit({"queued"}, {}, "queued").accepted);
  auto clear = std::async(std::launch::async, [&] { queue.PauseAndDrain(); });
  EXPECT_EQ(clear.wait_for(20ms), std::future_status::timeout);
  EXPECT_FALSE(queue.Submit({"rejected"}).accepted);
  EXPECT_EQ(queue.Status("queued").state, PrefetchState::kFailed);
  io.Release();
  ASSERT_EQ(clear.wait_for(2s), std::future_status::ready);
  clear.get();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(queue.Status("active").state, PrefetchState::kFailed);
  queue.Resume();
  ASSERT_TRUE(queue.Submit({"new"}, {}, "new").accepted);
  ASSERT_TRUE(WaitTask(queue, "new"));
  EXPECT_EQ(queue.Status("new").state, PrefetchState::kReady);
  queue.Stop();
  EXPECT_FALSE(queue.Submit({"stopped"}).accepted);
}

TEST(PrefetchQueue, CompletedHistoryDoesNotBlockAdmission) {
  PrefetchQueue queue(
      [](const auto& keys, auto, auto) { return std::vector<bool>(keys.size(), true); });
  for (int i = 0; i < 1024; ++i) {
    const auto id = std::to_string(i);
    ASSERT_TRUE(queue.Submit({"key"}, {}, id).accepted);
    ASSERT_TRUE(WaitTask(queue, id));
    ASSERT_EQ(queue.Status(id).state, PrefetchState::kReady);
  }
  EXPECT_TRUE(queue.Submit({"key"}, {}, "1023").accepted);  // Retained IDs remain idempotent.
  ASSERT_TRUE(queue.Submit({"key"}, {}, "next").accepted);
  EXPECT_EQ(queue.Status("0").state, PrefetchState::kUnknown);
  ASSERT_TRUE(WaitTask(queue, "next"));
  EXPECT_TRUE(queue.Submit({"key"}, {}, {}, false).accepted);  // Engine hint.
  queue.PauseAndDrain();
  queue.Resume();
  EXPECT_TRUE(queue.Submit({"key"}).accepted);
}

TEST(PrefetchQueue, FullQueueKeepsUnfinishedRequests) {
  Gate io;
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    io.Block();
    return std::vector<bool>(keys.size(), true);
  });
  ASSERT_TRUE(queue.Submit({"key"}, {}, "0").accepted);
  ASSERT_TRUE(io.Wait());
  for (int i = 1; i < 1024; ++i) ASSERT_TRUE(queue.Submit({"key"}, {}, std::to_string(i)).accepted);
  EXPECT_FALSE(queue.Submit({"key"}, {}, "overflow").accepted);
  EXPECT_EQ(queue.Status("0").state, PrefetchState::kRunning);
  EXPECT_EQ(queue.Status("1023").state, PrefetchState::kQueued);
  EXPECT_TRUE(queue.Submit({"key"}, {}, "1023").accepted);
  io.Release();
  ASSERT_TRUE(WaitTask(queue, "1023"));
}

TEST(PrefetchQueue, ReadsWaitOnlyForTheCurrentlyExecutingBatch) {
  Gate first, later;
  PrefetchQueue* executor = nullptr;
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    executor->BeginKeys({keys.front()});
    first.Block();
    executor->EndKeys();
    executor->BeginKeys({keys.back()});
    later.Block();
    executor->EndKeys();
    return std::vector<bool>(keys.size(), true);
  });
  executor = &queue;
  ASSERT_TRUE(queue.Submit({"first", "later"}, {}, "task").accepted);
  ASSERT_TRUE(first.Wait());
  ASSERT_TRUE(queue.Submit({"queued"}, {}, "queued").accepted);
  EXPECT_FALSE(queue.HasRunningKeys({"later", "queued"}));
  EXPECT_TRUE(queue.WaitForKeys({"later", "queued"}, 0ms));
  EXPECT_FALSE(queue.WaitForKeys({"first"}, 1ms));
  EXPECT_EQ(queue.Status("task").state, PrefetchState::kRunning);
  auto read = std::async(std::launch::async, [&] { return queue.WaitForKeys({"first"}, 2s); });
  EXPECT_EQ(read.wait_for(20ms), std::future_status::timeout);
  first.Release();
  ASSERT_TRUE(later.Wait());
  ASSERT_EQ(read.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(read.get());
  EXPECT_TRUE(queue.WaitForKeys({"first"}, 0ms));
  EXPECT_EQ(queue.Status("task").state, PrefetchState::kRunning);
  later.Release();
  ASSERT_TRUE(WaitTask(queue, "queued"));
}

TEST(PrefetchQueue, AReadDoesNotJoinALaterExecutionOfTheSameKey) {
  Gate first, later;
  std::atomic<int> calls{0};
  PrefetchQueue* executor = nullptr;
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    executor->BeginKeys(keys);
    (++calls == 1 ? first : later).Block();
    executor->EndKeys();
    return std::vector<bool>(keys.size(), true);
  });
  executor = &queue;
  ASSERT_TRUE(queue.Submit({"key"}, {}, "first").accepted);
  ASSERT_TRUE(first.Wait());
  ASSERT_TRUE(queue.Submit({"key"}, {}, "later").accepted);
  auto read = std::async(std::launch::async, [&] { return queue.WaitForKeys({"key"}, 2s); });
  EXPECT_EQ(read.wait_for(20ms), std::future_status::timeout);
  first.Release();
  ASSERT_TRUE(later.Wait());
  ASSERT_EQ(read.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(read.get());
  EXPECT_TRUE(queue.HasRunningKeys({"key"}));
  EXPECT_FALSE(queue.WaitForKeys({"key"}, 1ms));
  later.Release();
  ASSERT_TRUE(WaitTask(queue, "later"));
}

TEST(PrefetchQueue, LegacyHintsFilterSplitAndDeduplicatePendingKeys) {
  Gate io;
  std::atomic<int> calls{0};
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    ++calls;
    EXPECT_LE(keys.size(), 4096u);
    io.Block();
    return std::vector<bool>(keys.size(), true);
  });
  std::vector<std::string> keys{"", std::string(4097, 'x')};
  for (size_t i = 0; i < 4100; ++i) keys.push_back(std::to_string(i));
  keys.push_back("0");
  EXPECT_FALSE(queue.Submit(keys).accepted);
  EXPECT_FALSE(queue.Submit({"valid", ""}).accepted);
  EXPECT_EQ(queue.SubmitHint(keys), 4100u);
  ASSERT_TRUE(io.Wait());
  EXPECT_EQ(queue.SubmitHint(keys), 0u);
  EXPECT_EQ(queue.SubmitHint({"new", "new", "0", ""}), 1u);
  ASSERT_TRUE(queue.Submit({"barrier"}, {}, "barrier").accepted);
  io.Release();
  ASSERT_TRUE(WaitTask(queue, "barrier"));
  EXPECT_EQ(calls, 4);
  queue.Stop();
  EXPECT_EQ(queue.SubmitHint({"stopped"}), 0u);
}

}  // namespace
}  // namespace mori::umbp
