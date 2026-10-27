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
#include <stdexcept>

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

PrefetchStatus Finished(PrefetchQueue& queue, const std::string& id) {
  for (int i = 0; i < 2000; ++i) {
    auto status = queue.Status(id);
    if (PrefetchTerminal(status.state)) return status;
    std::this_thread::sleep_for(1ms);
  }
  ADD_FAILURE() << "prefetch did not finish";
  return {};
}

TEST(PrefetchQueue, AsyncResultsIdempotenceAndInputValidation) {
  Gate io;
  PrefetchQueue queue([&](const auto& keys, auto, auto) {
    io.Block();
    std::vector<bool> ok;
    for (const auto& key : keys) ok.push_back(key != "missing");
    return ok;
  });
  EXPECT_FALSE(queue.Submit({}).accepted);
  EXPECT_FALSE(queue.Submit({""}).accepted);
  EXPECT_FALSE(queue.Submit({"a"}, {1ms, 6min}).accepted);
  ASSERT_TRUE(queue.Submit({"a", "missing", "a"}, {}, "id").accepted);
  ASSERT_TRUE(io.Wait());
  EXPECT_EQ(queue.Status("id").state, PrefetchState::kRunning);
  EXPECT_TRUE(queue.Submit({"a", "missing", "a"}, {}, "id").accepted);
  EXPECT_FALSE(queue.Submit({"different"}, {}, "id").accepted);
  EXPECT_TRUE(queue.WaitForKeys({"unrelated"}, 0ms));
  EXPECT_FALSE(queue.WaitForKeys({"a"}, 1ms));
  io.Release();
  auto status = Finished(queue, "id");
  EXPECT_EQ(status.state, PrefetchState::kPartial);
  EXPECT_EQ(status.ok, (std::vector<bool>{true, false, true}));
  EXPECT_EQ(status.completed_keys, 2u);
  EXPECT_TRUE(queue.WaitForKeys({"a"}, 0ms));
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
  EXPECT_TRUE(queue.WaitForKeys({"expired"}, 2s));
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(Finished(queue, "active").state, PrefetchState::kReady);
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
  EXPECT_EQ(Finished(queue, "new").state, PrefetchState::kReady);
  queue.Stop();
  EXPECT_FALSE(queue.Submit({"stopped"}).accepted);
}

TEST(PrefetchQueue, ExceptionsReleaseWaitersAndCompletedHintsAreDiscarded) {
  PrefetchQueue queue([](const auto& keys, auto, auto) {
    if (keys[0] == "throw") throw std::runtime_error("I/O failed");
    return std::vector<bool>(keys.size(), true);
  });
  ASSERT_TRUE(queue.Submit({"throw"}, {}, "bad").accepted);
  EXPECT_EQ(Finished(queue, "bad").state, PrefetchState::kFailed);
  EXPECT_TRUE(queue.WaitForKeys({"throw"}, 0ms));
  ASSERT_TRUE(queue.Submit({"hint"}, {}, "hint", false).accepted);
  EXPECT_TRUE(queue.WaitForKeys({"hint"}, 2s));
  EXPECT_EQ(queue.Status("hint").state, PrefetchState::kUnknown);
}

}  // namespace
}  // namespace mori::umbp
