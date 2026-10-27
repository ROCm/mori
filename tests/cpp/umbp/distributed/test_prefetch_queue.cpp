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
  EXPECT_TRUE(queue.WaitForKeys({"new"}, 2s));
  EXPECT_EQ(queue.Status("new").state, PrefetchState::kReady);
  queue.Stop();
  EXPECT_FALSE(queue.Submit({"stopped"}).accepted);
}

}  // namespace
}  // namespace mori::umbp
