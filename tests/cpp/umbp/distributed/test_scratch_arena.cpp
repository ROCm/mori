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

#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "umbp/distributed/scratch_arena.h"

namespace {

using mori::umbp::ScratchArena;

constexpr size_t kArena = 64 * ScratchArena::kAlignment;

bool WaitQueued(const ScratchArena& arena, size_t calls) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (arena.Queued() != calls) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

TEST(ScratchArenaTest, ALoneCallGetsWhatItAsksForUpToTheWholeArena) {
  std::vector<char> buffer(kArena + 10);
  ScratchArena arena(buffer.data(), buffer.size(), -1, 4);
  EXPECT_EQ(arena.Bytes(), kArena);
  {
    auto lease = arena.Acquire(100, 1000);
    EXPECT_EQ(lease.base(), buffer.data());
    EXPECT_EQ(lease.size(), 1024u);
  }
  EXPECT_EQ(arena.Acquire(1, 10 * kArena).size(), kArena);
}

TEST(ScratchArenaTest, LiveSlicesShareTheArena) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, /*callers=*/4);
  auto first = arena.Acquire(1, kArena / 4);  // alone: all it asks for
  auto second = arena.Acquire(1, kArena);     // one of two: half
  auto third = arena.Acquire(1, kArena);      // one of three, but a quarter is left
  EXPECT_EQ(first.size(), kArena / 4);
  EXPECT_EQ(second.size(), kArena / 2);
  EXPECT_EQ(third.size(), kArena / 4);
  EXPECT_EQ(second.base(), first.base() + first.size());
  EXPECT_EQ(third.base(), second.base() + second.size());

  second = ScratchArena::Lease{};  // the largest gap is now where it was
  auto fourth = arena.Acquire(1, kArena);
  EXPECT_EQ(fourth.base(), first.base() + first.size());
  EXPECT_EQ(fourth.size(), kArena / 3 / ScratchArena::kAlignment * ScratchArena::kAlignment);
}

TEST(ScratchArenaTest, ACallWaitsForItsFloorRatherThanASliver) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, /*callers=*/2);  // a floor of half
  auto holder = arena.Acquire(1, 3 * kArena / 4);
  EXPECT_EQ(arena.Acquire(1, kArena / 4).size(), kArena / 4);  // all it wants fits
  size_t got = 0;
  std::thread waiter([&] { got = arena.Acquire(1, kArena).size(); });
  EXPECT_TRUE(WaitQueued(arena, 1));  // a quarter is free, below its floor
  holder = ScratchArena::Lease{};
  waiter.join();
  EXPECT_EQ(got, kArena);
}

TEST(ScratchArenaTest, AUnitLargerThanAShareStillGetsAllItNeeds) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, 2);
  auto small = arena.Acquire(1, 1);
  auto large = arena.Acquire(3 * kArena / 4, kArena);
  EXPECT_EQ(small.size(), ScratchArena::kAlignment);
  EXPECT_EQ(large.size(), 3 * kArena / 4);
}

TEST(ScratchArenaTest, CallsAreAdmittedInArrivalOrder) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, 2);
  auto holder = arena.Acquire(kArena, kArena);
  std::mutex mutex;
  std::vector<int> admitted;
  const auto call = [&](int id, size_t bytes) {
    auto lease = arena.Acquire(bytes, bytes);
    std::lock_guard<std::mutex> lock(mutex);
    admitted.push_back(id);
  };
  // The small call would fit the moment anything is freed, but the whole-arena
  // call queued ahead of it goes first.
  std::thread large(call, 1, kArena);
  EXPECT_TRUE(WaitQueued(arena, 1));
  std::thread small(call, 2, ScratchArena::kAlignment);
  EXPECT_TRUE(WaitQueued(arena, 2));
  holder = ScratchArena::Lease{};
  large.join();
  small.join();
  EXPECT_EQ(admitted, (std::vector<int>{1, 2}));
}

TEST(ScratchArenaTest, AMovedLeaseIsReleasedOnce) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, 2);
  ScratchArena::Lease kept;
  {
    auto lease = arena.Acquire(1, kArena / 4);
    kept = std::move(lease);
  }
  EXPECT_EQ(kept.size(), kArena / 4);
  EXPECT_EQ(arena.Acquire(1, kArena).base(), buffer.data() + kArena / 4);
}

TEST(ScratchArenaTest, NothingForWhatTheArenaCouldNeverHold) {
  std::vector<char> buffer(kArena);
  ScratchArena arena(buffer.data(), buffer.size(), -1, 2);
  EXPECT_EQ(arena.Acquire(kArena + 1, kArena + 1).base(), nullptr);
  ScratchArena none(nullptr, kArena, -1, 2);
  EXPECT_EQ(none.Bytes(), 0u);
  EXPECT_EQ(none.Acquire(1, 1).base(), nullptr);
}

}  // namespace
