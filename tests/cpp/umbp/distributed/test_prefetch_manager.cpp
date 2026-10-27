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

#include "umbp/common/prefetch_manager.h"

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

auto Ready(const std::vector<std::string>& keys) {
  return std::vector<PrefetchKeyResult>(keys.size(),
                                        {PrefetchCode::kReady, PrefetchSource::kDram, 1, 64});
}
void Protect(const std::vector<std::string>&, std::vector<PrefetchKeyResult>*,
             std::chrono::milliseconds) {}

PrefetchStatus Finished(PrefetchManager& manager, const std::string& id) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  PrefetchStatus status;
  do {
    status = manager.Status(id);
    if (PrefetchTerminal(status.state)) return status;
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < deadline);
  ADD_FAILURE() << "prefetch did not finish";
  return status;
}

TEST(PrefetchManager, DuplicateInputsAndCallersShareActualCompletion) {
  Gate io;
  std::atomic<size_t> transferred{0};
  PrefetchManager manager(
      [&](const auto& keys) {
        transferred += keys.size();
        io.Block();
        return Ready(keys);
      },
      Protect, {});
  auto first = manager.Submit({"a", "a"}, {}, "first");
  ASSERT_TRUE(first.accepted);
  ASSERT_TRUE(io.Wait());
  auto second = manager.Submit({"a"}, {}, "second");
  ASSERT_TRUE(second.accepted);
  EXPECT_EQ(manager.Status("second").state, PrefetchState::kRunning);
  EXPECT_EQ(manager.Status("first").completed_keys, 0u);
  EXPECT_TRUE(manager.WaitForKeys({"unrelated"}, 0ms));
  EXPECT_FALSE(manager.WaitForKeys({"a"}, 1ms));
  io.Release();
  const auto status = Finished(manager, "first");
  EXPECT_EQ(status.state, PrefetchState::kReady);
  ASSERT_EQ(status.results.size(), 2u);
  EXPECT_EQ(status.completed_keys, 2u);
  EXPECT_EQ(Finished(manager, "second").state, PrefetchState::kReady);
  EXPECT_EQ(transferred, 1u);
  EXPECT_TRUE(manager.WaitForKeys({"a"}, 0ms));
}

TEST(PrefetchManager, SlowRequestDoesNotBlockUnrelatedBatch) {
  Gate io;
  PrefetchManager manager(
      [&](const auto& keys) {
        if (keys[0] == "slow") io.Block();
        return Ready(keys);
      },
      Protect, {});
  ASSERT_TRUE(manager.Submit({"slow"}, {}, "slow").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(manager.Submit({"fast"}, {}, "fast").accepted);
  EXPECT_EQ(Finished(manager, "fast").state, PrefetchState::kReady);
  EXPECT_EQ(manager.Status("slow").state, PrefetchState::kRunning);
  io.Release();
}

TEST(PrefetchManager, ExpiredSubscriberDoesNotCancelSharedIoOrLaterLease) {
  Gate io;
  PrefetchManager manager(
      [&](const auto& keys) {
        io.Block();
        return Ready(keys);
      },
      Protect, {});
  ASSERT_TRUE(manager.Submit({"a"}, {100ms, 20ms}, "short").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(manager.Submit({"a"}, {300ms, 2s}, "long").accepted);
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(manager.Status("short").state, PrefetchState::kExpired);
  EXPECT_TRUE(manager.ShouldRun("a"));
  io.Release();
  auto status = Finished(manager, "long");
  EXPECT_EQ(status.state, PrefetchState::kReady);
  EXPECT_GT(status.lease_remaining_ms, 200u);
  EXPECT_EQ(manager.Status("short").state, PrefetchState::kExpired);
}

TEST(PrefetchManager, QueueDeadlineCoversTimeBeforeExecution) {
  Gate io;
  std::atomic<size_t> calls{0};
  PrefetchManager::Limits limits;
  limits.workers = 1;
  PrefetchManager manager(
      [&](const auto& keys) {
        ++calls;
        io.Block();
        return Ready(keys);
      },
      Protect, limits);
  ASSERT_TRUE(manager.Submit({"slow"}, {}, "slow").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(manager.Submit({"expired"}, {100ms, 10ms}, "expired").accepted);
  std::this_thread::sleep_for(20ms);
  io.Release();
  EXPECT_EQ(Finished(manager, "expired").state, PrefetchState::kExpired);
  EXPECT_EQ(Finished(manager, "slow").state, PrefetchState::kReady);
  manager.PauseAndDrain();
  EXPECT_EQ(calls, 1u);
}

TEST(PrefetchManager, ReadyRequiresFinalResidencyAndProtection) {
  Gate protection;
  PrefetchManager manager(Ready,
                          [&](const auto&, auto* results, auto) {
                            protection.Block();
                            results->back().code = PrefetchCode::kNotResident;
                          },
                          {});
  ASSERT_TRUE(manager.Submit({"kept", "evicted"}, {}, "mixed").accepted);
  ASSERT_TRUE(protection.Wait());
  EXPECT_EQ(manager.Status("mixed").state, PrefetchState::kRunning);
  protection.Release();
  const auto status = Finished(manager, "mixed");
  EXPECT_EQ(status.state, PrefetchState::kPartial);
  EXPECT_EQ(status.completed_keys, 1u);
  EXPECT_EQ(status.results.back().code, PrefetchCode::kNotResident);
}

TEST(PrefetchManager, IdempotenceBoundsRetentionAndValidation) {
  Gate io;
  PrefetchManager::Limits limits;
  limits.tasks = 2;
  limits.inflight_keys = 1;
  limits.retention = 10ms;
  PrefetchManager manager(
      [&](const auto& keys) {
        io.Block();
        return Ready(keys);
      },
      Protect, limits);
  EXPECT_FALSE(manager.Submit({}).accepted);
  EXPECT_FALSE(manager.Submit({""}).accepted);
  EXPECT_FALSE(manager.Submit({"a"}, {1ms, 6min}).accepted);
  ASSERT_TRUE(manager.Submit({"a"}, {}, "id").accepted);
  ASSERT_TRUE(io.Wait());
  EXPECT_TRUE(manager.Submit({"a"}, {}, "id").accepted);
  EXPECT_FALSE(manager.Submit({"b"}, {}, "id").accepted);
  EXPECT_FALSE(manager.Submit({"a"}, {1ms, 1s}, "id").accepted);
  EXPECT_FALSE(manager.Submit({"b"}, {}, "new-key").accepted);
  EXPECT_TRUE(manager.Submit({"a"}, {}, "join").accepted);
  EXPECT_FALSE(manager.Submit({"a"}, {}, "full").accepted);
  io.Release();
  EXPECT_EQ(Finished(manager, "id").state, PrefetchState::kReady);
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(manager.Status("id").state, PrefetchState::kUnknown);
  EXPECT_TRUE(manager.Submit({"b"}, {}, "reuse").accepted);
}

TEST(PrefetchManager, DrainWaitsForRealIoAndCancelsQueuedWork) {
  Gate io;
  std::atomic<size_t> calls{0};
  PrefetchManager::Limits limits;
  limits.workers = 1;
  PrefetchManager manager(
      [&](const auto& keys) {
        ++calls;
        io.Block();
        return Ready(keys);
      },
      Protect, limits);
  ASSERT_TRUE(manager.Submit({"running"}, {}, "running").accepted);
  ASSERT_TRUE(io.Wait());
  ASSERT_TRUE(manager.Submit({"queued"}, {}, "queued").accepted);
  auto clear = std::async(std::launch::async, [&] { manager.PauseAndDrain(); });
  EXPECT_EQ(clear.wait_for(10ms), std::future_status::timeout);
  EXPECT_FALSE(manager.Submit({"blocked"}).accepted);
  EXPECT_EQ(manager.Status("queued").state, PrefetchState::kFailed);
  io.Release();
  EXPECT_EQ(clear.wait_for(2s), std::future_status::ready);
  clear.get();
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(manager.Status("running").state, PrefetchState::kFailed);
  manager.Resume();
  ASSERT_TRUE(manager.Submit({"new"}, {}, "new").accepted);
  EXPECT_EQ(Finished(manager, "new").state, PrefetchState::kReady);
  manager.Stop();
  EXPECT_FALSE(manager.Submit({"stopped"}).accepted);
}

TEST(PrefetchManager, ExecutionFailureNotifiesWaitersAndWorkerSurvives) {
  PrefetchManager manager(
      [](const auto& keys) {
        if (keys[0] == "throw") throw std::runtime_error("I/O failed");
        return Ready(keys);
      },
      Protect, {});
  ASSERT_TRUE(manager.Submit({"throw"}, {}, "bad").accepted);
  EXPECT_EQ(Finished(manager, "bad").state, PrefetchState::kFailed);
  EXPECT_TRUE(manager.WaitForKeys({"throw"}, 1ms));
  ASSERT_TRUE(manager.Submit({"okay"}, {}, "good").accepted);
  EXPECT_EQ(Finished(manager, "good").state, PrefetchState::kReady);
}

TEST(PrefetchManager, ProtectionFailurePreservesExistingCopyFailure) {
  PrefetchManager manager(
      [](const auto& keys) {
        auto results = Ready(keys);
        results.back().code = PrefetchCode::kMissing;
        return results;
      },
      [](const auto&, auto*, auto) { throw std::runtime_error("lease failed"); }, {});
  ASSERT_TRUE(manager.Submit({"copied", "missing"}, {}, "lease-failure").accepted);
  const auto status = Finished(manager, "lease-failure");
  ASSERT_EQ(status.results.size(), 2u);
  EXPECT_EQ(status.state, PrefetchState::kFailed);
  EXPECT_EQ(status.results[0].code, PrefetchCode::kNotResident);
  EXPECT_EQ(status.results[1].code, PrefetchCode::kMissing);
}

TEST(PrefetchManager, EventsCountAdmissionsSharedKeysAndFinalResults) {
  std::atomic<uint64_t> admissions{0}, joined{0}, ready{0}, rejected{0};
  PrefetchManager manager(Ready, Protect, {}, [&](const char* event, uint64_t count) {
    const std::string name(event);
    if (name == "accepted_request") admissions += count;
    if (name == "joined_key") joined += count;
    if (name == "ready_key") ready += count;
    if (name == "rejected_request") rejected += count;
  });
  ASSERT_TRUE(manager.Submit({"a", "a"}, {}, "metrics").accepted);
  EXPECT_FALSE(manager.Submit({}).accepted);
  EXPECT_EQ(Finished(manager, "metrics").state, PrefetchState::kReady);
  EXPECT_EQ(admissions, 1u);
  EXPECT_EQ(joined, 1u);
  EXPECT_EQ(ready, 2u);
  EXPECT_EQ(rejected, 1u);
}

TEST(PrefetchManager, HistoricalReadyDoesNotPromiseAnUnboundedLease) {
  PrefetchManager manager(Ready, Protect, {});
  ASSERT_TRUE(manager.Submit({"a"}, {15ms, 2s}, "lease").accepted);
  EXPECT_EQ(Finished(manager, "lease").state, PrefetchState::kReady);
  std::this_thread::sleep_for(25ms);
  EXPECT_EQ(manager.Status("lease").state, PrefetchState::kReady);
  EXPECT_EQ(manager.Status("lease").lease_remaining_ms, 0u);
}

}  // namespace
}  // namespace mori::umbp
