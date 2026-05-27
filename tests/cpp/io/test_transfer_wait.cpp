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
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"

using namespace mori::io;

namespace {

using Clock = std::chrono::steady_clock;

struct TestFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Require(bool cond, const std::string& msg) {
  if (!cond) throw TestFailure(msg);
}

int64_t ElapsedMs(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

std::unique_ptr<IOEngine> MakeEngine(const std::string& key) {
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  return std::make_unique<IOEngine>(key, cfg);
}

std::thread DelayedUpdate(TransferStatus* status, int delayMs, StatusCode code,
                          std::string msg = "") {
  return std::thread([status, delayMs, code, msg = std::move(msg)] {
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    status->Update(code, msg);
  });
}

void SetInProgress(TransferStatus* status) { status->SetCode(StatusCode::IN_PROGRESS); }

template <size_t N>
std::vector<TransferStatus*> Ptrs(std::array<TransferStatus, N>* statuses) {
  std::vector<TransferStatus*> ptrs;
  ptrs.reserve(N);
  for (TransferStatus& status : *statuses) ptrs.push_back(&status);
  return ptrs;
}

void CaseWaitForFastPathSuccess() {
  TransferStatus status;
  status.SetCode(StatusCode::SUCCESS);
  auto start = Clock::now();
  StatusCode rc = status.WaitFor(-1);
  Require(rc == StatusCode::SUCCESS, "WaitFor(-1) should return preset success");
  Require(ElapsedMs(start) < 50, "WaitFor fast-path success took too long");
}

void CaseWaitForFastPathFailure() {
  TransferStatus status;
  status.SetCode(StatusCode::ERR_RDMA_OP);
  StatusCode rc = status.WaitFor(-1);
  Require(rc == StatusCode::ERR_RDMA_OP, "WaitFor(-1) should return preset failure");
}

void CaseWaitForZeroIsPollOnly() {
  TransferStatus status;
  SetInProgress(&status);
  std::atomic<bool> callbackCalled{false};
  status.SetWaitCallback([&] {
    callbackCalled.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    status.SetCode(StatusCode::SUCCESS);
  });

  auto start = Clock::now();
  StatusCode rc = status.WaitFor(0);
  int64_t elapsed = ElapsedMs(start);
  Require(rc == StatusCode::IN_PROGRESS, "WaitFor(0) should not wait for completion");
  Require(!callbackCalled.load(std::memory_order_acquire), "WaitFor(0) must not call waitCallback");
  Require(elapsed < 50, "WaitFor(0) took too long");
}

void CaseWaitForZeroPollsProgressCallback() {
  TransferStatus status;
  SetInProgress(&status);
  status.SetProgressCallback([&] { status.SetCode(StatusCode::SUCCESS); });
  StatusCode rc = status.WaitFor(0);
  Require(rc == StatusCode::SUCCESS, "WaitFor(0) should run progressCallback once");
}

void CaseWaitForBoundedIgnoresCallback() {
  TransferStatus status;
  SetInProgress(&status);
  std::atomic<bool> callbackCalled{false};
  status.SetWaitCallback([&] {
    callbackCalled.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    status.SetCode(StatusCode::SUCCESS);
  });

  auto start = Clock::now();
  StatusCode rc = status.WaitFor(30);
  int64_t elapsed = ElapsedMs(start);
  Require(rc == StatusCode::IN_PROGRESS, "bounded WaitFor should time out");
  Require(!callbackCalled.load(std::memory_order_acquire),
          "bounded WaitFor must not call waitCallback");
  Require(elapsed >= 20 && elapsed < 250, "bounded WaitFor did not respect timeout");
}

void CaseWaitForUnboundedHonoursCallback() {
  TransferStatus status;
  SetInProgress(&status);
  std::atomic<bool> callbackCalled{false};
  status.SetWaitCallback([&] {
    callbackCalled.store(true, std::memory_order_release);
    status.SetCode(StatusCode::SUCCESS);
  });

  StatusCode rc = status.WaitFor(-1);
  Require(callbackCalled.load(std::memory_order_acquire),
          "unbounded WaitFor should call waitCallback");
  Require(rc == StatusCode::SUCCESS, "unbounded WaitFor should return callback result");
}

void CaseWaitForBlockingSuccess() {
  TransferStatus status;
  SetInProgress(&status);
  std::thread updater = DelayedUpdate(&status, 10, StatusCode::SUCCESS);
  StatusCode rc = status.WaitFor(-1);
  updater.join();
  Require(rc == StatusCode::SUCCESS, "WaitFor(-1) should wake on Update(SUCCESS)");
}

void CaseWaitForTimeoutNoUpdate() {
  TransferStatus status;
  SetInProgress(&status);
  auto start = Clock::now();
  StatusCode rc = status.WaitFor(30);
  int64_t elapsed = ElapsedMs(start);
  Require(rc == StatusCode::IN_PROGRESS, "WaitFor should return IN_PROGRESS on timeout");
  Require(elapsed >= 20 && elapsed < 250, "WaitFor timeout duration out of range");
}

void CaseWaitForTimeoutThenUpdate() {
  TransferStatus status;
  SetInProgress(&status);
  Require(status.WaitFor(10) == StatusCode::IN_PROGRESS, "first bounded WaitFor should time out");
  status.Update(StatusCode::SUCCESS, "");
  Require(status.WaitFor(-1) == StatusCode::SUCCESS, "second WaitFor should observe later success");
}

void CaseWaitForMultipleWaiters() {
  TransferStatus status;
  SetInProgress(&status);
  std::array<StatusCode, 2> results{StatusCode::INIT, StatusCode::INIT};
  std::thread waiter0([&] { results[0] = status.WaitFor(-1); });
  std::thread waiter1([&] { results[1] = status.WaitFor(-1); });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  status.Update(StatusCode::SUCCESS, "");
  waiter0.join();
  waiter1.join();
  Require(results[0] == StatusCode::SUCCESS && results[1] == StatusCode::SUCCESS,
          "notify_all should wake every waiter");
}

void CaseWaitForRedundantUpdates() {
  TransferStatus status;
  SetInProgress(&status);
  std::atomic<bool> done{false};
  StatusCode result = StatusCode::INIT;
  std::thread waiter([&] {
    result = status.WaitFor(-1);
    done.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  status.Update(StatusCode::IN_PROGRESS, "msg-1");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  bool returnedAfterFirstUpdate = done.load(std::memory_order_acquire);
  status.Update(StatusCode::IN_PROGRESS, "msg-2");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  bool returnedAfterSecondUpdate = done.load(std::memory_order_acquire);
  status.Update(StatusCode::SUCCESS, "");
  waiter.join();

  Require(!returnedAfterFirstUpdate && !returnedAfterSecondUpdate,
          "redundant IN_PROGRESS updates must not complete WaitFor");
  Require(result == StatusCode::SUCCESS, "WaitFor should complete on terminal update");
}

void CaseSetCodeSuccessNotifies() {
  TransferStatus status;
  SetInProgress(&status);
  StatusCode result = StatusCode::INIT;
  std::thread waiter([&] { result = status.WaitFor(-1); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  status.SetCode(StatusCode::SUCCESS);
  waiter.join();
  Require(result == StatusCode::SUCCESS, "SetCode(SUCCESS) should wake WaitFor");
}

void CaseSetCodeInProgressDoesNotNotify() {
  TransferStatus status;
  SetInProgress(&status);
  std::atomic<bool> done{false};
  StatusCode result = StatusCode::INIT;
  std::thread waiter([&] {
    result = status.WaitFor(-1);
    done.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  status.SetCode(StatusCode::IN_PROGRESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  bool returnedAfterInProgress = done.load(std::memory_order_acquire);
  status.SetCode(StatusCode::SUCCESS);
  waiter.join();

  Require(!returnedAfterInProgress, "SetCode(IN_PROGRESS) must not wake WaitFor as terminal");
  Require(result == StatusCode::SUCCESS, "WaitFor should still complete on later success");
}

void CaseWaitAllEmpty() {
  auto engine = MakeEngine("wait_all_empty");
  std::vector<TransferStatus*> statuses;
  Require(engine->WaitAll(statuses) == StatusCode::SUCCESS, "WaitAll({}) should succeed");
}

void CaseWaitAllZeroTimeoutPollOnly() {
  auto engine = MakeEngine("wait_all_zero_poll");
  std::array<TransferStatus, 4> statuses;
  for (TransferStatus& status : statuses) SetInProgress(&status);
  auto ptrs = Ptrs(&statuses);

  auto start = Clock::now();
  StatusCode rc = engine->WaitAll(ptrs, 0);
  int64_t elapsed = ElapsedMs(start);
  Require(rc == StatusCode::IN_PROGRESS, "WaitAll(0) should report in-progress statuses");
  Require(elapsed < 50, "WaitAll(0) should not wait");
}

void CaseWaitAllZeroTimeoutFailureWins() {
  auto engine = MakeEngine("wait_all_zero_failure");
  TransferStatus success;
  TransferStatus inProgress;
  TransferStatus failed;
  success.SetCode(StatusCode::SUCCESS);
  SetInProgress(&inProgress);
  failed.SetCode(StatusCode::ERR_RDMA_OP);
  std::vector<TransferStatus*> statuses{&success, &inProgress, &failed};

  StatusCode rc = engine->WaitAll(statuses, 0);
  Require(rc == StatusCode::ERR_RDMA_OP,
          "WaitAll(0) should scan past IN_PROGRESS and return failure");
}

void CaseWaitAllAllSuccess() {
  auto engine = MakeEngine("wait_all_success");
  std::array<TransferStatus, 4> statuses;
  for (TransferStatus& status : statuses) SetInProgress(&status);
  auto ptrs = Ptrs(&statuses);

  std::vector<std::thread> updaters;
  for (size_t i = 0; i < statuses.size(); ++i) {
    updaters.push_back(
        DelayedUpdate(&statuses[i], static_cast<int>((i + 1) * 5), StatusCode::SUCCESS));
  }
  StatusCode rc = engine->WaitAll(ptrs, -1);
  for (std::thread& updater : updaters) updater.join();
  Require(rc == StatusCode::SUCCESS, "WaitAll should succeed when every status succeeds");
}

void CaseWaitAllOneFailure() {
  auto engine = MakeEngine("wait_all_failure");
  std::array<TransferStatus, 3> statuses;
  for (TransferStatus& status : statuses) SetInProgress(&status);
  auto ptrs = Ptrs(&statuses);

  std::thread success0 = DelayedUpdate(&statuses[0], 20, StatusCode::SUCCESS);
  std::thread failure = DelayedUpdate(&statuses[1], 10, StatusCode::ERR_GPU_OP, "gpu failure");
  std::thread success2 = DelayedUpdate(&statuses[2], 30, StatusCode::SUCCESS);
  StatusCode rc = engine->WaitAll(ptrs, -1);
  success0.join();
  failure.join();
  success2.join();

  Require(rc == StatusCode::ERR_GPU_OP, "WaitAll should return first observed failure");
  Require(!statuses[0].InProgress() && !statuses[1].InProgress() && !statuses[2].InProgress(),
          "unbounded WaitAll should wait for every status to become terminal");
}

void CaseWaitAllTimeout() {
  auto engine = MakeEngine("wait_all_timeout");
  TransferStatus neverDone;
  SetInProgress(&neverDone);
  std::vector<TransferStatus*> statuses{&neverDone};
  StatusCode rc = engine->WaitAll(statuses, 20);
  Require(rc == StatusCode::IN_PROGRESS, "WaitAll should return IN_PROGRESS on timeout");
}

void CaseWaitAllBudget() {
  {
    auto engine = MakeEngine("wait_all_budget_success");
    std::array<TransferStatus, 4> statuses;
    for (TransferStatus& status : statuses) SetInProgress(&status);
    auto ptrs = Ptrs(&statuses);
    std::vector<std::thread> updaters;
    for (size_t i = 0; i < statuses.size(); ++i) {
      updaters.push_back(
          DelayedUpdate(&statuses[i], static_cast<int>((i + 1) * 15), StatusCode::SUCCESS));
    }
    StatusCode rc = engine->WaitAll(ptrs, 120);
    for (std::thread& updater : updaters) updater.join();
    Require(rc == StatusCode::SUCCESS, "WaitAll should share enough budget across statuses");
  }

  {
    auto engine = MakeEngine("wait_all_budget_timeout");
    std::array<TransferStatus, 4> statuses;
    for (TransferStatus& status : statuses) SetInProgress(&status);
    auto ptrs = Ptrs(&statuses);
    std::vector<std::thread> updaters;
    for (size_t i = 0; i < statuses.size(); ++i) {
      updaters.push_back(
          DelayedUpdate(&statuses[i], static_cast<int>((i + 1) * 30), StatusCode::SUCCESS));
    }
    StatusCode rc = engine->WaitAll(ptrs, 50);
    for (std::thread& updater : updaters) updater.join();
    Require(rc == StatusCode::IN_PROGRESS,
            "WaitAll should stop when shared timeout budget is exhausted");
  }
}

void CaseWaitAllBoundedFailureBeatsTimeout() {
  auto engine = MakeEngine("wait_all_failure_beats_timeout");
  TransferStatus failed;
  TransferStatus neverDone;
  SetInProgress(&failed);
  SetInProgress(&neverDone);
  std::vector<TransferStatus*> statuses{&failed, &neverDone};

  std::thread failure = DelayedUpdate(&failed, 10, StatusCode::ERR_RDMA_OP, "rdma failure");
  StatusCode rc = engine->WaitAll(statuses, 50);
  failure.join();
  Require(rc == StatusCode::ERR_RDMA_OP, "recorded failure should beat a later timeout in WaitAll");
}

struct TestCase {
  const char* name;
  std::function<void()> run;
};

}  // namespace

int main() {
  std::vector<TestCase> cases = {
      {"WaitForFastPathSuccess", CaseWaitForFastPathSuccess},
      {"WaitForFastPathFailure", CaseWaitForFastPathFailure},
      {"WaitForZeroIsPollOnly", CaseWaitForZeroIsPollOnly},
      {"WaitForZeroPollsProgressCallback", CaseWaitForZeroPollsProgressCallback},
      {"WaitForBoundedIgnoresCallback", CaseWaitForBoundedIgnoresCallback},
      {"WaitForUnboundedHonoursCallback", CaseWaitForUnboundedHonoursCallback},
      {"WaitForBlockingSuccess", CaseWaitForBlockingSuccess},
      {"WaitForTimeoutNoUpdate", CaseWaitForTimeoutNoUpdate},
      {"WaitForTimeoutThenUpdate", CaseWaitForTimeoutThenUpdate},
      {"WaitForMultipleWaiters", CaseWaitForMultipleWaiters},
      {"WaitForRedundantUpdates", CaseWaitForRedundantUpdates},
      {"SetCodeSuccessNotifies", CaseSetCodeSuccessNotifies},
      {"SetCodeInProgressDoesNotNotify", CaseSetCodeInProgressDoesNotNotify},
      {"WaitAllEmpty", CaseWaitAllEmpty},
      {"WaitAllZeroTimeoutPollOnly", CaseWaitAllZeroTimeoutPollOnly},
      {"WaitAllZeroTimeoutFailureWins", CaseWaitAllZeroTimeoutFailureWins},
      {"WaitAllAllSuccess", CaseWaitAllAllSuccess},
      {"WaitAllOneFailure", CaseWaitAllOneFailure},
      {"WaitAllTimeout", CaseWaitAllTimeout},
      {"WaitAllBudget", CaseWaitAllBudget},
      {"WaitAllBoundedFailureBeatsTimeout", CaseWaitAllBoundedFailureBeatsTimeout},
  };

  for (const TestCase& test : cases) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception& e) {
      std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
      return 1;
    }
  }
  return 0;
}
