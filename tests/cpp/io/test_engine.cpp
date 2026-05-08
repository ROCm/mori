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
#include <arpa/inet.h>
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <limits.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/common.hpp"
#include "src/io/rdma/executor.hpp"

using namespace mori::io;

namespace mori {
namespace io {
void TestWorkerShutdownDrainsTokensAndPromisesForTest();
}  // namespace io
}  // namespace mori

namespace {

struct TestSkip : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TestFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Require(bool cond, const std::string& msg) {
  if (!cond) throw TestFailure(msg);
}

template <typename Predicate>
bool WaitUntil(Predicate pred, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::yield();
  }
  return pred();
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : key_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) {
      hadOld_ = true;
      oldValue_ = old;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnvVar() {
    if (hadOld_) {
      setenv(key_.c_str(), oldValue_.c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

 private:
  std::string key_;
  bool hadOld_{false};
  std::string oldValue_;
};

struct RegisteredGpuMem {
  IOEngine* owner{nullptr};
  MemoryDesc desc{};
  void* ptr{nullptr};

  ~RegisteredGpuMem() {
    if (owner != nullptr) owner->DeregisterMemory(desc);
    if (ptr != nullptr) HIP_RUNTIME_CHECK(hipFree(ptr));
  }
};

struct ConnectedEnginePair {
  std::unique_ptr<IOEngine> initiator;
  std::unique_ptr<IOEngine> target;

  ConnectedEnginePair(std::unique_ptr<IOEngine>&& i, std::unique_ptr<IOEngine>&& t)
      : initiator(std::move(i)), target(std::move(t)) {}
};

int GetGpuCount() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return 0;
  return count;
}

bool WaitTransferDone(TransferStatus* status, int timeoutMs, std::string* err) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!status->Init() && !status->InProgress()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (err) {
    *err = "transfer timeout, code=" + std::to_string(status->CodeUint32()) + ", msg='" +
           status->Message() + "'";
  }
  return false;
}

bool WaitInboundStatusWithTimeout(IOEngine* engine, const EngineKey& remoteKey, TransferUniqueId id,
                                  int timeoutMs, TransferStatus* out, std::string* err) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (engine->PopInboundTransferStatus(remoteKey, id, out)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (err) {
    *err = "inbound timeout for transfer_uid=" + std::to_string(id) +
           ", code=" + std::to_string(out->CodeUint32()) + ", msg='" + out->Message() + "'";
  }
  return false;
}

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    close(fd);
    return -1;
  }

  int port = ntohs(addr.sin_port);

  close(fd);
  return port;
}

ConnectedEnginePair CreateConnectedRdmaPair(const std::string& prefix, bool enableNotification) {
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = GetFreePort();
  Require(cfg.port > 0, "failed to allocate free tcp port for initiator");
  auto initiator = std::make_unique<IOEngine>(prefix + "_initiator", cfg);

  cfg.port = GetFreePort();
  Require(cfg.port > 0, "failed to allocate free tcp port for target");
  auto target = std::make_unique<IOEngine>(prefix + "_target", cfg);

  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.enableNotification = enableNotification;
  initiator->CreateBackend(BackendType::RDMA, rdmaCfg);
  target->CreateBackend(BackendType::RDMA, rdmaCfg);

  EngineDesc initiatorDesc = initiator->GetEngineDesc();
  EngineDesc targetDesc = target->GetEngineDesc();
  initiator->RegisterRemoteEngine(targetDesc);
  target->RegisterRemoteEngine(initiatorDesc);

  return ConnectedEnginePair(std::move(initiator), std::move(target));
}

RegisteredGpuMem RegisterGpuMemory(IOEngine* engine, size_t sizeBytes, int deviceId) {
  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  void* ptr = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&ptr, sizeBytes));
  HIP_RUNTIME_CHECK(hipMemset(ptr, 0, sizeBytes));

  RegisteredGpuMem m;
  m.owner = engine;
  m.ptr = ptr;
  m.desc = engine->RegisterMemory(ptr, sizeBytes, deviceId, MemoryLocationType::GPU);
  return m;
}

void CaseSubmissionLedgerBasic() {
  constexpr uint32_t kNotifPerQp = 16;
  SubmissionLedger ledger(kNotifPerQp);
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 101, 8);
  const uint64_t id = ledger.Insert(3, true, meta, 8);
  Require(id == kNotifPerQp, "first ledger record id should start at notifPerQp boundary");
  Require(ledger.MarkPosted(id), "ledger mark posted should accept tentative signaled record");
  SubmissionRecord released;
  Require(ledger.ReleaseByCqe(id, &released), "ledger release should find posted record");
  Require(released.meta != nullptr, "ledger release meta should not be null");
  Require(released.meta->id == 101, "unexpected transfer id from ledger release");
  Require(released.batchSize == 8, "unexpected batch size from ledger release");
  Require(released.postedWr == 3, "unexpected posted WR count from ledger release");

  SubmissionLedger ledger2(kNotifPerQp);
  auto meta2 = std::make_shared<CqCallbackMeta>(&status, 202, 16);
  uint64_t postedId = ledger2.Insert(4, true, meta2, 10);
  Require(postedId == kNotifPerQp, "posted record id should respect notifPerQp offset");
  Require(ledger2.MarkPosted(postedId), "posted record should be markable after tail post");
  uint64_t orphanedId = ledger2.InsertOrphaned(3, meta2, 6);
  Require(orphanedId == kNotifPerQp + 1, "orphaned record id should follow posted id");
  Require(!ledger2.MarkPosted(orphanedId), "orphaned record should not be markable as posted");
  Require(!ledger2.CancelTentative(orphanedId, nullptr),
          "orphaned record should not be cancelable as tentative");
  Require(ledger2.HasOrphaned(), "expected orphaned record in ledger");
  std::vector<SubmissionRecord> orphaned;
  ledger2.ExtractOrphanedRecords(&orphaned);
  Require(orphaned.size() == 1, "expected one orphaned record");
  Require(orphaned[0].postedWr == 3, "unexpected orphaned WR count");
  Require(orphaned[0].meta == meta2, "orphaned record should preserve callback meta");
  Require(!ledger2.HasOrphaned(), "orphaned records should be drained");
  // The Posted record should still be present and retrievable via ReleaseByCqe.
  SubmissionRecord posted;
  Require(ledger2.ReleaseByCqe(postedId, &posted), "posted record should survive recovery");
  Require(posted.meta != nullptr, "posted record meta should not be null");
  Require(posted.batchSize == 10, "posted record batch size mismatch");
  Require(posted.postedWr == 4, "posted record WR count mismatch");

  SubmissionLedger ledger3(kNotifPerQp);
  uint64_t tentativeId = ledger3.Insert(2, true, meta2, 2);
  SubmissionRecord tentative;
  Require(ledger3.CancelTentative(tentativeId, &tentative), "tentative cancel should find record");
  Require(tentative.postedWr == 2, "tentative cancel should return record contents");
  Require(!ledger3.ReleaseByCqe(tentativeId, nullptr), "canceled record should be erased");

  SubmissionLedger ledger4(kNotifPerQp);
  uint64_t postedTentativeId = ledger4.Insert(5, true, meta2, 5);
  Require(ledger4.MarkPosted(postedTentativeId), "posted tentative should transition to posted");
  Require(!ledger4.CancelTentative(postedTentativeId, nullptr),
          "posted record should not be cancelable as tentative");
  SubmissionRecord stillPosted;
  Require(ledger4.ReleaseByCqe(postedTentativeId, &stillPosted),
          "posted record should remain after failed tentative cancel");
  Require(stillPosted.postedWr == 5, "posted record contents should be preserved");
}

void CaseSqControllerReserveReleaseWait() {
  SqController sq(4, 0);
  ReserveOptions opts;
  opts.timeoutUs = 200000;
  ReserveResult result;
  Require(sq.Reserve(4, opts, &result), "initial reserve should fill SQ");
  Require(sq.Depth() == 4, "unexpected depth after initial reserve");

  std::atomic<bool> waiterDone{false};
  std::atomic<bool> waiterOk{false};
  std::promise<void> waiterStartedPromise;
  std::future<void> waiterStarted = waiterStartedPromise.get_future();
  std::thread waiter([&]() {
    waiterStartedPromise.set_value();
    ReserveResult waitResult;
    waiterOk.store(sq.Reserve(1, opts, &waitResult), std::memory_order_release);
    waiterDone.store(true, std::memory_order_release);
  });

  Require(waiterStarted.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "waiter thread should start before release");
  Require(!WaitUntil([&]() { return waiterDone.load(std::memory_order_acquire); },
                     std::chrono::milliseconds(20)),
          "waiter should block while SQ is full");
  sq.Release(4);
  waiter.join();
  Require(waiterOk.load(std::memory_order_acquire), "waiter should reserve after release");
  Require(sq.Depth() == 1, "waiter reserve should hold one credit");
  sq.Release(1);
  Require(sq.Depth() == 0, "release should drain controller depth");

  SqController watermarked(4, 2);
  Require(watermarked.Reserve(4, opts, &result), "watermark initial reserve should fill SQ");
  waiterDone.store(false, std::memory_order_release);
  waiterOk.store(false, std::memory_order_release);
  std::promise<void> watermarkWaiterStartedPromise;
  std::future<void> watermarkWaiterStarted = watermarkWaiterStartedPromise.get_future();
  std::thread watermarkWaiter([&]() {
    watermarkWaiterStartedPromise.set_value();
    ReserveResult waitResult;
    waiterOk.store(watermarked.Reserve(1, opts, &waitResult), std::memory_order_release);
    waiterDone.store(true, std::memory_order_release);
  });
  Require(watermarkWaiterStarted.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "watermark waiter thread should start before release");
  watermarked.Release(1);
  Require(!WaitUntil([&]() { return waiterDone.load(std::memory_order_acquire); },
                     std::chrono::milliseconds(20)),
          "waiter should respect resume watermark after pressure");
  watermarked.Release(1);
  watermarkWaiter.join();
  Require(waiterOk.load(std::memory_order_acquire),
          "waiter should reserve after resume watermark is available");
  Require(watermarked.Depth() == 3,
          "watermark waiter should reserve one credit after two releases");
  watermarked.Release(3);
}

void CaseSqControllerTerminalDegraded() {
  SqController sq(4, 2);
  ReserveOptions opts;
  opts.timeoutUs = 200000;
  ReserveResult result;
  Require(sq.Reserve(4, opts, &result), "initial reserve should fill SQ");

  std::atomic<bool> waiterDone{false};
  std::atomic<bool> waiterOk{true};
  std::atomic<SqReserveFailureKind> waiterKind{SqReserveFailureKind::None};
  std::promise<void> waiterStartedPromise;
  std::future<void> waiterStarted = waiterStartedPromise.get_future();
  std::thread waiter([&]() {
    waiterStartedPromise.set_value();
    ReserveResult waitResult;
    waiterOk.store(sq.Reserve(1, opts, &waitResult), std::memory_order_release);
    waiterKind.store(waitResult.kind, std::memory_order_release);
    waiterDone.store(true, std::memory_order_release);
  });

  Require(waiterStarted.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "degraded waiter thread should start before mark degraded");
  sq.MarkDegraded(SqDegradeReason::PartialPostOrphaned);
  Require(WaitUntil([&]() { return waiterDone.load(std::memory_order_acquire); },
                    std::chrono::seconds(1)),
          "waiter should wake on degraded");
  waiter.join();
  Require(!waiterOk.load(std::memory_order_acquire), "waiter should not reserve degraded SQ");
  Require(waiterKind.load(std::memory_order_acquire) == SqReserveFailureKind::TerminalDegraded,
          "waiter should fail with terminal degraded");
  Require(sq.IsTerminalDegraded(), "partial-post orphaned should be terminal degraded");
  Require(sq.Depth() == 4, "mark degraded should not release outstanding credits");
  sq.ReleaseDrainedOrphaned(4);
  Require(sq.Depth() == 0, "drained orphaned release should fix diagnostic depth");
  Require(sq.IsTerminalDegraded(), "drained orphaned release must not restore admission");
  Require(!sq.Reserve(1, opts, &result), "terminal degraded controller should reject reserve");
}

void CaseSqControllerRecheckRollsBack() {
  SqController sq(8, 0);
  ReserveOptions opts;
  opts.timeoutUs = 10000;
  ReserveResult result;
  Require(sq.Reserve(3, opts, &result), "reserve before recheck should succeed");
  sq.MarkDegraded(SqDegradeReason::FatalCqe);
  Require(!sq.RecheckBeforePost(3, &result), "recheck should fail after terminal degraded");
  Require(result.kind == SqReserveFailureKind::TerminalDegraded,
          "recheck should report terminal degraded");
  Require(sq.Depth() == 0, "failed recheck should roll back reserved credits");
}

void CaseSqControllerAdmissionCounters() {
  SqController sq(4, 2);
  AdmissionResult result;
  Require(sq.TryAcquireAdmission(2, 2, &result), "initial admission should succeed");
  Require(sq.QueuedDepth() == 2, "admission should increment queued depth");
  Require(sq.EffectiveDepth() == 2, "effective depth should include queued depth");
  Require(sq.FreeAdmissionSlots() == 2, "free admission slots should reflect queued depth");
  Require(result.snapshot.queuedDepth == 2, "admission result should report queued depth");

  Require(!sq.TryAcquireAdmission(3, 3, &result),
          "admission should fail when effective SQ is full");
  Require(result.kind == AdmissionFailureKind::NoCapacity,
          "full admission should report no capacity");
  const uint64_t observedEpoch = result.snapshot.epoch;

  std::atomic<bool> waiterDone{false};
  std::thread waiter([&]() {
    waiterDone.store(sq.WaitForAdmissionChange(
                         std::chrono::steady_clock::now() + std::chrono::seconds(1), observedEpoch),
                     std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sq.ReleaseAdmission(2);
  waiter.join();
  Require(waiterDone.load(std::memory_order_acquire), "admission waiter should wake on release");
  Require(sq.QueuedDepth() == 0, "admission release should decrement queued depth");

  ReserveOptions opts;
  opts.timeoutUs = 10000;
  ReserveResult reserveResult;
  Require(sq.Reserve(1, opts, &reserveResult), "hard reserve should still work");
  Require(sq.TryAcquireAdmission(2, 2, &result), "admission should account with hard depth");
  Require(sq.EffectiveDepth() == 3, "effective depth should be hard plus queued depth");
  sq.ReleaseAdmission(2);
  sq.Release(1);
}

void CaseSqControllerAdmissionDegraded() {
  SqController sq(4, 0);
  sq.MarkDegraded(SqDegradeReason::FatalCqe);
  AdmissionResult result;
  Require(!sq.TryAcquireAdmission(1, 1, &result), "degraded SQ should reject admission");
  Require(result.kind == AdmissionFailureKind::TerminalDegraded,
          "fatal degraded SQ should report terminal degraded admission failure");
}

void CaseSqControllerAdmissionMarkDegradedWakes() {
  SqController sq(1, 0);
  AdmissionResult result;
  Require(sq.TryAcquireAdmission(1, 1, &result), "initial admission should fill soft slots");
  Require(!sq.TryAcquireAdmission(1, 1, &result), "second admission should observe pressure");
  const uint64_t observedEpoch = result.snapshot.epoch;

  std::atomic<bool> waiterDone{false};
  std::thread waiter([&]() {
    waiterDone.store(sq.WaitForAdmissionChange(
                         std::chrono::steady_clock::now() + std::chrono::seconds(1), observedEpoch),
                     std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sq.MarkDegraded(SqDegradeReason::FatalCqe);
  waiter.join();

  Require(waiterDone.load(std::memory_order_acquire), "MarkDegraded should wake admission waiters");
  sq.ReleaseAdmission(1);
}

void CaseAdmissionTokenMoveReleasesOnce() {
  auto sq = std::make_shared<SqController>(8, 0);
  AdmissionResult result;
  Require(sq->TryAcquireAdmission(3, 3, &result), "token test admission should succeed");
  {
    AdmissionToken token(sq, 3);
    AdmissionToken moved(std::move(token));
    AdmissionToken assigned;
    assigned = std::move(moved);
    Require(sq->QueuedDepth() == 3, "moving token must not release admission early");
  }
  Require(sq->QueuedDepth() == 0, "moved token should release exactly once on destruction");

  Require(sq->TryAcquireAdmission(2, 2, &result), "manual release admission should succeed");
  AdmissionToken token(sq, 2);
  token.Release();
  token.Release();
  Require(sq->QueuedDepth() == 0, "manual token release should be idempotent");
}

void CaseWorkerShutdownDrainsTokensAndPromises() {
  mori::io::TestWorkerShutdownDrainsTokensAndPromisesForTest();
}

void RunPendingUnsignaledOrphaningCase(const char* context, int epCount) {
  constexpr uint32_t kNotifPerQp = 16;
  TransferStatus status;
  status.SetCode(StatusCode::IN_PROGRESS);
  auto meta = std::make_shared<CqCallbackMeta>(&status, 303, epCount * 3);

  EpPairVec eps;
  eps.reserve(epCount);

  ReserveOptions opts;
  opts.timeoutUs = 10000;
  ReserveResult result;
  std::vector<int> epWrsSinceSignal;
  std::vector<size_t> epMergedSinceSignal;
  epWrsSinceSignal.reserve(epCount);
  epMergedSinceSignal.reserve(epCount);

  for (int i = 0; i < epCount; ++i) {
    EpPair ep{};
    ep.sq = std::make_shared<SqController>(8, 0);
    ep.ledger = std::make_shared<SubmissionLedger>(kNotifPerQp);
    Require(ep.sq->Reserve(2, opts, &result), "simulated unsignaled reserve should succeed");
    eps.push_back(ep);
    epWrsSinceSignal.push_back(2);
    epMergedSinceSignal.push_back(3);
  }

  for (int i = 0; i < epCount; ++i) {
    MovePendingUnsignaledToOrphanedForEndpoint(eps, static_cast<size_t>(i), epWrsSinceSignal,
                                               epMergedSinceSignal, meta,
                                               std::string("simulated ") + context, context);
  }

  for (int i = 0; i < epCount; ++i) {
    Require(epWrsSinceSignal[i] == 0, "pending WR counter should be cleared after orphaning");
    Require(epMergedSinceSignal[i] == 0,
            "pending merged counter should be cleared after orphaning");
    Require(eps[i].sq->IsTerminalDegraded(),
            "pending unsignaled orphan should terminal degrade SQ");
    Require(eps[i].sq->Depth() == 2, "orphaned WR credits should remain held until proven drained");
    Require(eps[i].ledger->RecordCount() == 0, "orphaned records should be extracted for failure");
    eps[i].sq->ReleaseDrainedOrphaned(2);
    Require(eps[i].sq->Depth() == 0, "drained orphaned release should clear held credits");
  }

  Require(
      meta->finishedBatchSize.load(std::memory_order_relaxed) == static_cast<uint32_t>(epCount * 3),
      "orphaned meta should be counted as finished by failure path");
  Require(status.Failed(), "orphaned meta should fail transfer status");
}

void CasePendingUnsignaledRecheckFailureOrphans() {
  RunPendingUnsignaledOrphaningCase("RecheckBeforePost failed", 1);
}

void CasePendingUnsignaledReserveFailureOrphans() {
  RunPendingUnsignaledOrphaningCase("TryReserveSqDepth failed", 2);
}

void CasePendingUnsignaledOrphaningClosesAdmissionBeforeRecoveryGuard() {
  constexpr uint32_t kNotifPerQp = 16;
  TransferStatus status;
  status.SetCode(StatusCode::IN_PROGRESS);
  auto meta = std::make_shared<CqCallbackMeta>(&status, 404, 3);

  EpPair ep{};
  ep.sq = std::make_shared<SqController>(8, 0);
  ep.ledger = std::make_shared<SubmissionLedger>(kNotifPerQp);

  ReserveOptions opts;
  opts.timeoutUs = 10000;
  ReserveResult result;
  Require(ep.sq->Reserve(2, opts, &result), "simulated unsignaled reserve should succeed");

  EpPairVec eps{ep};
  std::vector<int> epWrsSinceSignal{2};
  std::vector<size_t> epMergedSinceSignal{3};
  std::shared_lock<std::shared_mutex> recoveryBlocker = ep.sq->AcquireSubmitGuard();
  std::atomic<bool> helperStarted{false};
  std::atomic<bool> helperDone{false};

  std::thread helper([&]() {
    std::shared_lock<std::shared_mutex> heldSubmitGuard = eps[0].sq->AcquireSubmitGuard();
    helperStarted.store(true, std::memory_order_release);
    MovePendingUnsignaledToOrphanedForEndpoint(eps, 0, epWrsSinceSignal, epMergedSinceSignal, meta,
                                               "blocked recovery", "admission-close test",
                                               &heldSubmitGuard);
    helperDone.store(true, std::memory_order_release);
  });

  Require(WaitUntil([&]() { return helperStarted.load(std::memory_order_acquire); },
                    std::chrono::seconds(1)),
          "helper thread should acquire submit guard");
  Require(WaitUntil([&]() { return ep.sq->IsTerminalDegraded(); }, std::chrono::seconds(1)),
          "helper should close admission before recovery guard drains");
  Require(!helperDone.load(std::memory_order_acquire),
          "helper should wait for recovery guard while admission is closed");
  Require(!ep.sq->Reserve(1, opts, &result), "reserve should fail after admission is closed");

  recoveryBlocker.unlock();
  helper.join();
  Require(helperDone.load(std::memory_order_acquire), "helper should finish after recovery drains");
  Require(status.Failed(), "orphaned meta should fail after recovery drains");
  ep.sq->ReleaseDrainedOrphaned(2);
  Require(ep.sq->Depth() == 0, "drained orphaned release should clear held credits");
}

void CaseRdmaBackendSessionAliveChecksTerminalSq() {
  RdmaBackendConfig cfg{};
  mori::application::RdmaMemoryRegion mr{};
  EpPair ep{};
  ep.sq = std::make_shared<SqController>(8, 0);
  RdmaBackendSession sess(cfg, mr, mr, EpPairVec{ep}, nullptr);
  Require(sess.Alive(), "session should be alive before terminal degrade");
  ep.sq->MarkDegraded(SqDegradeReason::PartialPostOrphaned);
  Require(!sess.Alive(), "session should be dead after terminal degraded endpoint");
}

void CaseWrIdNamespaceHelpers() {
  const uint64_t taggedZero = MakeNotifSendWrId(0);
  Require(taggedZero == kNotifSendWrIdTag, "tagged zero should only set the reserved high bit");
  Require(IsNotifSendWrId(taggedZero), "tagged zero should be recognized as notification SEND");
  Require(ExtractTransferIdFromWrId(taggedZero) == 0,
          "extracting transfer id from tagged zero should yield zero");

  const TransferUniqueId plainId = 1023;
  const uint64_t taggedPlain = MakeNotifSendWrId(plainId);
  Require(IsNotifSendWrId(taggedPlain), "tagged plain id should be recognized");
  Require(ExtractTransferIdFromWrId(taggedPlain) == plainId,
          "extracting transfer id should preserve the original low bits");

  const TransferUniqueId externalTaggedId = kNotifSendWrIdTag | TransferUniqueId{42};
  const uint64_t taggedMasked = MakeNotifSendWrId(externalTaggedId);
  Require(IsNotifSendWrId(taggedMasked), "masked tagged id should still carry the SEND tag");
  Require(ExtractTransferIdFromWrId(taggedMasked) == 42,
          "high-bit caller ids should be masked before tagging");
  Require(!IsNotifSendWrId(4096), "ledger-range ids without bit 63 should not be SEND-tagged");
}

void CaseRdmaNotificationRejectsZeroNotifPerQp() {
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("rdma_invalid_notif_per_qp", cfg);

  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.enableNotification = true;
  rdmaCfg.notifPerQp = 0;

  bool threw = false;
  try {
    engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  } catch (const std::runtime_error& e) {
    threw = true;
    Require(std::string(e.what()).find("notifPerQp") != std::string::npos,
            "zero notifPerQp failure should mention notifPerQp");
  }
  Require(threw, "notification-enabled RDMA backend should reject notifPerQp == 0");
}

void CaseRdmaTransferBasic() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_basic", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 1024 * 1024, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 1024 * 1024, 0);

  TransferStatus initStatus;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 1024 * 1024, &initStatus, uid);

  std::string err;
  Require(WaitTransferDone(&initStatus, 3000, &err), "rdma initiator status timeout: " + err);
  Require(initStatus.Succeeded(),
          "rdma initiator status failed: code=" + std::to_string(initStatus.CodeUint32()) +
              ", msg='" + initStatus.Message() + "'");

  TransferStatus inbound;
  Require(WaitInboundStatusWithTimeout(pair.target.get(), pair.initiator->GetEngineDesc().key, uid,
                                       3000, &inbound, &err),
          "rdma inbound status timeout: " + err);
  Require(inbound.Succeeded(),
          "rdma inbound status failed: code=" + std::to_string(inbound.CodeUint32()) + ", msg='" +
              inbound.Message() + "'");
}

void CaseRdmaNotificationDisabledBehavior() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_no_notif", false);
  auto src = RegisterGpuMemory(pair.initiator.get(), 64 * 1024, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 64 * 1024, 0);

  TransferStatus initStatus;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Write(src.desc, 0, dst.desc, 0, 64 * 1024, &initStatus, uid);

  std::string err;
  Require(WaitTransferDone(&initStatus, 3000, &err),
          "rdma(no_notif) initiator status timeout: " + err);
  Require(initStatus.Succeeded(), "rdma(no_notif) initiator status failed: code=" +
                                      std::to_string(initStatus.CodeUint32()) + ", msg='" +
                                      initStatus.Message() + "'");

  TransferStatus inbound;
  bool popped = WaitInboundStatusWithTimeout(pair.target.get(), pair.initiator->GetEngineDesc().key,
                                             uid, 200, &inbound, nullptr);
  Require(!popped, "inbound notification should be unavailable when notification is disabled");
}

void CaseRdmaNotificationEnvOverrideDisables() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ScopedEnvVar forceDisableNotif("MORI_IO_ENABLE_NOTIFICATION", "0");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_env_no_notif", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 64 * 1024, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 64 * 1024, 0);

  TransferStatus initStatus;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Write(src.desc, 0, dst.desc, 0, 64 * 1024, &initStatus, uid);

  std::string err;
  Require(WaitTransferDone(&initStatus, 3000, &err),
          "rdma(env_no_notif) initiator status timeout: " + err);
  Require(initStatus.Succeeded(), "rdma(env_no_notif) initiator status failed: code=" +
                                      std::to_string(initStatus.CodeUint32()) + ", msg='" +
                                      initStatus.Message() + "'");

  TransferStatus inbound;
  bool popped = WaitInboundStatusWithTimeout(pair.target.get(), pair.initiator->GetEngineDesc().key,
                                             uid, 200, &inbound, nullptr);
  Require(!popped, "inbound notification should be disabled by MORI_IO_ENABLE_NOTIFICATION=0");
}

void CaseRdmaNotificationInvalidEnvKeepsConfig() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ScopedEnvVar invalidNotif("MORI_IO_ENABLE_NOTIFICATION", "invalid");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_invalid_env_notif", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 64 * 1024, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 64 * 1024, 0);

  TransferStatus initStatus;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Write(src.desc, 0, dst.desc, 0, 64 * 1024, &initStatus, uid);

  std::string err;
  Require(WaitTransferDone(&initStatus, 3000, &err),
          "rdma(invalid_env_notif) initiator status timeout: " + err);
  Require(initStatus.Succeeded(), "rdma(invalid_env_notif) initiator status failed: code=" +
                                      std::to_string(initStatus.CodeUint32()) + ", msg='" +
                                      initStatus.Message() + "'");

  TransferStatus inbound;
  Require(WaitInboundStatusWithTimeout(pair.target.get(), pair.initiator->GetEngineDesc().key, uid,
                                       3000, &inbound, &err),
          "rdma(invalid_env_notif) inbound status timeout: " + err);
  Require(inbound.Succeeded(),
          "invalid MORI_IO_ENABLE_NOTIFICATION should keep config(true), inbound code=" +
              std::to_string(inbound.CodeUint32()) + ", msg='" + inbound.Message() + "'");
}

// Mirror of the NormalizeBusId logic in src/io/xgmi/backend_impl.cpp for testing.
std::string TestNormalizeBusId(const std::string& busId) {
  std::string result = busId;
  for (auto& c : result) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return result;
}

void CaseNormalizeBusId() {
  Require(TestNormalizeBusId("0000:C1:00.0") == "0000:c1:00.0",
          "uppercase hex should be lowercased");
  Require(TestNormalizeBusId("0000:c1:00.0") == "0000:c1:00.0",
          "already lowercase should be unchanged");
  Require(TestNormalizeBusId("0000:AB:CD.0") == "0000:ab:cd.0",
          "mixed case should be fully lowered");
  Require(TestNormalizeBusId("") == "", "empty string should remain empty");
}

void CaseIsIpcHandleEmpty() {
  constexpr size_t kSize = 64;
  std::array<char, kSize> zeroHandle{};
  Require(std::all_of(zeroHandle.begin(), zeroHandle.end(), [](char c) { return c == 0; }),
          "zero-initialized handle should be empty");

  std::array<char, kSize> nonZeroFirst{};
  nonZeroFirst[0] = 1;
  Require(!std::all_of(nonZeroFirst.begin(), nonZeroFirst.end(), [](char c) { return c == 0; }),
          "handle with non-zero first byte should not be empty");

  std::array<char, kSize> nonZeroLast{};
  nonZeroLast[kSize - 1] = 1;
  Require(!std::all_of(nonZeroLast.begin(), nonZeroLast.end(), [](char c) { return c == 0; }),
          "handle with non-zero last byte should not be empty");
}

void CaseXgmiVisibleDeviceRegression() {
  if (GetGpuCount() < 2) throw TestSkip("requires at least 2 GPUs");

  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("xgmi_visible_regression_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  auto src = RegisterGpuMemory(&engine, 1024 * 1024, 0);
  auto dst = RegisterGpuMemory(&engine, 1024 * 1024, 1);

  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  engine.Write(src.desc, 0, dst.desc, 0, 1024 * 1024, &status, uid);

  std::string err;
  Require(WaitTransferDone(&status, 5000, &err),
          "xgmi visible-device regression transfer timeout: " + err);
  Require(status.Succeeded(), "xgmi visible-device regression transfer failed: code=" +
                                  std::to_string(status.CodeUint32()) + ", msg='" +
                                  status.Message() + "'");
}

void CaseXgmiCrossEngineIpc() {
  // Tests cross-engine XGMI IPC: two IOEngines in the same process exchange
  // data between GPU 0 and GPU 1 using IPC handles.
  //
  // NOTE: This exercises the cross-engine IPC handle open/remap path but NOT
  // the hidden-device branch (LookupVisibleDevice returns nullopt).  In a
  // single process all GPUs are visible, so CreateSession always takes the
  // visible-remote path.  The true hidden-device path (split HIP_VISIBLE_DEVICES)
  // is tested by CaseXgmiHiddenDeviceSplitVisibility which launches a subprocess.
  if (GetGpuCount() < 2) throw TestSkip("requires at least 2 GPUs");

  IOEngineConfig cfgA;
  cfgA.host = "127.0.0.1";
  cfgA.port = 0;
  auto engineA = std::make_unique<IOEngine>("xgmi_cross_A", cfgA);

  IOEngineConfig cfgB;
  cfgB.host = "127.0.0.1";
  cfgB.port = 0;
  auto engineB = std::make_unique<IOEngine>("xgmi_cross_B", cfgB);

  XgmiBackendConfig xgmiCfg{};
  engineA->CreateBackend(BackendType::XGMI, xgmiCfg);
  engineB->CreateBackend(BackendType::XGMI, xgmiCfg);

  engineA->RegisterRemoteEngine(engineB->GetEngineDesc());
  engineB->RegisterRemoteEngine(engineA->GetEngineDesc());

  constexpr size_t kSize = 1024 * 1024;
  auto srcMem = RegisterGpuMemory(engineA.get(), kSize, 0);
  auto dstMem = RegisterGpuMemory(engineB.get(), kSize, 1);

  HIP_RUNTIME_CHECK(hipSetDevice(0));
  HIP_RUNTIME_CHECK(hipMemset(srcMem.ptr, 0xAB, kSize));
  HIP_RUNTIME_CHECK(hipSetDevice(1));
  HIP_RUNTIME_CHECK(hipMemset(dstMem.ptr, 0x00, kSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  TransferStatus status;
  TransferUniqueId uid = engineA->AllocateTransferUniqueId();
  engineA->Write(srcMem.desc, 0, dstMem.desc, 0, kSize, &status, uid);

  std::string err;
  Require(WaitTransferDone(&status, 5000, &err), "xgmi cross-engine IPC transfer timeout: " + err);
  Require(status.Succeeded(),
          "xgmi cross-engine IPC transfer failed: code=" + std::to_string(status.CodeUint32()) +
              ", msg='" + status.Message() + "'");

  std::vector<uint8_t> hostBuf(kSize);
  HIP_RUNTIME_CHECK(hipSetDevice(1));
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuf.data(), dstMem.ptr, kSize, hipMemcpyDeviceToHost));
  bool allMatch = true;
  for (size_t i = 0; i < kSize; ++i) {
    if (hostBuf[i] != 0xAB) {
      allMatch = false;
      break;
    }
  }
  Require(allMatch, "xgmi cross-engine IPC data verification failed");
}

// --------------------------------------------------------------------------
// Subprocess-based hidden-device test.
//
// The real hidden-device path requires a bus ID that is NOT in the importing
// process's HIP_VISIBLE_DEVICES.  In a single process all GPUs are visible, so
// we can never trigger LookupVisibleDevice() -> nullopt.  To test it properly
// we launch a subprocess with restricted HIP_VISIBLE_DEVICES.
//
// Protocol (via shared memory file in /dev/shm):
//   1. Exporter (this process, GPU 0):  allocates GPU memory, registers it with
//      an IOEngine to populate the IPC handle, writes a MemoryDesc blob to the
//      shared file, and waits for the importer to signal completion.
//   2. Importer (subprocess, HIP_VISIBLE_DEVICES=<last_gpu>):  reads the
//      MemoryDesc, creates its own IOEngine, and does a Write from its local
//      GPU to the exporter's memory.  The exporter's bus ID is NOT in the
//      importer's localDeviceByBusId, so it must go through the hidden-device
//      branch.
// --------------------------------------------------------------------------
int RunHiddenDeviceImporter(const char* shmPath) {
  // This function runs in a subprocess with restricted HIP_VISIBLE_DEVICES.
  // It only sees one GPU (the last physical GPU), while the exporter used GPU 0.
  SetLogLevel("info");
  int gpuCount = GetGpuCount();
  if (gpuCount < 1) {
    std::fprintf(stderr, "importer: no GPUs visible\n");
    return 1;
  }

  // Read serialized MemoryDesc from shared file
  int fd = open(shmPath, O_RDONLY);
  if (fd < 0) {
    std::fprintf(stderr, "importer: failed to open shm\n");
    return 1;
  }

  // Read msgpack blob
  char buf[4096];
  ssize_t n = read(fd, buf, sizeof(buf));
  close(fd);
  if (n <= 0) {
    std::fprintf(stderr, "importer: failed to read shm\n");
    return 1;
  }

  // Deserialize remote MemoryDesc
  msgpack::object_handle oh = msgpack::unpack(buf, static_cast<size_t>(n));
  MemoryDesc remoteDesc;
  oh.get().convert(remoteDesc);

  std::fprintf(stderr, "importer: remote bus_id=%s engineKey=%s ipcHandle[0]=%d\n",
               remoteDesc.deviceBusId.c_str(), remoteDesc.engineKey.c_str(),
               static_cast<int>(remoteDesc.ipcHandle[0]));

  // Create local engine with XGMI backend
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("importer_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  // Register the remote engine so IsSameNodeEngine returns true
  EngineDesc remoteEngDesc;
  remoteEngDesc.key = remoteDesc.engineKey;
  {
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    remoteEngDesc.hostname = std::string(hostname);
    remoteEngDesc.nodeId = remoteEngDesc.hostname;
  }
  remoteEngDesc.host = "127.0.0.1";
  remoteEngDesc.port = 0;
  remoteEngDesc.pid = 0;
  engine.RegisterRemoteEngine(remoteEngDesc);

  // Allocate local memory on device 0 (importer's only visible GPU)
  constexpr size_t kSize = 1024 * 1024;
  auto localMem = RegisterGpuMemory(&engine, kSize, 0);

  // Fill local source with 0xCD
  HIP_RUNTIME_CHECK(hipSetDevice(0));
  HIP_RUNTIME_CHECK(hipMemset(localMem.ptr, 0xCD, kSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Create a session from local to remote (hidden-device path!)
  auto session = engine.CreateSession(localMem.desc, remoteDesc);
  if (!session.has_value()) {
    std::fprintf(stderr, "importer: CreateSession failed\n");
    return 1;
  }

  // Write local data to remote memory
  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  session->Write(0, 0, kSize, &status, uid);

  std::string err;
  bool ok = WaitTransferDone(&status, 5000, &err);
  if (!ok || !status.Succeeded()) {
    std::fprintf(stderr, "importer: transfer failed: %s\n", err.c_str());
    return 1;
  }

  std::fprintf(stderr, "importer: transfer succeeded\n");

  // Signal completion by writing "done" to shm
  int wfd = open(shmPath, O_WRONLY | O_TRUNC);
  if (wfd >= 0) {
    const char* msg = "done";
    (void)write(wfd, msg, 4);
    close(wfd);
  }

  return 0;
}

void CaseXgmiHiddenDeviceSplitVisibility() {
  int totalGpus = GetGpuCount();
  if (totalGpus < 2) throw TestSkip("requires at least 2 GPUs");

  // Create shared memory file for IPC
  std::string shmPath = "/dev/shm/mori_test_hidden_" + std::to_string(getpid());
  int shmFd = open(shmPath.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  Require(shmFd >= 0, "failed to create shared memory file");

  // Exporter: allocate on GPU 0, register, and serialize the descriptor
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("exporter_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  constexpr size_t kSize = 1024 * 1024;
  auto exportMem = RegisterGpuMemory(&engine, kSize, 0);

  // Clear export buffer
  HIP_RUNTIME_CHECK(hipSetDevice(0));
  HIP_RUNTIME_CHECK(hipMemset(exportMem.ptr, 0x00, kSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Serialize MemoryDesc via msgpack
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, exportMem.desc);
  ssize_t written = write(shmFd, sbuf.data(), sbuf.size());
  close(shmFd);
  Require(written == static_cast<ssize_t>(sbuf.size()), "failed to write descriptor to shm");

  // Get our own executable path
  char selfExe[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", selfExe, sizeof(selfExe) - 1);
  Require(len > 0, "failed to read /proc/self/exe");
  selfExe[len] = '\0';

  // Launch importer subprocess with the LAST GPU only visible
  // (so GPU 0's bus ID is NOT in the importer's localDeviceByBusId)
  std::string visibleDevices = std::to_string(totalGpus - 1);
  std::string cmd = "HIP_VISIBLE_DEVICES=" + visibleDevices + " " + std::string(selfExe) +
                    " --hidden-device-importer " + shmPath + " 2>&1";
  int rc = system(cmd.c_str());
  int exitCode = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
  Require(exitCode == 0, "importer subprocess failed with exit code " + std::to_string(exitCode));

  // Read back the signal from shm
  shmFd = open(shmPath.c_str(), O_RDONLY);
  char doneBuf[8] = {};
  if (shmFd >= 0) {
    (void)read(shmFd, doneBuf, sizeof(doneBuf));
    close(shmFd);
  }
  unlink(shmPath.c_str());
  Require(std::string(doneBuf, 4) == "done", "importer did not signal completion");

  // Verify the exporter's GPU 0 buffer now contains 0xCD (written by importer)
  std::vector<uint8_t> hostBuf(kSize);
  HIP_RUNTIME_CHECK(hipSetDevice(0));
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuf.data(), exportMem.ptr, kSize, hipMemcpyDeviceToHost));
  bool allMatch = true;
  for (size_t i = 0; i < kSize; ++i) {
    if (hostBuf[i] != 0xCD) {
      allMatch = false;
      break;
    }
  }
  Require(allMatch, "hidden-device data verification failed: expected 0xCD in exporter buffer");
}

void CaseXgmiInboundNotificationIsUnsupported() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("xgmi_semantics_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  auto src = RegisterGpuMemory(&engine, 64 * 1024, 0);
  auto dst = RegisterGpuMemory(&engine, 64 * 1024, 0);

  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  engine.Write(src.desc, 0, dst.desc, 0, 64 * 1024, &status, uid);

  status.Wait();
  Require(status.Succeeded(), "xgmi transfer failed: code=" + std::to_string(status.CodeUint32()) +
                                  ", msg='" + status.Message() + "'");

  TransferStatus inbound;
  bool popped = engine.PopInboundTransferStatus("dummy_remote", uid, &inbound);
  Require(!popped, "xgmi pop inbound should return false");
}

void CaseXgmiConcurrentWaitAndPollIsSafe() {
  if (GetGpuCount() < 2) throw TestSkip("requires at least 2 GPUs");

  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("xgmi_concurrent_wait_poll_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  auto src = RegisterGpuMemory(&engine, 64 * 1024 * 1024, 0);
  auto dst = RegisterGpuMemory(&engine, 64 * 1024 * 1024, 1);

  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  engine.Write(src.desc, 0, dst.desc, 0, 64 * 1024 * 1024, &status, uid);

  std::atomic<bool> stopPolling{false};
  std::thread poller([&]() {
    while (!stopPolling.load(std::memory_order_acquire)) {
      (void)status.Code();
      if (!status.Init() && !status.InProgress()) {
        break;
      }
      std::this_thread::yield();
    }
  });

  status.Wait();
  stopPolling.store(true, std::memory_order_release);
  if (poller.joinable()) poller.join();

  Require(status.Succeeded(),
          "xgmi concurrent wait/poll transfer failed: code=" + std::to_string(status.CodeUint32()) +
              ", msg='" + status.Message() + "'");
}

struct TestCase {
  const char* name;
  std::function<void()> run;
};

}  // namespace

namespace mori {
namespace io {

void TestWorkerShutdownDrainsTokensAndPromisesForTest() {
  auto check = [](bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
  };

  MultithreadExecutor::Worker worker(0);
  worker.running.store(true, std::memory_order_release);

  auto sq = std::make_shared<SqController>(8, 0);
  AdmissionResult result;
  std::vector<std::future<RdmaOpRet>> futures;

  {
    std::lock_guard<std::mutex> lock(worker.mu);
    for (int i = 0; i < 3; ++i) {
      check(sq->TryAcquireAdmission(1, 1, &result),
            "test setup should acquire worker shutdown admission token");
      MultithreadExecutor::Task task{nullptr, 0, 0, 0, AdmissionToken(sq, 1)};
      futures.push_back(task.ret.get_future());
      worker.q.push(std::move(task));
    }
  }

  check(sq->QueuedDepth() == 3, "queued test tasks should hold admission tokens");
  worker.Shutdown();
  check(sq->QueuedDepth() == 0, "shutdown should release queued task admission tokens");

  for (auto& fut : futures) {
    check(fut.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "shutdown should complete queued task promise");
    RdmaOpRet ret = fut.get();
    check(ret.code == StatusCode::ERR_BAD_STATE, "shutdown should return ERR_BAD_STATE");
    check(ret.message == "executor shutdown", "shutdown should use executor shutdown message");
  }
}

}  // namespace io
}  // namespace mori

int main(int argc, char* argv[]) {
  // Subprocess entry point for hidden-device importer
  if (argc >= 3 && std::string(argv[1]) == "--hidden-device-importer") {
    return RunHiddenDeviceImporter(argv[2]);
  }

  SetLogLevel("info");
  std::vector<TestCase> cases = {
      {"submission_ledger_basic", CaseSubmissionLedgerBasic},
      {"sq_controller_reserve_release_wait", CaseSqControllerReserveReleaseWait},
      {"sq_controller_terminal_degraded", CaseSqControllerTerminalDegraded},
      {"sq_controller_recheck_rolls_back", CaseSqControllerRecheckRollsBack},
      {"sq_controller_admission_counters", CaseSqControllerAdmissionCounters},
      {"sq_controller_admission_degraded", CaseSqControllerAdmissionDegraded},
      {"sq_controller_admission_mark_degraded_wakes", CaseSqControllerAdmissionMarkDegradedWakes},
      {"admission_token_move_releases_once", CaseAdmissionTokenMoveReleasesOnce},
      {"worker_shutdown_drains_tokens_and_promises", CaseWorkerShutdownDrainsTokensAndPromises},
      {"pending_unsignaled_recheck_failure_orphans", CasePendingUnsignaledRecheckFailureOrphans},
      {"pending_unsignaled_reserve_failure_orphans", CasePendingUnsignaledReserveFailureOrphans},
      {"pending_unsignaled_orphaning_closes_admission_before_recovery",
       CasePendingUnsignaledOrphaningClosesAdmissionBeforeRecoveryGuard},
      {"rdma_session_alive_checks_terminal_sq", CaseRdmaBackendSessionAliveChecksTerminalSq},
      {"wr_id_namespace_helpers", CaseWrIdNamespaceHelpers},
      {"rdma_notification_rejects_zero_notif_per_qp", CaseRdmaNotificationRejectsZeroNotifPerQp},
      {"rdma_transfer_basic", CaseRdmaTransferBasic},
      {"rdma_notification_disabled_behavior", CaseRdmaNotificationDisabledBehavior},
      {"rdma_notification_env_override_disables", CaseRdmaNotificationEnvOverrideDisables},
      {"rdma_notification_invalid_env_keeps_config", CaseRdmaNotificationInvalidEnvKeepsConfig},
      {"normalize_bus_id", CaseNormalizeBusId},
      {"is_ipc_handle_empty", CaseIsIpcHandleEmpty},
      {"xgmi_visible_device_regression", CaseXgmiVisibleDeviceRegression},
      {"xgmi_cross_engine_ipc", CaseXgmiCrossEngineIpc},
      {"xgmi_hidden_device_split_visibility", CaseXgmiHiddenDeviceSplitVisibility},
      {"xgmi_inbound_notification_is_unsupported", CaseXgmiInboundNotificationIsUnsupported},
      {"xgmi_concurrent_wait_and_poll_is_safe", CaseXgmiConcurrentWaitAndPollIsSafe},
  };

  int passed = 0;
  int failed = 0;
  int skipped = 0;
  auto allStart = std::chrono::steady_clock::now();

  for (const auto& tc : cases) {
    auto st = std::chrono::steady_clock::now();
    try {
      tc.run();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - st)
                    .count();
      std::printf("[PASS] %s (%lld ms)\n", tc.name, static_cast<long long>(ms));
      passed++;
    } catch (const TestSkip& e) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - st)
                    .count();
      std::printf("[SKIP] %s (%lld ms): %s\n", tc.name, static_cast<long long>(ms), e.what());
      skipped++;
    } catch (const std::exception& e) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - st)
                    .count();
      std::printf("[FAIL] %s (%lld ms): %s\n", tc.name, static_cast<long long>(ms), e.what());
      failed++;
    }
  }

  auto allMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - allStart)
                   .count();
  std::printf("==== test_engine summary ====\n");
  std::printf("total=%zu passed=%d failed=%d skipped=%d elapsed_ms=%lld\n", cases.size(), passed,
              failed, skipped, static_cast<long long>(allMs));
  return failed == 0 ? 0 : 1;
}
