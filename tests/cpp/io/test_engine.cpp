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
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <limits.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"
#include "mori/utils/host_utils.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/common.hpp"
#include "src/io/rdma/verbs_ops.hpp"

namespace mori::io {
struct SubmissionLedgerTest {
  static void CorruptEntry(SubmissionLedger& ledger, uint64_t seq) {
    ledger.ring_[seq & ledger.mask_].seq.store(0, std::memory_order_release);
  }
};
}  // namespace mori::io

using namespace mori::io;

namespace {

constexpr const char* kNoRdmaDeviceFilter = "__mori_no_such_device_for_test__";

struct TestSkip : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TestFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Require(bool cond, const std::string& msg) {
  if (!cond) throw TestFailure(msg);
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

ConnectedEnginePair CreateConnectedRdmaPair(const std::string& prefix, bool enableNotification) {
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

  // Port 0 lets each backend bind its own ephemeral port. A port probed up front
  // is only free until CreateBackend binds it, and two probes can return the same one.
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  auto initiator = std::make_unique<IOEngine>(prefix + "_initiator", cfg);
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

class LedgerVerbs : public VerbsOps {
 public:
  std::function<int(ibv_send_wr*, ibv_send_wr**)> post;
  int PostSend(ibv_qp*, ibv_send_wr* wr, ibv_send_wr** bad) override {
    return post ? post(wr, bad) : 0;
  }
  int PostRecv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) override { return 0; }
  int PollCq(ibv_cq*, int, ibv_wc*) override { return 0; }
  int GetCqEvent(ibv_comp_channel*, ibv_cq**, void**) override { return 0; }
  int GetAsyncEvent(ibv_context*, ibv_async_event*) override { return 0; }
  void AckAsyncEvent(ibv_async_event*) override {}
  int QueryPort(ibv_context*, uint8_t, ibv_port_attr*) override { return 0; }
  int WatchAsyncContext(ibv_context*) override { return -1; }
  void UnwatchAsyncContext(ibv_context*) override {}
  void ReleaseAsyncContext(ibv_context*) override {}
  void ForgetAsyncEndpoint(ibv_context*, uint32_t, ibv_cq*) override {}
  void BeforeNotify(ibv_qp*, CqCallbackMeta*) override {}
};

SubmissionLedger::PostResult PostLedger(SubmissionLedger& ledger, LedgerVerbs& verbs, int count,
                                        const std::shared_ptr<CqCallbackMeta>& meta,
                                        bool final = true, int credit = 0) {
  Require(ledger.Reserve(count, 1000000, nullptr), "ledger reserve");
  std::vector<ibv_send_wr> wrs(count);
  for (int i = 0; i < count; ++i) {
    wrs[i].opcode = IBV_WR_RDMA_WRITE;
    wrs[i].next = i + 1 < count ? &wrs[i + 1] : nullptr;
  }
  return ledger.PostToSq(nullptr, wrs.data(), count, SubmissionLedger::Kind::Data, meta,
                         credit ? credit : count, final, meta->id, 0, verbs);
}

void CaseQuiescenceBusyPostRecordsTarget() {
  SubmissionLedger ledger(16);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 10);
  PostLedger(ledger, verbs, 1, meta, false);
  std::promise<void> posting, release;
  auto gate = release.get_future().share();
  verbs.post = [&](ibv_send_wr*, ibv_send_wr**) {
    posting.set_value();
    gate.wait();
    return 0;
  };
  std::thread poster([&] { PostLedger(ledger, verbs, 1, meta, false); });
  posting.get_future().wait();
  auto pending = ledger.EnsureCovered(1);
  release.set_value();
  poster.join();
  Require(pending == QuiescenceState::Pending, "busy post remains nonblocking");
  int markers = 0;
  verbs.post = [&](ibv_send_wr*, ibv_send_wr**) {
    ++markers;
    return 0;
  };
  ledger.DrivePending();
  Require(markers == 1, "busy-lock target survives without a caller waiting");
  ledger.OnSqCompletion(kSqTag | 3, IBV_WC_SUCCESS, "marker");
  Require(ledger.Used() == 0, "retry covers the unsignaled tail");
}

void CaseQuiescenceMarkers() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 8);
  PostLedger(ledger, verbs, 1, meta, false);
  Require(ledger.Reserve(6, 0, nullptr), "all ordinary credits occupied");
  std::vector<uint64_t> markers;
  verbs.post = [&](ibv_send_wr* wr, ibv_send_wr**) {
    Require(
        wr->num_sge == 0 && wr->opcode == IBV_WR_RDMA_WRITE && (wr->send_flags & IBV_SEND_SIGNALED),
        "zero-length signaled marker");
    markers.push_back(SqSequence(wr->wr_id));
    Require(ledger.EnsureCovered(1) == QuiescenceState::Pending, "query never self-locks");
    return 0;
  };
  ledger.EnsureCovered(1);
  Require(markers.size() == 1 && ledger.Used() == 8, "spare marker credit remains available");
  ledger.EnsureCovered(1);
  Require(markers.size() == 1, "one outstanding marker");
  ledger.CancelReservation(6);
  verbs.post = nullptr;
  PostLedger(ledger, verbs, 1, meta, false);  // sequence 3, behind marker 2
  verbs.post = [&](ibv_send_wr* wr, ibv_send_wr**) {
    markers.push_back(SqSequence(wr->wr_id));
    return 0;
  };
  ledger.EnsureCovered(3);
  Require(markers.size() == 1, "new target merged while marker outstanding");
  ledger.OnSqCompletion(kSqTag | 2, IBV_WC_SUCCESS, "first marker");
  Require(markers.size() == 2 && markers.back() == 4, "CQ drives next merged marker");
  ledger.OnSqCompletion(kSqTag | 4, IBV_WC_SUCCESS, "second marker");
  Require(ledger.Done() == 4 && ledger.Used() == 0, "all physical credits retired once");
}

void CaseQuiescenceMarkerRetryAndFreeze() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 8);
  PostLedger(ledger, verbs, 1, meta, false);
  int tries = 0;
  verbs.post = [&](ibv_send_wr* wr, ibv_send_wr** bad) {
    if (++tries == 1) {
      *bad = wr;
      return ENOMEM;
    }
    return 0;
  };
  Require(ledger.EnsureCovered(1) == QuiescenceState::Pending && ledger.Used() == 1,
          "ENOMEM remains pending and refunds unposted marker");
  ledger.DrivePending();
  Require(tries == 2 && ledger.Used() == 2, "periodic retry without query");
  // Losing the marker CQE and covering it later never refunds a separate credit.
  verbs.post = nullptr;
  auto next = PostLedger(ledger, verbs, 1, meta);
  ledger.OnSqCompletion(kSqTag | next.dataSeq, IBV_WC_SUCCESS, "marker successor");
  Require(ledger.Done() == 3 && ledger.Used() == 0, "lost marker prefix refund");
  ledger.Close(true);
  Require(ledger.EnsureCovered(4) == QuiescenceState::Unprovable, "freeze stops marker posts");
  auto done = ledger.Done();
  ledger.OnSqCompletion(kSqTag | 4, IBV_WC_SUCCESS, "untrusted");
  Require(ledger.Done() == done, "frozen prefix never advances");
}

void CaseQuiescenceMarkerRejected() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 8);
  PostLedger(ledger, verbs, 1, meta, false);
  verbs.post = [](ibv_send_wr* wr, ibv_send_wr** bad) {
    *bad = wr;
    return EINVAL;
  };
  Require(ledger.EnsureCovered(1) == QuiescenceState::Unprovable, "permanent marker rejection");
  Require(!ledger.Closed() && ledger.Used() == 1, "marker rejection cannot kill healthy QP");
}

void CaseReusableStatusContention() {
  TransferStatus status;
  std::promise<void> entered, resume;
  auto release = resume.get_future().share();
  std::atomic<bool> block{false};
  status.SetQuiescenceCallback([&](int) {
    if (block.load()) {
      entered.set_value();
      release.wait();
    }
    return QuiescenceState::Quiesced;
  });
  block = true;
  auto installer = std::async(std::launch::async, [&] {
    status.SetQuiescenceCallback([](int) { return QuiescenceState::Quiesced; });
  });
  entered.get_future().wait();
  auto reuse = std::async(std::launch::async, [&] { status.CheckReusable(); });
  auto waiting = reuse.wait_for(std::chrono::milliseconds(20));
  resume.set_value();
  installer.get();
  reuse.get();
  Require(waiting == std::future_status::timeout, "reuse blocks on snapshot contention");
  Require(status.Quiesced(), "already quiescent status remains reusable");
}

void CaseOperationLongWaitSlice() {
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 0);
  auto start = std::chrono::steady_clock::now();
  std::thread publisher([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    meta->ReportData(&status, StatusCode::SUCCESS, "empty");
    meta->Seal();
  });
  meta->Quiescence(1001);
  auto elapsed = std::chrono::steady_clock::now() - start;
  publisher.join();
  Require(meta->Quiescence() == QuiescenceState::Quiesced, "long slice accepts a valid timespec");
  Require(elapsed >= std::chrono::milliseconds(10),
          "long slice did not fail immediately with EINVAL");
}

void CaseOperationQuiescence() {
  TransferStatus status;
  Require(status.Quiesced(), "never submitted status is quiescent");
  auto ledger = std::make_shared<SubmissionLedger>(8);
  LedgerVerbs verbs;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 1);
  meta->target.emplace_back(ledger, 0);
  status.SetQuiescenceCallback([meta](int slice) { return meta->Quiescence(slice); });
  status.SetCode(StatusCode::IN_PROGRESS);
  Require(status.Quiescence() == QuiescenceState::Pending, "unsealed operation is pending");
  // An invalid target must remain unread while the posting phase is open.
  meta->target[0] = {std::weak_ptr<SubmissionLedger>{}, 7};
  Require(status.Quiescence() == QuiescenceState::Pending, "unsealed target never read");
  auto posted = PostLedger(*ledger, verbs, 1, meta);
  meta->target[0] = {ledger, posted.dataSeq};
  ledger->OnSqCompletion(kSqTag | posted.dataSeq, IBV_WC_SUCCESS, "done");
  Require(status.Succeeded() && !status.Quiesced(), "result alone does not seal submission");
  meta->Seal();
  Require(status.WaitQuiesced(10), "sealed published covered operation");
  ledger.reset();
  Require(status.Quiesced(), "certified evidence survives ledger destruction");
  status.SetCode(StatusCode::INIT);
  auto next = std::make_shared<CqCallbackMeta>(&status, 2, 0);
  status.SetQuiescenceCallback([next](int slice) { return next->Quiescence(slice); });
  status.SetCode(StatusCode::IN_PROGRESS);
  bool threw = false;
  try {
    status.CheckReusable();
  } catch (const std::logic_error&) {
    threw = true;
  }
  Require(threw && status.Code() == StatusCode::IN_PROGRESS,
          "reuse refuses without state mutation");
  status.SetCode(StatusCode::INIT);
  Require(!status.Quiesced(), "SetCode INIT cannot clear old proof");
  next->ReportData(&status, StatusCode::SUCCESS, "empty");
  next->Seal();
  Require(status.Quiesced(), "empty published operation");
}

void CaseOperationNotificationPublication() {
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 0, true);
  status.SetQuiescenceCallback([meta](int slice) { return meta->Quiescence(slice); });
  status.SetCode(StatusCode::IN_PROGRESS);
  meta->Seal();
  meta->ReportData(&status, StatusCode::SUCCESS, "data");
  Require(status.InProgress() && !status.Quiesced(), "notification still pending");
  meta->ReportNotification(&status, StatusCode::SUCCESS, "notification");
  Require(status.Succeeded() && status.Quiesced(), "publication completes both paths");
  TransferStatus lost;
  auto ledger = std::make_shared<SubmissionLedger>(8);
  auto pending = std::make_shared<CqCallbackMeta>(&lost, 2, 0);
  pending->target.emplace_back(ledger, 1);
  lost.SetQuiescenceCallback([pending](int slice) { return pending->Quiescence(slice); });
  pending->ReportData(&lost, StatusCode::ERR_RDMA_OP, "failed");
  pending->Seal();
  Require(lost.WaitQuiescence(1) == QuiescenceState::Pending, "absolute wait deadline");
  ledger.reset();
  Require(lost.Quiescence() == QuiescenceState::Unprovable, "uncertified lost source");
}

void CaseFreezeAfterCqTakesMeta() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus lost, taken, remaining;
  lost.SetCode(StatusCode::IN_PROGRESS);
  taken.SetCode(StatusCode::IN_PROGRESS);
  remaining.SetCode(StatusCode::IN_PROGRESS);
  bool unpublishedAtFailAll = false;
  auto first = std::shared_ptr<CqCallbackMeta>(new CqCallbackMeta(&lost, 1, 1), [&](auto* meta) {
    // The lost-tail snapshot dies after its result, before the successor advances the prefix.
    ledger.FailAll(StatusCode::ERR_RDMA_OP, "frozen during CQ processing", true);
    unpublishedAtFailAll = taken.InProgress() && remaining.Failed();
    delete meta;
  });
  auto second = std::make_shared<CqCallbackMeta>(&taken, 2, 1);
  auto third = std::make_shared<CqCallbackMeta>(&remaining, 3, 1);
  PostLedger(ledger, verbs, 1, first);
  PostLedger(ledger, verbs, 1, second);
  PostLedger(ledger, verbs, 1, third);
  first.reset();
  ledger.OnSqCompletion(kSqTag | 2, IBV_WC_SUCCESS, "taken before freeze");
  Require(unpublishedAtFailAll, "empty ring meta means CQ owns publication, not that it finished");
  Require(taken.Succeeded() && lost.Failed() && remaining.Failed(),
          "taken CQ meta still publishes");
  Require(ledger.Frozen() && ledger.Done() == 0 && ledger.Used() == 3, "frozen CAS never refunds");
}

void CaseLostNotificationCompletion() {
  auto ledger = std::make_shared<SubmissionLedger>(8);
  LedgerVerbs verbs;
  TransferStatus status, next;
  status.SetCode(StatusCode::IN_PROGRESS);
  auto data = std::make_shared<CqCallbackMeta>(&status, 1, 1, true);
  data->target.emplace_back(ledger, 1);
  status.SetQuiescenceCallback([data](int slice) { return data->Quiescence(slice); });
  PostLedger(*ledger, verbs, 1, data);
  Require(ledger->Reserve(1, 0, nullptr), "notification credit");
  ibv_send_wr notify{};
  notify.opcode = IBV_WR_SEND;
  ledger->PostToSq(nullptr, &notify, 1, SubmissionLedger::Kind::Notify, nullptr, 0, true, 1, 0,
                   verbs);
  data->ReportNotification(&status, StatusCode::SUCCESS, "posted");
  data->Seal();
  auto successor = std::make_shared<CqCallbackMeta>(&next, 2, 1);
  PostLedger(*ledger, verbs, 1, successor);
  ledger->OnSqCompletion(kSqTag | 3, IBV_WC_SUCCESS, "successor");
  Require(status.Failed() && status.Quiesced() && next.Succeeded(),
          "lost notify never reports twice");
  Require(ledger->Used() == 0 && ledger->Done() == 3, "notify credits follow the prefix once");
}

void CaseScanMismatchFailsOutstanding() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus first, current;
  auto a = std::make_shared<CqCallbackMeta>(&first, 1, 1);
  auto b = std::make_shared<CqCallbackMeta>(&current, 2, 1);
  PostLedger(ledger, verbs, 1, a);
  PostLedger(ledger, verbs, 1, b);
  SubmissionLedgerTest::CorruptEntry(ledger, 1);
  ledger.OnSqCompletion(kSqTag | 2, IBV_WC_SUCCESS, "bad gap");
  Require(first.Failed() && current.Succeeded() && ledger.Frozen(),
          "scan freeze fails unclaimed results");
  Require(ledger.Used() == 2 && ledger.Done() == 0, "untrusted gap has no credit evidence");
}

void CaseRdmaBatchClampsOrdinaryCapacity() {
  EpPair ep{};
  ep.ledger = std::make_shared<SubmissionLedger>(4);
  ep.local.handle.maxSge = 1;
  ep.local.maxMsgSize = 4096;
  ibv_context context{};
  ibv_qp qp{};
  qp.context = &context;
  std::vector<int> batches;
  struct State {
    SubmissionLedger* ledger;
    std::vector<int>* batches;
  } state{ep.ledger.get(), &batches};
  qp.qp_context = &state;
  context.ops.post_send = [](ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
    auto& state = *static_cast<State*>(qp->qp_context);
    int n = 0;
    ibv_send_wr* tail = wr;
    for (; tail->next; tail = tail->next) ++n;
    state.batches->push_back(n + 1);
    state.ledger->OnSqCompletion(tail->wr_id, IBV_WC_SUCCESS, "early batch completion");
    return 0;
  };
  ep.local.ibvHandle.qp = &qp;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 7);
  mori::application::RdmaMemoryRegion local{.addr = 0x10000, .lkey = 1, .length = 4096};
  mori::application::RdmaMemoryRegion remote{.addr = 0x20000, .rkey = 2, .length = 4096};
  SizeVec offsets{0, 64, 128, 192, 256, 320, 384}, sizes(7, 16);
  RdmaTransferControl control{};
  control.disableMerge = true;
  auto result = RdmaBatchReadWrite({ep}, {local}, {remote}, offsets, offsets, sizes, meta, 1, false,
                                   -1, control);
  Require(!result.Failed() && batches == std::vector<int>({3, 3, 1}), "batch clamp uses N-1");
  Require(status.Succeeded() && ep.ledger->Used() == 0, "clamped batches keep logical accounting");
}

void CaseRdmaEarlyReturnsSealOperation() {
  EpPair ep{};
  ep.ledger = std::make_shared<SubmissionLedger>(8);
  ep.ledger->Close();
  ep.local.handle.maxSge = 1;
  ep.local.maxMsgSize = 4096;
  mori::application::RdmaMemoryRegion local{.addr = 0x10000, .lkey = 1, .length = 4096};
  mori::application::RdmaMemoryRegion remote{.addr = 0x20000, .rkey = 2, .length = 4096};
  RdmaBackendConfig config{};
  config.enableNotification = false;
  RdmaBackendSession session(config, {local}, {remote}, {ep}, nullptr, nullptr);
  TransferStatus status;
  session.ReadWrite(4096, 0, 16, &status, 1, false);
  Require(status.Failed() && status.Quiesced(), "preflight failure seals and publishes");
  session.BatchReadWrite({}, {}, {}, &status, 2, false);
  Require(status.Succeeded() && status.Quiesced(), "empty batch seals and publishes");
  session.ReadWrite(0, 0, 16, &status, 3, false);
  Require(
      status.Failed() && status.Quiesced() && status.Message().find("closed") != std::string::npos,
      "closed endpoint reports admission failure and remains quiesced");
}

void CaseOperationOwnershipHasNoCycle() {
  std::weak_ptr<SubmissionLedger> ledgerWeak;
  std::weak_ptr<CqCallbackMeta> metaWeak;
  {
    TransferStatus status;
    auto ledger = std::make_shared<SubmissionLedger>(8);
    ledgerWeak = ledger;
    auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 1);
    metaWeak = meta;
    meta->target.emplace_back(ledger, 1);
    status.SetQuiescenceCallback([meta](int slice) { return meta->Quiescence(slice); });
    LedgerVerbs verbs;
    PostLedger(*ledger, verbs, 1, meta);
    ledger->FailAll(StatusCode::ERR_RDMA_OP, "engine stopped", true);
    meta->Seal();
    ledger.reset();
    Require(status.Quiescence() == QuiescenceState::Unprovable, "lost source remains unprovable");
  }
  Require(ledgerWeak.expired() && metaWeak.expired(),
          "ledger and operation release without a cycle");
}

void CaseSubmissionLedgerBasic() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 101, 3);
  auto result = PostLedger(ledger, verbs, 3, meta);
  Require(result.dataSeq == 3 && result.signaledSeq == 3, "contiguous SQ sequences");
  Require(ledger.Used() == 3, "posted WRs retain credits");
  ledger.OnSqCompletion(kSqTag | 3, IBV_WC_SUCCESS, "done");
  Require(status.Succeeded() && ledger.Done() == 3 && ledger.Used() == 0, "CQ retires prefix");
  // Reuse the ring repeatedly, then inject a duplicate from an older generation.
  for (int i = 0; i < 20; ++i) {
    TransferStatus next;
    auto m = std::make_shared<CqCallbackMeta>(&next, 102 + i, 3);
    auto posted = PostLedger(ledger, verbs, 3, m);
    ledger.OnSqCompletion(kSqTag | posted.dataSeq, IBV_WC_SUCCESS, "done");
    Require(next.Succeeded() && ledger.Used() == 0, "ring wrap preserves meta and credits");
  }
  auto done = ledger.Done();
  Require(!ledger.OnSqCompletion(kSqTag | 3, IBV_WC_SUCCESS, "duplicate"), "old CQE freezes");
  Require(ledger.Frozen() && ledger.Done() == done, "frozen prefix is monotonic");
}

void CaseSubmissionLedgerSharedQpSignals() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus a, b;
  auto ma = std::make_shared<CqCallbackMeta>(&a, 1, 4);
  auto mb = std::make_shared<CqCallbackMeta>(&b, 2, 2);
  Require(!PostLedger(ledger, verbs, 2, ma, false).signaledSeq, "short unsignaled batch");
  auto second = PostLedger(ledger, verbs, 2, mb, false);
  Require(second.signaledSeq == 4, "QP threshold combines concurrent operations");
  ledger.OnSqCompletion(kSqTag | 4, IBV_WC_SUCCESS, "covered");
  Require(ledger.Used() == 0 && b.Succeeded(), "physical credits span operations");
  // A's logical credit still belongs to A, despite B covering its physical WRs.
  Require(!a.Succeeded(), "cross-operation evidence cannot complete logical results");
  auto tail = PostLedger(ledger, verbs, 2, ma, true, 4);
  ledger.OnSqCompletion(kSqTag | tail.dataSeq, IBV_WC_SUCCESS, "tail");
  Require(a.Succeeded(), "own signaled tail completes logical result");
}

void CaseSubmissionLedgerTrackingFreezeFailsOutstanding() {
  for (uint64_t invalid : {0, 1, 3}) {
    SubmissionLedger ledger(8);
    LedgerVerbs verbs;
    TransferStatus first, pending;
    auto a = std::make_shared<CqCallbackMeta>(&first, 1, 1);
    auto b = std::make_shared<CqCallbackMeta>(&pending, 2, 1);
    PostLedger(ledger, verbs, 1, a);
    ledger.OnSqCompletion(kSqTag | 1, IBV_WC_SUCCESS, "first");
    PostLedger(ledger, verbs, 1, b);
    // Duplicate CQE and a CQE with no published entry both invalidate tracking.
    ledger.OnSqCompletion(kSqTag | invalid, IBV_WC_SUCCESS, "invalid");
    Require(ledger.Frozen() && pending.Failed(), "anomalous freeze terminates outstanding results");
    Require(pending.Message() == "completion tracking frozen", "tracking failure diagnosis");
    Require(ledger.Used() == 1 && ledger.Done() == 1, "freeze cannot certify or refund DMA");
    ledger.OnSqCompletion(kSqTag | 2, IBV_WC_SUCCESS, "late");
    Require(pending.Failed() && ledger.Used() == 1, "later CQE ignored after terminal freeze");
  }
}

void CaseSubmissionLedgerFailAll() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus a, b;
  auto ma = std::make_shared<CqCallbackMeta>(&a, 1, 2);
  auto mb = std::make_shared<CqCallbackMeta>(&b, 2, 1);
  PostLedger(ledger, verbs, 2, ma);
  PostLedger(ledger, verbs, 1, mb);
  Require(ledger.FailAll(StatusCode::ERR_RDMA_OP, "qp died") == 2, "fail every operation");
  Require(a.Failed() && b.Failed(), "failure published");
  Require(ledger.Used() == 3 && ledger.Done() == 0, "FailAll cannot retire physical credits");
  ledger.OnSqCompletion(kSqTag | 2, IBV_WC_WR_FLUSH_ERR, "flush");
  ledger.OnSqCompletion(kSqTag | 3, IBV_WC_WR_FLUSH_ERR, "flush");
  Require(ledger.Used() == 0 && ledger.Done() == 3 && !ledger.Frozen(), "closed CQ still retires");
  Require(!ledger.Reserve(1, 0, nullptr), "closed ledger rejects");
}

void CaseSubmissionLedgerPostCommitRace() {
  SubmissionLedger ledger(8);
  LedgerVerbs verbs;
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 1, 2);
  verbs.post = [&](ibv_send_wr* wr, ibv_send_wr**) {
    ledger.OnSqCompletion(wr->next->wr_id, IBV_WC_SUCCESS, "CQ before post return");
    return 0;
  };
  PostLedger(ledger, verbs, 2, meta);
  Require(status.Succeeded() && ledger.Used() == 0 && ledger.Done() == 2, "early CQ evidence");
  verbs.post = [](ibv_send_wr* wr, ibv_send_wr** bad) {
    *bad = wr->next;
    return ENOMEM;
  };
  TransferStatus partial;
  auto mp = std::make_shared<CqCallbackMeta>(&partial, 2, 3);
  auto result = PostLedger(ledger, verbs, 3, mp);
  Require(result.posted == 1 && result.dataSeq == 3 && result.signaledSeq == 0,
          "partial post commits only actual prefix");
  Require(ledger.Used() == 1, "refund only unposted suffix");
  verbs.post = nullptr;
  TransferStatus next;
  auto mn = std::make_shared<CqCallbackMeta>(&next, 3, 1);
  auto tail = PostLedger(ledger, verbs, 1, mn);
  Require(tail.dataSeq == 4, "unposted sequence reused");
  ledger.OnSqCompletion(kSqTag | 4, IBV_WC_SUCCESS, "covered partial");
  Require(ledger.Used() == 0 && next.Succeeded(), "prefix credits exactly once");
}

void CaseSubmissionLedgerLostCompletion() {
  SubmissionLedger ledger(16);
  LedgerVerbs verbs;
  TransferStatus a, b, c;
  auto ma = std::make_shared<CqCallbackMeta>(&a, 1, 1);
  auto mb = std::make_shared<CqCallbackMeta>(&b, 2, 1);
  auto mc = std::make_shared<CqCallbackMeta>(&c, 3, 1);
  PostLedger(ledger, verbs, 1, ma);
  PostLedger(ledger, verbs, 1, mb);
  PostLedger(ledger, verbs, 1, mc);
  ledger.OnSqCompletion(kSqTag | 3, IBV_WC_SUCCESS, "successor");
  Require(a.Failed() && b.Failed() && c.Succeeded(),
          "lost signaled results fail, never infer success");
  Require(ledger.Done() == 3 && ledger.Used() == 0 && !ledger.Frozen(), "successor covers gap");
}

void CaseSubmissionLedgerCloseRaceStress() {
  for (int run = 0; run < 100; ++run) {
    SubmissionLedger ledger(64);
    LedgerVerbs verbs;
    std::vector<TransferStatus> statuses(24);
    std::vector<std::shared_ptr<CqCallbackMeta>> metas;
    for (int i = 0; i < 24; ++i) {
      metas.push_back(std::make_shared<CqCallbackMeta>(&statuses[i], i, 1));
      PostLedger(ledger, verbs, 1, metas.back());
    }
    std::thread cq([&] {
      for (int i = 1; i <= 24; ++i) ledger.OnSqCompletion(kSqTag | i, IBV_WC_SUCCESS, "done");
    });
    std::thread fail([&] { ledger.FailAll(StatusCode::ERR_RDMA_OP, "fatal"); });
    cq.join();
    fail.join();
    for (auto& status : statuses) Require(status.Succeeded() || status.Failed(), "terminal");
    Require(ledger.Used() == 0 && ledger.Done() == 24, "concurrent FailAll never refunds twice");
  }
  SubmissionLedger ledger(16);
  LedgerVerbs verbs;
  std::vector<TransferStatus> statuses(12);
  std::vector<std::thread> submitters;
  std::vector<uint64_t> observed;
  verbs.post = [&](ibv_send_wr* wr, ibv_send_wr**) {
    observed.push_back(SqSequence(wr->wr_id));
    return 0;
  };
  for (int i = 0; i < 12; ++i)
    submitters.emplace_back([&, i] {
      PostLedger(ledger, verbs, 1, std::make_shared<CqCallbackMeta>(&statuses[i], i, 1));
    });
  for (auto& thread : submitters) thread.join();
  for (int i = 0; i < 12; ++i) {
    Require(observed[i] == i + 1, "concurrent posts serialized in SQ order");
    ledger.OnSqCompletion(kSqTag | (i + 1), IBV_WC_SUCCESS, "done");
  }
}

EpPair MakeAsyncEventEp(ibv_qp* qp, ibv_cq* cq, uint32_t qpn) {
  EpPair ep{};
  ep.local.ibvHandle.qp = qp;
  ep.local.ibvHandle.cq = cq;
  ep.local.handle.qpn = qpn;
  return ep;
}

void CaseAsyncEventScopeMatching() {
  // Two NICs that handed out the same qp_num, which is the normal case: qp_num is
  // only unique within one device.
  ibv_context ctxA{};
  ibv_context ctxB{};
  ibv_qp qpA{};
  ibv_qp qpB{};
  qpA.context = &ctxA;
  qpB.context = &ctxB;
  ibv_cq cqA{};
  ibv_cq cqB{};

  constexpr uint32_t kSharedQpn = 137;
  const EpPair epA = MakeAsyncEventEp(&qpA, &cqA, kSharedQpn);
  const EpPair epB = MakeAsyncEventEp(&qpB, &cqB, kSharedQpn);

  QpErrorEvent qpEvent;
  qpEvent.scope = QpErrorEvent::Scope::kQueuePair;
  qpEvent.qpNum = kSharedQpn;
  qpEvent.context = &ctxA;
  Require(EndpointAffectedByAsyncEvent(qpEvent, epA), "QP event must hit its own endpoint");
  Require(!EndpointAffectedByAsyncEvent(qpEvent, epB),
          "QP event must not hit a same-qpn endpoint on another device");

  QpErrorEvent cqEvent;
  cqEvent.scope = QpErrorEvent::Scope::kCompletionQueue;
  cqEvent.cq = &cqB;
  cqEvent.context = &ctxB;
  Require(EndpointAffectedByAsyncEvent(cqEvent, epB), "CQ event must hit the endpoint on that CQ");
  Require(!EndpointAffectedByAsyncEvent(cqEvent, epA), "CQ event must not hit another CQ");

  QpErrorEvent deviceEvent;
  deviceEvent.scope = QpErrorEvent::Scope::kDevice;
  deviceEvent.context = &ctxA;
  Require(EndpointAffectedByAsyncEvent(deviceEvent, epA),
          "device event must hit every endpoint on its context");
  Require(!EndpointAffectedByAsyncEvent(deviceEvent, epB),
          "device event must not cross into another context");

  const EpPair unbuilt = MakeAsyncEventEp(nullptr, nullptr, kSharedQpn);
  Require(!EndpointAffectedByAsyncEvent(qpEvent, unbuilt),
          "an endpoint with no verbs QP must never match");
}

EpPair MakeSqAdmissionEp(int capacity, int used) {
  EpPair ep{};
  ep.ledger = std::make_shared<SubmissionLedger>(capacity);
  Require(ep.ledger->Reserve(used, 0, nullptr), "initial SQ reservation");
  return ep;
}

bool WaitForSqAdmissionWaiter(const EpPair& ep) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (std::chrono::steady_clock::now() < deadline) {
    if (ep.ledger->Waiters()) return true;
    std::this_thread::yield();
  }
  return false;
}

void CaseSqAdmissionReleaseWakesWaiter() {
  auto ep = MakeSqAdmissionEp(2, 1);
  bool ok = false;
  std::thread worker([&] { ok = detail::TryReserveSqDepthForTesting(ep, 1); });
  bool waiting = WaitForSqAdmissionWaiter(ep);
  ep.ledger->CancelReservation(1);
  worker.join();
  Require(waiting && ok && ep.ledger->Used() == 1 && !ep.ledger->Waiters(), "credit wakeup");
}

void CaseSqAdmissionClosedWakesWaiter() {
  auto ep = MakeSqAdmissionEp(2, 1);
  bool ok = true;
  std::thread worker([&] { ok = detail::TryReserveSqDepthForTesting(ep, 1); });
  bool waiting = WaitForSqAdmissionWaiter(ep);
  ep.ledger->Close(true);
  worker.join();
  Require(waiting && !ok && !ep.ledger->Waiters(), "freeze wakes admission");
  Require(ep.ledger->Used() == 1, "freeze cannot release DMA credits");
}

void CaseWrIdNamespaceHelpers() {
  Require(IsSqWrId(kSqTag | 42) && SqSequence(kSqTag | 42) == 42, "SQ namespace");
  Require(!IsSqWrId(4096), "RQ namespace stays untagged");
  bool rejected = false;
  try {
    SubmissionLedger ledger(1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "N less than two rejected");
  SubmissionLedger ledger(8);
  Require(ledger.Reserve(7, 0, nullptr) && !ledger.Reserve(1, 0, nullptr), "N-1 ordinary limit");
}

void CaseRdmaBackendConfigChunkingFields() {
  RdmaBackendConfig defaultCfg{};
  Require(defaultCfg.chunkBytes == 65536, "default chunkBytes should be 64KB");

  RdmaBackendConfig cfg{4, -1, 2, PollCqMode::POLLING, true, 2048, true, 65536, 32, 2};
  Require(cfg.qpPerTransfer == 4, "qpPerTransfer constructor field mismatch");
  Require(cfg.postBatchSize == -1, "postBatchSize constructor field mismatch");
  Require(cfg.numWorkerThreads == 2, "numWorkerThreads constructor field mismatch");
  Require(cfg.enableNotification, "enableNotification constructor field mismatch");
  Require(cfg.notifPerQp == 2048, "notifPerQp constructor field mismatch");
  Require(cfg.enableTransferChunking, "enableTransferChunking constructor field mismatch");
  Require(cfg.chunkBytes == 65536, "chunkBytes constructor field mismatch");
  Require(cfg.maxChunksPerTransfer == 32, "maxChunksPerTransfer constructor field mismatch");
  Require(cfg.numNicsPerTransfer == 2, "numNicsPerTransfer constructor field mismatch");
}

void CaseResolveRequestedNics() {
  RdmaBackendConfig cfg{};
  cfg.numNicsPerTransfer = 4;

  TopoKey cpu0{0, MemoryLocationType::CPU, 0};
  TopoKey cpu1{1, MemoryLocationType::CPU, 1};
  TopoKey gpu0{0, MemoryLocationType::GPU, -1};

  Require(ResolveRequestedNics(cfg, cpu0, cpu1) == 4,
          "host-host session should honor configured NIC count");
  Require(ResolveRequestedNics(cfg, gpu0, cpu0) == 1, "GPU-local session should force single-NIC");
  Require(ResolveRequestedNics(cfg, cpu0, gpu0) == 1, "GPU-remote session should force single-NIC");
}

void RequireChunkPlanCoverage(const std::vector<std::pair<uint64_t, uint32_t>>& plan,
                              uint32_t total) {
  uint64_t expectedOffset = 0;
  uint64_t totalLength = 0;
  for (const auto& [offset, length] : plan) {
    Require(offset == expectedOffset, "chunk plan must be contiguous");
    expectedOffset += length;
    totalLength += length;
  }
  Require(totalLength == total, "chunk plan total length mismatch");
}

void CasePlanChunksBoundaries() {
  {
    auto plan = PlanChunks(0, 65536, 8);
    Require(plan.empty(), "zero-length plan should be empty");
  }
  {
    auto plan = PlanChunks(65536, 0, 8);
    Require(plan.size() == 1, "chunkBytes==0 should disable splitting");
    Require(plan[0].first == 0 && plan[0].second == 65536, "unsplit plan mismatch");
  }
  {
    auto plan = PlanChunks(65536, 65536, 8);
    Require(plan.size() == 1, "total==chunkBytes should not split");
    Require(plan[0].first == 0 && plan[0].second == 65536, "boundary non-split mismatch");
  }
  {
    auto plan = PlanChunks(65537, 65536, 8);
    Require(plan.size() == 2, "chunkBytes+1 should split into 2 chunks");
    RequireChunkPlanCoverage(plan, 65537);
    Require(plan[0].second == 32769 && plan[1].second == 32768,
            "unexpected chunkBytes+1 split geometry");
  }
  {
    auto plan = PlanChunks(1024 * 1024, 131072, 4);
    Require(plan.size() == 4, "maxChunks must cap chunk count");
    RequireChunkPlanCoverage(plan, 1024 * 1024);
    for (const auto& [_, length] : plan) {
      Require(length == 262144, "capped chunk plan should rebalance evenly");
    }
  }
  {
    auto plan = PlanChunks(65536, 65536, 0);
    Require(plan.empty(), "invalid maxChunks should produce empty plan");
  }
}

void CaseChunkGeometrySingleSgeCountMatchesPlanner() {
  constexpr uint64_t kKiB = 1024ull;
  constexpr uint64_t kMiB = 1024ull * kKiB;
  constexpr uint64_t kGiB = 1024ull * kMiB;

  const std::vector<size_t> chunkBytesValues{4 * kKiB, 64 * kKiB, kMiB, 0};
  const std::vector<int> maxChunksValues{1, 4, 64, 4096};
  const std::vector<uint64_t> maxMessageValues{8 * kKiB, kMiB, 0x80000000ull};
  const std::vector<uint64_t> totals{0, 1, 4095, 4096, 4097, 64 * kKiB, kMiB, kGiB};

  for (size_t chunkBytes : chunkBytesValues) {
    for (int maxChunks : maxChunksValues) {
      for (uint64_t maxMessageSize : maxMessageValues) {
        for (uint64_t total : totals) {
          const uint64_t actual = CountChunksForSize(total, chunkBytes, maxChunks, maxMessageSize);

          uint64_t expected = 1;
          const bool splittable =
              total != 0 && ((chunkBytes > 0 && total > chunkBytes) || total > maxMessageSize);
          if (splittable) {
            std::vector<ibv_sge> sges{
                ibv_sge{.addr = 0x1000000000ull, .length = static_cast<uint32_t>(total), .lkey = 1},
            };
            std::vector<ChunkedSgeSegment> segments;
            PlanSgeStreamChunks(segments, sges, total, chunkBytes, maxChunks, maxMessageSize);
            expected = segments.size();
          }

          Require(actual == expected,
                  "CountChunksForSize mismatch for total=" + std::to_string(total) +
                      " chunkBytes=" + std::to_string(chunkBytes) +
                      " maxChunks=" + std::to_string(maxChunks) + " maxMessageSize=" +
                      std::to_string(maxMessageSize) + " expected=" + std::to_string(expected) +
                      " actual=" + std::to_string(actual));
        }
      }
    }
  }
}

void CaseChunkGeometryProperties() {
  ChunkGeometry geometry = PlanChunkGeometry(6, 1, 4, 2);
  Require(geometry.finalCount == 4, "geometry repro should cap soft count at 4");
  Require(geometry.targetChunkBytes == 2, "geometry target should respect max message size");
  Require(CountChunksForSize(6, 1, 4, 2) == 3,
          "actual chunk count should be ceil(total/target), not finalCount");

  const std::vector<uint64_t> totals{1, 4096, 4097, 1024 * 1024, 17 * 1024 * 1024};
  const std::vector<uint64_t> maxMessages{2, 4096, 65536, 1024 * 1024};
  for (uint64_t total : totals) {
    for (uint64_t maxMessage : maxMessages) {
      ChunkGeometry g = PlanChunkGeometry(total, 4096, 64, maxMessage);
      Require(g.targetChunkBytes <= maxMessage,
              "target chunk bytes must not exceed max message size");
      Require(CountChunksForSize(total, 4096, 64, maxMessage) >= 1,
              "non-empty requests must emit at least one WR");
    }
  }
}

void CaseChunkCountSplitInvariant() {
  constexpr size_t kKiB = 1024ull;
  constexpr size_t kMiB = 1024ull * kKiB;
  const SizeVec sizes{0, 1, 4096, 4097, 64 * kKiB + 1, kMiB, 3 * kMiB + 7};
  constexpr size_t kChunkBytes = 64 * kKiB;
  constexpr int kMaxChunks = 64;
  constexpr uint64_t kMaxMessageSize = kMiB;

  auto countRange = [&](size_t begin, size_t end) {
    uint64_t sum = 0;
    for (size_t i = begin; i < end; ++i) {
      sum += CountChunksForSize(sizes[i], kChunkBytes, kMaxChunks, kMaxMessageSize);
    }
    return sum;
  };

  const uint64_t full = countRange(0, sizes.size());
  const std::vector<std::vector<size_t>> partitions{
      {0, sizes.size()},
      {0, 1, sizes.size()},
      {0, 2, 5, sizes.size()},
      {0, 3, 3, sizes.size()},
  };
  for (const auto& cuts : partitions) {
    uint64_t partitioned = 0;
    for (size_t i = 1; i < cuts.size(); ++i) {
      partitioned += countRange(cuts[i - 1], cuts[i]);
    }
    Require(partitioned == full, "chunk count must be invariant under worker splits");
  }
}

EpPair MakeRejectingEp(uint64_t maxMessageSize, uint32_t maxSge = 8) {
  EpPair ep{};
  ep.local.handle.maxSge = maxSge;
  ep.local.maxMsgSize = maxMessageSize;
  ep.ledger = std::make_shared<SubmissionLedger>(16);
  ep.ledger->Close();
  return ep;
}

void CaseRdmaTransferControlDisableMergeAndOwnsTotal() {
  EpPairVec eps{MakeRejectingEp(1024 * 1024)};
  mori::application::RdmaMemoryRegion localMr{
      .addr = 0x1000000000ull,
      .lkey = 1,
      .rkey = 0,
      .length = 4096,
  };
  mori::application::RdmaMemoryRegion remoteMr{
      .addr = 0x2000000000ull,
      .lkey = 0,
      .rkey = 2,
      .length = 4096,
  };
  SizeVec offsets{0, 1024, 2048};
  SizeVec sizes{1024, 1024, 1024};

  RdmaTransferControl control{};
  control.chunkBytes = 4096;
  control.maxChunks = 64;
  control.creditByWrCount = true;
  control.ownsTotalBatchSize = true;
  control.disableMerge = true;

  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 91, 99);
  RdmaOpRet ret = RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                                     std::vector<mori::application::RdmaMemoryRegion>{remoteMr},
                                     offsets, offsets, sizes, meta, 91, false, -1, control);
  Require(ret.Failed(), "rejecting EP should fail before post");
  Require(meta->totalBatchSize == 3, "disableMerge should keep one WR per request");
  Require(ret.message.find("requested=3") != std::string::npos,
          "disableMerge path should reserve three WRs; got: " + ret.message);

  control.disableMerge = false;
  meta = std::make_shared<CqCallbackMeta>(&status, 92, 99);
  ret = RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                           std::vector<mori::application::RdmaMemoryRegion>{remoteMr}, offsets,
                           offsets, sizes, meta, 92, false, -1, control);
  Require(ret.Failed(), "rejecting EP should fail before post");
  Require(meta->totalBatchSize == 1, "merge-enabled path should collapse contiguous requests");
  Require(ret.message.find("requested=1") != std::string::npos,
          "merge-enabled path should reserve one WR; got: " + ret.message);

  control.disableMerge = true;
  control.ownsTotalBatchSize = false;
  meta = std::make_shared<CqCallbackMeta>(&status, 93, 42);
  ret = RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                           std::vector<mori::application::RdmaMemoryRegion>{remoteMr}, offsets,
                           offsets, sizes, meta, 93, false, -1, control);
  Require(ret.Failed(), "rejecting EP should fail before post");
  Require(meta->totalBatchSize == 42,
          "worker control must not overwrite dispatcher-owned totalBatchSize");
  Require(ret.message.find("requested=3") != std::string::npos,
          "worker no-merge path should still post one WR per request; got: " + ret.message);
}

void CaseRdmaRejectsSingleRequestLargerThanUint32() {
  EpPairVec eps{MakeRejectingEp(1024 * 1024)};
  const size_t tooLarge = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  mori::application::RdmaMemoryRegion localMr{
      .addr = 0x1000000000ull,
      .lkey = 1,
      .rkey = 0,
      .length = tooLarge,
  };
  mori::application::RdmaMemoryRegion remoteMr{
      .addr = 0x2000000000ull,
      .lkey = 0,
      .rkey = 2,
      .length = tooLarge,
  };

  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 94, 1);
  RdmaTransferControl control{};
  RdmaOpRet ret =
      RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                         std::vector<mori::application::RdmaMemoryRegion>{remoteMr}, SizeVec{0},
                         SizeVec{0}, SizeVec{tooLarge}, meta, 94, false, -1, control);

  Require(ret.Failed(), "single requests larger than uint32 SGE length must be rejected");
  Require(ret.code == StatusCode::ERR_INVALID_ARGS, "oversized single request status mismatch");
  Require(ret.message.find("UINT32_MAX") != std::string::npos,
          "oversized single request error should mention UINT32_MAX");
}

void CaseChunkingDisabledOversizedWrReturnsError() {
  constexpr size_t kMiB = 1024ull * 1024ull;

  EpPair ep{};
  ep.local.handle.maxSge = 2;
  ep.local.maxMsgSize = 3 * kMiB;
  ep.ledger = std::make_shared<SubmissionLedger>(65536);
  ep.ledger->Close();
  EpPairVec eps{ep};

  mori::application::RdmaMemoryRegion localMr{
      .addr = 0x1000000000ull,
      .lkey = 1,
      .rkey = 0,
      .length = 10 * kMiB,
  };
  mori::application::RdmaMemoryRegion remoteMr{
      .addr = 0x2000000000ull,
      .lkey = 0,
      .rkey = 2,
      .length = 4 * kMiB,
  };
  SizeVec localOffsets{0, 8 * kMiB};
  SizeVec remoteOffsets{0, 2 * kMiB};
  SizeVec sizes{2 * kMiB, 2 * kMiB};

  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 77, static_cast<int>(sizes.size()));
  RdmaOpRet ret =
      RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                         std::vector<mori::application::RdmaMemoryRegion>{remoteMr}, localOffsets,
                         remoteOffsets, sizes, meta, 77, false /* isRead */, -1, 0, 1, false);

  Require(ret.Failed(), "oversized merged WR must fail before ibv_post_send when chunking is off");
  Require(ret.code == StatusCode::ERR_INVALID_ARGS, "unexpected status for oversized WR");
  Require(ret.message.find("max_msg_sz") != std::string::npos,
          "oversized WR error should mention max_msg_sz");
}

void RequireSgeSegmentCoverage(const std::vector<ChunkedSgeSegment>& segments,
                               const std::vector<ibv_sge>& sges, uint64_t total) {
  uint64_t sgeTotal = 0;
  for (const ibv_sge& sge : sges) sgeTotal += sge.length;
  Require(sgeTotal == total, "test SGE stream total length mismatch");

  uint64_t expectedOffset = 0;
  uint64_t totalLength = 0;
  uint64_t sgeStreamOffset = 0;
  size_t sgeIndex = 0;
  for (const ChunkedSgeSegment& segment : segments) {
    Require(segment.length > 0, "SGE chunk segment length must be non-zero");
    Require(segment.remoteOffset == expectedOffset, "SGE chunk segments must be contiguous");
    while (sgeIndex < sges.size() &&
           segment.remoteOffset >= sgeStreamOffset + sges[sgeIndex].length) {
      sgeStreamOffset += sges[sgeIndex].length;
      ++sgeIndex;
    }
    Require(sgeIndex < sges.size(), "SGE chunk segment offset exceeds SGE stream");

    const ibv_sge& sge = sges[sgeIndex];
    const uint64_t sgeOffset = segment.remoteOffset - sgeStreamOffset;
    Require(sgeOffset + segment.length <= sge.length,
            "SGE chunk segment must not cross an SGE boundary");
    Require(segment.localAddr == sge.addr + sgeOffset,
            "SGE chunk segment local address must match stream offset inside source SGE");

    expectedOffset += segment.length;
    totalLength += segment.length;
  }
  Require(totalLength == total, "SGE chunk segment total length mismatch");
}

std::vector<ibv_sge> MakeFourGiBTwoSgeStream() {
  constexpr uint64_t kMiB = 1024ull * 1024ull;
  constexpr uint64_t kPage = 2ull * kMiB;
  constexpr uint64_t kFirstLen = 2047ull * kPage;
  return {
      ibv_sge{.addr = 0x1000000000ull, .length = static_cast<uint32_t>(kFirstLen), .lkey = 1},
      ibv_sge{.addr = 0x2000000000ull, .length = static_cast<uint32_t>(kPage), .lkey = 1},
  };
}

void CaseSgeStreamChunkingCoversAndRespectsLimits() {
  constexpr uint64_t kMiB = 1024ull * 1024ull;
  constexpr uint64_t kTotal = 0x100000000ull;
  constexpr uint64_t kMaxMessageBytes = 0x80000000ull;
  std::vector<ibv_sge> sges = MakeFourGiBTwoSgeStream();

  std::vector<ChunkedSgeSegment> segments;
  PlanSgeStreamChunks(segments, sges, kTotal, 64 * 1024, 64, kMaxMessageBytes);

  RequireSgeSegmentCoverage(segments, sges, kTotal);
  Require(segments.size() >= 64 && segments.size() <= 65,
          "4GiB repro should stay near 64 target chunks, allowing one SGE-boundary split");
  uint32_t maxLen = 0;
  for (const ChunkedSgeSegment& segment : segments) {
    Require(segment.length <= kMaxMessageBytes, "segment exceeds max_msg_sz");
    maxLen = std::max(maxLen, segment.length);
  }
  Require(maxLen == 64ull * kMiB, "target chunk geometry should remain 64MiB");
}

void CaseSgeStreamChunkingNeverCrossesSgeBoundary() {
  std::vector<ibv_sge> sges = {
      ibv_sge{.addr = 0x1000000000ull, .length = 30 * 4096, .lkey = 1},
      ibv_sge{.addr = 0x2000000000ull, .length = 70 * 4096, .lkey = 1},
  };

  std::vector<ChunkedSgeSegment> segments;
  PlanSgeStreamChunks(segments, sges, 100 * 4096, 60 * 4096, 2, 1024 * 4096);

  RequireSgeSegmentCoverage(segments, sges, 100 * 4096);
  Require(segments.size() == 3, "target chunk crossing an SGE boundary should split at boundary");
  for (const ChunkedSgeSegment& segment : segments) {
    bool insideAnySge = false;
    for (const ibv_sge& sge : sges) {
      const uint64_t sgeBegin = sge.addr;
      const uint64_t sgeEnd = sge.addr + sge.length;
      const uint64_t segBegin = segment.localAddr;
      const uint64_t segEnd = segment.localAddr + segment.length;
      if (segBegin >= sgeBegin && segEnd <= sgeEnd) {
        insideAnySge = true;
        break;
      }
    }
    Require(insideAnySge, "chunk planner must not emit a segment crossing SGE boundaries");
  }
}

void CaseChunkingExpandsMultiSgeWrBeforePost() {
  constexpr size_t kMiB = 1024ull * 1024ull;
  constexpr size_t kPage = 2 * kMiB;
  constexpr size_t kFirstLen = 2047 * kPage;
  constexpr size_t kTotal = kFirstLen + kPage;

  EpPair ep{};
  ep.local.handle.maxSge = 2;
  ep.local.maxMsgSize = 0x80000000ull;
  ep.ledger = std::make_shared<SubmissionLedger>(65536);
  ep.ledger->Close();
  EpPairVec eps{ep};

  mori::application::RdmaMemoryRegion localMr{
      .addr = 0x1000000000ull,
      .lkey = 1,
      .rkey = 0,
      .length = kTotal + (8ull * kMiB),
  };
  mori::application::RdmaMemoryRegion remoteMr{
      .addr = 0x2000000000ull,
      .lkey = 0,
      .rkey = 2,
      .length = kTotal,
  };
  SizeVec localOffsets{0, 8ull * kMiB};
  SizeVec remoteOffsets{0, kFirstLen};
  SizeVec sizes{kFirstLen, kPage};

  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 88, static_cast<int>(sizes.size()));
  RdmaOpRet ret = RdmaBatchReadWrite(eps, std::vector<mori::application::RdmaMemoryRegion>{localMr},
                                     std::vector<mori::application::RdmaMemoryRegion>{remoteMr},
                                     localOffsets, remoteOffsets, sizes, meta, 88,
                                     false /* isRead */, -1, 64 * 1024, 64, true);

  Require(ret.Failed(), "zero SQ capacity should reject after chunk expansion");
  Require(ret.message.find("requested=65") != std::string::npos,
          "4GiB multi-SGE repro should expand to 65 WRs before posting; got: " + ret.message);
  Require(meta->totalBatchSize == 65, "chunked completion accounting should use expanded WR count");
}

void CaseRdmaEndpointCarriesLocalMaxMsgSize() {
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

  auto ctx =
      std::make_unique<mori::application::RdmaContext>(mori::application::RdmaBackendType::IBVerbs);
  RdmaBackendConfig cfg{};
  RdmaManager mgr(cfg, ctx.get());
  (void)ctx.release();

  if (mgr.NumAvailDevices() == 0) throw TestSkip("requires at least one active RDMA device");
  mori::application::RdmaEndpoint endpoint = mgr.CreateEndpoint(0);
  const ibv_port_attr* portAttr =
      mgr.GetRdmaDeviceContext(0)->GetRdmaDevice()->GetPortAttr(endpoint.handle.portId);

  Require(portAttr != nullptr, "port attr must be available");
  Require(portAttr->max_msg_sz > 0, "port max_msg_sz must be non-zero");
  Require(endpoint.maxMsgSize == portAttr->max_msg_sz,
          "endpoint maxMsgSize should match local device max_msg_sz");
}

void CaseBuildDesiredQpCounts() {
  {
    auto counts = BuildDesiredQpCounts(4, 3);
    Require(counts.size() == 3, "counts size mismatch");
    Require(counts[0] == 2 && counts[1] == 1 && counts[2] == 1,
            "4 QPs over 3 ranks should distribute as 2/1/1");
  }
  {
    auto counts = BuildDesiredQpCounts(8, 1);
    Require(counts.size() == 1 && counts[0] == 8, "single-rank distribution mismatch");
  }
  {
    auto counts = BuildDesiredQpCounts(2, 4);
    Require(counts.size() == 4, "counts size mismatch for sparse distribution");
    Require(counts[0] == 1 && counts[1] == 1 && counts[2] == 0 && counts[3] == 0,
            "2 QPs over 4 ranks should distribute as 1/1/0/0");
  }
  {
    auto counts = BuildDesiredQpCounts(0, 4);
    int total = 0;
    for (int v : counts) total += v;
    Require(total == 0, "zero-QP distribution should sum to zero");
  }
}

void CaseInterleaveEndpointsByLocalDevice() {
  EpPairVec eps;
  auto add = [&](int ldevId) {
    EpPair ep{};
    ep.ldevId = ldevId;
    eps.push_back(ep);
  };
  add(0);
  add(0);
  add(1);
  add(1);
  add(2);

  {
    auto interleaved = InterleaveEndpointsByLocalDevice(eps, {0, 1, 2}, {2, 1, 1});
    Require(interleaved.size() == 4, "interleaved endpoint count mismatch");
    Require(interleaved[0].ldevId == 0 && interleaved[1].ldevId == 1 &&
                interleaved[2].ldevId == 2 && interleaved[3].ldevId == 0,
            "unexpected interleave order for 0/1/2 buckets");
  }
  {
    auto interleaved = InterleaveEndpointsByLocalDevice(eps, {1, 0}, {1, 2});
    Require(interleaved.size() == 3, "rank-limited interleave endpoint count mismatch");
    Require(interleaved[0].ldevId == 1 && interleaved[1].ldevId == 0 && interleaved[2].ldevId == 0,
            "unexpected interleave order for reordered buckets");
  }
}

void CaseUsesInlineOnly() {
  RdmaBackendConfig cfg{};
  Require(!UsesInlineOnly(cfg), "default config should keep executor-compatible path");

  cfg.enableTransferChunking = true;
  Require(!UsesInlineOnly(cfg), "single-NIC chunking should keep executor-compatible path");

  cfg.enableTransferChunking = false;
  cfg.numNicsPerTransfer = 2;
  Require(UsesInlineOnly(cfg), "multi-NIC should force inline-only path");
}

void CaseValidateRdmaTransferConfig() {
  {
    RdmaBackendConfig cfg{};
    ValidateRdmaTransferConfig(cfg);
  }
  {
    RdmaBackendConfig cfg{};
    cfg.maxChunksPerTransfer = 0;
    bool threw = false;
    try {
      ValidateRdmaTransferConfig(cfg);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Require(threw, "maxChunksPerTransfer<1 should be rejected");
  }
  {
    RdmaBackendConfig cfg{};
    cfg.numNicsPerTransfer = 0;
    bool threw = false;
    try {
      ValidateRdmaTransferConfig(cfg);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Require(threw, "numNicsPerTransfer<1 should be rejected");
  }
  {
    RdmaBackendConfig cfg{};
    cfg.enableTransferChunking = true;
    cfg.chunkBytes = 1024;
    bool threw = false;
    try {
      ValidateRdmaTransferConfig(cfg);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Require(threw, "chunkBytes<4096 should be rejected when chunking is enabled");
  }
  {
    RdmaBackendConfig cfg{};
    cfg.enableTransferChunking = true;
    cfg.chunkBytes = 4096;
    cfg.maxChunksPerTransfer = 1;
    cfg.numNicsPerTransfer = 1;
    ValidateRdmaTransferConfig(cfg);
  }
}

void CaseRdmaNotificationRejectsZeroNotifPerQp() {
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

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

void CaseRdmaBackendHasActiveDevicesReturnsFalseWhenNoDevice() {
  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  Require(!RdmaBackend::HasActiveDevices(),
          "RdmaBackend::HasActiveDevices() should return false when MORI_RDMA_DEVICES filters "
          "out all devices");
}

void CaseRdmaManagerThrowsWhenNoActiveDevices() {
  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  auto ctx =
      std::make_unique<mori::application::RdmaContext>(mori::application::RdmaBackendType::IBVerbs);
  RdmaBackendConfig cfg{};

  bool threw = false;
  try {
    RdmaManager mgr(cfg, ctx.get());
    (void)ctx.release();
    (void)mgr;
  } catch (const std::runtime_error&) {
    threw = true;
  }

  Require(threw, "RdmaManager ctor must throw when no active RDMA device is available");
}

void CaseCreateBackendRdmaThrowsByDefaultWhenNoRdmaDevice() {
  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "1");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_default_no_rdma_fallback", cfg);

  RdmaBackendConfig rdmaCfg{};
  bool threw = false;
  std::string what;
  try {
    engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }

  Require(threw,
          "CreateBackend(RDMA) must throw when no RDMA device is available and fallback is not "
          "explicitly enabled");
  Require(what.find("MORI_DISABLE_AUTO_XGMI=0") != std::string::npos,
          "no-RDMA error should mention MORI_DISABLE_AUTO_XGMI=0; got: " + what);
}

void CaseCreateBackendRdmaFallsBackToXgmiWhenOptedIn() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "0");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_rdma_fallback_to_xgmi", cfg);

  RdmaBackendConfig rdmaCfg{};
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);

  EngineDesc desc = engine.GetEngineDesc();
  Require(desc.port == internal::kXgmiOnlyFallbackPlaceholderPort,
          "XGMI-only fallback should set engine_desc.port to sentinel; got " +
              std::to_string(desc.port));

  engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  desc = engine.GetEngineDesc();
  Require(desc.port == internal::kXgmiOnlyFallbackPlaceholderPort,
          "repeated fallback should keep engine_desc.port at sentinel; got " +
              std::to_string(desc.port));
}

void CaseCreateBackendRdmaThrowsWhenOptedInButNoXgmi() {
  if (GetGpuCount() != 0) {
    throw TestSkip("requires a no-GPU host to deterministically exercise no-XGMI fallback failure");
  }

  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "0");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_no_rdma_no_xgmi", cfg);

  RdmaBackendConfig rdmaCfg{};
  bool threw = false;
  std::string what;
  try {
    engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }

  Require(threw, "CreateBackend(RDMA) must throw when neither RDMA nor XGMI is usable");
  Require(what.find("XGMI") != std::string::npos || what.find("GPU P2P") != std::string::npos,
          "no-XGMI error should mention XGMI/GPU P2P; got: " + what);
}

void CaseExplicitXgmiThenRdmaWithoutOptInStillThrows() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "1");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_explicit_xgmi_then_rdma_no_optin", cfg);

  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  RdmaBackendConfig rdmaCfg{};
  bool threw = false;
  std::string what;
  try {
    engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }

  Require(threw, "explicit XGMI must not bypass the RDMA fallback env gate");
  Require(what.find("MORI_DISABLE_AUTO_XGMI=0") != std::string::npos,
          "env-gate error should remain actionable; got: " + what);
}

void CaseExplicitXgmiThenRdmaWithOptInRefreshesPort() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "0");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_explicit_xgmi_then_rdma_optin", cfg);

  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  RdmaBackendConfig rdmaCfg{};
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);

  EngineDesc desc = engine.GetEngineDesc();
  Require(desc.port == internal::kXgmiOnlyFallbackPlaceholderPort,
          "opted-in RDMA fallback should refresh desc.port to sentinel after explicit XGMI; got " +
              std::to_string(desc.port));
}

void CaseRdmaBackendRefusesSentinelPortConfig() {
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "1");
  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = internal::kXgmiOnlyFallbackPlaceholderPort;
  IOEngine engine("test_rdma_sentinel_port_refused", cfg);

  RdmaBackendConfig rdmaCfg{};
  bool threw = false;
  std::string what;
  try {
    engine.CreateBackend(BackendType::RDMA, rdmaCfg);
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }

  Require(threw, "real RDMA backend must refuse the XGMI-only sentinel port");
  Require(what.find("sentinel") != std::string::npos || what.find("reserved") != std::string::npos,
          "sentinel port error should explain that the port is reserved; got: " + what);
}

void CaseSelectBackendReturnsNullForCrossNodeUnderXgmiOnly() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar noRdma("MORI_RDMA_DEVICES", kNoRdmaDeviceFilter);
  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "0");

  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_xgmi_only_cross_node", cfg);

  RdmaBackendConfig rdmaCfg{};
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);

  EngineDesc fakeRemote{};
  fakeRemote.key = "fake_cross_node_remote";
  fakeRemote.nodeId = "different-node";
  fakeRemote.hostname = "different-host";
  fakeRemote.host = "127.0.0.1";
  fakeRemote.port = internal::kXgmiOnlyFallbackPlaceholderPort;
  fakeRemote.pid = 0;
  engine.RegisterRemoteEngine(fakeRemote);

  auto local = RegisterGpuMemory(&engine, 4096, 0);
  MemoryDesc remote{};
  remote.engineKey = fakeRemote.key;
  remote.id = 999;
  remote.deviceId = 0;
  remote.deviceBusId = local.desc.deviceBusId;
  remote.data = local.desc.data;
  remote.size = local.desc.size;
  remote.loc = MemoryLocationType::GPU;

  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  engine.Write(local.desc, 0, remote, 0, 16, &status, uid);

  Require(status.Code() == StatusCode::ERR_BAD_STATE,
          "cross-node transfer under XGMI-only fallback should return ERR_BAD_STATE; got " +
              std::to_string(status.CodeUint32()) + ", msg='" + status.Message() + "'");
  Require(status.Message().find("No available backend") != std::string::npos,
          "cross-node transfer under XGMI-only fallback should be rejected by route layer; got: " +
              status.Message());
}

void CaseRdmaBackendCanHandleRejectsSentinelPortRemote() {
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

  ScopedEnvVar gate("MORI_DISABLE_AUTO_XGMI", "1");
  IOEngineConfig cfg{};
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("test_rdma_rejects_sentinel_remote", cfg);

  RdmaBackendConfig rdmaCfg{};
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);

  EngineDesc fakeRemote{};
  fakeRemote.key = "remote_xgmi_only";
  fakeRemote.nodeId = "remote-node";
  fakeRemote.hostname = "remote-host";
  fakeRemote.host = "10.255.255.255";
  fakeRemote.port = internal::kXgmiOnlyFallbackPlaceholderPort;
  fakeRemote.pid = 0;
  engine.RegisterRemoteEngine(fakeRemote);

  int localValue = 0;
  int remoteValue = 0;
  MemoryDesc local{};
  local.engineKey = engine.GetEngineDesc().key;
  local.id = 1;
  local.deviceId = -1;
  local.data = reinterpret_cast<uintptr_t>(&localValue);
  local.size = sizeof(localValue);
  local.loc = MemoryLocationType::CPU;

  MemoryDesc remote{};
  remote.engineKey = fakeRemote.key;
  remote.id = 2;
  remote.deviceId = -1;
  remote.data = reinterpret_cast<uintptr_t>(&remoteValue);
  remote.size = sizeof(remoteValue);
  remote.loc = MemoryLocationType::CPU;

  TransferStatus status;
  TransferUniqueId uid = engine.AllocateTransferUniqueId();
  engine.Write(local, 0, remote, 0, sizeof(localValue), &status, uid);

  Require(status.Code() == StatusCode::ERR_BAD_STATE,
          "RDMA backend must reject sentinel-port remote before Connect; got " +
              std::to_string(status.CodeUint32()) + ", msg='" + status.Message() + "'");
  Require(status.Message().find("No available backend") != std::string::npos,
          "sentinel-port remote should be rejected by route layer; got: " + status.Message());
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

// Writes kXfer bytes from srcBase + offset to dstBase + offset over the dma-buf
// registration path and checks the payload landed at dstBase + offset, not at
// dstBase. Both bases point at allocations of at least allocSize bytes.
void CheckDmabufInteriorWrite(ConnectedEnginePair& pair, void* srcBase, void* dstBase,
                              size_t allocSize, size_t offset, size_t xfer) {
  void* srcInterior = static_cast<char*>(srcBase) + offset;
  void* dstInterior = static_cast<char*>(dstBase) + offset;

  // Base regions get a sentinel (0x11); the src sub-range gets the payload (0xCD),
  // the dst sub-range is cleared (0x00). If the MR wrongly resolves to the base,
  // the write lands at dstBase and the checks below catch it two ways: the dst
  // sub-range stays 0x00 (payload missing) and/or the dst base sentinel is clobbered.
  HIP_RUNTIME_CHECK(hipMemset(srcBase, 0x11, allocSize));
  HIP_RUNTIME_CHECK(hipMemset(dstBase, 0x11, allocSize));
  HIP_RUNTIME_CHECK(hipMemset(srcInterior, 0xCD, xfer));
  HIP_RUNTIME_CHECK(hipMemset(dstInterior, 0x00, xfer));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  MemoryDesc srcDesc =
      pair.initiator->RegisterMemory(srcInterior, xfer, 0, MemoryLocationType::GPU);
  MemoryDesc dstDesc = pair.target->RegisterMemory(dstInterior, xfer, 0, MemoryLocationType::GPU);

  TransferStatus status;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Write(srcDesc, 0, dstDesc, 0, xfer, &status, uid);

  std::string err;
  Require(WaitTransferDone(&status, 5000, &err), "dmabuf interior transfer timeout: " + err);
  Require(status.Succeeded(),
          "dmabuf interior transfer failed: code=" + std::to_string(status.CodeUint32()) +
              ", msg='" + status.Message() + "'");

  std::vector<uint8_t> host(xfer);
  HIP_RUNTIME_CHECK(hipMemcpy(host.data(), dstInterior, xfer, hipMemcpyDeviceToHost));
  size_t badAt = xfer;
  for (size_t i = 0; i < xfer; ++i) {
    if (host[i] != 0xCD) {
      badAt = i;
      break;
    }
  }
  Require(badAt == xfer,
          "dma-buf interior-range transfer wrote wrong data at byte " + std::to_string(badAt) +
              " (a zero dma-buf offset lands the write at the allocation base, not base+offset)");

  std::vector<uint8_t> hostBase(xfer);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBase.data(), dstBase, xfer, hipMemcpyDeviceToHost));
  size_t clobberAt = xfer;
  for (size_t i = 0; i < xfer; ++i) {
    if (hostBase[i] != 0x11) {
      clobberAt = i;
      break;
    }
  }
  Require(clobberAt == xfer,
          "dma-buf interior-range transfer clobbered the allocation base at byte " +
              std::to_string(clobberAt) + " (indicates the MR resolved to offset 0)");

  pair.initiator->DeregisterMemory(srcDesc);
  pair.target->DeregisterMemory(dstDesc);
}

void CaseRdmaDmabufInteriorRangeTransfer() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  // Force the dma-buf registration path (RegisterRdmaMemoryRegionAuto), which this
  // fix touches. The regression: TryExportDmabufFd exports a dma-buf fd for the
  // whole *backing* allocation, so registering an interior sub-range with offset 0
  // pointed the MR at the allocation base instead of base+offset. Transfers then
  // hit LOC_PROT or silently landed at the wrong address.
  ScopedEnvVar enableDmabuf("MORI_ENABLE_DMABUF_REG", "1");
  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");

  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_dmabuf_interior", true);

  constexpr size_t kAlloc = 64 * 1024 * 1024;   // 64 MiB backing allocation
  constexpr size_t kOffset = 32 * 1024 * 1024;  // nonzero interior offset
  constexpr size_t kXfer = 1 * 1024 * 1024;     // transferred sub-range

  HIP_RUNTIME_CHECK(hipSetDevice(0));
  void* srcBase = nullptr;
  void* dstBase = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&srcBase, kAlloc));
  HIP_RUNTIME_CHECK(hipMalloc(&dstBase, kAlloc));
  CheckDmabufInteriorWrite(pair, srcBase, dstBase, kAlloc, kOffset, kXfer);
  HIP_RUNTIME_CHECK(hipFree(srcBase));
  HIP_RUNTIME_CHECK(hipFree(dstBase));
}

// Two VMM allocations (POSIX fd handle type) mapped back to back into one VA
// reservation, the way vLLM's cuMem allocator backs a large tensor.
struct VmmPair {
  void* va{nullptr};
  size_t chunk{0};
  hipMemGenericAllocationHandle_t handles[2]{};

  explicit VmmPair(size_t chunkSize) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    size_t gran = 0;
    HIP_RUNTIME_CHECK(
        hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    chunk = (chunkSize + gran - 1) / gran * gran;
    HIP_RUNTIME_CHECK(hipMemAddressReserve(&va, 2 * chunk, 0, nullptr, 0));
    for (int i = 0; i < 2; ++i) {
      HIP_RUNTIME_CHECK(hipMemCreate(&handles[i], chunk, &prop, 0));
      HIP_RUNTIME_CHECK(hipMemMap(At(i), chunk, 0, handles[i], 0));
    }
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    HIP_RUNTIME_CHECK(hipMemSetAccess(va, 2 * chunk, &access, 1));
  }
  ~VmmPair() {
    for (int i = 0; i < 2; ++i) {
      (void)hipMemUnmap(At(i), chunk);
      (void)hipMemRelease(handles[i]);
    }
    (void)hipMemAddressFree(va, 2 * chunk);
  }
  void* At(int i) const { return static_cast<char*>(va) + i * chunk; }
};

void CaseRdmaDmabufVmmInteriorRangeTransfer() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  // Same check as CaseRdmaDmabufInteriorRangeTransfer for VMM memory, which
  // TryExportDmabufFd exports through its allocation handle. The registered range
  // sits inside the second allocation of the reservation, so its dma-buf offset
  // must be relative to that allocation, not to the reservation base.
  ScopedEnvVar enableDmabuf("MORI_ENABLE_DMABUF_REG", "1");
  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");

  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_dmabuf_vmm_interior", true);

  HIP_RUNTIME_CHECK(hipSetDevice(0));
  VmmPair src(64 * 1024 * 1024);
  VmmPair dst(64 * 1024 * 1024);
  CheckDmabufInteriorWrite(pair, src.At(1), dst.At(1), src.chunk, src.chunk / 2, 1 * 1024 * 1024);
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
//   1. Exporter (this process, GPU <exportDevice>):  allocates GPU memory,
//      registers it with an IOEngine to populate the IPC handle, writes a
//      MemoryDesc blob to the shared file, and waits for the importer to signal
//      completion.
//   2. Importer (subprocess, restricted HIP_VISIBLE_DEVICES that excludes the
//      exporter's GPU):  reads the MemoryDesc, creates its own IOEngine, and
//      does a Write from its local GPU <importerDevice> to the exporter's
//      memory.  The exporter's bus ID is NOT in the importer's
//      localDeviceByBusId, so it must go through the hidden-device branch.
// --------------------------------------------------------------------------
int RunHiddenDeviceImporter(const char* shmPath, int localDevice) {
  // This function runs in a subprocess with restricted HIP_VISIBLE_DEVICES.
  SetLogLevel("info");
  int gpuCount = GetGpuCount();
  if (gpuCount <= localDevice) {
    std::fprintf(stderr, "importer: device %d not visible (%d GPUs)\n", localDevice, gpuCount);
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
    // Match the node id the exporter's backend resolves (MORI_NODE_ID, then
    // boot_id, then hostname), or IsSameNodeEngine rejects the peer.
    remoteEngDesc.nodeId = mori::ResolveNodeId(remoteEngDesc.hostname);
  }
  remoteEngDesc.host = "127.0.0.1";
  remoteEngDesc.port = 0;
  remoteEngDesc.pid = 0;
  engine.RegisterRemoteEngine(remoteEngDesc);

  // Allocate local memory on the importer's chosen local device
  constexpr size_t kSize = 1024 * 1024;
  auto localMem = RegisterGpuMemory(&engine, kSize, localDevice);

  // Fill local source with 0xCD
  HIP_RUNTIME_CHECK(hipSetDevice(localDevice));
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

// Exports from physical GPU `exportDevice` in this process (which sees every
// GPU) and imports from a subprocess that sees only `importerVisible`, writing
// from that subprocess's local device `importerDevice`.
void RunHiddenDeviceSplitVisibility(int exportDevice, const std::string& importerVisible,
                                    int importerDevice) {
  // Create shared memory file for IPC
  std::string shmPath = "/dev/shm/mori_test_hidden_" + std::to_string(getpid());
  int shmFd = open(shmPath.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  Require(shmFd >= 0, "failed to create shared memory file");

  // Exporter: allocate on GPU exportDevice, register, and serialize the descriptor.
  // RegisterGpuMemory leaves exportDevice current, as a per-rank process would.
  IOEngineConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  IOEngine engine("exporter_engine", cfg);
  XgmiBackendConfig xgmiCfg{};
  engine.CreateBackend(BackendType::XGMI, xgmiCfg);

  constexpr size_t kSize = 1024 * 1024;
  auto exportMem = RegisterGpuMemory(&engine, kSize, exportDevice);

  // RegisterMemory switches to device 0 to export the IPC handle; it must hand
  // the caller's current device back.
  int currentDevice = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&currentDevice));
  Require(currentDevice == exportDevice, "RegisterMemory changed the current device from " +
                                             std::to_string(exportDevice) + " to " +
                                             std::to_string(currentDevice));

  // Clear export buffer
  HIP_RUNTIME_CHECK(hipSetDevice(exportDevice));
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

  // Launch the importer subprocess with only importerVisible visible
  // (so the exporter's bus ID is NOT in the importer's localDeviceByBusId)
  std::string cmd = "HIP_VISIBLE_DEVICES=" + importerVisible + " " + std::string(selfExe) +
                    " --hidden-device-importer " + shmPath + " " + std::to_string(importerDevice) +
                    " 2>&1";
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

  // Verify the exporter's buffer now contains 0xCD (written by importer)
  std::vector<uint8_t> hostBuf(kSize);
  HIP_RUNTIME_CHECK(hipSetDevice(exportDevice));
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

void CaseXgmiHiddenDeviceSplitVisibility() {
  int totalGpus = GetGpuCount();
  if (totalGpus < 2) throw TestSkip("requires at least 2 GPUs");
  // Exporter on GPU 0, importer sees only the last GPU.
  RunHiddenDeviceSplitVisibility(0, std::to_string(totalGpus - 1), 0);
}

// Regression: the exporter registers memory on a non-zero device that is also
// the current device, and the importer sees fewer GPUs than that ordinal.
// hipIpcGetMemHandle records the current ordinal in the handle, so before the
// fix hipIpcOpenMemHandle failed with hipErrorInvalidValue and CreateSession
// returned nullopt. This is prefill TP1 -> decode TP2 rank 1 on one node.
void CaseXgmiHiddenDeviceExportFromNonZeroDevice() {
  int totalGpus = GetGpuCount();
  if (totalGpus < 2) throw TestSkip("requires at least 2 GPUs");
  RunHiddenDeviceSplitVisibility(totalGpus - 1, "0", 0);
}

// Same as above, but the importer sees two GPUs and writes from its local
// device 1, so the importer's own device is not ordinal 0 either.
void CaseXgmiHiddenDeviceImporterNonZeroDevice() {
  int totalGpus = GetGpuCount();
  if (totalGpus < 4) throw TestSkip("requires at least 4 GPUs");
  RunHiddenDeviceSplitVisibility(totalGpus - 1, "0,1", 1);
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

int main(int argc, char* argv[]) {
  // Subprocess entry point for hidden-device importer
  if (argc >= 3 && std::string(argv[1]) == "--hidden-device-importer") {
    return RunHiddenDeviceImporter(argv[2], argc >= 4 ? std::atoi(argv[3]) : 0);
  }

  SetLogLevel("info");
  std::vector<TestCase> cases = {
      {"quiescence_markers", CaseQuiescenceMarkers},
      {"quiescence_busy_post_records_target", CaseQuiescenceBusyPostRecordsTarget},
      {"quiescence_marker_retry_and_freeze", CaseQuiescenceMarkerRetryAndFreeze},
      {"quiescence_marker_rejected", CaseQuiescenceMarkerRejected},
      {"operation_quiescence", CaseOperationQuiescence},
      {"reusable_status_contention", CaseReusableStatusContention},
      {"operation_long_wait_slice", CaseOperationLongWaitSlice},
      {"operation_notification_publication", CaseOperationNotificationPublication},
      {"freeze_after_cq_takes_meta", CaseFreezeAfterCqTakesMeta},
      {"lost_notification_completion", CaseLostNotificationCompletion},
      {"scan_mismatch_fails_outstanding", CaseScanMismatchFailsOutstanding},
      {"rdma_batch_clamps_ordinary_capacity", CaseRdmaBatchClampsOrdinaryCapacity},
      {"rdma_early_returns_seal_operation", CaseRdmaEarlyReturnsSealOperation},
      {"operation_ownership_has_no_cycle", CaseOperationOwnershipHasNoCycle},
      {"submission_ledger_basic", CaseSubmissionLedgerBasic},
      {"submission_ledger_shared_qp_signals", CaseSubmissionLedgerSharedQpSignals},
      {"submission_ledger_fail_all", CaseSubmissionLedgerFailAll},
      {"submission_ledger_tracking_freeze_fails_outstanding",
       CaseSubmissionLedgerTrackingFreezeFailsOutstanding},
      {"submission_ledger_lost_completion", CaseSubmissionLedgerLostCompletion},
      {"submission_ledger_post_commit_race", CaseSubmissionLedgerPostCommitRace},
      {"submission_ledger_close_race_stress", CaseSubmissionLedgerCloseRaceStress},
      {"async_event_scope_matching", CaseAsyncEventScopeMatching},
      {"sq_admission_release_wakes_waiter", CaseSqAdmissionReleaseWakesWaiter},
      {"sq_admission_closed_wakes_waiter", CaseSqAdmissionClosedWakesWaiter},
      {"wr_id_namespace_helpers", CaseWrIdNamespaceHelpers},
      {"rdma_backend_config_chunking_fields", CaseRdmaBackendConfigChunkingFields},
      {"resolve_requested_nics", CaseResolveRequestedNics},
      {"plan_chunks_boundaries", CasePlanChunksBoundaries},
      {"chunk_geometry_single_sge_count_matches_planner",
       CaseChunkGeometrySingleSgeCountMatchesPlanner},
      {"chunk_geometry_properties", CaseChunkGeometryProperties},
      {"chunk_count_split_invariant", CaseChunkCountSplitInvariant},
      {"rdma_transfer_control_disable_merge_and_owns_total",
       CaseRdmaTransferControlDisableMergeAndOwnsTotal},
      {"rdma_rejects_single_request_larger_than_uint32",
       CaseRdmaRejectsSingleRequestLargerThanUint32},
      {"chunking_disabled_oversized_wr_returns_error", CaseChunkingDisabledOversizedWrReturnsError},
      {"sge_stream_chunking_covers_and_respects_limits",
       CaseSgeStreamChunkingCoversAndRespectsLimits},
      {"sge_stream_chunking_never_crosses_sge_boundary",
       CaseSgeStreamChunkingNeverCrossesSgeBoundary},
      {"chunking_expands_multi_sge_wr_before_post", CaseChunkingExpandsMultiSgeWrBeforePost},
      {"rdma_endpoint_carries_local_max_msg_size", CaseRdmaEndpointCarriesLocalMaxMsgSize},
      {"build_desired_qp_counts", CaseBuildDesiredQpCounts},
      {"interleave_endpoints_by_local_device", CaseInterleaveEndpointsByLocalDevice},
      {"uses_inline_only", CaseUsesInlineOnly},
      {"validate_rdma_transfer_config", CaseValidateRdmaTransferConfig},
      {"rdma_notification_rejects_zero_notif_per_qp", CaseRdmaNotificationRejectsZeroNotifPerQp},
      {"rdma_backend_has_active_devices_returns_false_when_no_device",
       CaseRdmaBackendHasActiveDevicesReturnsFalseWhenNoDevice},
      {"rdma_manager_throws_when_no_active_devices", CaseRdmaManagerThrowsWhenNoActiveDevices},
      {"create_backend_rdma_throws_by_default_when_no_rdma_device",
       CaseCreateBackendRdmaThrowsByDefaultWhenNoRdmaDevice},
      {"create_backend_rdma_falls_back_to_xgmi_when_opted_in",
       CaseCreateBackendRdmaFallsBackToXgmiWhenOptedIn},
      {"create_backend_rdma_throws_when_opted_in_but_no_xgmi",
       CaseCreateBackendRdmaThrowsWhenOptedInButNoXgmi},
      {"explicit_xgmi_then_rdma_without_opt_in_still_throws",
       CaseExplicitXgmiThenRdmaWithoutOptInStillThrows},
      {"explicit_xgmi_then_rdma_with_opt_in_refreshes_port",
       CaseExplicitXgmiThenRdmaWithOptInRefreshesPort},
      {"rdma_backend_refuses_sentinel_port_config", CaseRdmaBackendRefusesSentinelPortConfig},
      {"select_backend_returns_null_for_cross_node_under_xgmi_only",
       CaseSelectBackendReturnsNullForCrossNodeUnderXgmiOnly},
      {"rdma_backend_can_handle_rejects_sentinel_port_remote",
       CaseRdmaBackendCanHandleRejectsSentinelPortRemote},
      {"rdma_transfer_basic", CaseRdmaTransferBasic},
      {"rdma_dmabuf_interior_range_transfer", CaseRdmaDmabufInteriorRangeTransfer},
      {"rdma_dmabuf_vmm_interior_range_transfer", CaseRdmaDmabufVmmInteriorRangeTransfer},
      {"rdma_notification_disabled_behavior", CaseRdmaNotificationDisabledBehavior},
      {"rdma_notification_env_override_disables", CaseRdmaNotificationEnvOverrideDisables},
      {"rdma_notification_invalid_env_keeps_config", CaseRdmaNotificationInvalidEnvKeepsConfig},
      {"normalize_bus_id", CaseNormalizeBusId},
      {"is_ipc_handle_empty", CaseIsIpcHandleEmpty},
      {"xgmi_visible_device_regression", CaseXgmiVisibleDeviceRegression},
      {"xgmi_cross_engine_ipc", CaseXgmiCrossEngineIpc},
      {"xgmi_hidden_device_split_visibility", CaseXgmiHiddenDeviceSplitVisibility},
      {"xgmi_hidden_device_export_from_nonzero_device",
       CaseXgmiHiddenDeviceExportFromNonZeroDevice},
      {"xgmi_hidden_device_importer_nonzero_device", CaseXgmiHiddenDeviceImporterNonZeroDevice},
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
