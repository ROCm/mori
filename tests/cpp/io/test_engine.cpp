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
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
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
  if (!RdmaBackend::HasActiveDevices()) {
    throw TestSkip("requires at least one active RDMA device");
  }

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
  std::atomic<int> sqDepth{5};
  TransferStatus status;
  auto meta = std::make_shared<CqCallbackMeta>(&status, 101, 8);
  const uint64_t id = ledger.Insert(3, true, meta, 8);
  Require(id == kNotifPerQp, "first ledger record id should start at notifPerQp boundary");
  int batchSize = 0;
  auto releasedMeta = ledger.ReleaseByCqe(id, &sqDepth, &batchSize);
  Require(releasedMeta != nullptr, "ledger release meta should not be null");
  Require(releasedMeta->id == 101, "unexpected transfer id from ledger release");
  Require(batchSize == 8, "unexpected batch size from ledger release");
  Require(sqDepth.load(std::memory_order_relaxed) == 2, "unexpected sq depth after release");

  SubmissionLedger ledger2(kNotifPerQp);
  std::atomic<int> sqDepth2{12};
  auto meta2 = std::make_shared<CqCallbackMeta>(&status, 202, 16);
  uint64_t postedId = ledger2.Insert(4, true, meta2, 10);
  Require(postedId == kNotifPerQp, "posted record id should respect notifPerQp offset");
  ledger2.InsertOrphaned(3, meta2, 6);
  Require(ledger2.HasOrphaned(), "expected orphaned record in ledger");
  int recovered = ledger2.ReleaseOrphanedByRecovery(&sqDepth2);
  // Only Orphaned record (3 WRs) should be released; Posted record (4 WRs) preserved.
  Require(recovered == 3, "unexpected recovered wr count (should only release orphaned)");
  Require(sqDepth2.load(std::memory_order_relaxed) == 9, "unexpected sq depth after recovery");
  Require(!ledger2.HasOrphaned(), "orphaned records should be drained");
  // The Posted record should still be present and retrievable via ReleaseByCqe.
  int postedBatch = 0;
  auto postedMeta = ledger2.ReleaseByCqe(postedId, &sqDepth2, &postedBatch);
  Require(postedMeta != nullptr, "posted record should survive recovery");
  Require(postedBatch == 10, "posted record batch size mismatch");
  Require(sqDepth2.load(std::memory_order_relaxed) == 5, "sq depth after posted CQE release");
}

void CaseNotificationCompletionFanIn() {
  TransferStatus status;
  status.SetCode(StatusCode::IN_PROGRESS);
  auto meta = std::make_shared<CqCallbackMeta>(&status, 42, 3);

  uint32_t finishedBefore = meta->finishedBatchSize.fetch_add(2);
  if (finishedBefore + 2 == meta->totalBatchSize) {
    status.Update(StatusCode::SUCCESS, "data complete");
  }
  Require(status.InProgress(), "data completion must not finish before notification SEND CQE");

  finishedBefore = meta->finishedBatchSize.fetch_add(1);
  if (finishedBefore + 1 == meta->totalBatchSize) {
    status.Update(StatusCode::SUCCESS, "notification complete");
  }
  Require(status.Succeeded(), "final notification completion should finish transfer");

  TransferStatus outOfOrderStatus;
  outOfOrderStatus.SetCode(StatusCode::IN_PROGRESS);
  auto outOfOrderMeta = std::make_shared<CqCallbackMeta>(&outOfOrderStatus, 44, 5);
  (void)outOfOrderMeta->finishedBatchSize.fetch_add(2);  // notification SEND CQEs first
  finishedBefore = outOfOrderMeta->finishedBatchSize.fetch_add(3);
  if (finishedBefore + 3 == outOfOrderMeta->totalBatchSize) {
    outOfOrderStatus.Update(StatusCode::SUCCESS, "data complete after notification");
  }
  Require(outOfOrderStatus.Succeeded(),
          "notification-first completion order must still finish at the exact total");

  TransferStatus failedStatus;
  failedStatus.SetCode(StatusCode::IN_PROGRESS);
  auto failedMeta = std::make_shared<CqCallbackMeta>(&failedStatus, 43, 2);
  (void)failedMeta->finishedBatchSize.fetch_add(1);
  failedStatus.Update(StatusCode::ERR_RDMA_OP, "notification failed");
  finishedBefore = failedMeta->finishedBatchSize.fetch_add(1);
  if (finishedBefore + 1 == failedMeta->totalBatchSize) {
    failedStatus.Update(StatusCode::SUCCESS, "late success");
  }
  Require(failedStatus.Failed(), "notification failure must not be overwritten by late success");
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
  ep.sqDepth = std::make_shared<std::atomic<int>>(0);
  ep.maxSqDepth = 0;
  ep.degraded = std::make_shared<std::atomic<bool>>(false);
  ep.ledger = std::make_shared<SubmissionLedger>(16);
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
  ep.sqDepth = std::make_shared<std::atomic<int>>(0);
  ep.maxSqDepth = 0;
  ep.degraded = std::make_shared<std::atomic<bool>>(false);
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

int main(int argc, char* argv[]) {
  // Subprocess entry point for hidden-device importer
  if (argc >= 3 && std::string(argv[1]) == "--hidden-device-importer") {
    return RunHiddenDeviceImporter(argv[2]);
  }

  SetLogLevel("info");
  std::vector<TestCase> cases = {
      {"submission_ledger_basic", CaseSubmissionLedgerBasic},
      {"notification_completion_fan_in", CaseNotificationCompletionFanIn},
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
