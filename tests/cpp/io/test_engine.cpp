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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
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
  Require(UsesInlineOnly(cfg), "chunking should force inline-only path");

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

// REVIEW_M #66-3. `a5d37786` turned `assert(remoteMr->length == remote.size)`
// into two named throws; until now that was compile-and-strings evidence only.
// The peer's `:1119` arm answers a default-constructed (zero) MR for ANY memory
// id it does not know, so a flipped peer that re-registered under new ids -- or
// simply an id it never had -- lands here with NO fault injection required.
//
// The whole point is the PROCESS SURVIVES: this used to abort() the engine
// (asserts are LIVE in this build, review #64-1), which on a real flip kills
// the inference server for a peer-side condition TransferStatus can express.
void CaseRdmaUnknownRemoteMemoryIdFailsTransferWithoutAbort() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_unknown_remote_id", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 4096, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 4096, 0);

  // Same descriptor the peer really registered, except for the id: the peer
  // has no such memory, so its control plane answers a zero-length MR.
  MemoryDesc bogus = dst.desc;
  bogus.id = dst.desc.id + 4242;

  TransferStatus status;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, bogus, 0, 64, &status, uid);

  Require(status.Failed(),
          "transfer against an unknown remote memory id must FAIL, not succeed; code=" +
              std::to_string(status.CodeUint32()) + ", msg='" + status.Message() + "'");
  Require(status.Code() == StatusCode::ERR_BAD_STATE,
          "unknown remote memory id should surface as ERR_BAD_STATE; got " +
              std::to_string(status.CodeUint32()) + ", msg='" + status.Message() + "'");
  Require(status.Message().find("does not know this id") != std::string::npos,
          "expected a5d37786's zero-MR wording; got: " + status.Message());

  // NON-VACUITY: prove the pair still works, i.e. we measured a rejected
  // transfer on a live engine and not a wedged one. A test that only asserts a
  // failure would also pass against an engine that fails everything.
  TransferStatus good;
  TransferUniqueId uid2 = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &good, uid2);
  std::string err;
  Require(WaitTransferDone(&good, 5000, &err),
          "engine must still serve real transfers after the bad-id rejection: " + err);
  Require(good.Succeeded(), "post-rejection transfer failed: code=" +
                                std::to_string(good.CodeUint32()) + ", msg='" + good.Message() +
                                "'");
}

// REVIEW_M #66-2 + #66-3, the FLIP race itself: `DeregisterRemoteEngine` while
// the initiator still holds a warm session for that peer. Two things are under
// test and they are ordered as they occur on a flip:
//   1. a transfer issued AFTER the deregistration must fail cleanly rather than
//      abort (`065d5764`'s BuildRdmaConn throw) or RDMA-write against the dead
//      peer's rkeys (this turn's engine-scoped invalidation, `4534e67c`);
//   2. re-registering the engine -- what the peer's flip completion does --
//      must restore service, which only holds if the caches really were
//      dropped rather than left stale.
//
// WHAT THIS CASE DOES **NOT** ESTABLISH -- measured in T34, do not read the
// green as more than it is. The post-deregistration transfer is rejected by the
// ROUTE layer, not by the caches: `CanHandle` -> `TryGetRemoteEnginePort`
// misses as soon as `engines` is erased, so the log line is "No available
// backend found" and `BuildRdmaConn`'s `065d5764` throw is never reached. A
// build WITHOUT `4534e67c`'s invalidation would fail this transfer too. So:
//  - step 1's assertion is carried by the route layer => NOT a discriminator
//    for the stale-cache bug (#66-2);
//  - step 2 succeeds either way, because this fixture's target never
//    re-registers its MEMORY, so the pre-flip rkeys are still valid.
// The invalidation IS proven to run and to have had something to drop, but by
// the INFO line "dropped 1 cached remote memory region(s)" (1, not 0) rather
// than by an assertion here. A true discriminator needs the peer to re-register
// its memory under the same id so a stale rkey becomes WRONG rather than
// merely old; that is the next test, not this one.
void CaseRdmaDeregisteredEngineFailsTransferThenRecovers() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_dereg_engine", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 4096, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 4096, 0);

  // Warm the session cache + the remote MR table FIRST -- that is the state
  // the invalidation exists to clear, and without this the test would only
  // exercise the cold path.
  TransferStatus warm;
  TransferUniqueId warmUid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &warm, warmUid);
  std::string err;
  Require(WaitTransferDone(&warm, 5000, &err), "warm-up transfer timed out: " + err);
  Require(warm.Succeeded(), "warm-up transfer failed: code=" +
                                std::to_string(warm.CodeUint32()) + ", msg='" + warm.Message() +
                                "'");

  EngineDesc targetDesc = pair.target->GetEngineDesc();
  pair.initiator->DeregisterRemoteEngine(targetDesc);

  TransferStatus afterDereg;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &afterDereg, uid);
  Require(afterDereg.Failed(),
          "a transfer to a DEREGISTERED engine must fail -- if it succeeded, the session/MR "
          "caches were reused across the flip; code=" +
              std::to_string(afterDereg.CodeUint32()) + ", msg='" + afterDereg.Message() + "'");

  // Re-register: this is the peer's flip completing.
  pair.initiator->RegisterRemoteEngine(targetDesc);
  TransferStatus afterReReg;
  TransferUniqueId uid2 = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &afterReReg, uid2);
  Require(WaitTransferDone(&afterReReg, 5000, &err),
          "transfer after re-registering the flipped peer timed out: " + err);
  Require(afterReReg.Succeeded(),
          "engine must serve again once the peer re-registers; code=" +
              std::to_string(afterReReg.CodeUint32()) + ", msg='" + afterReReg.Message() + "'");
}

// The REAL PD flip, modelled as sglang actually performs it -- and the first
// test in this campaign to do so. Every prior flip test deregistered and then
// re-registered the SAME engine key, which is a shape sglang never produces:
//
//   * `deregister_remote_engine` has ZERO callers in all of sglang (measured
//     2026-07-30T11:55Z @38ad45fe; the control grep for the register variant
//     returns 1). Nothing tells the survivor that the old peer died.
//   * A flip destroys the whole IOEngine (`MoriKVManager.teardown()` ->
//     `self.engine = None`) and builds a NEW one whose key embeds a fresh
//     `uuid4().hex[:8]`. The peer comes back under a DIFFERENT key.
//
// So the survivor's per-engine state is not stale-and-reused (that was my
// turn-39 reading, retracted in 2a7e61eb) -- it is ORPHANED and unreachable,
// and nothing ever collects it. This test asserts on the numbers rather than
// on a source reading, because T3 taught this team what a leak assertion is
// worth when the instrument cannot move: it re-registers a peer under a new
// key N times, exactly as N flips would, and reports the growth in each
// retained structure.
//
// It is deliberately written to PASS while the leak exists, and to say so: the
// point of this turn is to produce the NUMBER, honestly, not to assert a fix I
// have not written. The measured per-flip deltas are printed and the only hard
// assertion is the one that cannot be vacuous -- that the counters moved at
// all, i.e. that the instrument works.
void CaseRdmaPerFlipRetentionIsMeasured() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_flip_retention", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 4096, 0);
  auto dst = RegisterGpuMemory(pair.target.get(), 4096, 0);

  auto* backend = dynamic_cast<RdmaBackend*>(pair.initiator->GetBackend(BackendType::RDMA));
  Require(backend != nullptr, "initiator must have an RDMA backend to read retention stats from");

  // One real transfer so the caches are WARM before the first measurement --
  // otherwise the baseline is empty and every later number is trivially larger.
  std::string err;
  {
    TransferStatus warm;
    TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
    pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &warm, uid);
    Require(WaitTransferDone(&warm, 5000, &err), "warm-up transfer timed out: " + err);
    Require(warm.Succeeded(), "warm-up transfer failed: " + warm.Message());
  }

  RdmaBackend::RemoteRetentionStats base = backend->GetRemoteRetentionStats();
  std::printf(
      "[retention] baseline: engines=%zu metas=%zu endpoints=%zu sessions=%zu notifQps=%zu "
      "notifBytes=%zu\n",
      base.numRemoteEngines, base.numRemoteMetas, base.numEndpointRuntimes, base.numSessions,
      base.numNotifContexts, base.notifBufferBytes);

  // NON-VACUITY: the warm transfer must have built something, or "it did not
  // grow" later would be a statement about an instrument that reads zero.
  Require(base.numRemoteEngines > 0,
          "instrument check: a connected+warmed pair must retain at least one remote engine");
  Require(base.numEndpointRuntimes > 0,
          "instrument check: a warmed transfer must have built at least one endpoint");

  // N flips. Each one gives the peer a brand-new engine key, exactly as
  // `_init_engine`'s uuid4 does, and drives a real transfer against it so the
  // survivor builds the same per-engine state a real flip would make it build.
  constexpr int kFlips = 5;
  EngineDesc realTarget = pair.target->GetEngineDesc();
  for (int i = 0; i < kFlips; ++i) {
    EngineDesc flipped = realTarget;
    flipped.key = realTarget.key + "_flip" + std::to_string(i);
    // NOTE: no DeregisterRemoteEngine here, and that is the whole point --
    // sglang never calls it. The survivor simply learns a new key.
    pair.initiator->RegisterRemoteEngine(flipped);

    MemoryDesc remoteUnderNewKey = dst.desc;
    remoteUnderNewKey.engineKey = flipped.key;

    TransferStatus st;
    TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
    pair.initiator->Read(src.desc, 0, remoteUnderNewKey, 0, 64, &st, uid);
    // Whether this SUCCEEDS is not the assertion -- the peer is the same
    // process, so it may well serve it. What matters is the state left behind.
    WaitTransferDone(&st, 5000, &err);
  }

  RdmaBackend::RemoteRetentionStats after = backend->GetRemoteRetentionStats();
  std::printf(
      "[retention] after %d flips: engines=%zu metas=%zu endpoints=%zu sessions=%zu notifQps=%zu "
      "notifBytes=%zu\n",
      kFlips, after.numRemoteEngines, after.numRemoteMetas, after.numEndpointRuntimes,
      after.numSessions, after.numNotifContexts, after.notifBufferBytes);
  std::printf(
      "[retention] delta over %d flips: engines=+%zu metas=+%zu endpoints=+%zu sessions=+%zu "
      "notifQps=+%zu notifBytes=+%zu\n",
      kFlips, after.numRemoteEngines - base.numRemoteEngines,
      after.numRemoteMetas - base.numRemoteMetas,
      after.numEndpointRuntimes - base.numEndpointRuntimes,
      after.numSessions - base.numSessions, after.numNotifContexts - base.numNotifContexts,
      after.notifBufferBytes - base.notifBufferBytes);

  // The one hard assertion, and it is about the LEAK being real rather than
  // about it being fixed: a survivor that is never told the old key died must
  // be holding strictly more remote-engine state than before. If this ever
  // fails, either someone added the collection this comment says is missing
  // (good -- then flip this to an equality) or the instrument stopped reading.
  Require(after.numRemoteEngines >= base.numRemoteEngines + kFlips,
          "expected one orphaned remote-engine entry per flip; baseline=" +
              std::to_string(base.numRemoteEngines) + " after=" +
              std::to_string(after.numRemoteEngines) + " flips=" + std::to_string(kFlips));

  // Still serving on the ORIGINAL key -- the orphaned state must not have
  // broken the live path. Without this the test could pass on a wedged engine.
  TransferStatus good;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &good, uid);
  Require(WaitTransferDone(&good, 5000, &err),
          "engine must still serve the original peer after the flips: " + err);
  Require(good.Succeeded(), "post-flip transfer on the original key failed: " + good.Message());
}

// REVIEW_M #68-1 / #67-2, the use-after-free `8f2d80b2` fixes. The interleaving
// under test is the one sglang's flip actually produces:
//
//   transfer thread : GetOrCreateSessionCachedNoThrow returns the session and
//                     DROPS sessionCacheMu, then dereferences it.
//   flip thread     : MoriKVManager.teardown() -> engine.deregister_memory(desc)
//                     -> RdmaBackend::DeregisterMemory -> InvalidateSessionsForMemory
//                     erases the cache entry.
//
// Before the fix the cache owned the session through a `unique_ptr` and handed
// out `it->second.get()`, so that erase DESTROYED the object the transfer
// thread was about to use. After it, the getter returns a `shared_ptr` by
// value and the erase only unpublishes.
//
// HONESTY ABOUT WHAT THIS PROVES: a use-after-free is not observable by
// assertion -- freed memory usually still reads plausibly. Run under
// -fsanitize=address this case is a two-sided discriminator (RED at 8f2d80b2^,
// GREEN at 8f2d80b2). Run WITHOUT a sanitizer it is a stress that can pass on
// broken code, and it is labelled so nobody cites a plain green as a proof.
void CaseRdmaTransferSurvivesConcurrentDeregister() {
  if (GetGpuCount() < 1) throw TestSkip("requires at least one GPU");

  ScopedEnvVar disableAutoXgmi("MORI_DISABLE_AUTO_XGMI", "1");
  ConnectedEnginePair pair = CreateConnectedRdmaPair("rdma_dereg_race", true);
  auto src = RegisterGpuMemory(pair.initiator.get(), 4096, 0);

  // Many SHORT-LIVED remote registrations: each cycle warms a session keyed on
  // that memory id, then deregisters it while a transfer against it is in
  // flight. Distinct ids per cycle is what makes each erase hit a live entry
  // rather than a cold miss.
  constexpr int kCycles = 24;
  std::atomic<int> transfersIssued{0};
  std::atomic<int> deregsDone{0};

  for (int i = 0; i < kCycles; ++i) {
    void* rptr = nullptr;
    HIP_RUNTIME_CHECK(hipSetDevice(0));
    HIP_RUNTIME_CHECK(hipMalloc(&rptr, 4096));
    HIP_RUNTIME_CHECK(hipMemset(rptr, 0, 4096));
    MemoryDesc rdesc =
        pair.target->RegisterMemory(rptr, 4096, 0, MemoryLocationType::GPU);

    // Warm the cache entry so the racing DeregisterMemory has something to erase.
    {
      TransferStatus warm;
      TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
      pair.initiator->Read(src.desc, 0, rdesc, 0, 64, &warm, uid);
      std::string werr;
      WaitTransferDone(&warm, 5000, &werr);
    }

    std::atomic<bool> go{false};
    std::thread transferThread([&]() {
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int k = 0; k < 8; ++k) {
        TransferStatus st;
        TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
        // Reaching a freed session is exactly what this call used to do; a
        // FAILED status here is fine and expected once the memory is gone.
        pair.initiator->Read(src.desc, 0, rdesc, 0, 64, &st, uid);
        std::string terr;
        WaitTransferDone(&st, 5000, &terr);
        transfersIssued.fetch_add(1, std::memory_order_relaxed);
      }
    });

    go.store(true, std::memory_order_release);
    std::this_thread::yield();
    pair.initiator->DeregisterMemory(rdesc);  // erases the session under the transfer
    pair.target->DeregisterMemory(rdesc);
    deregsDone.fetch_add(1, std::memory_order_relaxed);

    transferThread.join();
    HIP_RUNTIME_CHECK(hipFree(rptr));
  }

  // NON-VACUITY on both sides: the race must actually have been driven. If the
  // transfer thread never ran or no deregister landed, a green means nothing.
  Require(transfersIssued.load() == kCycles * 8,
          "instrument check: expected " + std::to_string(kCycles * 8) +
              " transfers, got " + std::to_string(transfersIssued.load()));
  Require(deregsDone.load() == kCycles,
          "instrument check: expected " + std::to_string(kCycles) +
              " deregisters, got " + std::to_string(deregsDone.load()));

  // The engine must still be USABLE afterwards. THIS IS THE ARM THAT FAILS, and
  // it found a defect BIGGER than the one the case was written for -- see the
  // T36 note below. Kept RED on purpose: it is the honest state of the code.
  auto dst = RegisterGpuMemory(pair.target.get(), 4096, 0);
  TransferStatus good;
  TransferUniqueId uid = pair.initiator->AllocateTransferUniqueId();
  pair.initiator->Read(src.desc, 0, dst.desc, 0, 64, &good, uid);
  std::string err;
  Require(WaitTransferDone(&good, 5000, &err),
          "engine must still serve after the deregister race: " + err);
  Require(good.Succeeded(),
          "post-race transfer failed: " + good.Message());
  std::printf("[dereg-race] %d transfers across %d deregister cycles, engine still serving\n",
              transfersIssued.load(), deregsDone.load());
}

// T36 MEASURED RESULT of the case above, recorded AT the code so nobody has to
// find the log: it FAILS, and NOT from the use-after-free.
//
//   [FAIL] rdma_transfer_survives_concurrent_deregister (1421 ms):
//          post-race transfer failed: Work Request Flushed Error
//   ProcessOneCqe: [ROOT CAUSE] CQE error: wr_id=1025 status=10(remote access
//          error) qp_num=42863 vendor_err=136
//   ... then ~700 identical "1 flush errors ... representative eid=1
//          qp_num=42863" rounds, the SAME qp_num, forever.
//
// The mechanism, and why it matters far more than a leak:
//  1. A transfer is in flight against a memory region; DeregisterMemory
//     destroys its MR on the TARGET side.
//  2. The in-flight RDMA read lands on a dead rkey -> CQE status 10
//     IBV_WC_REM_ACCESS_ERR. On a **Reliable Connected** QP that transitions
//     the QP to ERROR state -- an RC QP does not fail one work request, it
//     fails the CONNECTION.
//  3. Every subsequent WR on that QP completes IBV_WC_WR_FLUSH_ERR. mori has
//     NO recovery: `grep ibv_modify_qp src/io` is EMPTY (all matches are in
//     src/application/transport), so nothing ever brings the QP back through
//     RESET->INIT->RTR->RTS.
//  4. `CreateSession` (backend_impl.cpp:1621) reuses whatever endpoints
//     already exist -- `CountEndpoint` >= qpPerTransfer means BuildRdmaConn is
//     skipped -- so even a brand-new session for a brand-new memory id is
//     handed the SAME dead QP. That is why the final transfer, on a freshly
//     registered descriptor, still fails.
//
// Consequence for the campaign: sglang's flip teardown deregisters every
// kv/aux/state desc while the peer may still have reads outstanding. If that
// race is hit, the SURVIVING peer's QP to that engine is dead for the lifetime
// of the process and every later transfer fails -- not slow, not leaky: down.
// That is a flip-robustness blocker, and it is the next thing to fix.
//
// T36b: I said the sglang half was "not yet measured", so I measured it rather
// than leaving the reader to. It is REACHABLE, and by a 3-second timeout:
//
//   conn.py:1574   _run_chunk -> rc = self._wait_chunk(statuses)   # DOES wait
//   conn.py:691-692  teardown: for t in self._worker_threads: t.join(timeout=3.0)
//   conn.py:693      self._worker_threads = []                     # unconditional
//   conn.py:706-718  for desc in ...: self.engine.deregister_memory(desc)
//
// So the drain exists -- and it is BOUNDED. `join(timeout=3.0)` returns whether
// the thread finished or not, there is no `t.is_alive()` check after it, and
// the very next statements deregister every kv/aux/state desc. A chunk still in
// _wait_chunk at t+3s therefore has its MR pulled out from under it, which is
// step 1 of the sequence above. On dsv3-full, with large KV chunks and a busy
// fabric, 3 seconds is not a comfortable margin -- it is the margin between a
// clean flip and a dead QP.
//
// This does NOT need the use-after-free to be present; the QP wedge is a
// property of RC semantics plus a missing recovery path, and it survives
// 8f2d80b2.
//
// T36c: THE ASAN DISCRIMINATOR WAS RUN AND IT IS INCONCLUSIVE. Reporting the
// non-result rather than quietly dropping it. Pre-fix side, built at
// `8f2d80b2^` with HEAD's test overlaid (one variable changed):
//   CMAKE_ASAN_RC=0 BUILD_ASAN_RC=0 ASAN_RACE_RC_pre=1
//   census of `SUMMARY: AddressSanitizer:` lines = **374 bad-free, 0 of any
//   other kind**, and in particular ZERO heap-use-after-free.
//   log: logs/mori_io_M_t36asan_pre.log
// Every bad-free stack is inside /opt/rocm/lib/libamdhip64.so.7 with a 4096-byte
// region allocated by the same library -- ASAN interposing malloc under HIP,
// which manages that memory itself. It is instrument noise, not a mori defect,
// and 374 of them bury any real report.
//
// So `8f2d80b2` remains justified by SOURCE (the interleaving is written out at
// its declaration) and is NOT sanitizer-proven. Two things would have to change
// to get the proof: suppress the HIP allocator (ASAN_OPTIONS suppressions or a
// host-memory-only fixture that never calls hipMalloc), AND fix the QP wedge
// first, since the case now dies at the wedge before any post-free window.
// That ordering is itself a finding: the wedge blocks its own diagnosis.

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
      {"wr_id_namespace_helpers", CaseWrIdNamespaceHelpers},
      {"rdma_backend_config_chunking_fields", CaseRdmaBackendConfigChunkingFields},
      {"resolve_requested_nics", CaseResolveRequestedNics},
      {"plan_chunks_boundaries", CasePlanChunksBoundaries},
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
      {"rdma_unknown_remote_memory_id_fails_without_abort",
       CaseRdmaUnknownRemoteMemoryIdFailsTransferWithoutAbort},
      {"rdma_deregistered_engine_fails_then_recovers",
       CaseRdmaDeregisteredEngineFailsTransferThenRecovers},
      {"rdma_per_flip_retention_is_measured", CaseRdmaPerFlipRetentionIsMeasured},
      {"rdma_transfer_survives_concurrent_deregister",
       CaseRdmaTransferSurvivesConcurrentDeregister},
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
