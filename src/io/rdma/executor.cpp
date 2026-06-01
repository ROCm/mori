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
#include "src/io/rdma/executor.hpp"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/logging.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori {
namespace io {

AdmissionToken::AdmissionToken(std::shared_ptr<SqController> sq_, int wr)
    : sq(std::move(sq_)), estimatedWr(wr), active(sq != nullptr && wr > 0) {}

AdmissionToken::AdmissionToken(AdmissionToken&& rhs) noexcept
    : sq(std::move(rhs.sq)), estimatedWr(rhs.estimatedWr), active(rhs.active) {
  rhs.estimatedWr = 0;
  rhs.active = false;
}

AdmissionToken& AdmissionToken::operator=(AdmissionToken&& rhs) noexcept {
  if (this == &rhs) return *this;
  Release();
  sq = std::move(rhs.sq);
  estimatedWr = rhs.estimatedWr;
  active = rhs.active;
  rhs.estimatedWr = 0;
  rhs.active = false;
  return *this;
}

AdmissionToken::~AdmissionToken() { Release(); }

void AdmissionToken::Release() {
  if (!active) return;
  if (sq) sq->ReleaseAdmission(estimatedWr);
  active = false;
  estimatedWr = 0;
  sq.reset();
}

namespace {

enum class ExecutorSplitPolicy {
  Static,
  LeastLoaded,
  Capacity,
};

struct ExecutorAdmissionConfig {
  bool enabled{false};
  ExecutorSplitPolicy policy{ExecutorSplitPolicy::Static};
  int timeoutUs{1000000};
  int resumeWatermarkWr{2048};
  int waitSliceUs{100};
  int admissionChunkWr{0};
  int maxChunksPerCall{0};
};

struct AdmissionCandidate {
  int epId{-1};
  int workerId{-1};
  std::shared_ptr<SqController> sq;
  int depth{0};
  int queuedDepth{0};
  int effectiveLoad{0};
  int maxDepth{0};
  int effectiveFree{0};
};

struct AcquiredAdmission {
  bool ok{false};
  RdmaOpRet error;
  int epId{-1};
  int workerId{-1};
  int begin{-1};
  int end{-1};
  int admissionWr{0};
  AdmissionToken token;
};

struct AdmissionWaitStats {
  uint64_t waitCount{0};
  uint64_t waitUs{0};
};

std::optional<int> ParseNonNegativeInt(const char* raw) {
  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || errno != 0 || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }
  return static_cast<int>(parsed);
}

int GetNonNegativeEnv(const char* name, int fallback) {
  const char* raw = mori::env::Get(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  auto parsed = ParseNonNegativeInt(raw);
  if (!parsed.has_value()) {
    MORI_IO_WARN("Ignore invalid env {}={}", name, raw);
    return fallback;
  }
  return *parsed;
}

bool GetBoolEnv(const char* name, bool fallback) {
  const char* raw = mori::env::Get(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  auto parsed = mori::env::detail::ParseBool(raw);
  if (!parsed.has_value()) {
    MORI_IO_WARN("Ignore invalid env {}={}", name, raw);
    return fallback;
  }
  return *parsed;
}

ExecutorSplitPolicy GetSplitPolicyEnv() {
  const char* raw = mori::env::Get("MORI_IO_SQ_SPLIT_POLICY");
  if (raw == nullptr || raw[0] == '\0') return ExecutorSplitPolicy::Static;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "static") return ExecutorSplitPolicy::Static;
  if (value == "least_loaded") return ExecutorSplitPolicy::LeastLoaded;
  if (value == "capacity") {
    if (GetBoolEnv("MORI_IO_SQ_ENABLE_EXPERIMENTAL_CAPACITY", false)) {
      return ExecutorSplitPolicy::Capacity;
    }
    MORI_IO_WARN(
        "MORI_IO_SQ_SPLIT_POLICY=capacity requires "
        "MORI_IO_SQ_ENABLE_EXPERIMENTAL_CAPACITY=1; falling back to static");
    return ExecutorSplitPolicy::Static;
  }
  MORI_IO_WARN("Ignore invalid env MORI_IO_SQ_SPLIT_POLICY={}", raw);
  return ExecutorSplitPolicy::Static;
}

const char* SplitPolicyName(ExecutorSplitPolicy policy) {
  switch (policy) {
    case ExecutorSplitPolicy::Static:
      return "static";
    case ExecutorSplitPolicy::LeastLoaded:
      return "least_loaded";
    case ExecutorSplitPolicy::Capacity:
      return "capacity";
  }
  return "unknown";
}

ExecutorAdmissionConfig LoadAdmissionConfig() {
  ExecutorAdmissionConfig cfg;
  cfg.enabled = GetBoolEnv("MORI_IO_SQ_EXECUTOR_ADMISSION", false);
  cfg.policy = GetSplitPolicyEnv();
  cfg.timeoutUs = GetNonNegativeEnv("MORI_IO_SQ_ADMISSION_TIMEOUT_US", 1000000);
  cfg.resumeWatermarkWr = GetNonNegativeEnv("MORI_IO_SQ_ADMISSION_RESUME_WATERMARK_WR", 2048);
  cfg.waitSliceUs = std::max(1, GetNonNegativeEnv("MORI_IO_SQ_ADMISSION_WAIT_SLICE_US", 100));
  cfg.admissionChunkWr = GetNonNegativeEnv("MORI_IO_SQ_ADMISSION_CHUNK_WR", 0);
  cfg.maxChunksPerCall = GetNonNegativeEnv("MORI_IO_SQ_ADMISSION_MAX_CHUNKS_PER_CALL", 0);
  return cfg;
}

const ExecutorAdmissionConfig& GetCachedAdmissionConfig() {
  static const ExecutorAdmissionConfig kConfig = LoadAdmissionConfig();
  return kConfig;
}

int ActiveQpCount(const ExecutorReq& req, int numWorker) {
  return std::min(static_cast<int>(req.eps.size()), numWorker);
}

void RecordAdmissionWait(const EpPair& ep, uint64_t waitUs) {
  if (!ep.sqCqeDiagnostics) return;
  ep.sqCqeDiagnostics->executorAdmissionWaitCount.fetch_add(1, std::memory_order_relaxed);
  ep.sqCqeDiagnostics->executorAdmissionWaitUs.fetch_add(waitUs, std::memory_order_relaxed);
}

void RecordAdmissionTimeout(const EpPair& ep) {
  if (!ep.sqCqeDiagnostics) return;
  ep.sqCqeDiagnostics->executorAdmissionTimeoutCount.fetch_add(1, std::memory_order_relaxed);
}

void RecordLeastLoadedSelection(const EpPair& ep) {
  if (!ep.sqCqeDiagnostics) return;
  ep.sqCqeDiagnostics->leastLoadedSelectionCount.fetch_add(1, std::memory_order_relaxed);
}

void RecordQueuedWrHighWatermark(const EpPair& ep, int queuedDepth) {
  if (!ep.sqCqeDiagnostics) return;
  int cur = ep.sqCqeDiagnostics->queuedWrHighWatermark.load(std::memory_order_relaxed);
  while (queuedDepth > cur && !ep.sqCqeDiagnostics->queuedWrHighWatermark.compare_exchange_weak(
                                  cur, queuedDepth, std::memory_order_relaxed)) {
  }
}

void WarnMaxChunksCappedOnce(int configured, int activeQps, int effective) {
  static std::atomic<bool> warned{false};
  if (warned.exchange(true, std::memory_order_relaxed)) return;
  MORI_IO_WARN(
      "MORI_IO_SQ_ADMISSION_MAX_CHUNKS_PER_CALL={} is above the current cap activeQps*2={} "
      "(activeQps={}); using effective maxOutstandingChunks={}",
      configured, activeQps * 2, activeQps, effective);
}

int ComputeAdmissionChunkBudgetWr(const ExecutorReq& req, int epId,
                                  const ExecutorAdmissionConfig& cfg) {
  assert(epId >= 0 && epId < static_cast<int>(req.eps.size()));
  const EpPair& ep = req.eps[epId];
  assert(ep.sq != nullptr);
  int budget = ep.sq->MaxDepth();
  if (budget <= 0) budget = 1;

  if (req.postBatchSize > 0) {
    budget = std::min(budget, req.postBatchSize);
  } else if (req.postBatchSize == 0) {
    budget = std::min(budget, 1);
  }

  const int signalIntervalWr = GetSqSignalIntervalWr();
  if (signalIntervalWr > 0) budget = std::min(budget, signalIntervalWr);
  if (cfg.admissionChunkWr > 0) budget = std::min(budget, cfg.admissionChunkWr);
  return std::max(1, budget);
}

int ComputeMinAdmissionChunkBudgetWr(const ExecutorReq& req, int activeQps,
                                     const ExecutorAdmissionConfig& cfg) {
  int budget = std::numeric_limits<int>::max();
  for (int epId = 0; epId < activeQps; ++epId) {
    budget = std::min(budget, ComputeAdmissionChunkBudgetWr(req, epId, cfg));
  }
  return budget == std::numeric_limits<int>::max() ? 1 : std::max(1, budget);
}

int EstimateWrForRange(const ExecutorReq& req, int begin, int end, int epId) {
  if (end <= begin) return 0;
  assert(epId >= 0 && epId < static_cast<int>(req.eps.size()));
  const uint32_t maxSge = std::max(req.eps[epId].local.handle.maxSge, 1u);
  SizeVec localOffsets(req.localOffsets.begin() + begin, req.localOffsets.begin() + end);
  SizeVec remoteOffsets(req.remoteOffsets.begin() + begin, req.remoteOffsets.begin() + end);
  SizeVec sizes(req.sizes.begin() + begin, req.sizes.begin() + end);
  const int estimated = EstimateMergedWrCount(localOffsets, remoteOffsets, sizes, maxSge);
  return std::max(1, estimated);
}

std::vector<AdmissionCandidate> BuildHealthyCandidates(const ExecutorReq& req, int numWorker,
                                                       ExecutorSplitPolicy policy) {
  const int activeQps = ActiveQpCount(req, numWorker);
  std::vector<AdmissionCandidate> candidates;
  candidates.reserve(activeQps);
  for (int epId = 0; epId < activeQps; ++epId) {
    const EpPair& ep = req.eps[epId];
    assert(ep.sq != nullptr);
    if (ep.sq->IsDegraded()) continue;
    AdmissionCandidate candidate;
    candidate.epId = epId;
    candidate.workerId = epId;
    candidate.sq = ep.sq;
    candidate.depth = ep.sq->Depth();
    candidate.queuedDepth = ep.sq->QueuedDepth();
    candidate.effectiveLoad = candidate.depth + candidate.queuedDepth;
    candidate.maxDepth = ep.sq->MaxDepth();
    candidate.effectiveFree = std::max(0, candidate.maxDepth - candidate.effectiveLoad);
    candidates.push_back(std::move(candidate));
  }

  if (policy == ExecutorSplitPolicy::Capacity) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.effectiveFree != rhs.effectiveFree) return lhs.effectiveFree > rhs.effectiveFree;
      if (lhs.effectiveLoad != rhs.effectiveLoad) return lhs.effectiveLoad < rhs.effectiveLoad;
      return lhs.epId < rhs.epId;
    });
  } else {
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.effectiveLoad != rhs.effectiveLoad) return lhs.effectiveLoad < rhs.effectiveLoad;
      return lhs.epId < rhs.epId;
    });
  }
  return candidates;
}

std::vector<AdmissionCandidate> BuildStaticCandidate(const ExecutorReq& req, int epId) {
  assert(epId >= 0 && epId < static_cast<int>(req.eps.size()));
  const EpPair& ep = req.eps[epId];
  assert(ep.sq != nullptr);
  if (ep.sq->IsDegraded()) return {};
  AdmissionCandidate candidate;
  candidate.epId = epId;
  candidate.workerId = epId;
  candidate.sq = ep.sq;
  candidate.depth = ep.sq->Depth();
  candidate.queuedDepth = ep.sq->QueuedDepth();
  candidate.effectiveLoad = candidate.depth + candidate.queuedDepth;
  candidate.maxDepth = ep.sq->MaxDepth();
  candidate.effectiveFree = std::max(0, candidate.maxDepth - candidate.effectiveLoad);
  return {std::move(candidate)};
}

int ComputeRequiredFree(const AdmissionCandidate& candidate, int admissionWr, bool pressureSeen,
                        int chunkBudgetWr, const ExecutorAdmissionConfig& cfg) {
  int requiredFree = admissionWr;
  const int watermarkCap = std::min(candidate.maxDepth, chunkBudgetWr);
  const int effectiveWatermark = std::max(0, std::min(cfg.resumeWatermarkWr, watermarkCap));
  if (pressureSeen && effectiveWatermark > 0) {
    requiredFree = std::max(requiredFree, effectiveWatermark);
  }
  return std::max(1, std::min(requiredFree, candidate.maxDepth));
}

std::string FormatAdmissionQps(const ExecutorReq& req, int activeQps) {
  std::ostringstream os;
  os << "[";
  for (int epId = 0; epId < activeQps; ++epId) {
    if (epId > 0) os << ", ";
    const EpPair& ep = req.eps[epId];
    if (!ep.sq) {
      os << "{ep=" << epId << " sq=null}";
      continue;
    }
    const int depth = ep.sq->Depth();
    const int queued = ep.sq->QueuedDepth();
    const int queuedHwm =
        ep.sqCqeDiagnostics
            ? ep.sqCqeDiagnostics->queuedWrHighWatermark.load(std::memory_order_relaxed)
            : 0;
    os << "{ep=" << epId << " depth=" << depth << " queued=" << queued << " load=" << depth + queued
       << " max=" << ep.sq->MaxDepth() << " queuedHwm=" << queuedHwm
       << " degraded=" << (ep.sq->IsDegraded() ? 1 : 0) << "}";
  }
  os << "]";
  return os.str();
}

RdmaOpRet BuildAdmissionTimeoutRet(const ExecutorReq& req, int activeQps,
                                   const ExecutorAdmissionConfig& cfg, int requestedWr,
                                   int requiredFree, int chunkBudgetWr,
                                   const AdmissionWaitStats& stats, int diagnosticEpId) {
  int effectiveResumeWatermarkWr = 0;
  if (diagnosticEpId >= 0 && diagnosticEpId < activeQps && req.eps[diagnosticEpId].sq) {
    const int watermarkCap = std::min(req.eps[diagnosticEpId].sq->MaxDepth(), chunkBudgetWr);
    effectiveResumeWatermarkWr = std::max(0, std::min(cfg.resumeWatermarkWr, watermarkCap));
  }
  std::string message =
      "SQ admission timeout: requestedWr=" + std::to_string(requestedWr) +
      " activeQps=" + std::to_string(activeQps) + " timeoutUs=" + std::to_string(cfg.timeoutUs) +
      " policy=" + SplitPolicyName(cfg.policy) +
      " resumeWatermarkWr=" + std::to_string(cfg.resumeWatermarkWr) +
      " effectiveResumeWatermarkWr=" + std::to_string(effectiveResumeWatermarkWr) +
      " requiredFree=" + std::to_string(requiredFree) +
      " admissionChunkBudgetWr=" + std::to_string(chunkBudgetWr) +
      " waitCount=" + std::to_string(stats.waitCount) + " waitUs=" + std::to_string(stats.waitUs) +
      " qps=" + FormatAdmissionQps(req, activeQps);
  if (diagnosticEpId >= 0 && diagnosticEpId < activeQps) {
    message += " diagnosticEp=" + std::to_string(diagnosticEpId) + ". ";
    message += BuildSqCqeDiagnosticHint(req.eps[diagnosticEpId]);
  }
  return {StatusCode::ERR_RDMA_OP, std::move(message)};
}

AcquiredAdmission TryAcquireAdmissionForCandidates(const ExecutorReq& req, int begin, int end,
                                                   int admissionWr, std::optional<int> fixedEpId,
                                                   const ExecutorAdmissionConfig& cfg,
                                                   int numWorker,
                                                   std::chrono::steady_clock::time_point deadline) {
  AdmissionWaitStats stats;
  bool pressureSeen = false;
  int lastRequiredFree = admissionWr;
  int lastChunkBudgetWr = 1;
  const int activeQps = ActiveQpCount(req, numWorker);

  while (true) {
    AdmissionResult bestFailure;
    AdmissionCandidate bestFailedCandidate;
    bool hasBestFailure = false;
    std::vector<AdmissionCandidate> candidates =
        fixedEpId.has_value() ? BuildStaticCandidate(req, *fixedEpId)
                              : BuildHealthyCandidates(req, numWorker, cfg.policy);
    if (candidates.empty()) {
      return {
          false,
          {StatusCode::ERR_RDMA_OP, "SQ admission failed: no healthy active endpoint for policy " +
                                        std::string(SplitPolicyName(cfg.policy))},
          -1,
          -1,
          begin,
          end,
          admissionWr,
          AdmissionToken{}};
    }

    for (const auto& candidate : candidates) {
      const int chunkBudgetWr = ComputeAdmissionChunkBudgetWr(req, candidate.epId, cfg);
      const int requiredFree =
          ComputeRequiredFree(candidate, admissionWr, pressureSeen, chunkBudgetWr, cfg);
      lastRequiredFree = requiredFree;
      lastChunkBudgetWr = chunkBudgetWr;
      AdmissionResult result;
      if (candidate.sq->TryAcquireAdmission(admissionWr, requiredFree, &result)) {
        RecordQueuedWrHighWatermark(req.eps[candidate.epId], candidate.sq->QueuedDepth());
        if (cfg.policy == ExecutorSplitPolicy::LeastLoaded ||
            cfg.policy == ExecutorSplitPolicy::Capacity) {
          RecordLeastLoadedSelection(req.eps[candidate.epId]);
        }
        AcquiredAdmission acquired;
        acquired.ok = true;
        acquired.epId = candidate.epId;
        acquired.workerId = candidate.workerId;
        acquired.begin = begin;
        acquired.end = end;
        acquired.admissionWr = admissionWr;
        acquired.token = AdmissionToken(candidate.sq, admissionWr);
        return acquired;
      }

      if (!hasBestFailure || result.snapshot.effectiveLoad < bestFailure.snapshot.effectiveLoad) {
        hasBestFailure = true;
        bestFailure = result;
        bestFailedCandidate = candidate;
      }
    }

    pressureSeen = true;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      if (hasBestFailure) RecordAdmissionTimeout(req.eps[bestFailedCandidate.epId]);
      return {false,
              BuildAdmissionTimeoutRet(req, activeQps, cfg, admissionWr, lastRequiredFree,
                                       lastChunkBudgetWr, stats,
                                       hasBestFailure ? bestFailedCandidate.epId : -1),
              -1,
              -1,
              begin,
              end,
              admissionWr,
              AdmissionToken{}};
    }

    const auto sliceDeadline = std::min(deadline, now + std::chrono::microseconds(cfg.waitSliceUs));
    const auto waitStart = std::chrono::steady_clock::now();
    if (hasBestFailure) {
      bestFailedCandidate.sq->WaitForAdmissionChange(sliceDeadline, bestFailure.snapshot.epoch);
      const auto waitedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - waitStart)
                                .count();
      stats.waitCount++;
      stats.waitUs += static_cast<uint64_t>(std::max<int64_t>(0, waitedUs));
      RecordAdmissionWait(req.eps[bestFailedCandidate.epId],
                          static_cast<uint64_t>(std::max<int64_t>(0, waitedUs)));
    } else {
      std::this_thread::sleep_until(sliceDeadline);
    }
  }
}

AcquiredAdmission TryAcquireCapacityAdmission(const ExecutorReq& req, int begin, int total,
                                              const ExecutorAdmissionConfig& cfg, int numWorker,
                                              std::chrono::steady_clock::time_point deadline) {
  AdmissionWaitStats stats;
  bool pressureSeen = false;
  int lastRequestedWr = 1;
  int lastRequiredFree = 1;
  int lastChunkBudgetWr = 1;
  const int activeQps = ActiveQpCount(req, numWorker);

  while (true) {
    AdmissionResult bestFailure;
    AdmissionCandidate bestFailedCandidate;
    bool hasBestFailure = false;
    auto candidates = BuildHealthyCandidates(req, numWorker, ExecutorSplitPolicy::Capacity);
    if (candidates.empty()) {
      return {false,
              {StatusCode::ERR_RDMA_OP,
               "SQ admission failed: no healthy active endpoint for policy capacity"},
              -1,
              -1,
              begin,
              begin,
              0,
              AdmissionToken{}};
    }

    for (const auto& candidate : candidates) {
      const int chunkBudgetWr = ComputeAdmissionChunkBudgetWr(req, candidate.epId, cfg);
      int requestBudget = std::min(total - begin, chunkBudgetWr);
      if (candidate.effectiveFree > 0) {
        requestBudget = std::min(requestBudget, candidate.effectiveFree);
      }
      if (requestBudget <= 0) requestBudget = std::min(total - begin, chunkBudgetWr);
      requestBudget = std::max(1, requestBudget);
      const int end = std::min(total, begin + requestBudget);
      const int admissionWr = EstimateWrForRange(req, begin, end, candidate.epId);
      const int requiredFree =
          ComputeRequiredFree(candidate, admissionWr, pressureSeen, chunkBudgetWr, cfg);
      lastRequestedWr = admissionWr;
      lastRequiredFree = requiredFree;
      lastChunkBudgetWr = chunkBudgetWr;

      AdmissionResult result;
      if (candidate.sq->TryAcquireAdmission(admissionWr, requiredFree, &result)) {
        RecordQueuedWrHighWatermark(req.eps[candidate.epId], candidate.sq->QueuedDepth());
        RecordLeastLoadedSelection(req.eps[candidate.epId]);
        AcquiredAdmission acquired;
        acquired.ok = true;
        acquired.epId = candidate.epId;
        acquired.workerId = candidate.workerId;
        acquired.begin = begin;
        acquired.end = end;
        acquired.admissionWr = admissionWr;
        acquired.token = AdmissionToken(candidate.sq, admissionWr);
        return acquired;
      }

      if (!hasBestFailure || result.snapshot.effectiveLoad < bestFailure.snapshot.effectiveLoad) {
        hasBestFailure = true;
        bestFailure = result;
        bestFailedCandidate = candidate;
      }
    }

    pressureSeen = true;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      if (hasBestFailure) RecordAdmissionTimeout(req.eps[bestFailedCandidate.epId]);
      return {false,
              BuildAdmissionTimeoutRet(req, activeQps, cfg, lastRequestedWr, lastRequiredFree,
                                       lastChunkBudgetWr, stats,
                                       hasBestFailure ? bestFailedCandidate.epId : -1),
              -1,
              -1,
              begin,
              begin,
              lastRequestedWr,
              AdmissionToken{}};
    }

    const auto sliceDeadline = std::min(deadline, now + std::chrono::microseconds(cfg.waitSliceUs));
    const auto waitStart = std::chrono::steady_clock::now();
    if (hasBestFailure) {
      bestFailedCandidate.sq->WaitForAdmissionChange(sliceDeadline, bestFailure.snapshot.epoch);
      const auto waitedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - waitStart)
                                .count();
      stats.waitCount++;
      stats.waitUs += static_cast<uint64_t>(std::max<int64_t>(0, waitedUs));
      RecordAdmissionWait(req.eps[bestFailedCandidate.epId],
                          static_cast<uint64_t>(std::max<int64_t>(0, waitedUs)));
    } else {
      std::this_thread::sleep_until(sliceDeadline);
    }
  }
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                   MultithreadExecutor::Worker                                  */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::Worker::Worker(int wid) : workerId(wid) {}

MultithreadExecutor::Worker::~Worker() { Shutdown(); }

void MultithreadExecutor::Worker::Start() {
  if (running.load()) return;
  running.store(true);
  thd = std::thread([this] { MainLoop(); });
}

void MultithreadExecutor::Worker::Shutdown() {
  std::queue<Task> abandoned;
  {
    std::lock_guard<std::mutex> lock(mu);
    running.store(false);
    std::swap(abandoned, q);
    cond.notify_all();
  }

  while (!abandoned.empty()) {
    Task task = std::move(abandoned.front());
    abandoned.pop();
    task.admissionToken.Release();
    task.ret.set_value({StatusCode::ERR_BAD_STATE, "executor shutdown"});
  }

  if (thd.joinable()) thd.join();
}

void MultithreadExecutor::Worker::MainLoop() {
  int coreOffset = 0;
  const char* env = std::getenv("MORI_CORE_OFFSET");
  if (env) {
    coreOffset = std::stoi(env);
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int targetCore = workerId + coreOffset;
  CPU_SET(targetCore, &cpuset);

  int rc = pthread_setaffinity_np(thd.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    MORI_IO_WARN(
        "worker {} failed to set affinity to core {}: errno={} ({}). "
        "Worker will run on any available core. "
        "This is usually caused by: CPU not available in cpuset, "
        "NUMA configuration, or container CPU limits.",
        workerId, targetCore, rc, strerror(rc));
  }

  MORI_IO_INFO("worker {} enter main loop, running on core {}", workerId, sched_getcpu());

  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(mu);
      cond.wait(lock, [this]() { return !q.empty() || !running.load(); });

      if (q.empty() && !running.load()) {
        MORI_IO_INFO("worker {} shutdown", workerId);
        break;
      }
      task = std::move(q.front());
      q.pop();
    }

    SizeVec tLoclOffsets(task.req->localOffsets.begin() + task.begin,
                         task.req->localOffsets.begin() + task.end);
    SizeVec tRemoteOffsets(task.req->remoteOffsets.begin() + task.begin,
                           task.req->remoteOffsets.begin() + task.end);
    SizeVec tSizes(task.req->sizes.begin() + task.begin, task.req->sizes.begin() + task.end);

    RdmaOpRet ret = mori::io::RdmaBatchReadWrite(
        {task.req->eps[task.epId]}, task.req->local, tLoclOffsets, task.req->remote, tRemoteOffsets,
        tSizes, task.req->callbackMeta, task.req->id, task.req->isRead, task.req->postBatchSize);
    task.admissionToken.Release();
    task.ret.set_value(ret);
    MORI_IO_TRACE("Worker {} execute task {} begin {} end {} ret code {}", workerId, task.req->id,
                  task.begin, task.end, static_cast<uint32_t>(ret.code));
  }
}

void MultithreadExecutor::Worker::Submit(Task&& task) {
  MORI_IO_FUNCTION_TIMER;
  const TransferUniqueId taskId = task.req != nullptr ? task.req->id : 0;
  const int taskBegin = task.begin;
  const int taskEnd = task.end;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load()) {
      task.admissionToken.Release();
      task.ret.set_value({StatusCode::ERR_BAD_STATE, "worker not started yet"});
      return;
    }
    q.push(std::move(task));
    cond.notify_all();
  }
  MORI_IO_TRACE("Submit to worker {} task {} begin {} end {}", workerId, taskId, taskBegin,
                taskEnd);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       MultithreadExecutor                                      */
/* ---------------------------------------------------------------------------------------------- */
MultithreadExecutor::MultithreadExecutor(int n) : numWorker(n) {
  assert(n > 0);
  for (int i = 0; i < numWorker; i++) {
    pool.emplace_back(new Worker(i));
  }
}

MultithreadExecutor::~MultithreadExecutor() { Shutdown(); }

std::vector<MultithreadExecutor::WorkSplit> MultithreadExecutor::SplitWork(const ExecutorReq& req) {
  int numEps = req.eps.size();
  int totalBatchSize = req.sizes.size();

  assert(numEps > 0);

  int numActiveWorkers = std::min(numEps, numWorker);
  int perWorkerBatchSize = (totalBatchSize + numActiveWorkers - 1) / numActiveWorkers;
  int startWorker = totalBatchSize < numActiveWorkers
                        ? static_cast<int>(req.id % static_cast<TransferUniqueId>(numActiveWorkers))
                        : 0;

  std::vector<WorkSplit> splits;
  for (int i = 0; i < numActiveWorkers; i++) {
    int begin = i * perWorkerBatchSize;
    int end = std::min(begin + perWorkerBatchSize, totalBatchSize);
    int workerId = totalBatchSize < numActiveWorkers ? (startWorker + i) % numActiveWorkers : i;
    splits.push_back({workerId, workerId, begin, end});
    if (end >= totalBatchSize) break;
  }

  return splits;
}

RdmaOpRet MultithreadExecutor::RdmaBatchReadWrite(const ExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  const ExecutorAdmissionConfig& admissionCfg = GetCachedAdmissionConfig();
  if (admissionCfg.enabled) {
    return RdmaBatchReadWriteWithAdmission(req);
  }

  auto splits = SplitWork(req);
  int numSplits = splits.size();
  std::vector<std::future<RdmaOpRet>> futs;

  for (int i = 0; i < numSplits; i++) {
    Task task{&req, splits[i].epId, splits[i].begin, splits[i].end};
    futs.push_back(std::move(task.ret.get_future()));
    pool[splits[i].workerId]->Submit(std::move(task));
  }

  bool hasFail = false;
  int numSucc = 0;
  RdmaOpRet failedRet;
  for (auto& fut : futs) {
    RdmaOpRet ret = fut.get();
    if (ret.Failed()) {
      hasFail = true;
      failedRet = ret;
    } else if (ret.Succeeded()) {
      numSucc++;
    }
  }
  if (hasFail) return failedRet;

  if (numSucc == numSplits) {
    return {StatusCode::SUCCESS, ""};
  }

  MORI_IO_TRACE("MultithreadExecutor submit request for RdmaBatchReadWrite done");
  return {StatusCode::IN_PROGRESS, ""};
}

RdmaOpRet MultithreadExecutor::RdmaBatchReadWriteWithAdmission(const ExecutorReq& req) {
  MORI_IO_FUNCTION_TIMER;

  // Admission may split one user batch into multiple sub-chunks. Each sub-chunk
  // invokes RdmaBatchReadWrite with the same callbackMeta, whose totalBatchSize
  // remains the original request size. On a later admission/worker failure this
  // call returns an error and backend_impl updates TransferStatus to ERR_*.
  // TransferStatus::Update ignores subsequent updates once failed, so late CQE
  // SUCCESS from already-posted earlier chunks cannot overwrite the error; those
  // chunks still drain SQ credit through the normal ledger/CQE path.
  const ExecutorAdmissionConfig& cfg = GetCachedAdmissionConfig();
  const int totalBatchSize = static_cast<int>(req.sizes.size());
  if (totalBatchSize == 0) return {StatusCode::SUCCESS, ""};
  const int activeQps = ActiveQpCount(req, numWorker);
  if (activeQps <= 0) return {StatusCode::ERR_INVALID_ARGS, "no active RDMA endpoints"};

  int maxOutstandingChunks = cfg.maxChunksPerCall > 0 ? cfg.maxChunksPerCall : activeQps;
  const int maxOutstandingCap = activeQps * 2;
  if (cfg.maxChunksPerCall > maxOutstandingCap) {
    WarnMaxChunksCappedOnce(cfg.maxChunksPerCall, activeQps, maxOutstandingCap);
  }
  maxOutstandingChunks = std::max(1, std::min(maxOutstandingChunks, maxOutstandingCap));
  // The admission timeout is scoped to the whole user batch, not each chunk, so
  // later chunks share the original call's remaining wait budget.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg.timeoutUs);

  std::deque<std::future<RdmaOpRet>> futs;
  bool hasFail = false;
  RdmaOpRet failedRet;

  auto drainOne = [&]() {
    RdmaOpRet ret = futs.front().get();
    futs.pop_front();
    if (ret.Failed()) {
      hasFail = true;
      failedRet = ret;
    }
  };

  auto drainAll = [&]() {
    while (!futs.empty()) drainOne();
  };

  auto submitAcquired = [&](AcquiredAdmission&& acquired) {
    Task task{&req, acquired.epId, acquired.begin, acquired.end, std::move(acquired.token)};
    futs.push_back(task.ret.get_future());
    pool[acquired.workerId]->Submit(std::move(task));
  };

  auto drainForPipelineLimit = [&]() -> bool {
    while (static_cast<int>(futs.size()) >= maxOutstandingChunks) {
      drainOne();
      if (hasFail) {
        drainAll();
        return false;
      }
    }
    return true;
  };

  if (cfg.policy == ExecutorSplitPolicy::Static) {
    auto splits = SplitWork(req);
    std::vector<int> nextBegin;
    std::vector<bool> pressureSeen;
    nextBegin.reserve(splits.size());
    for (const auto& split : splits) {
      nextBegin.push_back(split.begin);
      pressureSeen.push_back(false);
    }

    AdmissionWaitStats staticWaitStats;
    bool hasRemaining = true;
    while (hasRemaining) {
      hasRemaining = false;
      bool admittedAny = false;
      AdmissionResult bestFailure;
      AdmissionCandidate bestFailedCandidate;
      bool hasBestFailure = false;
      int lastAdmissionWr = 1;
      int lastRequiredFree = 1;
      int lastChunkBudgetWr = 1;

      for (size_t splitIdx = 0; splitIdx < splits.size(); ++splitIdx) {
        const WorkSplit& split = splits[splitIdx];
        int begin = nextBegin[splitIdx];
        if (begin >= split.end) continue;
        hasRemaining = true;
        if (!drainForPipelineLimit()) return failedRet;

        auto candidates = BuildStaticCandidate(req, split.epId);
        if (candidates.empty()) {
          drainAll();
          return {StatusCode::ERR_RDMA_OP,
                  "SQ admission failed: static target ep=" + std::to_string(split.epId) +
                      " is degraded or unavailable"};
        }
        const AdmissionCandidate& candidate = candidates.front();
        const int chunkBudgetWr = ComputeAdmissionChunkBudgetWr(req, split.epId, cfg);
        const int end = std::min(split.end, begin + chunkBudgetWr);
        const int admissionWr = EstimateWrForRange(req, begin, end, split.epId);
        const int requiredFree =
            ComputeRequiredFree(candidate, admissionWr, pressureSeen[splitIdx], chunkBudgetWr, cfg);
        lastAdmissionWr = admissionWr;
        lastRequiredFree = requiredFree;
        lastChunkBudgetWr = chunkBudgetWr;

        AdmissionResult result;
        if (candidate.sq->TryAcquireAdmission(admissionWr, requiredFree, &result)) {
          AcquiredAdmission acquired;
          acquired.ok = true;
          acquired.epId = split.epId;
          acquired.workerId = split.workerId;
          acquired.begin = begin;
          acquired.end = end;
          acquired.admissionWr = admissionWr;
          acquired.token = AdmissionToken(candidate.sq, admissionWr);
          RecordQueuedWrHighWatermark(req.eps[split.epId], candidate.sq->QueuedDepth());
          submitAcquired(std::move(acquired));
          nextBegin[splitIdx] = end;
          pressureSeen[splitIdx] = false;
          admittedAny = true;
          continue;
        }

        pressureSeen[splitIdx] = true;
        if (!hasBestFailure || result.snapshot.effectiveLoad < bestFailure.snapshot.effectiveLoad) {
          hasBestFailure = true;
          bestFailure = result;
          bestFailedCandidate = candidate;
        }
      }

      if (!hasRemaining || admittedAny) continue;

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        if (hasBestFailure) RecordAdmissionTimeout(req.eps[bestFailedCandidate.epId]);
        drainAll();
        return hasFail
                   ? failedRet
                   : BuildAdmissionTimeoutRet(req, activeQps, cfg, lastAdmissionWr,
                                              lastRequiredFree, lastChunkBudgetWr, staticWaitStats,
                                              hasBestFailure ? bestFailedCandidate.epId : -1);
      }

      if (!hasBestFailure) {
        std::this_thread::sleep_until(
            std::min(deadline, now + std::chrono::microseconds(cfg.waitSliceUs)));
        continue;
      }

      const auto sliceDeadline =
          std::min(deadline, now + std::chrono::microseconds(cfg.waitSliceUs));
      const auto waitStart = std::chrono::steady_clock::now();
      bestFailedCandidate.sq->WaitForAdmissionChange(sliceDeadline, bestFailure.snapshot.epoch);
      const auto waitedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - waitStart)
                                .count();
      staticWaitStats.waitCount++;
      staticWaitStats.waitUs += static_cast<uint64_t>(std::max<int64_t>(0, waitedUs));
      RecordAdmissionWait(req.eps[bestFailedCandidate.epId],
                          static_cast<uint64_t>(std::max<int64_t>(0, waitedUs)));
    }
  } else if (cfg.policy == ExecutorSplitPolicy::LeastLoaded) {
    const int chunkBudgetWr = ComputeMinAdmissionChunkBudgetWr(req, activeQps, cfg);
    int begin = 0;
    while (begin < totalBatchSize) {
      if (!drainForPipelineLimit()) return failedRet;
      const int end = std::min(totalBatchSize, begin + chunkBudgetWr);
      // Endpoint maxSge is assumed uniform across a session, matching the
      // verbs post path, so ep0 is a stable estimator before final QP choice.
      const int admissionWr = EstimateWrForRange(req, begin, end, 0);
      AcquiredAdmission acquired = TryAcquireAdmissionForCandidates(
          req, begin, end, admissionWr, std::nullopt, cfg, numWorker, deadline);
      if (!acquired.ok) {
        drainAll();
        return hasFail ? failedRet : acquired.error;
      }
      submitAcquired(std::move(acquired));
      begin = end;
    }
  } else {
    int begin = 0;
    while (begin < totalBatchSize) {
      if (!drainForPipelineLimit()) return failedRet;
      AcquiredAdmission acquired =
          TryAcquireCapacityAdmission(req, begin, totalBatchSize, cfg, numWorker, deadline);
      if (!acquired.ok) {
        drainAll();
        return hasFail ? failedRet : acquired.error;
      }
      begin = acquired.end;
      submitAcquired(std::move(acquired));
    }
  }

  drainAll();
  if (hasFail) return failedRet;
  MORI_IO_TRACE("MultithreadExecutor admission submit request for RdmaBatchReadWrite done");
  return {StatusCode::IN_PROGRESS, ""};
}

void MultithreadExecutor::Start() {
  for (auto& worker : pool) {
    worker->Start();
  }
}

void MultithreadExecutor::Shutdown() {
  for (auto& worker : pool) {
    worker->Shutdown();
  }
}

}  // namespace io
}  // namespace mori
