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
#include "src/io/rdma/common.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <utility>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

enum class PostSendOpKind : uint8_t {
  BatchData = 0,
  Notification,
};

static int GetSqBackoffTimeoutUs() {
  static const int kBackoffTimeoutUs = []() {
    int v = 10000;
    env::Override("MORI_IO_SQ_BACKOFF_TIMEOUT_US", v, mori::env::detail::ParsePositiveInt);
    const char* raw = std::getenv("MORI_IO_SQ_BACKOFF_TIMEOUT_US");
    MORI_IO_INFO("MORI_IO_SQ_BACKOFF_TIMEOUT_US raw={} effective={} us",
                 raw != nullptr ? raw : "<unset>", v);
    return v;
  }();
  return kBackoffTimeoutUs;
}

static int GetSqSignalIntervalWr() {
  static const int kSignalIntervalWr = []() {
    int v = 0;
    const char* raw = std::getenv("MORI_IO_SQ_SIGNAL_INTERVAL_WR");
    if (raw != nullptr && raw[0] != '\0') {
      errno = 0;
      char* end = nullptr;
      long parsed = std::strtol(raw, &end, 10);
      if (end != raw && *end == '\0' && errno == 0 && parsed >= 0 &&
          parsed <= std::numeric_limits<int>::max()) {
        v = static_cast<int>(parsed);
      } else {
        MORI_IO_WARN("Ignore invalid env MORI_IO_SQ_SIGNAL_INTERVAL_WR={}", raw);
      }
    }
    MORI_IO_INFO("MORI_IO_SQ_SIGNAL_INTERVAL_WR raw={} configured={} WR",
                 raw != nullptr ? raw : "<unset>", v);
    return v;
  }();
  return kSignalIntervalWr;
}

int GetSqResumeWatermarkWrForDepth(int maxDepth) {
  static const int kResumeWatermarkWr = []() {
    int v = 0;
    const char* raw = std::getenv("MORI_IO_SQ_RESUME_WATERMARK_WR");
    if (raw != nullptr && raw[0] != '\0') {
      errno = 0;
      char* end = nullptr;
      long parsed = std::strtol(raw, &end, 10);
      if (end != raw && *end == '\0' && errno == 0 && parsed >= 0 &&
          parsed <= std::numeric_limits<int>::max()) {
        v = static_cast<int>(parsed);
      } else {
        MORI_IO_WARN("Ignore invalid env MORI_IO_SQ_RESUME_WATERMARK_WR={}", raw);
      }
    }
    MORI_IO_INFO("MORI_IO_SQ_RESUME_WATERMARK_WR raw={} configured={} WR",
                 raw != nullptr ? raw : "<unset>", v);
    return v;
  }();
  if (maxDepth > 0) return std::min(kResumeWatermarkWr, maxDepth);
  return kResumeWatermarkWr;
}

static int64_t SteadyClockNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static bool IsTerminalDegradeReason(SqDegradeReason reason) {
  return reason == SqDegradeReason::PartialPostOrphaned || reason == SqDegradeReason::FatalCqe ||
         reason == SqDegradeReason::EndpointTeardown;
}

static void SetReserveResult(ReserveResult* result, SqReserveFailureKind kind, int depth,
                             int requested, int maxDepth, int backoffCount) {
  if (result == nullptr) return;
  result->kind = kind;
  result->depth = depth;
  result->requested = requested;
  result->maxDepth = maxDepth;
  result->backoffCount = backoffCount;
}

SqController::SqController(int maxDepth, int resumeWatermark)
    : maxDepth_(std::max(0, maxDepth)),
      resumeWatermark_(maxDepth_ > 0 ? std::max(0, std::min(resumeWatermark, maxDepth_))
                                     : std::max(0, resumeWatermark)) {}

bool SqController::Reserve(int wrCount, ReserveOptions opts, ReserveResult* result) {
  SetReserveResult(result, SqReserveFailureKind::None, Depth(), wrCount, maxDepth_, 0);
  if (wrCount <= 0 || maxDepth_ <= 0) return true;
  if (wrCount > maxDepth_) {
    SetReserveResult(result, SqReserveFailureKind::ExceedsCapacity, Depth(), wrCount, maxDepth_, 0);
    return false;
  }

  const int timeoutUs = opts.timeoutUs > 0 ? opts.timeoutUs : GetSqBackoffTimeoutUs();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeoutUs);
  int backoff = 0;
  bool pressureSeen = false;

  while (true) {
    if (IsDegraded()) {
      SetReserveResult(result,
                       IsTerminalDegraded() ? SqReserveFailureKind::TerminalDegraded
                                            : SqReserveFailureKind::Degraded,
                       Depth(), wrCount, maxDepth_, backoff);
      return false;
    }

    int cur = depth_.load(std::memory_order_relaxed);
    if (cur < 0) cur = 0;
    const int freeSlots = std::max(0, maxDepth_ - cur);
    int requiredFree = wrCount;
    if (pressureSeen && resumeWatermark_ > 0) {
      requiredFree = std::max(requiredFree, resumeWatermark_);
    }
    if (freeSlots >= requiredFree) {
      int expected = cur;
      if (depth_.compare_exchange_weak(expected, cur + wrCount, std::memory_order_relaxed)) {
        SetReserveResult(result, SqReserveFailureKind::None, cur + wrCount, wrCount, maxDepth_,
                         backoff);
        return true;
      }
      backoff++;
      continue;
    }

    pressureSeen = true;
    if (std::chrono::steady_clock::now() >= deadline) {
      SetReserveResult(result, SqReserveFailureKind::Timeout, cur, wrCount, maxDepth_, backoff);
      return false;
    }

    std::unique_lock<std::mutex> lock(mu_);
    const uint64_t observedEpoch = epoch_;
    auto waitPred = [&]() -> bool {
      if (degraded_.load(std::memory_order_relaxed)) return true;
      int depth = depth_.load(std::memory_order_relaxed);
      if (depth < 0) depth = 0;
      const int freeSlots = std::max(0, maxDepth_ - depth);
      int requiredFree = wrCount;
      if (pressureSeen && resumeWatermark_ > 0) {
        requiredFree = std::max(requiredFree, resumeWatermark_);
      }
      if (freeSlots >= requiredFree) return true;
      return epoch_ != observedEpoch;
    };
    if (!cv_.wait_until(lock, deadline, waitPred)) {
      SetReserveResult(result, SqReserveFailureKind::Timeout, Depth(), wrCount, maxDepth_, backoff);
      return false;
    }
    backoff++;
  }
}

bool SqController::RecheckBeforePost(int reservedWrCount, ReserveResult* result) {
  if (!IsDegraded()) {
    SetReserveResult(result, SqReserveFailureKind::None, Depth(), reservedWrCount, maxDepth_, 0);
    return true;
  }
  Release(reservedWrCount);
  SetReserveResult(result,
                   IsTerminalDegraded() ? SqReserveFailureKind::TerminalDegraded
                                        : SqReserveFailureKind::Degraded,
                   Depth(), reservedWrCount, maxDepth_, 0);
  return false;
}

void SqController::ReleaseInternal(int wrCount) {
  if (wrCount <= 0) return;
  int cur = depth_.load(std::memory_order_relaxed);
  while (true) {
    const int next = std::max(0, cur - wrCount);
    if (depth_.compare_exchange_weak(cur, next, std::memory_order_relaxed)) break;
  }
}

void SqController::Release(int wrCount) {
  ReleaseInternal(wrCount);
  std::lock_guard<std::mutex> lock(mu_);
  NotifyStateChangedLocked();
}

bool SqController::MarkDegraded(SqDegradeReason reason) {
  bool changed = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const bool wasDegraded = degraded_.exchange(true, std::memory_order_relaxed);
    changed = !wasDegraded;
    if (IsTerminalDegradeReason(reason)) {
      const bool wasTerminal = terminalDegraded_.exchange(true, std::memory_order_relaxed);
      changed = changed || !wasTerminal;
    }
    NotifyStateChangedLocked();
  }
  return changed;
}

void SqController::ReleaseDrainedOrphaned(int wrCount) {
  ReleaseInternal(wrCount);
  std::lock_guard<std::mutex> lock(mu_);
  NotifyStateChangedLocked();
}

std::shared_lock<std::shared_mutex> SqController::AcquireSubmitGuard() {
  return std::shared_lock<std::shared_mutex>(submitMu_);
}

std::unique_lock<std::shared_mutex> SqController::AcquireRecoveryGuard() {
  return std::unique_lock<std::shared_mutex>(submitMu_);
}

int SqController::Depth() const { return depth_.load(std::memory_order_relaxed); }

int SqController::MaxDepth() const { return maxDepth_; }

bool SqController::IsDegraded() const { return degraded_.load(std::memory_order_relaxed); }

bool SqController::IsTerminalDegraded() const {
  return terminalDegraded_.load(std::memory_order_relaxed);
}

void SqController::NotifyStateChangedLocked() {
  epoch_++;
  cv_.notify_all();
}

void RecordSqPollAttempt(const EpPair& ep) {
  if (!ep.sqCqeDiagnostics) return;
  ep.sqCqeDiagnostics->lastPollAttemptTimeUs.store(SteadyClockNowUs(), std::memory_order_relaxed);
}

void RecordSqPollCqes(const EpPair& ep, int cqeCount) {
  if (!ep.sqCqeDiagnostics || cqeCount <= 0) return;
  ep.sqCqeDiagnostics->lastNonEmptyCqeTimeUs.store(SteadyClockNowUs(), std::memory_order_relaxed);
  ep.sqCqeDiagnostics->recentCqeCount.fetch_add(static_cast<uint64_t>(cqeCount),
                                                std::memory_order_relaxed);
}

void RecordSqBatchReleaseWr(const EpPair& ep, int wrCount) {
  if (!ep.sqCqeDiagnostics || wrCount <= 0) return;
  ep.sqCqeDiagnostics->recentBatchReleaseWr.fetch_add(static_cast<uint64_t>(wrCount),
                                                      std::memory_order_relaxed);
}

static int EpSqDepth(const EpPair& ep) {
  if (ep.sq) return ep.sq->Depth();
  return ep.sqDepth ? ep.sqDepth->load(std::memory_order_relaxed) : -1;
}

static int EpMaxSqDepth(const EpPair& ep) {
  if (ep.sq) return ep.sq->MaxDepth();
  return ep.maxSqDepth;
}

static bool EpIsDegraded(const EpPair& ep) {
  if (ep.sq) return ep.sq->IsDegraded();
  return ep.degraded && ep.degraded->load(std::memory_order_relaxed);
}

static bool EpIsTerminalDegraded(const EpPair& ep) {
  if (ep.sq) return ep.sq->IsTerminalDegraded();
  return false;
}

static void SetSqReserveFailureKind(SqReserveFailureKind* out, SqReserveFailureKind kind) {
  if (out != nullptr) *out = kind;
}

static void AppendHint(std::string* message, const std::string& hint) {
  if (message == nullptr || hint.empty()) return;
  message->append(" Hint: ");
  message->append(hint);
}

static std::string BuildNotifyHint() {
  return "consider increasing notifPerQp / MORI_IO_QP_MAX_RECV_WR, or setting "
         "MORI_IO_ENABLE_NOTIFICATION=0 if inbound notification is not required";
}

static std::string FormatDiagnosticTimeAge(int64_t nowUs, int64_t timestampUs) {
  if (timestampUs <= 0) return "never";
  return std::to_string(std::max<int64_t>(0, nowUs - timestampUs)) + "us_ago";
}

static std::string BuildSqCqeDiagnosticHint(const EpPair& ep) {
  if (!ep.sqCqeDiagnostics) return "SQ/CQE diagnostics unavailable.";

  const int64_t nowUs = SteadyClockNowUs();
  const int64_t lastPollAttempt =
      ep.sqCqeDiagnostics->lastPollAttemptTimeUs.load(std::memory_order_relaxed);
  const int64_t lastNonEmptyCqe =
      ep.sqCqeDiagnostics->lastNonEmptyCqeTimeUs.load(std::memory_order_relaxed);
  const uint64_t recentCqeCount =
      ep.sqCqeDiagnostics->recentCqeCount.exchange(0, std::memory_order_relaxed);
  const uint64_t recentBatchReleaseWr =
      ep.sqCqeDiagnostics->recentBatchReleaseWr.exchange(0, std::memory_order_relaxed);
  const size_t recordCount = ep.ledger ? ep.ledger->RecordCount() : 0;
  const int sqDepth = EpSqDepth(ep);

  return "SQ/CQE diagnostics since previous SQ-full timeout: lastPollAttemptTime=" +
         FormatDiagnosticTimeAge(nowUs, lastPollAttempt) +
         ", lastNonEmptyCqeTime=" + FormatDiagnosticTimeAge(nowUs, lastNonEmptyCqe) +
         ", recentCqeCount=" + std::to_string(recentCqeCount) +
         ", recentBatchReleaseWr=" + std::to_string(recentBatchReleaseWr) +
         ", ledgerRecordCount=" + std::to_string(recordCount) +
         ", currentSqDepth=" + std::to_string(sqDepth) + ".";
}

static std::string BuildSqDepthHint(const EpPair& ep, size_t workerLocalQpCount,
                                    int effectivePostBatchSize, int effectiveSignalIntervalWr,
                                    PostSendOpKind opKind, SqReserveFailureKind failureKind) {
  if (failureKind == SqReserveFailureKind::Degraded ||
      failureKind == SqReserveFailureKind::TerminalDegraded)
    return {};

  if (opKind == PostSendOpKind::Notification) {
    std::string hint;
    if (failureKind == SqReserveFailureKind::Timeout) {
      hint =
          "if notification SEND completions are expected to drain shortly, try increasing "
          "MORI_IO_SQ_BACKOFF_TIMEOUT_US (current value " +
          std::to_string(GetSqBackoffTimeoutUs()) +
          " us); otherwise try increasing MORI_IO_QP_MAX_SEND_WR";
    } else {
      hint = "try increasing MORI_IO_QP_MAX_SEND_WR";
    }
    hint += ", and " + BuildNotifyHint() + ".";
    if (failureKind == SqReserveFailureKind::Timeout) {
      hint += " " + BuildSqCqeDiagnosticHint(ep);
    }
    return hint;
  }

  std::string hint;
  if (failureKind == SqReserveFailureKind::Timeout) {
    hint =
        "if completions are expected to drain shortly, try increasing "
        "MORI_IO_SQ_BACKOFF_TIMEOUT_US (current value " +
        std::to_string(GetSqBackoffTimeoutUs()) + " us); otherwise try increasing ";
  } else {
    hint = "try increasing ";
  }

  hint += "MORI_IO_QP_MAX_SEND_WR";
  if (effectivePostBatchSize > 0) {
    hint += ", reducing RdmaBackendConfig::postBatchSize (current effective value " +
            std::to_string(effectivePostBatchSize) + ")";
  }
  if (workerLocalQpCount > 0) {
    const int sessionQpPerTransfer =
        ep.qpPerTransfer > 0 ? ep.qpPerTransfer : static_cast<int>(workerLocalQpCount);
    const int numWorkerThreads = ep.numWorkerThreads > 0 ? ep.numWorkerThreads : 0;
    hint +=
        ", or increasing RdmaBackendConfig::qpPerTransfer only if additional QPs are "
        "actually used by workers";
    hint += " (worker-local QP count for this call=" + std::to_string(workerLocalQpCount) +
            ", session qpPerTransfer=" + std::to_string(sessionQpPerTransfer) +
            ", numWorkerThreads=" + std::to_string(numWorkerThreads) + ")";
  }
  hint += ". Current per-QP send WR limit is " + std::to_string(EpMaxSqDepth(ep)) + ".";
  hint += " MORI_IO_SQ_SIGNAL_INTERVAL_WR configured=" + std::to_string(GetSqSignalIntervalWr()) +
          ", effective=" + std::to_string(effectiveSignalIntervalWr) + " WR.";
  if (failureKind == SqReserveFailureKind::Timeout) {
    hint += " " + BuildSqCqeDiagnosticHint(ep);
  }
  return hint;
}

static std::string BuildPostSendFailureHint(int ret, const EpPair& ep, size_t qpCount,
                                            int effectivePostBatchSize, const ibv_send_wr* badWr,
                                            PostSendOpKind opKind) {
  if (ret == ENOMEM) {
    std::string hint;
    if (opKind == PostSendOpKind::Notification) {
      hint =
          "provider reported ENOMEM while posting notification SENDs; try increasing "
          "MORI_IO_QP_MAX_SEND_WR and MORI_IO_QP_MAX_CQE";
      hint += ", and " + BuildNotifyHint();
      hint += ".";
      return hint;
    }

    hint =
        "provider reported ENOMEM while posting WRs; try increasing MORI_IO_QP_MAX_SEND_WR and "
        "MORI_IO_QP_MAX_CQE";
    if (effectivePostBatchSize > 0) {
      hint += ", reducing RdmaBackendConfig::postBatchSize (current effective value " +
              std::to_string(effectivePostBatchSize) + ")";
    }
    if (qpCount > 0) {
      hint += ", or increasing RdmaBackendConfig::qpPerTransfer (current transfer uses " +
              std::to_string(qpCount) + " QP(s)) if additional QPs are available";
    }
    hint += ".";
    return hint;
  }

  if (ret == EINVAL) {
    if (opKind == PostSendOpKind::Notification) {
      std::string hint =
          "provider rejected the notification SEND as invalid; verify notification is enabled on "
          "both peers and the endpoint/QP is still healthy";
      hint += ", or disable MORI_IO_ENABLE_NOTIFICATION if inbound notification is not required";
      hint += ".";
      return hint;
    }

    if (badWr != nullptr && static_cast<uint32_t>(badWr->num_sge) > ep.local.handle.maxSge) {
      return "the failing WR uses num_sge=" + std::to_string(badWr->num_sge) +
             ", which exceeds endpoint max_send_sge=" + std::to_string(ep.local.handle.maxSge) +
             "; reduce scatter/gather fan-out or, if the device supports it, increase "
             "MORI_IO_QP_MAX_MSG_SGE / MORI_IO_QP_MAX_SGE.";
    }

    std::string hint =
        "provider rejected the WR as invalid; verify WR flags/opcode, lkey/rkey, and scatter/"
        "gather layout";
    if (badWr != nullptr) {
      hint += " (failing WR num_sge=" + std::to_string(badWr->num_sge) + ")";
    }
    hint += ".";
    return hint;
  }

  return {};
}

uint64_t MakeNotifSendWrId(TransferUniqueId id) {
  if ((id & kNotifSendWrIdTag) != 0) {
    MORI_IO_ERROR("MakeNotifSendWrId: TransferUniqueId {} has bit 63 set; masking reserved tag",
                  id);
    id &= ~kNotifSendWrIdTag;
  }
  return kNotifSendWrIdTag | id;
}

// SQ depth is an admission counter only. It does not publish data dependencies
// across threads, so relaxed atomics are sufficient for correctness.
static bool TryReserveSqDepth(const EpPair& ep, int wrCount, int epId, const char* opTag,
                              std::string* errMsg, SqReserveFailureKind* failureKind = nullptr) {
  if (wrCount <= 0) return true;
  const int kBackoffTimeoutUs = GetSqBackoffTimeoutUs();
  if (ep.sq) {
    ReserveOptions opts;
    opts.timeoutUs = kBackoffTimeoutUs;
    ReserveResult result;
    if (ep.sq->Reserve(wrCount, opts, &result)) return true;

    SetSqReserveFailureKind(failureKind, result.kind);
    if (result.kind == SqReserveFailureKind::TerminalDegraded ||
        result.kind == SqReserveFailureKind::Degraded) {
      if (errMsg) {
        *errMsg = EpIsTerminalDegraded(ep) ? "EP is terminal degraded, rejecting new submissions"
                                           : "EP is degraded, rejecting new submissions";
      }
      return false;
    }
    if (result.kind == SqReserveFailureKind::ExceedsCapacity) {
      MORI_IO_WARN("SQ request exceeds capacity ({}): ep={} requested={} max={}", opTag, epId,
                   wrCount, result.maxDepth);
      if (errMsg) {
        *errMsg = "SQ request exceeds capacity (" + std::string(opTag) +
                  "): ep=" + std::to_string(epId) + " requested=" + std::to_string(wrCount) +
                  " max=" + std::to_string(result.maxDepth);
      }
      return false;
    }
    if (result.kind == SqReserveFailureKind::Timeout) {
      MORI_IO_WARN(
          "SQ full timeout ({}): ep={} depth={} requested={} max={} after {} us (backoff={})",
          opTag, epId, result.depth, wrCount, result.maxDepth, kBackoffTimeoutUs,
          result.backoffCount);
      if (errMsg) {
        *errMsg = "SQ full (" + std::string(opTag) + "): ep=" + std::to_string(epId) +
                  " depth=" + std::to_string(result.depth) +
                  " requested=" + std::to_string(wrCount) +
                  " max=" + std::to_string(result.maxDepth);
      }
      return false;
    }
    if (errMsg) *errMsg = "SQ reserve failed";
    return false;
  }

  if (!ep.sqDepth) return true;
  if (EpIsDegraded(ep)) {
    SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Degraded);
    if (errMsg) *errMsg = "EP is degraded, rejecting new submissions";
    return false;
  }
  const int maxSqDepth = EpMaxSqDepth(ep);
  if (wrCount > maxSqDepth) {
    SetSqReserveFailureKind(failureKind, SqReserveFailureKind::ExceedsCapacity);
    MORI_IO_WARN("SQ request exceeds capacity ({}): ep={} requested={} max={}", opTag, epId,
                 wrCount, maxSqDepth);
    if (errMsg) {
      *errMsg = "SQ request exceeds capacity (" + std::string(opTag) +
                "): ep=" + std::to_string(epId) + " requested=" + std::to_string(wrCount) +
                " max=" + std::to_string(maxSqDepth);
    }
    return false;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(kBackoffTimeoutUs);
  int backoff = 0;
  int cur = ep.sqDepth->load(std::memory_order_relaxed);
  if (cur < 0) cur = 0;  // defensive: clamp stale negative depth
  while (true) {
    // Re-check degraded state while waiting to avoid accepting new submissions
    // after another thread has marked this EP as degraded.
    if (EpIsDegraded(ep)) {
      SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Degraded);
      if (errMsg) *errMsg = "EP is degraded, rejecting new submissions";
      return false;
    }
    if (cur + wrCount > maxSqDepth) {
      if (std::chrono::steady_clock::now() >= deadline) {
        SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Timeout);
        MORI_IO_WARN(
            "SQ full timeout ({}): ep={} depth={} requested={} max={} after {} us (backoff={})",
            opTag, epId, cur, wrCount, maxSqDepth, kBackoffTimeoutUs, backoff);
        if (errMsg) {
          *errMsg = "SQ full (" + std::string(opTag) + "): ep=" + std::to_string(epId) +
                    " depth=" + std::to_string(cur) + " requested=" + std::to_string(wrCount) +
                    " max=" + std::to_string(maxSqDepth);
        }
        return false;
      }
      // Phased backoff: short polite yields, then tiny sleeps to reduce CPU burn.
      if (backoff < 16) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(2));
      }
      backoff++;
      cur = ep.sqDepth->load(std::memory_order_relaxed);
      continue;
    }
    if (ep.sqDepth->compare_exchange_weak(cur, cur + wrCount, std::memory_order_relaxed))
      return true;
    if (backoff < 16) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(2));
    }
    backoff++;
  }
}

static void ReleaseSqDepth(const EpPair& ep, int wrCount) {
  if (wrCount <= 0) return;
  if (ep.sq) {
    ep.sq->Release(wrCount);
    return;
  }
  if (!ep.sqDepth) return;
  ep.sqDepth->fetch_sub(wrCount, std::memory_order_relaxed);
}

static bool RecheckSqBeforePost(const EpPair& ep, int reservedWrCount, int epId, const char* opTag,
                                std::string* errMsg, SqReserveFailureKind* failureKind = nullptr) {
  if (reservedWrCount <= 0) return true;
  if (ep.sq) {
    ReserveResult result;
    if (ep.sq->RecheckBeforePost(reservedWrCount, &result)) return true;
    SetSqReserveFailureKind(failureKind, result.kind);
    if (errMsg) {
      *errMsg =
          "EP is " +
          std::string(result.kind == SqReserveFailureKind::TerminalDegraded ? "terminal degraded"
                                                                            : "degraded") +
          ", rejecting reserved " + std::string(opTag) +
          " submission on ep=" + std::to_string(epId);
    }
    return false;
  }
  if (EpIsDegraded(ep)) {
    ReleaseSqDepth(ep, reservedWrCount);
    SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Degraded);
    if (errMsg) {
      *errMsg = "EP is degraded, rejecting reserved " + std::string(opTag) +
                " submission on ep=" + std::to_string(epId);
    }
    return false;
  }
  return true;
}

static void MarkEpDegradedFromSubmitFailure(const EpPair& ep, SqDegradeReason reason) {
  if (ep.sq) {
    ep.sq->MarkDegraded(reason);
  } else if (ep.degraded) {
    ep.degraded->store(true, std::memory_order_relaxed);
  }
}

static void FailUniqueSubmissionMetasFromSubmitFailure(const std::vector<SubmissionRecord>& records,
                                                       const std::string& message) {
  std::unordered_set<CqCallbackMeta*> seen;
  for (const auto& record : records) {
    if (!record.meta || !seen.insert(record.meta.get()).second) continue;
    if (record.batchSize > 0) {
      (void)record.meta->finishedBatchSize.fetch_add(static_cast<uint32_t>(record.batchSize),
                                                     std::memory_order_relaxed);
    }
    TransferStatus* statusPtr = record.meta->status;
    if (statusPtr != nullptr) {
      statusPtr->Update(StatusCode::ERR_RDMA_OP, message);
      record.meta->status = nullptr;
    }
  }
}

static void ExtractAndFailOrphanedRecordsFromSubmitFailure(const EpPair& ep,
                                                           const std::string& message) {
  if (!ep.ledger) return;
  std::vector<SubmissionRecord> orphaned;
  ep.ledger->ExtractOrphanedRecords(&orphaned);
  if (!orphaned.empty()) {
    FailUniqueSubmissionMetasFromSubmitFailure(orphaned, message);
  }
}

void MovePendingUnsignaledToOrphanedForEndpoint(
    const EpPairVec& eps, size_t epId, std::vector<int>& epWrsSinceSignal,
    std::vector<size_t>& epMergedSinceSignal, const std::shared_ptr<CqCallbackMeta>& callbackMeta,
    const std::string& message, const char* context,
    std::shared_lock<std::shared_mutex>* heldSubmitGuard) {
  if (epId >= eps.size() || epId >= epWrsSinceSignal.size() || epId >= epMergedSinceSignal.size()) {
    return;
  }
  if (epWrsSinceSignal[epId] <= 0) return;

  const int wrCount = epWrsSinceSignal[epId];
  const size_t mergedReq = epMergedSinceSignal[epId];
  if (heldSubmitGuard != nullptr && heldSubmitGuard->owns_lock()) {
    heldSubmitGuard->unlock();
  }

  std::unique_lock<std::shared_mutex> recoveryGuard;
  if (eps[epId].sq) recoveryGuard = eps[epId].sq->AcquireRecoveryGuard();
  MarkEpDegradedFromSubmitFailure(eps[epId], SqDegradeReason::PartialPostOrphaned);

  MORI_IO_WARN(
      "{}: moving pending unsignaled WRs on ep {} (wrCount={}, mergedReq={}) to orphaned and "
      "marking terminal degraded",
      context != nullptr ? context : "RDMA submit failure", epId, wrCount, mergedReq);
  if (eps[epId].ledger) {
    eps[epId].ledger->InsertOrphaned(wrCount, callbackMeta, static_cast<int>(mergedReq));
    ExtractAndFailOrphanedRecordsFromSubmitFailure(eps[epId], message);
  } else {
    MORI_IO_WARN(
        "EP {} has pending unsignaled WRs but no submission ledger; SQ credit may remain stale "
        "until endpoint restart",
        epId);
  }

  epWrsSinceSignal[epId] = 0;
  epMergedSinceSignal[epId] = 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Rdma Utilities                                         */
/* ---------------------------------------------------------------------------------------------- */

RdmaOpRet RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;
  (void)status;

  std::string reserveErr;
  int reserved = 0;
  for (size_t i = 0; i < eps.size(); i++) {
    SqReserveFailureKind reserveFailure = SqReserveFailureKind::None;
    if (!TryReserveSqDepth(eps[i], 1, i, "notify", &reserveErr, &reserveFailure)) {
      AppendHint(&reserveErr, BuildSqDepthHint(eps[i], eps.size(), -1, 0,
                                               PostSendOpKind::Notification, reserveFailure));
      for (int j = 0; j < reserved; ++j) ReleaseSqDepth(eps[j], 1);
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }
    reserved++;
  }

  for (size_t i = 0; i < eps.size(); i++) {
    std::shared_lock<std::shared_mutex> submitGuard;
    if (eps[i].sq) submitGuard = eps[i].sq->AcquireSubmitGuard();
    SqReserveFailureKind recheckFailure = SqReserveFailureKind::None;
    if (!RecheckSqBeforePost(eps[i], 1, static_cast<int>(i), "notify", &reserveErr,
                             &recheckFailure)) {
      AppendHint(&reserveErr, BuildSqDepthHint(eps[i], eps.size(), -1, 0,
                                               PostSendOpKind::Notification, recheckFailure));
      for (int j = static_cast<int>(i) + 1; j < static_cast<int>(eps.size()); ++j) {
        ReleaseSqDepth(eps[j], 1);
      }
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }

    const application::RdmaEndpoint& ep = eps[i].local;
    NotifMessage msg{id, static_cast<int>(i), static_cast<int>(eps.size())};

    struct ibv_sge sge{};
    sge.addr = reinterpret_cast<uintptr_t>(&msg);
    sge.length = sizeof(NotifMessage);
    sge.lkey = 0;

    struct ibv_send_wr wr{};
    wr.wr_id = MakeNotifSendWrId(id);
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
    if (ret != 0) {
      // WR i was reserved but failed to post if bad_wr points at this WR.
      if (bad_wr == &wr) ReleaseSqDepth(eps[i], 1);
      // Any remaining endpoints are reserved but not posted yet.
      for (int j = i + 1; j < eps.size(); ++j) ReleaseSqDepth(eps[j], 1);
      std::string message =
          "ibv_post_send (notify) failed with " + std::to_string(ret) + ": " + strerror(ret);
      AppendHint(&message, BuildPostSendFailureHint(ret, eps[i], eps.size(), -1, bad_wr,
                                                    PostSendOpKind::Notification));
      return {StatusCode::ERR_RDMA_OP, std::move(message)};
    }
  }

  return {StatusCode::IN_PROGRESS, ""};
}

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                             const SizeVec& localOffsets,
                             const application::RdmaMemoryRegion& remote,
                             const SizeVec& remoteOffsets, const SizeVec& sizes,
                             std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                             bool isRead, int postBatchSize) {
  MORI_IO_FUNCTION_TIMER;

  if ((localOffsets.size() != remoteOffsets.size()) || (sizes.size() != remoteOffsets.size())) {
    return {StatusCode::ERR_INVALID_ARGS,
            "lengths of local offsets, remote offsets or sizes mismatch"};
  }

  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    return {StatusCode::SUCCESS, ""};
  }

  for (size_t i = 0; i < batchSize; i++) {
    if (((localOffsets[i] + sizes[i]) > local.length) ||
        ((remoteOffsets[i] + sizes[i]) > remote.length)) {
      return {StatusCode::ERR_INVALID_ARGS, "length out of range"};
    }
  }

  if (eps.empty()) {
    return {StatusCode::ERR_INVALID_ARGS, "no endpoints"};
  }

  std::vector<size_t> indices(batchSize);
  std::iota(indices.begin(), indices.end(), 0);

  if (std::is_sorted(remoteOffsets.begin(), remoteOffsets.end()) == false)
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return remoteOffsets[a] < remoteOffsets[b]; });

  struct MergedWorkRequest {
    ibv_send_wr wr{};
    std::vector<ibv_sge> sges;
    size_t totalRemoteLength = 0;
    size_t mergedRequests = 1;
  };

  const uint64_t localBaseAddr = reinterpret_cast<uint64_t>(local.addr);
  const uint64_t remoteBaseAddr = reinterpret_cast<uint64_t>(remote.addr);
  const uint32_t maxSge =
      std::max(eps[0].local.handle.maxSge, 1u);  // We assume all endpoints have the same maxSge

  std::vector<MergedWorkRequest> mergedWrs;
  mergedWrs.reserve(batchSize);

  auto start_new_wr = [&](uint64_t remoteAddr, uint64_t localAddr, uint32_t len) {
    mergedWrs.emplace_back();
    MergedWorkRequest& newWr = mergedWrs.back();
    newWr.sges.reserve(maxSge);  // keep sg_list stable
    newWr.sges.push_back(ibv_sge{.addr = localAddr, .length = len, .lkey = local.lkey});
    newWr.totalRemoteLength = len;

    newWr.wr.sg_list = newWr.sges.data();
    newWr.wr.num_sge = 1;
    newWr.wr.opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
    newWr.wr.send_flags = 0;
    newWr.wr.wr.rdma.remote_addr = remoteAddr;
    newWr.wr.wr.rdma.rkey = remote.rkey;
  };

  for (size_t i = 0; i < batchSize; ++i) {
    const size_t idx = indices[i];
    const uint64_t currentLocalAddr = localBaseAddr + localOffsets[idx];
    const uint64_t currentRemoteAddr = remoteBaseAddr + remoteOffsets[idx];
    const uint32_t currentSize32 = static_cast<uint32_t>(sizes[idx]);

    bool merged = false;
    if (!mergedWrs.empty()) {
      MergedWorkRequest& lastWr = mergedWrs.back();
      const uint64_t expectedRemoteAddr = lastWr.wr.wr.rdma.remote_addr + lastWr.totalRemoteLength;
      if (expectedRemoteAddr == currentRemoteAddr) {
        // Try to merge into last WR
        ibv_sge& lastSge = lastWr.sges.back();
        const bool localContiguous = (lastSge.addr + lastSge.length) == currentLocalAddr;

        if (localContiguous) {
          // Ensure SGE length doesn't overflow uint32_t
          const uint64_t newLen = static_cast<uint64_t>(lastSge.length) + currentSize32;
          if (newLen <= std::numeric_limits<uint32_t>::max()) {
            lastSge.length = static_cast<uint32_t>(newLen);
            lastWr.mergedRequests += 1;
            lastWr.totalRemoteLength += currentSize32;
            merged = true;
          }
        }
        if (!merged) {
          if (lastWr.sges.size() < maxSge) {
            // Append a new SGE into the same WR
            lastWr.sges.push_back(
                ibv_sge{.addr = currentLocalAddr, .length = currentSize32, .lkey = local.lkey});
            lastWr.wr.num_sge = static_cast<int>(lastWr.sges.size());
            lastWr.mergedRequests += 1;
            lastWr.totalRemoteLength += currentSize32;
            merged = true;
          }
        }
      }
    }
    if (!merged) {
      start_new_wr(currentRemoteAddr, currentLocalAddr, currentSize32);
    }
  }

  size_t mergedWrCount = mergedWrs.size();
  size_t epNum = eps.size();
  size_t epBatchSize = (mergedWrCount + epNum - 1) / epNum;
  int minMaxSqDepth = std::numeric_limits<int>::max();
  for (size_t epId = 0; epId < epNum; ++epId) {
    const int maxSqDepth = EpMaxSqDepth(eps[epId]);
    if ((eps[epId].sq || eps[epId].sqDepth) && maxSqDepth > 0) {
      minMaxSqDepth = std::min(minMaxSqDepth, maxSqDepth);
    }
  }

  if (postBatchSize == -1) {
    postBatchSize = (epBatchSize > static_cast<size_t>(std::numeric_limits<int>::max()))
                        ? std::numeric_limits<int>::max()
                        : static_cast<int>(epBatchSize);
  }
  if (minMaxSqDepth != std::numeric_limits<int>::max() && postBatchSize > minMaxSqDepth) {
    postBatchSize = minMaxSqDepth;
  }

  int effectiveSignalIntervalWr = 0;
  const int configuredSignalIntervalWr = GetSqSignalIntervalWr();
  if (configuredSignalIntervalWr > 0) {
    effectiveSignalIntervalWr = configuredSignalIntervalWr;
    if (minMaxSqDepth != std::numeric_limits<int>::max()) {
      effectiveSignalIntervalWr = std::min(effectiveSignalIntervalWr, minMaxSqDepth);
    }
    if (effectiveSignalIntervalWr > 0 && postBatchSize > effectiveSignalIntervalWr) {
      postBatchSize = effectiveSignalIntervalWr;
    }
  }
  if (postBatchSize <= 0) postBatchSize = 1;
  int numPostBatch = (mergedWrCount + postBatchSize - 1) / postBatchSize;

  // Per-EP state for adaptive signaling: track WRs and merged requests
  // accumulated since the last signaled WR on each EP.
  std::vector<int> epWrsSinceSignal(epNum, 0);
  std::vector<size_t> epMergedSinceSignal(epNum, 0);

  for (int i = 0; i < numPostBatch; i++) {
    int st = i * postBatchSize;
    int end = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    if (end - st == 0) break;
    int epId = i % epNum;
    int batchWrNum = end - st;

    // Reserve SQ depth for this batch; blocks with backoff if the SQ is full,
    // waiting for CQEs from earlier signaled WRs to drain depth.
    std::string reserveErr;
    SqReserveFailureKind reserveFailure = SqReserveFailureKind::None;
    if (!TryReserveSqDepth(eps[epId], batchWrNum, epId, "batch", &reserveErr, &reserveFailure)) {
      AppendHint(&reserveErr,
                 BuildSqDepthHint(eps[epId], epNum, postBatchSize, effectiveSignalIntervalWr,
                                  PostSendOpKind::BatchData, reserveFailure));
      for (size_t pendingEpId = 0; pendingEpId < epNum; ++pendingEpId) {
        MovePendingUnsignaledToOrphanedForEndpoint(eps, pendingEpId, epWrsSinceSignal,
                                                   epMergedSinceSignal, callbackMeta, reserveErr,
                                                   "TryReserveSqDepth failed");
      }
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }

    std::shared_lock<std::shared_mutex> submitGuard;
    if (eps[epId].sq) submitGuard = eps[epId].sq->AcquireSubmitGuard();
    SqReserveFailureKind recheckFailure = SqReserveFailureKind::None;
    if (!RecheckSqBeforePost(eps[epId], batchWrNum, epId, "batch", &reserveErr, &recheckFailure)) {
      AppendHint(&reserveErr,
                 BuildSqDepthHint(eps[epId], epNum, postBatchSize, effectiveSignalIntervalWr,
                                  PostSendOpKind::BatchData, recheckFailure));
      for (size_t pendingEpId = 0; pendingEpId < epNum; ++pendingEpId) {
        MovePendingUnsignaledToOrphanedForEndpoint(
            eps, pendingEpId, epWrsSinceSignal, epMergedSinceSignal, callbackMeta, reserveErr,
            "RecheckBeforePost failed",
            static_cast<int>(pendingEpId) == epId ? &submitGuard : nullptr);
      }
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }

    size_t mergedReqSize = 0;
    for (int j = st; j < end; j++) {
      struct ibv_send_wr& wr = mergedWrs[j].wr;
      wr.wr_id = 0;
      wr.next = (j + 1 < end) ? &mergedWrs[j + 1].wr : nullptr;
      mergedReqSize += mergedWrs[j].mergedRequests;
    }

    epWrsSinceSignal[epId] += batchWrNum;
    epMergedSinceSignal[epId] += mergedReqSize;

    bool isLastBatchForEp = ((i + epNum) >= numPostBatch);
    bool signalIntervalReached =
        effectiveSignalIntervalWr > 0 && epWrsSinceSignal[epId] >= effectiveSignalIntervalWr;
    const int epMaxSqDepth = EpMaxSqDepth(eps[epId]);
    bool sqNearFull = (eps[epId].sq || eps[epId].sqDepth) && epMaxSqDepth > 0 &&
                      (epWrsSinceSignal[epId] >= epMaxSqDepth);
    bool needSignal = isLastBatchForEp || signalIntervalReached || sqNearFull;

    struct ibv_send_wr& last = mergedWrs[end - 1].wr;
    uint64_t recordId = 0;
    if (needSignal) {
      if (!eps[epId].ledger) {
        ReleaseSqDepth(eps[epId], batchWrNum);
        return {StatusCode::ERR_RDMA_OP,
                "submission ledger is not initialized for signaled WR tracking"};
      }
      recordId = eps[epId].ledger->Insert(epWrsSinceSignal[epId], true, callbackMeta,
                                          static_cast<int>(epMergedSinceSignal[epId]));
      last.wr_id = recordId;
      last.send_flags = IBV_SEND_SIGNALED;
    }

    struct ibv_send_wr* badWr = nullptr;
    int ret = ibv_post_send(eps[epId].local.ibvHandle.qp, &mergedWrs[st].wr, &badWr);
    if (ret != 0) {
      int postedCount = 0;
      if (badWr != nullptr) {
        struct ibv_send_wr* cur = &mergedWrs[st].wr;
        while (cur != nullptr && cur != badWr && postedCount < batchWrNum) {
          ++postedCount;
          cur = cur->next;
        }
      }
      postedCount = std::max(0, std::min(postedCount, batchWrNum));
      const int unpostedCount = batchWrNum - postedCount;
      if (unpostedCount > 0) {
        ReleaseSqDepth(eps[epId], unpostedCount);
        epWrsSinceSignal[epId] = std::max(0, epWrsSinceSignal[epId] - unpostedCount);

        size_t mergedUnposted = 0;
        for (int j = st + postedCount; j < end; ++j) {
          mergedUnposted += mergedWrs[j].mergedRequests;
        }
        if (epMergedSinceSignal[epId] >= mergedUnposted) {
          epMergedSinceSignal[epId] -= mergedUnposted;
        } else {
          epMergedSinceSignal[epId] = 0;
        }
      }

      const bool lastWasPosted = (postedCount == batchWrNum);
      if (needSignal && lastWasPosted) {
        // Signaled WR was posted; CQ path (ledger->ReleaseByCqe) owns the release.
        (void)eps[epId].ledger->MarkPosted(recordId);
      } else if (needSignal) {
        // Signaled WR itself was NOT posted; remove the tentative record.
        SubmissionRecord canceled;
        (void)eps[epId].ledger->CancelTentative(recordId, &canceled);
      }

      std::string message = "ibv_post_send failed with " + std::to_string(ret) + ": " +
                            strerror(ret) + " (posted " + std::to_string(postedCount) + "/" +
                            std::to_string(batchWrNum) + " WRs)";
      AppendHint(&message, BuildPostSendFailureHint(ret, eps[epId], epNum, postBatchSize, badWr,
                                                    PostSendOpKind::BatchData));

      if (epWrsSinceSignal[epId] > 0 && (!needSignal || !lastWasPosted)) {
        MovePendingUnsignaledToOrphanedForEndpoint(eps, epId, epWrsSinceSignal, epMergedSinceSignal,
                                                   callbackMeta, message, "ibv_post_send failed",
                                                   &submitGuard);
      }

      for (size_t otherEpId = 0; otherEpId < epNum; ++otherEpId) {
        if (static_cast<int>(otherEpId) == epId) continue;
        MovePendingUnsignaledToOrphanedForEndpoint(eps, otherEpId, epWrsSinceSignal,
                                                   epMergedSinceSignal, callbackMeta, message,
                                                   "ibv_post_send failed");
      }

      return {StatusCode::ERR_RDMA_OP, std::move(message)};
    }

    if (needSignal) {
      (void)eps[epId].ledger->MarkPosted(recordId);
      epWrsSinceSignal[epId] = 0;
      epMergedSinceSignal[epId] = 0;
    }
    MORI_IO_TRACE("ibv_post_send ep index {} batch index range [{}, {})", epId, st, end);
  }
  return {StatusCode::IN_PROGRESS, ""};
}

}  // namespace io
}  // namespace mori
