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
#include <utility>

#include "mori/io/env.hpp"
#include "mori/io/logging.hpp"

namespace mori {
namespace io {

enum class SqReserveFailureKind : uint8_t {
  None = 0,
  Degraded,
  ExceedsCapacity,
  Timeout,
};

enum class PostSendOpKind : uint8_t {
  BatchData = 0,
  Notification,
};

static int GetSqBackoffTimeoutUs() {
  static const int kBackoffTimeoutUs = []() {
    int v = 10000;
    env::Override("MORI_IO_SQ_BACKOFF_TIMEOUT_US", v, mori::env::detail::ParsePositiveInt);
    return v;
  }();
  return kBackoffTimeoutUs;
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

static std::string BuildSqDepthHint(const EpPair& ep, size_t qpCount, int effectivePostBatchSize,
                                    PostSendOpKind opKind, SqReserveFailureKind failureKind) {
  if (failureKind == SqReserveFailureKind::Degraded) return {};

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
  if (qpCount > 0) {
    hint += ", or increasing RdmaBackendConfig::qpPerTransfer (current transfer uses " +
            std::to_string(qpCount) + " QP(s)) if additional QPs are available";
  }
  hint += ". Current per-QP send WR limit is " + std::to_string(ep.maxSqDepth) + ".";
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
  if (wrCount <= 0 || !ep.sqDepth) return true;
  if (ep.degraded && ep.degraded->load(std::memory_order_relaxed)) {
    SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Degraded);
    if (errMsg) *errMsg = "EP is degraded, rejecting new submissions";
    return false;
  }
  if (wrCount > ep.maxSqDepth) {
    SetSqReserveFailureKind(failureKind, SqReserveFailureKind::ExceedsCapacity);
    MORI_IO_WARN("SQ request exceeds capacity ({}): ep={} requested={} max={}", opTag, epId,
                 wrCount, ep.maxSqDepth);
    if (errMsg) {
      *errMsg = "SQ request exceeds capacity (" + std::string(opTag) +
                "): ep=" + std::to_string(epId) + " requested=" + std::to_string(wrCount) +
                " max=" + std::to_string(ep.maxSqDepth);
    }
    return false;
  }
  const int kBackoffTimeoutUs = GetSqBackoffTimeoutUs();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(kBackoffTimeoutUs);
  int backoff = 0;
  int cur = ep.sqDepth->load(std::memory_order_relaxed);
  if (cur < 0) cur = 0;  // defensive: clamp stale negative depth
  while (true) {
    // Re-check degraded state while waiting to avoid accepting new submissions
    // after another thread has marked this EP as degraded.
    if (ep.degraded && ep.degraded->load(std::memory_order_relaxed)) {
      SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Degraded);
      if (errMsg) *errMsg = "EP is degraded, rejecting new submissions";
      return false;
    }
    if (cur + wrCount > ep.maxSqDepth) {
      if (std::chrono::steady_clock::now() >= deadline) {
        SetSqReserveFailureKind(failureKind, SqReserveFailureKind::Timeout);
        MORI_IO_WARN(
            "SQ full timeout ({}): ep={} depth={} requested={} max={} after {} us (backoff={})",
            opTag, epId, cur, wrCount, ep.maxSqDepth, kBackoffTimeoutUs, backoff);
        if (errMsg) {
          *errMsg = "SQ full (" + std::string(opTag) + "): ep=" + std::to_string(epId) +
                    " depth=" + std::to_string(cur) + " requested=" + std::to_string(wrCount) +
                    " max=" + std::to_string(ep.maxSqDepth);
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
  if (wrCount <= 0 || !ep.sqDepth) return;
  ep.sqDepth->fetch_sub(wrCount, std::memory_order_relaxed);
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
      AppendHint(&reserveErr, BuildSqDepthHint(eps[i], eps.size(), -1, PostSendOpKind::Notification,
                                               reserveFailure));
      for (int j = 0; j < reserved; ++j) ReleaseSqDepth(eps[j], 1);
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }
    reserved++;
  }

  for (size_t i = 0; i < eps.size(); i++) {
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

  if (postBatchSize == -1) {
    postBatchSize = (epBatchSize > static_cast<size_t>(std::numeric_limits<int>::max()))
                        ? std::numeric_limits<int>::max()
                        : static_cast<int>(epBatchSize);
  }
  {
    int minMaxSqDepth = std::numeric_limits<int>::max();
    for (size_t epId = 0; epId < epNum; ++epId) {
      if (eps[epId].sqDepth && eps[epId].maxSqDepth > 0) {
        minMaxSqDepth = std::min(minMaxSqDepth, eps[epId].maxSqDepth);
      }
    }
    if (minMaxSqDepth != std::numeric_limits<int>::max() && postBatchSize > minMaxSqDepth)
      postBatchSize = minMaxSqDepth;
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
      AppendHint(&reserveErr, BuildSqDepthHint(eps[epId], epNum, postBatchSize,
                                               PostSendOpKind::BatchData, reserveFailure));
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
    bool sqNearFull = eps[epId].sqDepth && (epWrsSinceSignal[epId] >= eps[epId].maxSqDepth);
    bool needSignal = isLastBatchForEp || sqNearFull;

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
        // The record inserted above remains in Posted state — nothing to do here.
      } else if (needSignal) {
        // Signaled WR itself was NOT posted; remove the record we just inserted
        // and release whatever was actually posted (tracked via unpostedCount above).
        int dummy = 0;
        eps[epId].ledger->ReleaseByCqe(recordId, nullptr, &dummy);
      }

      if (postedCount > 0 && (!needSignal || !lastWasPosted)) {
        MORI_IO_WARN(
            "ibv_post_send partially posted {} / {} WRs without a posted signaled tail; "
            "marking EP {} as degraded until recovery",
            postedCount, batchWrNum, epId);
        if (eps[epId].degraded) {
          eps[epId].degraded->store(true, std::memory_order_relaxed);
        }
        // Ledger record for ALL orphaned posted WRs (including prior unsignaled batches
        // on this EP): sqDepth held by ledger until recovery.
        if (eps[epId].ledger) {
          eps[epId].ledger->InsertOrphaned(epWrsSinceSignal[epId], callbackMeta,
                                           static_cast<int>(epMergedSinceSignal[epId]));
        }
      }

      for (size_t otherEpId = 0; otherEpId < epNum; ++otherEpId) {
        if (static_cast<int>(otherEpId) == epId) continue;
        if (epWrsSinceSignal[otherEpId] <= 0) continue;
        MORI_IO_WARN(
            "ibv_post_send failed on ep {}: moving pending unsignaled WRs on ep {} "
            "(wrCount={}, mergedReq={}) to orphaned and marking degraded",
            epId, otherEpId, epWrsSinceSignal[otherEpId], epMergedSinceSignal[otherEpId]);
        if (eps[otherEpId].degraded) {
          eps[otherEpId].degraded->store(true, std::memory_order_relaxed);
        }
        if (eps[otherEpId].ledger) {
          eps[otherEpId].ledger->InsertOrphaned(epWrsSinceSignal[otherEpId], callbackMeta,
                                                static_cast<int>(epMergedSinceSignal[otherEpId]));
        } else {
          MORI_IO_WARN(
              "EP {} has pending unsignaled WRs but no submission ledger; "
              "sqDepth may remain stale until endpoint restart",
              otherEpId);
        }
      }

      std::string message = "ibv_post_send failed with " + std::to_string(ret) + ": " +
                            strerror(ret) + " (posted " + std::to_string(postedCount) + "/" +
                            std::to_string(batchWrNum) + " WRs)";
      AppendHint(&message, BuildPostSendFailureHint(ret, eps[epId], epNum, postBatchSize, badWr,
                                                    PostSendOpKind::BatchData));
      return {StatusCode::ERR_RDMA_OP, std::move(message)};
    }

    if (needSignal) {
      epWrsSinceSignal[epId] = 0;
      epMergedSinceSignal[epId] = 0;
    }
    MORI_IO_TRACE("ibv_post_send ep index {} batch index range [{}, {})", epId, st, end);
  }
  return {StatusCode::IN_PROGRESS, ""};
}

/* ---------------------------------------------------------------------------------------------- */
/*                              RdmaLocalMemoryRegistration                                       */
/* ---------------------------------------------------------------------------------------------- */
RdmaLocalMemoryRegistration::RdmaLocalMemoryRegistration(
    RdmaMemoryLayout layout, std::vector<application::RdmaOwnedMr> ownedMrs)
    : layout_(std::move(layout)), ownedMrs_(std::move(ownedMrs)) {}

/* ---------------------------------------------------------------------------------------------- */
/*                                  Slice Resolution Algorithm                                    */
/* ---------------------------------------------------------------------------------------------- */
static size_t FindContainingChunk(const std::vector<RdmaMemoryChunk>& chunks, size_t offset) {
  size_t lo = 0, hi = chunks.size();
  while (lo + 1 < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (chunks[mid].offset <= offset)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

std::vector<RdmaTransferSlice> ResolveTransferSlices(const RdmaMemoryLayout& local,
                                                     size_t localOffset,
                                                     const RdmaMemoryLayout& remote,
                                                     size_t remoteOffset, size_t totalSize) {
  if (totalSize == 0) return {};

  if (localOffset > local.logicalSize || totalSize > local.logicalSize - localOffset ||
      remoteOffset > remote.logicalSize || totalSize > remote.logicalSize - remoteOffset) {
    return {};
  }

  std::vector<RdmaTransferSlice> slices;
  size_t estChunks = (local.chunks.size() > 1 || remote.chunks.size() > 1) ? 4 : 1;
  slices.reserve(estChunks);

  size_t li = FindContainingChunk(local.chunks, localOffset);
  size_t ri = FindContainingChunk(remote.chunks, remoteOffset);
  size_t remaining = totalSize;

  while (remaining > 0) {
    const auto& lc = local.chunks[li];
    const auto& rc = remote.chunks[ri];

    uintptr_t localAddr = local.baseAddr + localOffset;
    uintptr_t remoteAddr = remote.baseAddr + remoteOffset;

    size_t localChunkEnd = lc.offset + lc.mr.length;
    size_t remoteChunkEnd = rc.offset + rc.mr.length;

    size_t localAvail = localChunkEnd - localOffset;
    size_t remoteAvail = remoteChunkEnd - remoteOffset;

    size_t sliceLen = std::min({remaining, localAvail, remoteAvail,
                                static_cast<size_t>(std::numeric_limits<uint32_t>::max())});

    slices.push_back(
        {localAddr, lc.mr.lkey, remoteAddr, rc.mr.rkey, static_cast<uint32_t>(sliceLen)});

    remaining -= sliceLen;
    localOffset += sliceLen;
    remoteOffset += sliceLen;

    if (localOffset >= localChunkEnd && remaining > 0) ++li;
    if (remoteOffset >= remoteChunkEnd && remaining > 0) ++ri;
  }

  return slices;
}

static std::vector<ResolvedLaneWork> BuildResolvedLaneWork(size_t totalSlices, size_t epCount,
                                                           size_t maxLanes) {
  std::vector<ResolvedLaneWork> lanes;
  if (totalSlices == 0 || epCount == 0) return lanes;

  size_t laneCount = epCount;
  if (maxLanes > 0) laneCount = std::min(laneCount, maxLanes);
  laneCount = std::min(laneCount, totalSlices);
  if (laneCount == 0) return lanes;

  lanes.reserve(laneCount);
  size_t begin = 0;
  size_t base = totalSlices / laneCount;
  size_t remainder = totalSlices % laneCount;
  for (size_t laneIdx = 0; laneIdx < laneCount; ++laneIdx) {
    size_t laneSize = base + (laneIdx < remainder ? 1 : 0);
    size_t end = begin + laneSize;
    lanes.push_back({static_cast<int>(laneIdx), begin, end});
    begin = end;
  }
  return lanes;
}

ResolvedTransferPlan BuildResolvedTransferPlan(const EpPairVec& eps, const RdmaMemoryLayout& local,
                                               const SizeVec& localOffsets,
                                               const RdmaMemoryLayout& remote,
                                               const SizeVec& remoteOffsets, const SizeVec& sizes,
                                               size_t maxLanes) {
  ResolvedTransferPlan plan;
  if ((localOffsets.size() != remoteOffsets.size()) || (sizes.size() != remoteOffsets.size())) {
    return plan;
  }

  for (size_t i = 0; i < sizes.size(); ++i) {
    auto slices = ResolveTransferSlices(local, localOffsets[i], remote, remoteOffsets[i], sizes[i]);
    if (slices.empty() && sizes[i] > 0) {
      plan.slices.clear();
      plan.lanes.clear();
      return plan;
    }
    plan.slices.insert(plan.slices.end(), slices.begin(), slices.end());
  }
  if (plan.slices.empty()) return plan;

  size_t laneCap = maxLanes > 0 ? maxLanes : eps.size();
  plan.lanes = BuildResolvedLaneWork(plan.slices.size(), eps.size(), laneCap);
  return plan;
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Resolved Slice Lane Submission                                    */
/* ---------------------------------------------------------------------------------------------- */
RdmaOpRet RdmaPostResolvedSlicesToSingleEp(const EpPair& ep,
                                           const std::vector<RdmaTransferSlice>& slices,
                                           size_t begin, size_t end,
                                           std::shared_ptr<CqCallbackMeta> callbackMeta,
                                           TransferUniqueId id, bool isRead, int postBatchSize) {
  MORI_IO_FUNCTION_TIMER;

  if (begin > end || end > slices.size()) {
    return {StatusCode::ERR_INVALID_ARGS, "resolved slice range is invalid"};
  }
  if (begin == end) return {StatusCode::SUCCESS, ""};

  struct MergedWorkRequest {
    ibv_send_wr wr{};
    std::vector<ibv_sge> sges;
    size_t totalRemoteLength{0};
    size_t mergedSlices{1};
  };

  const uint32_t maxSge = std::max(ep.local.handle.maxSge, 1u);
  std::vector<MergedWorkRequest> mergedWrs;
  mergedWrs.reserve(end - begin);

  auto startNewWr = [&](const RdmaTransferSlice& slice) {
    mergedWrs.emplace_back();
    MergedWorkRequest& mwr = mergedWrs.back();
    mwr.sges.reserve(maxSge);
    mwr.sges.push_back(
        ibv_sge{.addr = slice.localAddr, .length = slice.len, .lkey = slice.localLkey});
    mwr.totalRemoteLength = slice.len;

    mwr.wr.sg_list = mwr.sges.data();
    mwr.wr.num_sge = 1;
    mwr.wr.opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
    mwr.wr.send_flags = 0;
    mwr.wr.wr.rdma.remote_addr = slice.remoteAddr;
    mwr.wr.wr.rdma.rkey = slice.remoteRkey;
  };

  for (size_t idx = begin; idx < end; ++idx) {
    const RdmaTransferSlice& slice = slices[idx];
    bool merged = false;

    if (!mergedWrs.empty()) {
      MergedWorkRequest& last = mergedWrs.back();
      uint64_t expectedRemote = last.wr.wr.rdma.remote_addr + last.totalRemoteLength;
      if (expectedRemote == slice.remoteAddr && last.wr.wr.rdma.rkey == slice.remoteRkey) {
        ibv_sge& lastSge = last.sges.back();
        bool localContig = (lastSge.addr + lastSge.length) == slice.localAddr;
        bool sameLkey = lastSge.lkey == slice.localLkey;

        if (localContig && sameLkey) {
          uint64_t newLen = static_cast<uint64_t>(lastSge.length) + slice.len;
          if (newLen <= std::numeric_limits<uint32_t>::max()) {
            lastSge.length = static_cast<uint32_t>(newLen);
            last.mergedSlices++;
            last.totalRemoteLength += slice.len;
            merged = true;
          }
        }
        if (!merged && last.sges.size() < maxSge) {
          last.sges.push_back(
              ibv_sge{.addr = slice.localAddr, .length = slice.len, .lkey = slice.localLkey});
          last.wr.num_sge = static_cast<int>(last.sges.size());
          last.mergedSlices++;
          last.totalRemoteLength += slice.len;
          merged = true;
        }
      }
    }
    if (!merged) startNewWr(slice);
  }

  for (auto& mwr : mergedWrs) {
    mwr.wr.sg_list = mwr.sges.data();
  }

  size_t mergedWrCount = mergedWrs.size();
  if (postBatchSize == -1) {
    postBatchSize = (mergedWrCount > static_cast<size_t>(std::numeric_limits<int>::max()))
                        ? std::numeric_limits<int>::max()
                        : static_cast<int>(mergedWrCount);
  }
  if (ep.sqDepth && ep.maxSqDepth > 0 && postBatchSize > ep.maxSqDepth) {
    postBatchSize = ep.maxSqDepth;
  }
  if (postBatchSize <= 0) postBatchSize = 1;
  int numPostBatch = (mergedWrCount + postBatchSize - 1) / postBatchSize;

  int wrsSinceSignal = 0;
  size_t slicesSinceSignal = 0;

  for (int batchIdx = 0; batchIdx < numPostBatch; ++batchIdx) {
    int st = batchIdx * postBatchSize;
    int batchEnd = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    int batchWrNum = batchEnd - st;
    if (batchWrNum <= 0) break;

    std::string reserveErr;
    SqReserveFailureKind reserveFailure = SqReserveFailureKind::None;
    if (!TryReserveSqDepth(ep, batchWrNum, 0, "resolved-lane", &reserveErr, &reserveFailure)) {
      AppendHint(&reserveErr,
                 BuildSqDepthHint(ep, 1, postBatchSize, PostSendOpKind::BatchData, reserveFailure));
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }

    size_t mergedSliceSize = 0;
    for (int wrIdx = st; wrIdx < batchEnd; ++wrIdx) {
      mergedWrs[wrIdx].wr.wr_id = 0;
      mergedWrs[wrIdx].wr.next = (wrIdx + 1 < batchEnd) ? &mergedWrs[wrIdx + 1].wr : nullptr;
      mergedSliceSize += mergedWrs[wrIdx].mergedSlices;
    }

    wrsSinceSignal += batchWrNum;
    slicesSinceSignal += mergedSliceSize;

    bool isLastBatch = (batchIdx + 1 == numPostBatch);
    bool sqNearFull = ep.sqDepth && (wrsSinceSignal >= ep.maxSqDepth);
    bool needSignal = isLastBatch || sqNearFull;

    ibv_send_wr& last = mergedWrs[batchEnd - 1].wr;
    uint64_t recordId = 0;
    if (needSignal) {
      if (!ep.ledger) {
        ReleaseSqDepth(ep, batchWrNum);
        return {StatusCode::ERR_RDMA_OP,
                "submission ledger is not initialized for signaled WR tracking"};
      }
      recordId = ep.ledger->Insert(wrsSinceSignal, true, callbackMeta,
                                   static_cast<int>(slicesSinceSignal));
      last.wr_id = recordId;
      last.send_flags = IBV_SEND_SIGNALED;
    }

    ibv_send_wr* badWr = nullptr;
    int ret = ibv_post_send(ep.local.ibvHandle.qp, &mergedWrs[st].wr, &badWr);
    if (ret != 0) {
      int postedCount = 0;
      if (badWr != nullptr) {
        ibv_send_wr* cur = &mergedWrs[st].wr;
        while (cur != nullptr && cur != badWr && postedCount < batchWrNum) {
          ++postedCount;
          cur = cur->next;
        }
      }
      postedCount = std::max(0, std::min(postedCount, batchWrNum));
      int unpostedCount = batchWrNum - postedCount;
      if (unpostedCount > 0) {
        ReleaseSqDepth(ep, unpostedCount);
        wrsSinceSignal = std::max(0, wrsSinceSignal - unpostedCount);
        size_t mergedUnposted = 0;
        for (int wrIdx = st + postedCount; wrIdx < batchEnd; ++wrIdx) {
          mergedUnposted += mergedWrs[wrIdx].mergedSlices;
        }
        slicesSinceSignal =
            slicesSinceSignal >= mergedUnposted ? slicesSinceSignal - mergedUnposted : 0;
      }

      bool lastWasPosted = (postedCount == batchWrNum);
      if (needSignal && !lastWasPosted) {
        int dummy = 0;
        ep.ledger->ReleaseByCqe(recordId, nullptr, &dummy);
      }

      if (postedCount > 0 && (!needSignal || !lastWasPosted)) {
        if (ep.degraded) ep.degraded->store(true, std::memory_order_relaxed);
        if (ep.ledger) {
          ep.ledger->InsertOrphaned(wrsSinceSignal, callbackMeta,
                                    static_cast<int>(slicesSinceSignal));
        }
      }

      std::string message = "ibv_post_send (resolved lane) failed with " + std::to_string(ret) +
                            ": " + strerror(ret) + " (posted " + std::to_string(postedCount) + "/" +
                            std::to_string(batchWrNum) + " WRs)";
      AppendHint(&message, BuildPostSendFailureHint(ret, ep, 1, postBatchSize, badWr,
                                                    PostSendOpKind::BatchData));
      return {StatusCode::ERR_RDMA_OP, std::move(message)};
    }

    if (needSignal) {
      wrsSinceSignal = 0;
      slicesSinceSignal = 0;
    }
    MORI_IO_TRACE("ibv_post_send resolved lane task {} batch range [{}, {})", id, st, batchEnd);
  }

  return {StatusCode::IN_PROGRESS, ""};
}

RdmaOpRet RdmaSubmitResolvedTransferPlanInline(const EpPairVec& eps,
                                               const std::vector<RdmaTransferSlice>& slices,
                                               const std::vector<ResolvedLaneWork>& lanes,
                                               std::shared_ptr<CqCallbackMeta> callbackMeta,
                                               TransferUniqueId id, bool isRead,
                                               int postBatchSize) {
  MORI_IO_FUNCTION_TIMER;

  if (slices.empty()) return {StatusCode::SUCCESS, ""};
  if (eps.empty()) return {StatusCode::ERR_INVALID_ARGS, "no endpoints"};
  if (lanes.empty()) return {StatusCode::ERR_INVALID_ARGS, "resolved transfer lanes are empty"};

  bool hasFail = false;
  int numSucc = 0;
  RdmaOpRet failedRet;

  for (const auto& lane : lanes) {
    if (lane.epId < 0 || static_cast<size_t>(lane.epId) >= eps.size()) {
      return {StatusCode::ERR_INVALID_ARGS, "resolved lane endpoint is invalid"};
    }
    RdmaOpRet ret = RdmaPostResolvedSlicesToSingleEp(eps[lane.epId], slices, lane.begin, lane.end,
                                                     callbackMeta, id, isRead, postBatchSize);
    if (ret.Failed()) {
      hasFail = true;
      failedRet = ret;
    } else if (ret.Succeeded()) {
      numSucc++;
    }
  }

  if (hasFail) return failedRet;
  if (numSucc == static_cast<int>(lanes.size())) return {StatusCode::SUCCESS, ""};
  return {StatusCode::IN_PROGRESS, ""};
}

}  // namespace io
}  // namespace mori
