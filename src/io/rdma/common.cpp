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

#include <infiniband/verbs.h>  // dereferences ibvHandle.qp (forward-declared in core)

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
    int v = 5000000;
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

// Fill `plan` with (offset, len) chunks for a transfer of `total` bytes. Clears
// `plan` first but keeps its capacity so callers can reuse a pooled buffer.
static void PlanChunksInto(std::vector<std::pair<uint64_t, uint32_t>>& plan, uint32_t total,
                           size_t chunkBytes, int maxChunks) {
  plan.clear();
  if (total == 0) return;
  if (chunkBytes == 0) {
    plan.push_back({0, total});
    return;
  }
  if (maxChunks <= 0) return;
  if (total <= chunkBytes) {
    plan.push_back({0, total});
    return;
  }

  size_t chunkCount = (static_cast<size_t>(total) + chunkBytes - 1) / chunkBytes;
  chunkCount = std::max<size_t>(1, std::min(chunkCount, static_cast<size_t>(maxChunks)));
  const size_t perChunk = (static_cast<size_t>(total) + chunkCount - 1) / chunkCount;

  if (plan.capacity() < chunkCount) plan.reserve(chunkCount);
  for (size_t offset = 0; offset < total; offset += perChunk) {
    const uint32_t len =
        static_cast<uint32_t>(std::min(perChunk, static_cast<size_t>(total) - offset));
    plan.push_back({offset, len});
  }
}

std::vector<std::pair<uint64_t, uint32_t>> PlanChunks(uint32_t total, size_t chunkBytes,
                                                      int maxChunks) {
  std::vector<std::pair<uint64_t, uint32_t>> plan;
  PlanChunksInto(plan, total, chunkBytes, maxChunks);
  return plan;
}

void PlanSgeStreamChunks(std::vector<ChunkedSgeSegment>& plan, const std::vector<ibv_sge>& sges,
                         uint64_t totalLength, size_t chunkBytes, int maxChunks,
                         uint64_t maxMessageSize) {
  plan.clear();
  if (totalLength == 0 || maxChunks <= 0 || maxMessageSize == 0) return;

  auto ceilDiv = [](uint64_t value, uint64_t divisor) {
    return value / divisor + ((value % divisor) != 0);
  };

  uint64_t softCount = 1;
  if (chunkBytes > 0 && totalLength > chunkBytes) {
    softCount = ceilDiv(totalLength, static_cast<uint64_t>(chunkBytes));
    softCount = std::min<uint64_t>(softCount, static_cast<uint64_t>(maxChunks));
  }

  const uint64_t hardMinCount = std::max<uint64_t>(ceilDiv(totalLength, maxMessageSize), 1);
  const uint64_t finalCount = std::max<uint64_t>(softCount, hardMinCount);
  const uint64_t targetChunkBytes = std::max<uint64_t>(ceilDiv(totalLength, finalCount), 1);
  const uint64_t maxReserve = static_cast<uint64_t>(plan.max_size());
  uint64_t reserveHint = finalCount;
  const uint64_t sgeCount = static_cast<uint64_t>(sges.size());
  if (reserveHint <= maxReserve && sgeCount <= maxReserve - reserveHint) reserveHint += sgeCount;
  if (reserveHint <= maxReserve && plan.capacity() < reserveHint) {
    plan.reserve(static_cast<size_t>(reserveHint));
  }

  uint64_t remoteStreamOffset = 0;
  uint64_t targetRemaining = std::min(targetChunkBytes, totalLength);
  for (const ibv_sge& sge : sges) {
    uint64_t sgeRemaining = sge.length;
    uint64_t sgeOffset = 0;
    while (sgeRemaining > 0) {
      const uint64_t len64 = std::min({targetRemaining, sgeRemaining, maxMessageSize});
      if (len64 == 0) return;
      plan.push_back(ChunkedSgeSegment{
          .remoteOffset = remoteStreamOffset + sgeOffset,
          .localAddr = sge.addr + sgeOffset,
          .length = static_cast<uint32_t>(len64),
      });

      sgeRemaining -= len64;
      sgeOffset += len64;
      targetRemaining -= len64;
      if (targetRemaining == 0 && remoteStreamOffset + sgeOffset < totalLength) {
        const uint64_t streamRemaining = totalLength - (remoteStreamOffset + sgeOffset);
        targetRemaining = std::min(targetChunkBytes, streamRemaining);
      }
    }
    remoteStreamOffset += sge.length;
  }
}

struct MergedWorkRequest {
  ibv_send_wr wr{};
  std::vector<ibv_sge> sges;
  size_t totalRemoteLength = 0;
  size_t mergedRequests = 1;
};

static void ResetMergedWorkRequestPointers(MergedWorkRequest* wr) {
  if (wr == nullptr) return;
  wr->wr.sg_list = wr->sges.empty() ? nullptr : wr->sges.data();
  wr->wr.num_sge = static_cast<int>(wr->sges.size());
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

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps,
                             const std::vector<application::RdmaMemoryRegion>& localMrPerEp,
                             const std::vector<application::RdmaMemoryRegion>& remoteMrPerEp,
                             const SizeVec& localOffsets, const SizeVec& remoteOffsets,
                             const SizeVec& sizes, std::shared_ptr<CqCallbackMeta> callbackMeta,
                             TransferUniqueId id, bool isRead, int postBatchSize, size_t chunkBytes,
                             int maxChunks, bool creditByWrCount) {
  MORI_IO_FUNCTION_TIMER;

  if ((localOffsets.size() != remoteOffsets.size()) || (sizes.size() != remoteOffsets.size())) {
    return {StatusCode::ERR_INVALID_ARGS,
            "lengths of local offsets, remote offsets or sizes mismatch"};
  }

  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    return {StatusCode::SUCCESS, ""};
  }

  if (eps.empty()) {
    return {StatusCode::ERR_INVALID_ARGS, "no endpoints"};
  }

  if (localMrPerEp.size() != eps.size() || remoteMrPerEp.size() != eps.size()) {
    return {StatusCode::ERR_INVALID_ARGS, "memory-region vectors must align with endpoints"};
  }

  if (maxChunks <= 0) {
    return {StatusCode::ERR_INVALID_ARGS, "maxChunks must be >= 1"};
  }

  const application::RdmaMemoryRegion& baseLocalMr = localMrPerEp.front();
  const application::RdmaMemoryRegion& baseRemoteMr = remoteMrPerEp.front();
  for (size_t i = 0; i < batchSize; i++) {
    if (((localOffsets[i] + sizes[i]) > baseLocalMr.length) ||
        ((remoteOffsets[i] + sizes[i]) > baseRemoteMr.length)) {
      return {StatusCode::ERR_INVALID_ARGS, "length out of range"};
    }
  }

  // [tls-scratch] Per worker-thread scratch pools eliminate per-batch heap
  // allocations on the RDMA hot path (slot reuse / resize() / clear() retain
  // capacity across calls). The pools belong to the OUTERMOST call on a thread;
  // if RdmaBatchReadWrite is ever re-entered on the same thread (e.g. from a
  // completion callback), the nested call transparently falls back to local
  // buffers so the outer call's pools are never clobbered.
  thread_local std::vector<size_t> tlIndices;
  thread_local std::vector<MergedWorkRequest> tlMergedPool;
  thread_local std::vector<MergedWorkRequest> tlChunkedPool;
  thread_local std::vector<ChunkedSgeSegment> tlChunkPlan;
  thread_local std::vector<int> tlEpWrsSinceSignal;
  thread_local std::vector<size_t> tlEpMergedSinceSignal;
  thread_local int reentryDepth = 0;

  struct ReentryGuard {
    int& depth;
    explicit ReentryGuard(int& d) : depth(d) { ++depth; }
    ~ReentryGuard() { --depth; }
  } reentryGuard(reentryDepth);
  const bool usePool = (reentryDepth == 1);

  // Used only on (currently non-existent) same-thread re-entry.
  std::vector<size_t> localIndices;
  std::vector<MergedWorkRequest> localMergedPool;
  std::vector<MergedWorkRequest> localChunkedPool;
  std::vector<ChunkedSgeSegment> localChunkPlan;
  std::vector<int> localEpWrsSinceSignal;
  std::vector<size_t> localEpMergedSinceSignal;

  std::vector<size_t>& indices = usePool ? tlIndices : localIndices;
  std::vector<MergedWorkRequest>& mergedPool = usePool ? tlMergedPool : localMergedPool;
  std::vector<MergedWorkRequest>& chunkedPool = usePool ? tlChunkedPool : localChunkedPool;
  std::vector<ChunkedSgeSegment>& chunkPlan = usePool ? tlChunkPlan : localChunkPlan;
  std::vector<int>& epWrsSinceSignal = usePool ? tlEpWrsSinceSignal : localEpWrsSinceSignal;
  std::vector<size_t>& epMergedSinceSignal =
      usePool ? tlEpMergedSinceSignal : localEpMergedSinceSignal;

  // Bound peak retained memory: if an earlier very large batch grew the pools far
  // beyond the current need, release the excess so it doesn't stay resident.
  constexpr size_t kPoolHighWater = 8192;
  if (usePool && batchSize <= kPoolHighWater / 2) {
    if (mergedPool.size() > kPoolHighWater) {
      mergedPool.resize(kPoolHighWater);
      mergedPool.shrink_to_fit();
    }
    if (chunkedPool.size() > kPoolHighWater) {
      chunkedPool.resize(kPoolHighWater);
      chunkedPool.shrink_to_fit();
    }
  }

  indices.resize(batchSize);
  std::iota(indices.begin(), indices.end(), 0);

  if (!std::is_sorted(remoteOffsets.begin(), remoteOffsets.end())) {
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return remoteOffsets[a] < remoteOffsets[b]; });
  }

  const uint64_t localBaseAddr = reinterpret_cast<uint64_t>(baseLocalMr.addr);
  const uint64_t remoteBaseAddr = reinterpret_cast<uint64_t>(baseRemoteMr.addr);
  const uint32_t maxSge =
      std::max(eps[0].local.handle.maxSge, 1u);  // We assume all endpoints have the same maxSge
  const ibv_wr_opcode opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
  uint64_t localMaxMessageSize = std::numeric_limits<uint64_t>::max();
  for (size_t epId = 0; epId < eps.size(); ++epId) {
    const uint64_t epMaxMessageSize = eps[epId].local.maxMsgSize;
    if (epMaxMessageSize == 0) {
      return {StatusCode::ERR_BAD_STATE,
              "RDMA endpoint local max_msg_sz is unavailable for ep " + std::to_string(epId)};
    }
    localMaxMessageSize = std::min(localMaxMessageSize, epMaxMessageSize);
  }
  auto exceedsLocalMaxMessageSize = [&](const MergedWorkRequest& wr) {
    return static_cast<uint64_t>(wr.totalRemoteLength) > localMaxMessageSize;
  };
  auto hasOversizedSge = [&](const MergedWorkRequest& wr) {
    if (chunkBytes == 0) return false;
    for (const ibv_sge& sge : wr.sges) {
      if (sge.length > chunkBytes) return true;
    }
    return false;
  };

  // Initialize a pooled slot as a single-SGE WR (shared by the merge builder and
  // the chunk expander); clears but keeps the slot's sges capacity.
  auto initSingleSgeWr = [](MergedWorkRequest& w, ibv_wr_opcode op, uint64_t remoteAddr,
                            uint64_t localAddr, uint32_t len, uint32_t sgeCap) {
    w.sges.clear();
    if (w.sges.capacity() < sgeCap) w.sges.reserve(sgeCap);
    w.sges.push_back(ibv_sge{.addr = localAddr, .length = len, .lkey = 0});
    w.totalRemoteLength = len;
    w.mergedRequests = 1;
    w.wr = ibv_send_wr{};
    w.wr.opcode = op;
    w.wr.send_flags = 0;
    w.wr.wr.rdma.remote_addr = remoteAddr;
    w.wr.wr.rdma.rkey = 0;
    ResetMergedWorkRequestPointers(&w);
  };

  size_t wrCount = 0;
  auto start_new_wr = [&](uint64_t remoteAddr, uint64_t localAddr, uint32_t len) {
    if (wrCount >= mergedPool.size()) mergedPool.emplace_back();
    initSingleSgeWr(mergedPool[wrCount], opcode, remoteAddr, localAddr, len, maxSge);
    ++wrCount;
  };

  for (size_t i = 0; i < batchSize; ++i) {
    const size_t idx = indices[i];
    const uint64_t currentLocalAddr = localBaseAddr + localOffsets[idx];
    const uint64_t currentRemoteAddr = remoteBaseAddr + remoteOffsets[idx];
    const uint32_t currentSize32 = static_cast<uint32_t>(sizes[idx]);

    bool merged = false;
    if (wrCount > 0) {
      MergedWorkRequest& lastWr = mergedPool[wrCount - 1];
      const uint64_t expectedRemoteAddr = lastWr.wr.wr.rdma.remote_addr + lastWr.totalRemoteLength;
      if (expectedRemoteAddr == currentRemoteAddr) {
        ibv_sge& lastSge = lastWr.sges.back();
        const bool localContiguous = (lastSge.addr + lastSge.length) == currentLocalAddr;

        if (localContiguous) {
          const uint64_t newLen = static_cast<uint64_t>(lastSge.length) + currentSize32;
          if (newLen <= std::numeric_limits<uint32_t>::max()) {
            lastSge.length = static_cast<uint32_t>(newLen);
            lastWr.mergedRequests += 1;
            lastWr.totalRemoteLength += currentSize32;
            merged = true;
          }
        }
        if (!merged && lastWr.sges.size() < maxSge) {
          lastWr.sges.push_back(
              ibv_sge{.addr = currentLocalAddr, .length = currentSize32, .lkey = 0});
          ResetMergedWorkRequestPointers(&lastWr);
          lastWr.mergedRequests += 1;
          lastWr.totalRemoteLength += currentSize32;
          merged = true;
        }
      }
    }
    if (!merged) {
      start_new_wr(currentRemoteAddr, currentLocalAddr, currentSize32);
    }
  }

  // [expand-chunked-precompute] Expand oversized WRs into the pooled `chunkedPool`
  // in place (slot reuse); `chunkPlan` is reused by PlanSgeStreamChunks. Note: small
  // WRs are still copied as-is into chunkedPool when any WR needs splitting.
  const bool canExpandChunks = creditByWrCount && chunkBytes > 0;
  if (!canExpandChunks) {
    for (size_t k = 0; k < wrCount; ++k) {
      const MergedWorkRequest& wr = mergedPool[k];
      if (!exceedsLocalMaxMessageSize(wr)) continue;
      return {StatusCode::ERR_INVALID_ARGS,
              "merged RDMA WR " + std::to_string(k) + " length " +
                  std::to_string(wr.totalRemoteLength) + " exceeds local max_msg_sz " +
                  std::to_string(localMaxMessageSize) +
                  "; enable RDMA transfer chunking or reduce batch size"};
    }
  }

  bool useChunked = false;
  if (canExpandChunks) {
    for (size_t k = 0; k < wrCount; ++k) {
      const MergedWorkRequest& wr = mergedPool[k];
      if (hasOversizedSge(wr) || exceedsLocalMaxMessageSize(wr)) {
        useChunked = true;
        break;
      }
    }
  }
  size_t chunkedCount = 0;
  if (useChunked) {
    auto emit = [&](ibv_wr_opcode op, uint64_t remoteAddr, uint64_t localAddr, uint32_t len) {
      if (chunkedCount >= chunkedPool.size()) chunkedPool.emplace_back();
      initSingleSgeWr(chunkedPool[chunkedCount], op, remoteAddr, localAddr, len, 1);
      ++chunkedCount;
    };
    for (size_t k = 0; k < wrCount; ++k) {
      MergedWorkRequest& wr = mergedPool[k];
      if (!hasOversizedSge(wr) && !exceedsLocalMaxMessageSize(wr)) {
        // Copy as-is (preserves multi-sge / small WRs) into the pooled slot.
        if (chunkedCount >= chunkedPool.size()) chunkedPool.emplace_back();
        MergedWorkRequest& c = chunkedPool[chunkedCount];
        c.sges.clear();
        if (c.sges.capacity() < wr.sges.size()) c.sges.reserve(wr.sges.size());
        for (const auto& s : wr.sges) c.sges.push_back(s);
        c.totalRemoteLength = wr.totalRemoteLength;
        c.mergedRequests = wr.mergedRequests;
        c.wr = wr.wr;
        ResetMergedWorkRequestPointers(&c);
        ++chunkedCount;
        continue;
      }
      const uint64_t remoteBase = wr.wr.wr.rdma.remote_addr;
      PlanSgeStreamChunks(chunkPlan, wr.sges, static_cast<uint64_t>(wr.totalRemoteLength),
                          chunkBytes, maxChunks, localMaxMessageSize);
      if (chunkPlan.empty() && wr.totalRemoteLength != 0) {
        return {StatusCode::ERR_BAD_STATE, "failed to plan RDMA chunks for non-empty SGE stream"};
      }
      for (const ChunkedSgeSegment& segment : chunkPlan) {
        emit(wr.wr.opcode, remoteBase + segment.remoteOffset, segment.localAddr, segment.length);
      }
    }
  }

  std::vector<MergedWorkRequest>& mergedWrs = useChunked ? chunkedPool : mergedPool;
  size_t mergedWrCount = useChunked ? chunkedCount : wrCount;

  if (creditByWrCount) {
    if (mergedWrCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
      return {StatusCode::ERR_INVALID_ARGS, "final WR count exceeds int range"};
    }
    for (size_t k = 0; k < mergedWrCount; ++k) mergedWrs[k].mergedRequests = 1;
    callbackMeta->totalBatchSize = static_cast<int>(mergedWrCount);
  }

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

  epWrsSinceSignal.assign(epNum, 0);
  epMergedSinceSignal.assign(epNum, 0);

  // Rotate the starting EP by transfer id so single-segment (single WR)
  // transfers spread evenly across all QPs instead of always landing on eps[0].
  int epStartOffset = static_cast<int>(id % static_cast<uint64_t>(epNum));
  for (int i = 0; i < numPostBatch; i++) {
    int st = i * postBatchSize;
    int end = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    if (end - st == 0) break;
    int epId = (i + epStartOffset) % static_cast<int>(epNum);
    int batchWrNum = end - st;

    std::string reserveErr;
    SqReserveFailureKind reserveFailure = SqReserveFailureKind::None;
    if (!TryReserveSqDepth(eps[epId], batchWrNum, epId, "batch", &reserveErr, &reserveFailure)) {
      AppendHint(&reserveErr, BuildSqDepthHint(eps[epId], epNum, postBatchSize,
                                               PostSendOpKind::BatchData, reserveFailure));
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }

    const auto& localMr = localMrPerEp[epId];
    const auto& remoteMr = remoteMrPerEp[epId];
    size_t mergedReqSize = 0;
    for (int j = st; j < end; j++) {
      MergedWorkRequest& mergedWr = mergedWrs[j];
      for (auto& sge : mergedWr.sges) sge.lkey = localMr.lkey;
      mergedWr.wr.wr.rdma.rkey = remoteMr.rkey;
      ResetMergedWorkRequestPointers(&mergedWr);

      struct ibv_send_wr& wr = mergedWr.wr;
      wr.wr_id = 0;
      wr.next = (j + 1 < end) ? &mergedWrs[j + 1].wr : nullptr;
      mergedReqSize += mergedWr.mergedRequests;
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
      } else if (needSignal) {
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

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                             const SizeVec& localOffsets,
                             const application::RdmaMemoryRegion& remote,
                             const SizeVec& remoteOffsets, const SizeVec& sizes,
                             std::shared_ptr<CqCallbackMeta> callbackMeta, TransferUniqueId id,
                             bool isRead, int postBatchSize) {
  std::vector<application::RdmaMemoryRegion> localVec(eps.size(), local);
  std::vector<application::RdmaMemoryRegion> remoteVec(eps.size(), remote);
  return RdmaBatchReadWrite(eps, localVec, remoteVec, localOffsets, remoteOffsets, sizes,
                            callbackMeta, id, isRead, postBatchSize, 0, 1, false);
}

}  // namespace io
}  // namespace mori
