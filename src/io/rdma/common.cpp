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

#include <chrono>
#include <cstdlib>
#include <numeric>
#include <thread>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {

static int GetSqBackoffTimeoutUs() {
  static const int value = []() {
    const char* env = std::getenv("MORI_IO_SQ_BACKOFF_TIMEOUT_US");
    if (env && env[0] != '\0') {
      int v = std::atoi(env);
      if (v > 0) return v;
    }
    return 10000;
  }();
  return value;
}
// SQ depth is an admission counter only. It does not publish data dependencies
// across threads, so relaxed atomics are sufficient for correctness.
static bool TryReserveSqDepth(const EpPair& ep, int wrCount, int epId, const char* opTag,
                              std::string* errMsg) {
  if (wrCount <= 0 || !ep.sqDepth) return true;
  const int kBackoffTimeoutUs = GetSqBackoffTimeoutUs();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(kBackoffTimeoutUs);
  int backoff = 0;
  int cur = ep.sqDepth->load(std::memory_order_relaxed);
  while (true) {
    if (cur + wrCount > ep.maxSqDepth) {
      if (std::chrono::steady_clock::now() >= deadline) {
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

  for (int i = 0; i < eps.size(); i++) {
    const application::RdmaEndpoint& ep = eps[i].local;

    {
      std::string reserveErr;
      if (!TryReserveSqDepth(eps[i], 1, i, "notify", &reserveErr)) {
        return {StatusCode::ERR_RDMA_OP, reserveErr};
      }
    }

    NotifMessage msg{id, i, static_cast<int>(eps.size())};

    struct ibv_sge sge{};
    sge.addr = reinterpret_cast<uintptr_t>(&msg);
    sge.length = sizeof(NotifMessage);
    sge.lkey = 0;

    struct ibv_send_wr wr{};
    wr.wr_id = id;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(ep.ibvHandle.qp, &wr, &bad_wr);
    if (ret != 0) {
      ReleaseSqDepth(eps[i], 1);
      return {StatusCode::ERR_RDMA_OP,
              "ibv_post_send (notify) failed with " + std::to_string(ret) + ": " + strerror(ret)};
    }
  }

  return {StatusCode::IN_PROGRESS, ""};
}

RdmaOpRet RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                             const SizeVec& localOffsets,
                             const application::RdmaMemoryRegion& remote,
                             const SizeVec& remoteOffsets, const SizeVec& sizes,
                             CqCallbackMeta* callbackMeta, TransferUniqueId id, bool isRead,
                             int postBatchSize) {
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

  if (postBatchSize == -1) postBatchSize = epBatchSize;
  int numPostBatch = (mergedWrCount + postBatchSize - 1) / postBatchSize;

  // Pre-compute per-EP ibv WR counts for SQ depth reservation
  std::vector<int> epWrCounts(epNum, 0);
  for (int i = 0; i < numPostBatch; i++) {
    int st = i * postBatchSize;
    int end = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    epWrCounts[i % epNum] += (end - st);
  }

  // Atomically reserve SQ capacity for all EPs (all-or-nothing) with bounded backoff
  size_t reservedUpTo = 0;
  for (size_t i = 0; i < epNum; i++) {
    if (epWrCounts[i] == 0 || !eps[i].sqDepth) {
      reservedUpTo = i + 1;
      continue;
    }
    std::string reserveErr;
    if (!TryReserveSqDepth(eps[i], epWrCounts[i], i, "batch", &reserveErr)) {
      for (size_t j = 0; j < reservedUpTo; j++) {
        ReleaseSqDepth(eps[j], epWrCounts[j]);
      }
      return {StatusCode::ERR_RDMA_OP, reserveErr};
    }
    reservedUpTo = i + 1;
  }

  std::vector<size_t> epPostedCounts(epNum, 0);
  std::vector<int> epActualWrPosted(epNum, 0);
  std::vector<bool> epSignaledPosted(epNum, false);

  for (int i = 0; i < numPostBatch; i++) {
    int st = i * postBatchSize;
    int end = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    if (end - st == 0) break;
    int epId = i % epNum;
    int batchWrNum = end - st;
    size_t mergedReqSize = 0;
    for (int j = st; j < end; j++) {
      struct ibv_send_wr& wr = mergedWrs[j].wr;
      wr.wr_id = 0;
      wr.next = (j + 1 < end) ? &mergedWrs[j + 1].wr : nullptr;
      mergedReqSize += mergedWrs[j].mergedRequests;
    }

    struct ibv_send_wr& last = mergedWrs[end - 1].wr;

    epPostedCounts[epId] += mergedReqSize;
    bool isLastBatchForEp = ((i + epNum) >= numPostBatch);
    CqCallbackMessage* cbMsg = nullptr;
    if (isLastBatchForEp) {
      int epTotalBatchSize = static_cast<int>(epPostedCounts[epId]);
      cbMsg = new CqCallbackMessage(callbackMeta, epTotalBatchSize, epWrCounts[epId]);
      last.wr_id = reinterpret_cast<uint64_t>(cbMsg);
      last.send_flags = IBV_SEND_SIGNALED;
    }
    struct ibv_send_wr* badWr = nullptr;
    int ret = ibv_post_send(eps[epId].local.ibvHandle.qp, &mergedWrs[st].wr, &badWr);
    if (ret != 0) {
      delete cbMsg;
      for (size_t j = 0; j < epNum; j++) {
        if (!eps[j].sqDepth || epWrCounts[j] == 0) continue;
        if (epSignaledPosted[j]) continue;
        int unposted = epWrCounts[j] - epActualWrPosted[j];
        if (unposted > 0) ReleaseSqDepth(eps[j], unposted);
      }
      return {StatusCode::ERR_RDMA_OP,
              "ibv_post_send failed with " + std::to_string(ret) + ": " + strerror(ret)};
    }
    epActualWrPosted[epId] += batchWrNum;
    if (isLastBatchForEp) epSignaledPosted[epId] = true;
    MORI_IO_TRACE("ibv_post_send ep index {} batch index range [{}, {})", epId, st, end);
  }
  return {StatusCode::IN_PROGRESS, ""};
}

};  // namespace io
};  // namespace mori
