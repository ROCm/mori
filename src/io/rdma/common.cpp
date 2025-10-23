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

#include "mori/io/logging.hpp"

namespace mori {
namespace io {
/* ---------------------------------------------------------------------------------------------- */
/*                                         Rdma Utilities                                         */
/* ---------------------------------------------------------------------------------------------- */

RdmaOpRet RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* status, TransferUniqueId id) {
  MORI_IO_FUNCTION_TIMER;

  for (int i = 0; i < eps.size(); i++) {
    const application::RdmaEndpoint& ep = eps[i].local;
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
      // TODO: set callback status here?
      return {StatusCode::ERR_RDMA_OP, strerror(errno)};
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

  struct MergedWorkRequest {
    ibv_send_wr wr{};
    std::vector<ibv_sge> sges;
    size_t totalRemoteLength = 0;
    size_t mergedRequests = 1;
  };

  std::vector<MergedWorkRequest> mergedWrs;
  mergedWrs.reserve(batchSize);

  const uint64_t localBaseAddr = reinterpret_cast<uint64_t>(local.addr);
  const uint64_t remoteBaseAddr = reinterpret_cast<uint64_t>(remote.addr);
  const uint32_t maxSge = eps[0].local.handle.maxSge;

  for (size_t i = 0; i < batchSize; ++i) {
    uint64_t currentLocalAddr = localBaseAddr + localOffsets[i];
    uint64_t currentRemoteAddr = remoteBaseAddr + remoteOffsets[i];
    size_t currentSize = sizes[i];

    bool canMerge = !mergedWrs.empty() && (mergedWrs.back().wr.wr.rdma.remote_addr +
                                           mergedWrs.back().totalRemoteLength) == currentRemoteAddr;

    if (canMerge) {
      MergedWorkRequest& lastWr = mergedWrs.back();
      ibv_sge& lastSge = lastWr.sges.back();

      bool localContiguous = (lastSge.addr + lastSge.length) == currentLocalAddr;

      if (localContiguous) {
        lastSge.length += currentSize;
        lastWr.mergedRequests += 1;
      } else if (lastWr.sges.size() < maxSge) {
        lastWr.sges.push_back(
            {.addr = currentLocalAddr, .length = (uint32_t)currentSize, .lkey = local.lkey});
        lastWr.wr.num_sge = static_cast<int>(lastWr.sges.size());
        lastWr.mergedRequests += 1;
      } else {
        canMerge = false;
      }

      if (canMerge) {
        lastWr.totalRemoteLength += currentSize;
      }
    }

    if (!canMerge) {
      mergedWrs.emplace_back();
      MergedWorkRequest& newWr = mergedWrs.back();
      newWr.sges.reserve(maxSge);  // Make sure sges.data() is stable
      newWr.sges.push_back(
          {.addr = currentLocalAddr, .length = (uint32_t)currentSize, .lkey = local.lkey});

      newWr.totalRemoteLength = currentSize;  // Initialize remote length

      newWr.wr.sg_list = newWr.sges.data();
      newWr.wr.num_sge = 1;
      newWr.wr.opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
      newWr.wr.send_flags = 0;
      newWr.wr.wr.rdma.remote_addr = currentRemoteAddr;
      newWr.wr.wr.rdma.rkey = remote.rkey;
    }
  }

  size_t mergedWrCount = mergedWrs.size();
  size_t epNum = eps.size();
  size_t epBatchSize = (mergedWrCount + epNum - 1) / epNum;

  if (postBatchSize == -1) postBatchSize = epBatchSize;
  int numPostBatch = (mergedWrCount + postBatchSize - 1) / postBatchSize;

  std::vector<std::unique_ptr<CqCallbackMessage>> callbackMessages;
  callbackMessages.reserve(epNum);               // At most one callback per endpoint
  std::vector<size_t> epPostedCounts(epNum, 0);  // Actual posted requests count per endpoint

  for (int i = 0; i < numPostBatch; i++) {
    int st = i * postBatchSize;
    int end = std::min(static_cast<size_t>(st) + postBatchSize, mergedWrCount);
    if (end - st == 0) break;
    int epId = i % epNum;
    size_t mergedReqSize = 0;
    for (int j = st; j < end; j++) {
      struct ibv_send_wr& wr = mergedWrs[j].wr;
      wr.wr_id = 0;
      wr.next = (j + 1 < end) ? &mergedWrs[j + 1].wr : nullptr;
      mergedReqSize += mergedWrs[j].mergedRequests;
    }

    struct ibv_send_wr& last = mergedWrs[end - 1].wr;

    epPostedCounts[epId] += mergedReqSize;
    if ((i + epNum) >= numPostBatch) {
      int epTotalBatchSize = static_cast<int>(epPostedCounts[epId]);
      last.wr_id =
          reinterpret_cast<uint64_t>(new CqCallbackMessage(callbackMeta, epTotalBatchSize));
      last.send_flags = IBV_SEND_SIGNALED;
    }
    int ret = ibv_post_send(eps[epId].local.ibvHandle.qp, &mergedWrs[st].wr, nullptr);
    if (ret != 0) {
      return {StatusCode::ERR_RDMA_OP, strerror(errno)};
    }
    MORI_IO_TRACE("ibv_post_send ep index {} batch index range [{}, {})", epId, st, end);
  }
  return {StatusCode::IN_PROGRESS, ""};
}

};  // namespace io
};  // namespace mori
