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

void RdmaNotifyTransfer(const EpPairVec& eps, TransferStatus* status, TransferUniqueId id) {
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
      status->SetCode(StatusCode::ERR_RDMA_OP);
      status->SetMessage(strerror(errno));
      return;
    }
  }
}

void RdmaBatchReadWrite(const EpPairVec& eps, const application::RdmaMemoryRegion& local,
                        const SizeVec& localOffsets, const application::RdmaMemoryRegion& remote,
                        const SizeVec& remoteOffsets, const SizeVec& sizes, TransferStatus* status,
                        TransferUniqueId id, int postBatchSize, bool isRead) {
  MORI_IO_FUNCTION_TIMER;

  // Check sizes
  if ((localOffsets.size() != remoteOffsets.size()) || (sizes.size() != remoteOffsets.size())) {
    status->SetCode(StatusCode::ERR_INVALID_ARGS);
    status->SetMessage("lengths of local offsets, remote offsets or sizes mismatch");
    return;
  }

  size_t batchSize = sizes.size();
  if (batchSize == 0) {
    status->SetCode(StatusCode::SUCCESS);
    return;
  }

  // Check offset and size is in range
  for (int i = 0; i < batchSize; i++) {
    if (((localOffsets[i] + sizes[i]) > local.length) ||
        ((remoteOffsets[i] + sizes[i]) > remote.length)) {
      status->SetCode(StatusCode::ERR_INVALID_ARGS);
      status->SetMessage("length out of range");
      return;
    }
  }
  status->SetCode(StatusCode::IN_PROGRESS);

  size_t epNum = eps.size();
  size_t epBatchSize = (batchSize + epNum - 1) / epNum;

  RdmaOpStatusHandle* internalStatus = new RdmaOpStatusHandle();
  internalStatus->status = status;
  internalStatus->expectedNumCqe = std::min(batchSize, epNum);

  std::vector<struct ibv_sge> sges(batchSize, ibv_sge{});
  std::vector<struct ibv_send_wr> wrs(batchSize, ibv_send_wr{});
  for (int i = 0; i < epNum; i++) {
    int st = epBatchSize * i;
    int end = std::min(static_cast<size_t>(st) + epBatchSize, batchSize);
    if ((end - st) == 0) break;

    if (postBatchSize == -1) postBatchSize = end - st;
    for (int j = st; j < end; j++) {
      struct ibv_sge& sge = sges[j];
      sge.addr = reinterpret_cast<uint64_t>(local.addr) + localOffsets[j];
      sge.length = sizes[j];
      sge.lkey = local.lkey;

      struct ibv_send_wr& wr = wrs[j];
      wr.wr_id = (j < (end - 1)) ? 0 : reinterpret_cast<uint64_t>(internalStatus);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = isRead ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
      wr.send_flags = (j < (end - 1)) ? 0 : IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remote.addr) + remoteOffsets[j];
      wr.wr.rdma.rkey = remote.rkey;
      wr.next = (j < (end - 1)) ? wrs.data() + j + 1 : nullptr;

      if ((j > st) && ((j - st) % postBatchSize == 0) || (j == (end - 1))) {
        int ret = ibv_post_send(eps[i].local.ibvHandle.qp, wrs.data() + st, nullptr);
        if (ret != 0) {
          status->SetCode(StatusCode::ERR_RDMA_OP);
          status->SetMessage(strerror(errno));
          return;
        }
        MORI_IO_TRACE("ibv_post_send ep index {} batch index {}", i, j);
      }
    }
  }
}

};  // namespace io
};  // namespace mori
