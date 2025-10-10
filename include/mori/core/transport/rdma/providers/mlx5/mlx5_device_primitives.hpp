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
#pragma once

#include <hip/hip_runtime.h>

#include "infiniband/mlx5dv.h"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp"
#include "mori/core/transport/rdma/providers/utils.h"
#include "mori/core/utils.hpp"

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
/*                                        Send / Recv APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t Mlx5PostSend(WorkQueueHandle& wq, uint32_t curPostIdx, bool cqeSignal,
                                        uint32_t qpn, uintptr_t laddr, uint64_t lkey,
                                        size_t bytes) {
  uint8_t signalFlag = cqeSignal ? MLX5_WQE_CTRL_CQ_UPDATE : 0x00;
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;
  constexpr int sendWqeSize = sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_data_seg);
  constexpr int numOctoWords = CeilDiv(sendWqeSize, 16);
  constexpr int numWqeBb = CeilDiv(numOctoWords * 16, int(MLX5_SEND_WQE_BB));

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | MLX5_OPCODE_SEND);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | numOctoWords);
  wqeCtrlSeg->fm_ce_se = signalFlag;

  mlx5_wqe_data_seg* wqeDataSeg =
      reinterpret_cast<mlx5_wqe_data_seg*>(wqeAddr + sizeof(mlx5_wqe_ctrl_seg));
  wqeDataSeg->byte_count = HTOBE32(bytes);
  wqeDataSeg->addr = HTOBE64(laddr);
  wqeDataSeg->lkey = HTOBE32(lkey);

  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

template <>
inline __device__ uint64_t PostSend<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t postIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, size_t bytes) {
  return Mlx5PostSend(wq, postIdx, cqeSignal, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostSend<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return Mlx5PostSend(wq, curPostIdx, true, qpn, laddr, lkey, bytes);
}

inline __device__ uint64_t Mlx5PostRecv(WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t qpn,
                                        uintptr_t laddr, uint64_t lkey, size_t bytes) {
  void* queueBuffAddr = wq.rqAddr;
  uint32_t wqeNum = wq.rqWqeNum;

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);

  void* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + wqeIdx * sizeof(mlx5_wqe_data_seg);
  mlx5_wqe_data_seg* wqe_data_seg = reinterpret_cast<mlx5_wqe_data_seg*>(wqeAddr);
  wqe_data_seg->byte_count = HTOBE32(bytes);
  wqe_data_seg->lkey = HTOBE32(lkey);
  wqe_data_seg->addr = HTOBE64(laddr);
  return reinterpret_cast<uint64_t*>(wqe_data_seg)[0];
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        bool cqeSignal, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  return Mlx5PostRecv(wq, curPostIdx, qpn, laddr, lkey, bytes);
}

template <>
inline __device__ uint64_t PostRecv<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return Mlx5PostRecv(wq, curPostIdx, qpn, laddr, lkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Read / Write APIs                                       */
/* ---------------------------------------------------------------------------------------------- */
static constexpr int SendWqeSize =
    sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) + sizeof(mlx5_wqe_data_seg);
static constexpr int SendWqeNumOctoWords = CeilDiv(SendWqeSize, 16);
static constexpr int SendWqeNumWqeBb = CeilDiv(SendWqeNumOctoWords * 16, int(MLX5_SEND_WQE_BB));

// TODO: convert raddr/rkey laddr/lkey to big endien in advance to save cycles
inline __device__ uint64_t Mlx5PostReadWrite(WorkQueueHandle& wq, uint32_t curPostIdx,
                                             bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                             uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                             size_t bytes, bool isRead) {
  uint32_t opcode = isRead ? MLX5_OPCODE_RDMA_READ : MLX5_OPCODE_RDMA_WRITE;
  uint8_t signalFlag = cqeSignal ? MLX5_WQE_CTRL_CQ_UPDATE : 0x00;
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | opcode);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | SendWqeNumOctoWords);
  wqeCtrlSeg->fm_ce_se = signalFlag;

  mlx5_wqe_raddr_seg* wqeRaddrSeg =
      reinterpret_cast<mlx5_wqe_raddr_seg*>(wqeAddr + sizeof(mlx5_wqe_ctrl_seg));
  wqeRaddrSeg->raddr = HTOBE64(raddr);
  wqeRaddrSeg->rkey = HTOBE32(rkey);

  mlx5_wqe_data_seg* wqeDataSeg = reinterpret_cast<mlx5_wqe_data_seg*>(
      wqeAddr + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  wqeDataSeg->byte_count = HTOBE32(bytes);
  wqeDataSeg->addr = HTOBE64(laddr);
  wqeDataSeg->lkey = HTOBE32(lkey);

  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                         uint32_t curMsntblSlotIdx,
                                                         uint32_t curPsnIdx, bool cqeSignal,
                                                         uint32_t qpn, uintptr_t laddr,
                                                         uint64_t lkey, uintptr_t raddr,
                                                         uint64_t rkey, size_t bytes) {
  return Mlx5PostReadWrite(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                        uint32_t curMsntblSlotIdx,
                                                        uint32_t curPsnIdx, bool cqeSignal,
                                                        uint32_t qpn, uintptr_t laddr,
                                                        uint64_t lkey, uintptr_t raddr,
                                                        uint64_t rkey, size_t bytes) {
  return Mlx5PostReadWrite(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, bytes, true);
}

template <>
inline __device__ uint64_t PostWrite<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                         uintptr_t laddr, uint64_t lkey,
                                                         uintptr_t raddr, uint64_t rkey,
                                                         size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return Mlx5PostReadWrite(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes, false);
}

template <>
inline __device__ uint64_t PostRead<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                        uintptr_t laddr, uint64_t lkey,
                                                        uintptr_t raddr, uint64_t rkey,
                                                        size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return Mlx5PostReadWrite(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, bytes, true);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        WriteInline APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
static constexpr uint32_t MaxInlineDataSizePerWqe =
    sizeof(mlx5_wqe_data_seg) - sizeof(mlx5_wqe_inl_data_seg);

inline __device__ uint64_t Mlx5PostWriteInline(WorkQueueHandle& wq, uint32_t curPostIdx,
                                               bool cqeSignal, uint32_t qpn, void* val,
                                               uintptr_t raddr, uint64_t rkey, size_t bytes) {
  assert(bytes <= MaxInlineDataSizePerWqe);
  uint8_t signalFlag = cqeSignal ? MLX5_WQE_CTRL_CQ_UPDATE : 0x00;
  void* queueBuffAddr = wq.sqAddr;
  uint32_t wqeNum = wq.sqWqeNum;

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | MLX5_OPCODE_RDMA_WRITE);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | SendWqeNumOctoWords);
  wqeCtrlSeg->fm_ce_se = signalFlag;

  mlx5_wqe_raddr_seg* wqeRaddrSeg =
      reinterpret_cast<mlx5_wqe_raddr_seg*>(wqeAddr + sizeof(mlx5_wqe_ctrl_seg));
  wqeRaddrSeg->raddr = HTOBE64(raddr);
  wqeRaddrSeg->rkey = HTOBE32(rkey);

  mlx5_wqe_inl_data_seg* wqeInlDataSeg = reinterpret_cast<mlx5_wqe_inl_data_seg*>(
      wqeAddr + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  wqeInlDataSeg->byte_count = HTOBE32(bytes | MLX5_INLINE_SEG);

  void* wqeDataPtr =
      reinterpret_cast<void*>(wqeAddr + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) +
                              sizeof(mlx5_wqe_inl_data_seg));

  // TODO: support other size
  if (bytes == 4) {
    AtomicStoreRelaxed(reinterpret_cast<uint32_t*>(wqeDataPtr),
                       reinterpret_cast<uint32_t*>(val)[0]);
  } else if (bytes == 8) {
    AtomicStoreRelaxed(reinterpret_cast<uint64_t*>(wqeDataPtr),
                       reinterpret_cast<uint64_t*>(val)[0]);
  } else {
    for (int i = 0; i < bytes; i++) {
      AtomicStoreRelaxed(reinterpret_cast<uint8_t*>(wqeDataPtr) + i,
                         reinterpret_cast<uint8_t*>(val)[i]);
    }
  }
  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::MLX5>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, void* val, uintptr_t raddr, uint64_t rkey, size_t bytes) {
  return Mlx5PostWriteInline(wq, curPostIdx, cqeSignal, qpn, val, raddr, rkey, bytes);
}

template <>
inline __device__ uint64_t PostWriteInline<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                               void* val, uintptr_t raddr,
                                                               uint64_t rkey, size_t bytes) {
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  return Mlx5PostWriteInline(wq, curPostIdx, true, qpn, val, raddr, rkey, bytes);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Atomic APIs                                             */
/* ---------------------------------------------------------------------------------------------- */

inline __device__ uint64_t mlx5PrepareAtomicWqe(WorkQueueHandle& wq, uint32_t curPostIdx,
                                                bool cqeSignal, uint32_t qpn, uintptr_t laddr,
                                                uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                void* val_1, void* val_2, uint32_t bytes,
                                                atomicType amo_op) {
  uint8_t signalFlag = cqeSignal ? MLX5_WQE_CTRL_CQ_UPDATE : 0x00;
  void* queueBuffAddr = wq.rqAddr;
  uint32_t wqeNum = wq.rqWqeNum;

  uint32_t numWqesPerCmd = get_num_wqes_in_atomic(amo_op, bytes);
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  void* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  void* addition_wqe_addr =
      reinterpret_cast<char*>(queueBuffAddr) + ((wqeIdx + 1) << MLX5_SEND_WQE_SHIFT);

  int atomicWqeSize =
      sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) + 2 * sizeof(mlx5_wqe_atomic_seg);

  struct mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  struct mlx5_wqe_raddr_seg* wqeRaddrSeg = reinterpret_cast<mlx5_wqe_raddr_seg*>(
      reinterpret_cast<char*>(wqeAddr) + sizeof(mlx5_wqe_ctrl_seg));
  struct mlx5_wqe_atomic_seg* wqeAtomicSeg1 = reinterpret_cast<mlx5_wqe_atomic_seg*>(
      reinterpret_cast<char*>(wqeAddr) + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  struct mlx5_wqe_atomic_seg* wqeAtomicSeg2 = reinterpret_cast<mlx5_wqe_atomic_seg*>(
      reinterpret_cast<char*>(wqeAddr) + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) +
      sizeof(mlx5_wqe_atomic_seg));

  struct mlx5_wqe_data_seg* wqeDataSeg = (struct mlx5_wqe_data_seg*)wqeAtomicSeg2;

  wqeRaddrSeg->raddr = HTOBE64(raddr);
  wqeRaddrSeg->rkey = HTOBE32(rkey);
  wqeRaddrSeg->reserved = 0;

  switch (amo_op) {
    case AMO_FETCH_INC:
    case AMO_INC: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_fa_seg_t* atomic_32_masked_fa_seg =
            (ibgda_atomic_32_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_fa_seg->add_data = HTOBE32((uint32_t)1);
        atomic_32_masked_fa_seg->field_boundary = 0;
      } else {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_fa_seg_t* atomic_64_masked_fa_seg =
            (ibgda_atomic_64_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_fa_seg->add_data = HTOBE64((uint64_t)1);
        atomic_64_masked_fa_seg->field_boundary = 0;
      }
      break;
    }
    case AMO_SIGNAL:
    case AMO_SIGNAL_SET:
    case AMO_SWAP:
    case AMO_SET: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_cs_seg_t* atomic_32_masked_cs_seg =
            (ibgda_atomic_32_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_cs_seg->compare_data = 0;
        atomic_32_masked_cs_seg->compare_mask = 0;
        atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
      } else {
        atomicWqeSize += sizeof(mlx5_wqe_data_seg);
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_data_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_cs_data_seg->compare = 0;

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_mask_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg2;
        atomic_64_masked_cs_mask_seg->swap = UINT64_MAX;
        atomic_64_masked_cs_mask_seg->compare = 0;

        wqeDataSeg = (struct mlx5_wqe_data_seg*)addition_wqe_addr;
      }
      break;
    }
    case AMO_SIGNAL_ADD:
    case AMO_ADD: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_fa_seg_t* atomic_32_masked_fa_seg =
            (ibgda_atomic_32_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_fa_seg->field_boundary = 0;
      } else {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_fa_seg_t* atomic_64_masked_fa_seg =
            (ibgda_atomic_64_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_fa_seg->field_boundary = 0;
      }
      break;
    }
    case AMO_FETCH_AND:
    case AMO_AND: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_cs_seg_t* atomic_32_masked_cs_seg =
            (ibgda_atomic_32_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_cs_seg->compare_data = 0;
        atomic_32_masked_cs_seg->compare_mask = 0;
        atomic_32_masked_cs_seg->swap_mask = HTOBE32(~(*(uint32_t*)val_1));
      } else {
        atomicWqeSize += sizeof(mlx5_wqe_data_seg);
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_data_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_cs_data_seg->compare = 0;

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_mask_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg2;
        atomic_64_masked_cs_mask_seg->swap = HTOBE64(~(*(uint64_t*)val_1));
        atomic_64_masked_cs_mask_seg->compare = 0;
        wqeDataSeg = (struct mlx5_wqe_data_seg*)addition_wqe_addr;
      }
      break;
    }
    case AMO_FETCH_OR:
    case AMO_OR: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_cs_seg_t* atomic_32_masked_cs_seg =
            (ibgda_atomic_32_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_cs_seg->compare_data = 0;
        atomic_32_masked_cs_seg->compare_mask = 0;
        atomic_32_masked_cs_seg->swap_mask = HTOBE32(*(uint32_t*)val_1);
      } else {
        atomicWqeSize += sizeof(mlx5_wqe_data_seg);
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_data_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_cs_data_seg->compare = 0;

        ibgda_atomic_64_masked_cs_seg_t* atomic_64_masked_cs_mask_seg =
            (ibgda_atomic_64_masked_cs_seg_t*)wqeAtomicSeg2;
        atomic_64_masked_cs_mask_seg->swap = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_cs_mask_seg->compare = 0;
        wqeDataSeg = (struct mlx5_wqe_data_seg*)addition_wqe_addr;
      }
      break;
    }
    case AMO_FETCH_XOR:
    case AMO_XOR: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_fa_seg_t* atomic_32_masked_fa_seg =
            (ibgda_atomic_32_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_fa_seg->field_boundary = UINT32_MAX;
      } else {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_fa_seg_t* atomic_64_masked_fa_seg =
            (ibgda_atomic_64_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t*)val_1);
        atomic_64_masked_fa_seg->field_boundary = UINT64_MAX;
      }
      break;
    }
    case AMO_FETCH: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_fa_seg_t* atomic_32_masked_fa_seg =
            (ibgda_atomic_32_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_fa_seg->add_data = 0;
        atomic_32_masked_fa_seg->field_boundary = 0;
      } else {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_8_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_64_masked_fa_seg_t* atomic_64_masked_fa_seg =
            (ibgda_atomic_64_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_64_masked_fa_seg->add_data = 0;
        atomic_64_masked_fa_seg->field_boundary = 0;
      }
      break;
    }
    case AMO_FETCH_ADD: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_fa_seg_t* atomic_32_masked_fa_seg =
            (ibgda_atomic_32_masked_fa_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_fa_seg->field_boundary = 0;
      } else {
        wqeCtrlSeg->opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_FA | (wqeIdx << 8));
        wqeAtomicSeg1->swap_add = HTOBE64(*(uint64_t*)val_1);
      }
      break;
    }
    case AMO_COMPARE_SWAP: {
      if (bytes == 4) {
        wqeCtrlSeg->opmod_idx_opcode =
            HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqeIdx << 8) | IBGDA_4_BYTE_EXT_AMO_OPMOD);

        ibgda_atomic_32_masked_cs_seg_t* atomic_32_masked_cs_seg =
            (ibgda_atomic_32_masked_cs_seg_t*)wqeAtomicSeg1;
        atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t*)val_1);
        atomic_32_masked_cs_seg->compare_data = HTOBE32(*(uint32_t*)val_2);
        atomic_32_masked_cs_seg->compare_mask = UINT32_MAX;
        atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
      } else {
        wqeCtrlSeg->opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_CS | (wqeIdx << 8));
        wqeAtomicSeg1->swap_add = HTOBE64(*(uint64_t*)val_1);
        wqeAtomicSeg1->compare = HTOBE64(*(uint64_t*)val_2);
      }
      break;
    }
    default: {
      assert(0);
    }
  }

  int numOctoWords = CeilDiv(atomicWqeSize, 16);
  int numWqeBb = CeilDiv(numOctoWords * 16, int(MLX5_SEND_WQE_BB));
  assert(numWqeBb == numWqesPerCmd);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | numOctoWords);
  wqeCtrlSeg->fm_ce_se = signalFlag;

  wqeDataSeg->byte_count = HTOBE32(bytes);
  wqeDataSeg->addr = HTOBE64(laddr);
  wqeDataSeg->lkey = HTOBE32(lkey);
  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::MLX5>(
    WorkQueueHandle& wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx,
    bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr, uint64_t rkey,
    void* val_1, void* val_2, uint32_t typeBytes, atomicType amo_op) {
  return mlx5PrepareAtomicWqe(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey, val_1,
                              val_2, typeBytes, amo_op);
}

template <>
inline __device__ uint64_t PostAtomic<ProviderType::MLX5>(WorkQueueHandle& wq, uint32_t qpn,
                                                          uintptr_t laddr, uint64_t lkey,
                                                          uintptr_t raddr, uint64_t rkey,
                                                          void* val_1, void* val_2,
                                                          uint32_t typeBytes, atomicType amo_op) {
  uint32_t numWqesPerCmd = get_num_wqes_in_atomic(amo_op, typeBytes);
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, numWqesPerCmd);
  return mlx5PrepareAtomicWqe(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey, val_1, val_2,
                              typeBytes, amo_op);
}

#define DEFINE_MLX5_POST_ATOMIC_SPEC(TYPE)                                                      \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::MLX5, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t curPostIdx, uint32_t curMsntblSlotIdx, uint32_t curPsnIdx, \
      bool cqeSignal, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,            \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    return mlx5PrepareAtomicWqe(wq, curPostIdx, cqeSignal, qpn, laddr, lkey, raddr, rkey,       \
                                (void*)&val_1, (void*)&val_2, sizeof(TYPE), amo_op);            \
  }                                                                                             \
  template <>                                                                                   \
  inline __device__ uint64_t PostAtomic<ProviderType::MLX5, TYPE>(                              \
      WorkQueueHandle & wq, uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,      \
      uint64_t rkey, const TYPE val_1, const TYPE val_2, atomicType amo_op) {                   \
    uint32_t typeBytes = sizeof(TYPE);                                                          \
    uint32_t numWqesPerCmd = get_num_wqes_in_atomic(amo_op, typeBytes);                         \
    uint32_t curPostIdx = atomicAdd(&wq.postIdx, numWqesPerCmd);                                \
    return mlx5PrepareAtomicWqe(wq, curPostIdx, true, qpn, laddr, lkey, raddr, rkey,            \
                                (void*)&val_1, (void*)&val_2, typeBytes, amo_op);               \
  }

DEFINE_MLX5_POST_ATOMIC_SPEC(uint32_t)
DEFINE_MLX5_POST_ATOMIC_SPEC(uint64_t)
DEFINE_MLX5_POST_ATOMIC_SPEC(int32_t)
DEFINE_MLX5_POST_ATOMIC_SPEC(int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void UpdateSendDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx) {
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint32_t*>(dbrRecAddr) + MLX5_SND_DBR,
                                HTOBE32(wqeIdx & 0xffff));
}

template <>
inline __device__ void UpdateRecvDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx) {
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint32_t*>(dbrRecAddr) + MLX5_RCV_DBR,
                                HTOBE32(wqeIdx & 0xffff));
}

template <>
inline __device__ void RingDoorbell<ProviderType::MLX5>(void* dbrAddr, uint64_t dbrVal) {
  core::AtomicStoreSeqCstSystem(reinterpret_cast<uint64_t*>(dbrAddr), dbrVal);
}

template <>
inline __device__ void UpdateDbrAndRingDbSend<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                  void* dbrAddr, uint64_t dbrVal,
                                                                  uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateSendDbrRecord<ProviderType::MLX5>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::MLX5>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

template <>
inline __device__ void UpdateDbrAndRingDbRecv<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx,
                                                                  void* dbrAddr, uint64_t dbrVal,
                                                                  uint32_t* lockVar) {
  AcquireLock(lockVar);

  UpdateRecvDbrRecord<ProviderType::MLX5>(dbrRecAddr, wqeIdx);
  __threadfence_system();
  RingDoorbell<ProviderType::MLX5>(dbrAddr, dbrVal);

  ReleaseLock(lockVar);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ int PollCqOnce<ProviderType::MLX5>(void* cqeAddr, uint32_t cqeNum,
                                                     uint32_t consIdx, uint32_t* wqeIdx) {
  uint64_t* cqeQwords = reinterpret_cast<uint64_t*>(cqeAddr);
  auto* lastBytePtr = reinterpret_cast<uint8_t*>(cqeQwords + 7) + 7;
  uint8_t opOwn = *lastBytePtr;

  uint8_t opcode = opOwn >> 4;
  uint8_t owner = opOwn & MLX5_CQE_OWNER_MASK;

  bool is_empty = true;
  for (int i = 0; i < (sizeof(mlx5_cqe64) / sizeof(uint64_t)); i++) {
    if (atomicAdd(&reinterpret_cast<uint64_t*>(cqeAddr)[i], 0) != 0) {
      is_empty = false;
      break;
    }
  }

  // TODO: check if cqeNum should be power of 2?
  //   int cq_owner_flip = !!(consIdx & (cqeNum + 1));
  int cq_owner_flip = !!(consIdx & cqeNum);
  if ((opcode == MLX5_CQE_INVALID) || (owner ^ cq_owner_flip) || is_empty) {
    return -1;
  }

  *lastBytePtr = (MLX5_CQE_INVALID << 4) | (cq_owner_flip & 1);
  return opcode;
}

template <>
inline __device__ int PollCq<ProviderType::MLX5>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx) {
  uint32_t curConsIdx = atomicAdd(consIdx, 1);
  uint32_t cqeIdx = curConsIdx % cqeNum;
  void* cqeAddr = reinterpret_cast<char*>(cqAddr) + cqeIdx * sizeof(mlx5_cqe64);

  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::MLX5>(cqeAddr, cqeNum, curConsIdx, nullptr);
    // TODO: Explain clearly why adding a compiler barrier fix hang issue
    asm volatile("" ::: "memory");
  } while (opcode < 0);

  if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
    auto error = Mlx5HandleErrorCqe(reinterpret_cast<mlx5_err_cqe*>(cqeAddr));
    printf("(%s:%d) CQE error: %s\n", __FILE__, __LINE__, IbvWcStatusString(error));
    return opcode;
  }
  return opcode;
}

template <>
inline __device__ int PollCq<ProviderType::MLX5>(void* cqAddr, uint32_t cqeNum, uint32_t* consIdx,
                                                 uint16_t* wqeCounter) {
  uint32_t curConsIdx = *consIdx;
  uint32_t cqeIdx = curConsIdx % cqeNum;
  void* cqeAddr = reinterpret_cast<char*>(cqAddr) + cqeIdx * sizeof(mlx5_cqe64);
  // mlx5_cqe64* cqeAddr = reinterpret_cast<mlx5_cqe64*>(cqAddr) + cqeIdx;

  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::MLX5>(cqeAddr, cqeNum, curConsIdx, nullptr);
    asm volatile("" ::: "memory");
  } while (opcode < 0);

  if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
    auto error = Mlx5HandleErrorCqe(reinterpret_cast<mlx5_err_cqe*>(cqeAddr));
    printf("(%s:%d) CQE error: %s\n", __FILE__, __LINE__, IbvWcStatusString(error));
    return opcode;
  }
  *wqeCounter = BE16TOH(reinterpret_cast<mlx5_cqe64*>(cqeAddr)->wqe_counter);
  return opcode;
}

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t cons_idx, uint32_t cqeNum) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_CQ_SET_CI] = HTOBE32(cons_idx & 0xffffff);
}

template <>
inline __device__ int PollCqAndUpdateDbr<ProviderType::MLX5>(void* cqAddr, uint32_t cqeSize,
                                                             uint32_t cqeNum, uint32_t* consIdx,
                                                             void* dbrRecAddr, uint32_t* lockVar) {
  AcquireLock(lockVar);

  int opcode = PollCq<ProviderType::MLX5>(cqAddr, cqeNum, consIdx);
  if (opcode >= 0) {
    UpdateCqDbrRecord<ProviderType::MLX5>(dbrRecAddr, *consIdx, cqeNum);
  }

  ReleaseLock(lockVar);
  return opcode;
}

}  // namespace core
}  // namespace mori
