#pragma once

#include <hip/hip_runtime.h>

#include "infiniband/mlx5dv.h"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp"
#include "mori/core/transport/rdma/providers/mlx5/utils.h"
#include "mori/core/utils.hpp"

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ uint64_t PostSend<ProviderType::MLX5>(void* queueBuffAddr, uint32_t* postIdx,
                                                 uint32_t wqeNum, uint32_t qpn, uintptr_t laddr,
                                                 uint64_t lkey, size_t bytes) {
  constexpr int sendWqeSize = sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_data_seg);
  constexpr int numOctoWords = CeilDiv(sendWqeSize, 16);
  constexpr int numWqeBb = CeilDiv(numOctoWords * 16, int(MLX5_SEND_WQE_BB));

  uint32_t curPostIdx = atomicAdd(postIdx, numWqeBb);

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | MLX5_OPCODE_SEND);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | numOctoWords);
  wqeCtrlSeg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  mlx5_wqe_data_seg* wqeDataSeg =
      reinterpret_cast<mlx5_wqe_data_seg*>(wqeAddr + sizeof(mlx5_wqe_ctrl_seg));
  wqeDataSeg->byte_count = HTOBE32(bytes);
  wqeDataSeg->addr = HTOBE64(laddr);
  wqeDataSeg->lkey = HTOBE32(lkey);

  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

template <>
__device__ void PostRecv<ProviderType::MLX5>(void* queueBuffAddr, uint32_t wqeNum,
                                             uint32_t* postIdx, uintptr_t laddr, uint64_t lkey,
                                             size_t bytes) {
  uint32_t curPostIdx = atomicAdd(postIdx, 1);
  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);

  void* wqeAddr = reinterpret_cast<char*>(queueBuffAddr) + wqeIdx * sizeof(mlx5_wqe_data_seg);
  mlx5_wqe_data_seg* wqe_data_seg = reinterpret_cast<mlx5_wqe_data_seg*>(wqeAddr);
  wqe_data_seg->byte_count = HTOBE32(bytes);
  wqe_data_seg->lkey = HTOBE32(lkey);
  wqe_data_seg->addr = HTOBE64(laddr);
}

__device__ uint64_t PostReadWrite(void* queueBuffAddr, uint32_t wqeNum, uint32_t* postIdx,
                                  uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                  uint64_t rkey, size_t bytes, bool isRead) {
  uint32_t opcode = isRead ? MLX5_OPCODE_RDMA_READ : MLX5_OPCODE_RDMA_WRITE;

  constexpr int sendWqeSize =
      sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) + sizeof(mlx5_wqe_data_seg);
  constexpr int numOctoWords = CeilDiv(sendWqeSize, 16);
  constexpr int numWqeBb = CeilDiv(numOctoWords * 16, int(MLX5_SEND_WQE_BB));

  uint32_t curPostIdx = atomicAdd(postIdx, numWqeBb);

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | opcode);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | numOctoWords);
  wqeCtrlSeg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

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
__device__ uint64_t PostWrite<ProviderType::MLX5>(void* queueBuffAddr, uint32_t wqeNum,
                                                  uint32_t* postIdx, uint32_t qpn, uintptr_t laddr,
                                                  uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                  size_t bytes) {
  return PostReadWrite(queueBuffAddr, wqeNum, postIdx, qpn, laddr, lkey, raddr, rkey, bytes, false);
}

template <>
__device__ uint64_t PostRead<ProviderType::MLX5>(void* queueBuffAddr, uint32_t wqeNum,
                                                 uint32_t* postIdx, uint32_t qpn, uintptr_t laddr,
                                                 uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                 size_t bytes) {
  return PostReadWrite(queueBuffAddr, wqeNum, postIdx, qpn, laddr, lkey, raddr, rkey, bytes, true);
}

constexpr uint32_t MaxInlineDataSizePerWqe =
    sizeof(mlx5_wqe_data_seg) - sizeof(mlx5_wqe_inl_data_seg);

template <ProviderType PrvdType>
static __device__ uint64_t PostWriteInline(void* queueBuffAddr, uint32_t wqeNum, uint32_t* postIdx,
                                           uint32_t qpn, void* val, uintptr_t raddr, uint64_t rkey,
                                           size_t bytes) {
  constexpr int sendWqeSize =
      sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg) + sizeof(mlx5_wqe_data_seg);
  assert(bytes <= MaxInlineDataSizePerWqe);

  constexpr int numOctoWords = CeilDiv(sendWqeSize, 16);
  constexpr int numWqeBb = CeilDiv(numOctoWords * 16, int(MLX5_SEND_WQE_BB));

  uint32_t curPostIdx = atomicAdd(postIdx, numWqeBb);

  uint32_t wqeIdx = curPostIdx & (wqeNum - 1);
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(queueBuffAddr) + (wqeIdx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  wqeCtrlSeg->opmod_idx_opcode = HTOBE32(((curPostIdx & 0xffff) << 8) | MLX5_OPCODE_RDMA_WRITE);
  wqeCtrlSeg->qpn_ds = HTOBE32((qpn << 8) | numOctoWords);
  wqeCtrlSeg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

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
  } else {
    for (int i = 0; i < bytes; i++) {
      AtomicStoreRelaxed(reinterpret_cast<uint8_t*>(wqeDataPtr) + i,
                         reinterpret_cast<uint8_t*>(val)[i]);
    }
  }
  return reinterpret_cast<uint64_t*>(wqeCtrlSeg)[0];
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ void UpdateSendDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_SND_DBR] = HTOBE32(wqeIdx & 0xffff);
}

template <>
__device__ void UpdateRecvDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqeIdx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_RCV_DBR] = HTOBE32(wqeIdx & 0xffff);
}

template <>
__device__ void RingDoorbell<ProviderType::MLX5>(void* dbrAddr, uint64_t dbrVal) {
  reinterpret_cast<uint64_t*>(dbrAddr)[0] = dbrVal;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ int PollCqOnce<ProviderType::MLX5>(void* cqeAddr, uint32_t cqeSize, uint32_t cqeNum,
                                              uint32_t consIdx) {
  uint64_t lastDword = atomicAdd(reinterpret_cast<uint64_t*>(cqeAddr) + 7, 0);
  uint8_t opOwn = reinterpret_cast<char*>(&lastDword)[7];

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
  return opcode;
}

template <>
__device__ int PollCq<ProviderType::MLX5>(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum,
                                          uint32_t* consIdx) {
  uint32_t curConsIdx = atomicAdd(consIdx, 1);
  int cqeIdx = curConsIdx % cqeNum;
  void* cqeAddr = reinterpret_cast<char*>(cqAddr) + cqeIdx * cqeSize;

  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::MLX5>(cqeAddr, cqeSize, cqeNum, curConsIdx);
  } while (opcode < 0);

  if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
    auto error = Mlx5HandleErrorCqe(reinterpret_cast<mlx5_err_cqe*>(cqeAddr));
    printf("%s\n", IbvWcStatusString(error));
    return opcode;
  }
  return opcode;
}

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t cons_idx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_CQ_SET_CI] = HTOBE32(cons_idx & 0xffffff);
}

}  // namespace core
}  // namespace mori