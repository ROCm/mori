#pragma once

#include <hip/hip_runtime.h>

#include "infiniband/mlx5dv.h"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp"
#include "mori/core/transport/rdma/providers/mlx5/utils.h"

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ uint64_t PostSend<ProviderType::MLX5>(void* queue_buff_addr, uint32_t& post_idx,
                                                 uint32_t wqe_num, uint32_t qpn, uintptr_t laddr,
                                                 uint64_t lkey, size_t bytes_count) {
  uint32_t opcode = MLX5_OPCODE_SEND_IMM;

  uint32_t wqe_idx = post_idx & (wqe_num - 1);
  void* wqe_addr = reinterpret_cast<char*>(queue_buff_addr) + (wqe_idx << MLX5_SEND_WQE_SHIFT);

  mlx5_wqe_ctrl_seg* wqe_ctrl_seg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqe_addr);
  wqe_ctrl_seg[0] = mlx5_wqe_ctrl_seg{};
  wqe_ctrl_seg->opmod_idx_opcode = HTOBE32(((post_idx & 0xffff) << 8) | opcode);
  int size_in_octowords = int((sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_data_seg)) / 16);
  wqe_ctrl_seg->qpn_ds = HTOBE32((qpn << 8) | size_in_octowords);
  wqe_ctrl_seg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
  wqe_ctrl_seg->imm = std::numeric_limits<uint32_t>::max();

  mlx5_wqe_data_seg* wqe_data_seg = reinterpret_cast<mlx5_wqe_data_seg*>(
      reinterpret_cast<char*>(wqe_addr) + sizeof(mlx5_wqe_ctrl_seg));
  wqe_data_seg->byte_count = HTOBE32(bytes_count);
  wqe_data_seg->addr = HTOBE64(laddr);
  wqe_data_seg->lkey = HTOBE32(lkey);

  post_idx += int((size_in_octowords * 16 + MLX5_SEND_WQE_BB - 1) / MLX5_SEND_WQE_BB);
  return reinterpret_cast<uint64_t*>(wqe_ctrl_seg)[0];
}

template <>
__device__ void PostRecv<ProviderType::MLX5>(void* queue_buff_addr, uint32_t wqe_num,
                                             uint32_t& post_idx, uintptr_t laddr, uint64_t lkey,
                                             size_t bytes_count) {
  uint32_t wqe_idx = post_idx & (wqe_num - 1);
  void* wqe_addr = reinterpret_cast<char*>(queue_buff_addr) + wqe_idx * sizeof(mlx5_wqe_data_seg);
  mlx5_wqe_data_seg* wqe_data_seg = reinterpret_cast<mlx5_wqe_data_seg*>(wqe_addr);
  wqe_data_seg->byte_count = HTOBE32(bytes_count);
  wqe_data_seg->lkey = HTOBE32(lkey);
  wqe_data_seg->addr = HTOBE64(laddr);

  post_idx += 1;
}

__device__ uint64_t PostReadWrite(void* queue_buff_addr, uint32_t wqe_num, uint32_t& post_idx,
                                  uint32_t qpn, uintptr_t laddr, uint64_t lkey, uintptr_t raddr,
                                  uint64_t rkey, size_t bytes_count, bool is_read) {
  uint32_t opcode = is_read ? MLX5_OPCODE_RDMA_READ : MLX5_OPCODE_RDMA_WRITE;

  mlx5_wqe_ctrl_seg* wqe_ctrl_seg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(queue_buff_addr);
  wqe_ctrl_seg->opmod_idx_opcode = HTOBE32(((post_idx & 0xffff) << 8) | opcode);
  int size_in_octowords = int(
      (sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_data_seg) + sizeof(mlx5_wqe_raddr_seg)) / 16);
  wqe_ctrl_seg->qpn_ds = HTOBE32((qpn << 8) | size_in_octowords);
  wqe_ctrl_seg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  mlx5_wqe_raddr_seg* wqe_raddr_seg = reinterpret_cast<mlx5_wqe_raddr_seg*>(
      reinterpret_cast<char*>(queue_buff_addr) + sizeof(mlx5_wqe_ctrl_seg));
  wqe_raddr_seg->raddr = HTOBE64(raddr);
  wqe_raddr_seg->rkey = HTOBE32(rkey);

  mlx5_wqe_data_seg* wqe_data_seg =
      reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<char*>(queue_buff_addr) +
                                           sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  wqe_data_seg->byte_count = HTOBE32(bytes_count);
  wqe_data_seg->addr = HTOBE64(laddr);
  wqe_data_seg->lkey = HTOBE32(lkey);

  return reinterpret_cast<uint64_t*>(queue_buff_addr)[0];
}

template <>
__device__ uint64_t PostWrite<ProviderType::MLX5>(void* queue_buff_addr, uint32_t wqe_num,
                                                  uint32_t& post_idx, uint32_t qpn, uintptr_t laddr,
                                                  uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                  size_t bytes_count) {
  return PostReadWrite(queue_buff_addr, wqe_num, post_idx, qpn, laddr, lkey, raddr, rkey,
                       bytes_count, false);
}

template <>
__device__ uint64_t PostRead<ProviderType::MLX5>(void* queue_buff_addr, uint32_t wqe_num,
                                                 uint32_t& post_idx, uint32_t qpn, uintptr_t laddr,
                                                 uint64_t lkey, uintptr_t raddr, uint64_t rkey,
                                                 size_t bytes_count) {
  return PostReadWrite(queue_buff_addr, wqe_num, post_idx, qpn, laddr, lkey, raddr, rkey,
                       bytes_count, true);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ void UpdateSendDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqe_idx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_SND_DBR] = HTOBE32(wqe_idx & 0xffff);
}

template <>
__device__ void UpdateRecvDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t wqe_idx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_RCV_DBR] = HTOBE32(wqe_idx & 0xffff);
}

template <>
__device__ void RingDoorbell<ProviderType::MLX5>(void* dbr_addr, uint64_t dbr_val) {
  // TODO: make this atomic
  __be32 first_dword = HTOBE32(BE64TOH(dbr_val) >> 32);
  __be32 second_dword = HTOBE32(BE64TOH(dbr_val));
  reinterpret_cast<uint32_t*>(dbr_addr)[0] = first_dword;
  reinterpret_cast<uint32_t*>(dbr_addr)[1] = second_dword;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ int PollCqOnce<ProviderType::MLX5>(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum,
                                              uint32_t& consIdx) {
  int idx = consIdx % cqeNum;
  void* cqe_addr = reinterpret_cast<char*>(cqAddr) + idx * cqeSize;

  uint64_t last_dword = atomicAdd(reinterpret_cast<uint64_t*>(cqe_addr) + 7, 0);
  uint8_t op_own = reinterpret_cast<char*>(&last_dword)[7];

  uint8_t opcode = op_own >> 4;
  uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;

  bool is_empty = true;
  for (int i = 0; i < (sizeof(mlx5_cqe64) / sizeof(uint64_t)); i++) {
    if (atomicAdd(&reinterpret_cast<uint64_t*>(cqe_addr)[i], 0) != 0) {
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
__device__ int PoolCq<ProviderType::MLX5>(void* cqAddr, uint32_t cqeSize, uint32_t cqeNum,
                                          uint32_t& consIdx) {
  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::MLX5>(cqAddr, cqeSize, cqeNum, consIdx);
  } while (opcode < 0);

  if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
    int idx = consIdx % cqeNum;
    void* cqe_addr = reinterpret_cast<char*>(cqAddr) + idx * cqeSize;
    mlx5_err_cqe* ecqe = reinterpret_cast<mlx5_err_cqe*>(cqe_addr);
    auto error = Mlx5HandleErrorCqe(ecqe);
    printf("%s\n", IbvWcStatusString(error));
    assert(false);
  }
  return opcode;
}

template <>
inline __device__ void UpdateCqDbrRecord<ProviderType::MLX5>(void* dbrRecAddr, uint32_t cons_idx) {
  reinterpret_cast<uint32_t*>(dbrRecAddr)[MLX5_CQ_SET_CI] = HTOBE32(cons_idx & 0xffffff);
}

}  // namespace core
}  // namespace mori