#pragma once

#include <hip/hip_runtime.h>

#include "infiniband/mlx5dv.h"
#include "mori/core/transport/ibgda/device_primitives.hpp"

namespace mori {
namespace core {
namespace transport {
namespace ibgda {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ uint64_t PostWrite<ProviderType::MLX5>(const IbgdaWriteReq& req) {
  uint32_t opcode = MLX5_OPCODE_NOP;  // MLX5_OPCODE_RDMA_WRITE;

  mlx5_wqe_ctrl_seg* wqe_ctrl_seg =
      reinterpret_cast<mlx5_wqe_ctrl_seg*>(req.qp_handle.next_wqe_addr);
  wqe_ctrl_seg->opmod_idx_opcode = HTOBE32(((req.qp_handle.post_idx & 0xffff) << 8) | opcode);
  int size_in_octowords = int(
      (sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_data_seg) + sizeof(mlx5_wqe_raddr_seg)) / 16);
  wqe_ctrl_seg->qpn_ds = HTOBE32((req.qp_handle.qpn << 8) | size_in_octowords);
  wqe_ctrl_seg->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  mlx5_wqe_raddr_seg* wqe_raddr_seg = reinterpret_cast<mlx5_wqe_raddr_seg*>(
      reinterpret_cast<char*>(req.qp_handle.next_wqe_addr) + sizeof(mlx5_wqe_ctrl_seg));
  wqe_raddr_seg->raddr = HTOBE64(req.remote_mr.addr);
  wqe_raddr_seg->rkey = HTOBE32(req.remote_mr.rkey);

  mlx5_wqe_data_seg* wqe_data_seg =
      reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<char*>(req.qp_handle.next_wqe_addr) +
                                           sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  wqe_data_seg->byte_count = HTOBE32(req.bytes_count);
  wqe_data_seg->addr = HTOBE64(req.local_mr.addr);
  wqe_data_seg->lkey = HTOBE32(req.local_mr.lkey);

  return reinterpret_cast<uint64_t*>(req.qp_handle.next_wqe_addr)[0];
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            Doorbell                                            */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ void UpdateSendDbrRecord<ProviderType::MLX5>(void* dbr_rec_addr, uint32_t wqe_idx) {
  reinterpret_cast<uint32_t*>(dbr_rec_addr)[MLX5_SND_DBR] = HTOBE32(wqe_idx & 0xffff);
}

template <>
__device__ void UpdateRecvDbrRecord<ProviderType::MLX5>(void* dbr_rec_addr, uint32_t wqe_idx) {
  reinterpret_cast<uint32_t*>(dbr_rec_addr)[MLX5_RCV_DBR] = HTOBE32(wqe_idx & 0xffff);
}

template <>
__device__ void RingDoorbell<ProviderType::MLX5>(void* dbr_addr, uint64_t dbr_val) {
  __be32 first_dword = HTOBE32(BE64TOH(dbr_val) >> 32);
  __be32 second_dword = HTOBE32(BE64TOH(dbr_val));
  reinterpret_cast<uint32_t*>(dbr_addr)[0] = first_dword;
  reinterpret_cast<uint32_t*>(dbr_addr)[1] = second_dword;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Completion Queue                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
__device__ int PollCqOnce<ProviderType::MLX5>(CompletionQueueHandle cq) {
  int idx = cq.consumer_idx % cq.cqe_num;
  void* cqe_addr = reinterpret_cast<char*>(cq.cq_addr) + idx * cq.cqe_size;

  //   void* cqe_last_dword_addr =
  //       reinterpret_cast<char*>(cqe_addr) + sizeof(mlx5_cqe64) - sizeof(uint64_t);
  //   uint64_t last_dword_val = atomicAdd(reinterpret_cast<uint64_t*>(cqe_last_dword_addr), 0);

  uint64_t last_dword_val = atomicAdd(reinterpret_cast<uint64_t*>(cqe_addr) + 7, 0);

  uint8_t op_own = reinterpret_cast<char*>(&last_dword_val)[7];
  uint8_t opcode = op_own >> 4;
  uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;

  // TODO: check if cqe_num should be power of 2?
  //   int cq_owner_flip = !!(cq.consumer_idx & (cq.cqe_num + 1));
  int cq_owner_flip = ~(~(cq.consumer_idx & cq.cqe_num));
  if ((opcode == MLX5_CQE_INVALID) || (owner ^ cq_owner_flip)) {
    return -1;
  }

  return opcode;
}

template <>
__device__ int PoolCq<ProviderType::MLX5>(CompletionQueueHandle cq) {
  int opcode = -1;
  do {
    opcode = PollCqOnce<ProviderType::MLX5>(cq);
  } while (opcode < 0);

  return opcode;
}

}  // namespace ibgda
}  // namespace transport
}  // namespace core
}  // namespace mori