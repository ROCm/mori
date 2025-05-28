#pragma once

#include <iostream>
#include <string>

#include "infiniband/mlx5dv.h"
#include "mori/core/transport/rdma/device_primitives.hpp"
namespace mori {
namespace core {

static __device__ __host__ enum ibv_wc_status Mlx5HandleErrorCqe(struct mlx5_err_cqe* cqe) {
  switch (cqe->syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
      return IBV_WC_LOC_LEN_ERR;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
      return IBV_WC_LOC_QP_OP_ERR;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
      return IBV_WC_LOC_PROT_ERR;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
      return IBV_WC_WR_FLUSH_ERR;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
      return IBV_WC_MW_BIND_ERR;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
      return IBV_WC_BAD_RESP_ERR;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
      return IBV_WC_LOC_ACCESS_ERR;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
      return IBV_WC_REM_INV_REQ_ERR;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
      return IBV_WC_REM_ACCESS_ERR;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
      return IBV_WC_REM_OP_ERR;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
      return IBV_WC_RETRY_EXC_ERR;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
      return IBV_WC_RNR_RETRY_EXC_ERR;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
      return IBV_WC_REM_ABORT_ERR;
    default:
      return IBV_WC_GENERAL_ERR;
  }
}

static __device__ __host__ const char* IbvWcStatusString(enum ibv_wc_status status) {
  static const char* const wc_status_str[] = {
      /* IBV_WC_SUCCESS*/ "success",
      /* IBV_WC_LOC_LEN_ERR*/ "local length error",
      /* IBV_WC_LOC_QP_OP_ERR*/ "local QP operation error",
      /* IBV_WC_LOC_EEC_OP_ERR*/ "local EE context operation error",
      /* IBV_WC_LOC_PROT_ERR*/ "local protection error",
      /* IBV_WC_WR_FLUSH_ERR*/ "Work Request Flushed Error",
      /* IBV_WC_MW_BIND_ERR*/ "memory management operation error",
      /* IBV_WC_BAD_RESP_ERR*/ "bad response error",
      /* IBV_WC_LOC_ACCESS_ERR*/ "local access error",
      /* IBV_WC_REM_INV_REQ_ERR*/ "remote invalid request error",
      /* IBV_WC_REM_ACCESS_ERR*/ "remote access error",
      /* IBV_WC_REM_OP_ERR*/ "remote operation error",
      /* IBV_WC_RETRY_EXC_ERR*/ "transport retry counter exceeded",
      /* IBV_WC_RNR_RETRY_EXC_ERR*/ "RNR retry counter exceeded",
      /* IBV_WC_LOC_RDD_VIOL_ERR*/ "local RDD violation error",
      /* IBV_WC_REM_INV_RD_REQ_ERR*/ "remote invalid RD request",
      /* IBV_WC_REM_ABORT_ERR*/ "aborted error",
      /* IBV_WC_INV_EECN_ERR*/ "invalid EE context number",
      /* IBV_WC_INV_EEC_STATE_ERR*/ "invalid EE context state",
      /* IBV_WC_FATAL_ERR*/ "fatal error",
      /* IBV_WC_RESP_TIMEOUT_ERR*/ "response timeout error",
      /* IBV_WC_GENERAL_ERR*/ "general error",
      /* IBV_WC_TM_ERR*/ "TM error",
      /* IBV_WC_TM_RNDV_INCOMPLETE*/ "TM software rendezvous",
  };

  if (status < IBV_WC_SUCCESS || status > IBV_WC_TM_RNDV_INCOMPLETE) return "unknown";

  return wc_status_str[status];
}

// TODO: write a better verison
static __device__ __host__ void DumpWqe(void* wqeBaseAddr, uint32_t idx) {
  uintptr_t wqeAddr = reinterpret_cast<uintptr_t>(wqeBaseAddr) + (idx << MLX5_SEND_WQE_SHIFT);
  mlx5_wqe_ctrl_seg* wqeCtrlSeg = reinterpret_cast<mlx5_wqe_ctrl_seg*>(wqeAddr);
  uint32_t opmodIdxOpCode = BE32TOH(wqeCtrlSeg->opmod_idx_opcode);

  mlx5_wqe_data_seg* wqeDataSeg = reinterpret_cast<mlx5_wqe_data_seg*>(
      wqeAddr + sizeof(mlx5_wqe_ctrl_seg) + sizeof(mlx5_wqe_raddr_seg));
  uint32_t bytes = BE32TOH(wqeDataSeg->byte_count);
  printf("Wqe: post idx %d opcode %d bytes %d\n", opmodIdxOpCode >> 24, opmodIdxOpCode & 0xff,
         bytes);
}

static __device__ __host__ uint32_t get_num_wqes_in_atomic(atomicType amo_op, uint32_t bytes) {
  if (bytes == 8) {
    // RC
    switch (amo_op) {
      case AMO_SIGNAL:
      case AMO_SIGNAL_SET:
      case AMO_SWAP:
      case AMO_SET:
      case AMO_FETCH_AND:
      case AMO_AND:
      case AMO_FETCH_OR:
      case AMO_OR:
        return 2;
      default:
        break;
    }
  }
  return 1;
}

}  // namespace core
}  // namespace mori