#pragma once

#include <iostream>
#include <string>

#include "infiniband/mlx5dv.h"
#include "transport/providers/mlx5/mlx5.h"

static enum ibv_wc_status Mlx5HandleErrorCqe(struct mlx5_err_cqe* cqe) {
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

static std::string DumpCqe(void* cqe) {
  std::ostringstream ss;
  for (int i = 0; i < 4; i++) {
    ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(cqe)[i * 4 + 0]) << " ";
    ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(cqe)[i * 4 + 1]) << " ";
    ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(cqe)[i * 4 + 2]) << " ";
    ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(cqe)[i * 4 + 3]) << std::endl;
  }
  return ss.str();
}

static std::string DumpWqeCtrlSeq(mlx5_wqe_ctrl_seg* seg) {
  std::ostringstream ss;
  ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(seg)[0]) << " ";
  ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(seg)[1]) << " ";
  ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(seg)[2]) << " ";
  ss << std::hex << be32toh(reinterpret_cast<uint32_t*>(seg)[3]) << std::endl;
  return ss.str();
}

static std::string GetGidTypeString(ibv_gid_type_sysfs type) {
  if (type == IBV_GID_TYPE_SYSFS_IB_ROCE_V1) {
    return "IB/RoCEv1";
  } else if (type == IBV_GID_TYPE_SYSFS_ROCE_V2) {
    return "RoCEv2";
  }
  assert(false);
}