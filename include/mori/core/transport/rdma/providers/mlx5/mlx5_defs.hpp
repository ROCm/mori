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

// Vendored mlx5 WQE/CQE hardware-ABI subset.
//
// These structs/constants are a verbatim, device-safe copy of the small, stable
// subset of mlx5's WQE/CQE layout that the mori device code touches. Vendoring
// them lets the mlx5 device path drop <infiniband/mlx5dv.h> (and the system
// <infiniband/verbs.h> it drags in), the same way bnxt/ionic get their ABI from
// repo-vendored firmware headers. Only <stdint.h> is needed here.
//
// The names are mori-prefixed on purpose (Mlx5*, MORI_MLX5_*). A naive same-name
// copy (mlx5_cqe64, MLX5_OPCODE_SEND, ...) was tried and reverted: host/device/
// example TUs mix the system ::mlx5_* with the vendored ones and reference them
// unqualified, which is ambiguous. Distinct names can never collide.
//
// These structs overlay NIC hardware memory, so the layout MUST match the system
// header byte-for-byte. That parity is enforced at compile time by the
// static_asserts in
// src/application/transport/rdma/providers/mlx5/mlx5_abi_parity.cpp, which is the
// one place that includes both this header and the system mlx5dv.h. Any layout
// drift there is a build error — fix it before trusting these definitions.
//
// __beN fields in the system header are kept as plain uintN_t here; the device
// code byte-swaps explicitly (HTOBE*/BE*TOH).

#include <stdint.h>

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                        WQE/CQE structs                                          */
/* ---------------------------------------------------------------------------------------------- */

// ::mlx5_wqe_ctrl_seg (16 B)
struct Mlx5WqeCtrlSeg {
  uint32_t opmod_idx_opcode;
  uint32_t qpn_ds;
  uint8_t signature;
  uint16_t dci_stream_channel_id;
  uint8_t fm_ce_se;
  uint32_t imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

// ::mlx5_wqe_data_seg (16 B)
struct Mlx5WqeDataSeg {
  uint32_t byte_count;
  uint32_t lkey;
  uint64_t addr;
};

// ::mlx5_wqe_raddr_seg (16 B)
struct Mlx5WqeRaddrSeg {
  uint64_t raddr;
  uint32_t rkey;
  uint32_t reserved;
};

// ::mlx5_wqe_atomic_seg (16 B)
struct Mlx5WqeAtomicSeg {
  uint64_t swap_add;
  uint64_t compare;
};

// ::mlx5_wqe_inl_data_seg (4 B)
struct Mlx5WqeInlDataSeg {
  uint32_t byte_count;
};

// ::mlx5_err_cqe (64 B)
struct Mlx5ErrCqe {
  uint8_t rsvd0[32];
  uint32_t srqn;
  uint8_t rsvd1[18];
  uint8_t vendor_err_synd;
  uint8_t syndrome;
  uint32_t s_wqe_opcode_qpn;
  uint16_t wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

// ::mlx5_cqe64 (64 B). mlx5dv's leading union { anon hdr / mlx5_tm_cqe / ibv_tmh }
// is 32 B; the device code never reads its members (only the trailer +
// wqe_counter/op_own), so keep it opaque — avoids vendoring mlx5_tm_cqe / ibv_tmh.
struct Mlx5Cqe64 {
  uint8_t rsvd_hdr[32];
  uint32_t srqn_uidx;
  uint32_t imm_inval_pkey;
  uint8_t app;
  uint8_t app_op;
  uint16_t app_info;
  uint32_t byte_cnt;
  uint64_t timestamp;
  uint32_t sop_drop_qpn;
  uint16_t wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                           Constants                                             */
/* ---------------------------------------------------------------------------------------------- */

enum {
  MORI_MLX5_SEND_WQE_BB = 64,
  MORI_MLX5_SEND_WQE_SHIFT = 6,

  MORI_MLX5_RCV_DBR = 0,
  MORI_MLX5_SND_DBR = 1,

  MORI_MLX5_CQ_SET_CI = 0,
  MORI_MLX5_CQ_ARM_DB = 1,

  MORI_MLX5_WQE_CTRL_CQ_UPDATE = 2 << 2,

  MORI_MLX5_CQE_OWNER_MASK = 1,
  MORI_MLX5_CQE_REQ_ERR = 13,
  MORI_MLX5_CQE_RESP_ERR = 14,
  MORI_MLX5_CQE_INVALID = 15,

  MORI_MLX5_OPCODE_RDMA_WRITE = 0x08,
  MORI_MLX5_OPCODE_SEND = 0x0a,
  MORI_MLX5_OPCODE_RDMA_READ = 0x10,
  MORI_MLX5_OPCODE_ATOMIC_CS = 0x11,
  MORI_MLX5_OPCODE_ATOMIC_FA = 0x12,
  MORI_MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
  MORI_MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,

  MORI_MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR = 0x01,
  MORI_MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR = 0x02,
  MORI_MLX5_CQE_SYNDROME_LOCAL_PROT_ERR = 0x04,
  MORI_MLX5_CQE_SYNDROME_WR_FLUSH_ERR = 0x05,
  MORI_MLX5_CQE_SYNDROME_MW_BIND_ERR = 0x06,
  MORI_MLX5_CQE_SYNDROME_BAD_RESP_ERR = 0x10,
  MORI_MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR = 0x11,
  MORI_MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR = 0x12,
  MORI_MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR = 0x13,
  MORI_MLX5_CQE_SYNDROME_REMOTE_OP_ERR = 0x14,
  MORI_MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR = 0x15,
  MORI_MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR = 0x16,
  MORI_MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR = 0x22,
};

// 0x80000000 does not fit in a (signed) plain enum's underlying type portably;
// keep it as a typed constant.
constexpr uint32_t MORI_MLX5_INLINE_SEG = 0x80000000u;

}  // namespace core
}  // namespace mori
