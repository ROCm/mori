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

// MLX5 vendored-ABI parity guard.
//
// The mori::core::Mlx5* structs in mlx5_defs.hpp overlay NIC hardware memory, so
// their layout MUST match the system ::mlx5_* structs byte-for-byte. A wrong
// field offset does not fail to compile — it silently corrupts the RDMA WQE/CQE
// the hardware reads. This is the ONE translation unit that includes BOTH the
// system mlx5dv.h and the vendored header; the static_asserts below turn any
// layout drift into a compile error. Keep it isolated (no other mlx5 code) so the
// dual visibility of system + vendored names causes no ambiguity.

#include <cstddef>

#include <infiniband/mlx5dv.h>                                    // system ::mlx5_*

#include "mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp"  // vendored mori::core::Mlx5*

#define SZ(v, s) static_assert(sizeof(::mori::core::v) == sizeof(::s), "size drift " #v)
#define OFF(v, s, vm, sm) \
  static_assert(offsetof(::mori::core::v, vm) == offsetof(::s, sm), "offset drift " #v "::" #vm)

SZ(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg);
SZ(Mlx5WqeDataSeg, mlx5_wqe_data_seg);
SZ(Mlx5WqeRaddrSeg, mlx5_wqe_raddr_seg);
SZ(Mlx5WqeAtomicSeg, mlx5_wqe_atomic_seg);
SZ(Mlx5WqeInlDataSeg, mlx5_wqe_inl_data_seg);
SZ(Mlx5ErrCqe, mlx5_err_cqe);
SZ(Mlx5Cqe64, mlx5_cqe64);

OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, opmod_idx_opcode, opmod_idx_opcode);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, qpn_ds, qpn_ds);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, signature, signature);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, fm_ce_se, fm_ce_se);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, imm, imm);

OFF(Mlx5WqeDataSeg, mlx5_wqe_data_seg, byte_count, byte_count);
OFF(Mlx5WqeDataSeg, mlx5_wqe_data_seg, lkey, lkey);
OFF(Mlx5WqeDataSeg, mlx5_wqe_data_seg, addr, addr);

OFF(Mlx5WqeRaddrSeg, mlx5_wqe_raddr_seg, raddr, raddr);
OFF(Mlx5WqeRaddrSeg, mlx5_wqe_raddr_seg, rkey, rkey);
OFF(Mlx5WqeRaddrSeg, mlx5_wqe_raddr_seg, reserved, reserved);

OFF(Mlx5WqeAtomicSeg, mlx5_wqe_atomic_seg, swap_add, swap_add);
OFF(Mlx5WqeAtomicSeg, mlx5_wqe_atomic_seg, compare, compare);

OFF(Mlx5ErrCqe, mlx5_err_cqe, syndrome, syndrome);
OFF(Mlx5ErrCqe, mlx5_err_cqe, wqe_counter, wqe_counter);
OFF(Mlx5ErrCqe, mlx5_err_cqe, op_own, op_own);

OFF(Mlx5Cqe64, mlx5_cqe64, srqn_uidx, srqn_uidx);
OFF(Mlx5Cqe64, mlx5_cqe64, byte_cnt, byte_cnt);
OFF(Mlx5Cqe64, mlx5_cqe64, wqe_counter, wqe_counter);
OFF(Mlx5Cqe64, mlx5_cqe64, signature, signature);
OFF(Mlx5Cqe64, mlx5_cqe64, op_own, op_own);

#undef SZ
#undef OFF
