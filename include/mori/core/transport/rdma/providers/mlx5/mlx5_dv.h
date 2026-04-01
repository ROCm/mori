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

// Internal mlx5 direct-verbs type definitions.
// These mirror the definitions from <infiniband/mlx5dv.h> so that we can
// compile without the system mlx5 development headers installed.
// The actual library functions are loaded at runtime via dlopen/dlsym.

#pragma once

// If the system mlx5dv.h is available, use it directly.
// Otherwise, provide our own minimal definitions.
#if __has_include(<infiniband/mlx5dv.h>)
#include <infiniband/mlx5dv.h>
#else

#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/* -------------------------------------------------------------------------- */
/*  Struct definitions from mlx5dv.h                                          */
/* -------------------------------------------------------------------------- */

struct mlx5dv_pd {
  uint32_t pdn;
  uint64_t comp_mask;
};

struct mlx5dv_obj {
  struct {
    struct ibv_qp* in;
    void* out;
  } qp;
  struct {
    struct ibv_cq* in;
    void* out;
  } cq;
  struct {
    struct ibv_srq* in;
    void* out;
  } srq;
  struct {
    struct ibv_wq* in;
    void* out;
  } rwq;
  struct {
    struct ibv_dm* in;
    void* out;
  } dm;
  struct {
    struct ibv_ah* in;
    void* out;
  } ah;
  struct {
    struct ibv_pd* in;
    struct mlx5dv_pd* out;
  } pd;
};

// Opaque handle — only forward-declared in the real header too.
struct mlx5dv_devx_obj;

struct mlx5dv_devx_umem {
  uint32_t umem_id;
};

struct mlx5dv_devx_uar {
  void* reg_addr;
  void* base_addr;
  uint32_t page_id;
  off_t mmap_off;
  uint64_t comp_mask;
};

/* -------------------------------------------------------------------------- */
/*  Constants                                                                 */
/* -------------------------------------------------------------------------- */

enum mlx5dv_obj_type {
  MLX5DV_OBJ_QP = 1 << 0,
  MLX5DV_OBJ_CQ = 1 << 1,
  MLX5DV_OBJ_SRQ = 1 << 2,
  MLX5DV_OBJ_RWQ = 1 << 3,
  MLX5DV_OBJ_DM = 1 << 4,
  MLX5DV_OBJ_AH = 1 << 5,
  MLX5DV_OBJ_PD = 1 << 6,
};

// UAR allocation flags
#ifndef MLX5DV_UAR_ALLOC_TYPE_NC
#define MLX5DV_UAR_ALLOC_TYPE_NC 0x1
#endif

#endif  // __has_include(<infiniband/mlx5dv.h>)
