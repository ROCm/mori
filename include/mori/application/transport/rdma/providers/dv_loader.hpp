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

// Runtime loader for vendor-specific RDMA direct-verbs libraries.
// Uses dlopen/dlsym so that no dv library is needed at compile time.

#pragma once

#include <dlfcn.h>

#include <string>

#include "mori/utils/mori_log.hpp"

// Load a shared library at runtime. Returns the handle, or nullptr on failure.
inline void* DvLoadLibrary(const char* lib_name) {
  void* handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MORI_APP_TRACE("dlopen({}) failed: {}", lib_name, dlerror());
  } else {
    MORI_APP_TRACE("dlopen({}) succeeded", lib_name);
  }
  return handle;
}

// Load a symbol from a library handle. Returns nullptr on failure.
inline void* DvLoadSymbol(void* handle, const char* symbol_name) {
  void* sym = dlsym(handle, symbol_name);
  if (!sym) {
    MORI_APP_WARN("dlsym({}) failed: {}", symbol_name, dlerror());
  }
  return sym;
}

// ============================================================================
// MLX5 direct-verbs function pointers
// ============================================================================
struct Mlx5DvApi {
  using devx_general_cmd_t = int (*)(struct ibv_context*, const void*, size_t, void*, size_t);
  using devx_umem_reg_t = struct mlx5dv_devx_umem* (*)(struct ibv_context*, void*, size_t,
                                                       uint32_t);
  using devx_umem_dereg_t = int (*)(struct mlx5dv_devx_umem*);
  using devx_alloc_uar_t = struct mlx5dv_devx_uar* (*)(struct ibv_context*, uint32_t);
  using devx_free_uar_t = void (*)(struct mlx5dv_devx_uar*);
  using devx_query_eqn_t = int (*)(struct ibv_context*, uint32_t, uint32_t*);
  using devx_obj_create_t = struct mlx5dv_devx_obj* (*)(struct ibv_context*, const void*, size_t,
                                                        void*, size_t);
  using devx_obj_modify_t = int (*)(struct mlx5dv_devx_obj*, const void*, size_t, void*, size_t);
  using devx_obj_destroy_t = int (*)(struct mlx5dv_devx_obj*);
  using init_obj_t = int (*)(struct mlx5dv_obj*, uint64_t);

  devx_general_cmd_t devx_general_cmd = nullptr;
  devx_umem_reg_t devx_umem_reg = nullptr;
  devx_umem_dereg_t devx_umem_dereg = nullptr;
  devx_alloc_uar_t devx_alloc_uar = nullptr;
  devx_free_uar_t devx_free_uar = nullptr;
  devx_query_eqn_t devx_query_eqn = nullptr;
  devx_obj_create_t devx_obj_create = nullptr;
  devx_obj_modify_t devx_obj_modify = nullptr;
  devx_obj_destroy_t devx_obj_destroy = nullptr;
  init_obj_t init_obj = nullptr;

  void* handle = nullptr;

  bool Load() {
    handle = DvLoadLibrary("libmlx5.so");
    if (!handle) return false;

    devx_general_cmd = (devx_general_cmd_t)DvLoadSymbol(handle, "mlx5dv_devx_general_cmd");
    devx_umem_reg = (devx_umem_reg_t)DvLoadSymbol(handle, "mlx5dv_devx_umem_reg");
    devx_umem_dereg = (devx_umem_dereg_t)DvLoadSymbol(handle, "mlx5dv_devx_umem_dereg");
    devx_alloc_uar = (devx_alloc_uar_t)DvLoadSymbol(handle, "mlx5dv_devx_alloc_uar");
    devx_free_uar = (devx_free_uar_t)DvLoadSymbol(handle, "mlx5dv_devx_free_uar");
    devx_query_eqn = (devx_query_eqn_t)DvLoadSymbol(handle, "mlx5dv_devx_query_eqn");
    devx_obj_create = (devx_obj_create_t)DvLoadSymbol(handle, "mlx5dv_devx_obj_create");
    devx_obj_modify = (devx_obj_modify_t)DvLoadSymbol(handle, "mlx5dv_devx_obj_modify");
    devx_obj_destroy = (devx_obj_destroy_t)DvLoadSymbol(handle, "mlx5dv_devx_obj_destroy");
    init_obj = (init_obj_t)DvLoadSymbol(handle, "mlx5dv_init_obj");

    return devx_general_cmd && devx_umem_reg && devx_umem_dereg && devx_alloc_uar &&
           devx_free_uar && devx_query_eqn && devx_obj_create && devx_obj_modify &&
           devx_obj_destroy && init_obj;
  }

  static Mlx5DvApi& Instance() {
    static Mlx5DvApi api;
    return api;
  }

  static bool Available() {
    static bool loaded = Instance().Load();
    return loaded;
  }
};

// ============================================================================
// BNXT direct-verbs function pointers
// ============================================================================
struct BnxtDvApi {
  using umem_reg_t = void* (*)(struct ibv_context*, struct bnxt_re_dv_umem_reg_attr*);
  using umem_dereg_t = int (*)(void*);
  using create_cq_t = struct ibv_cq* (*)(struct ibv_context*, struct bnxt_re_dv_cq_init_attr*);
  using destroy_cq_t = int (*)(struct ibv_cq*);
  using init_obj_t = int (*)(struct bnxt_re_dv_obj*, uint32_t);
  using create_qp_t = struct ibv_qp* (*)(struct ibv_pd*, struct bnxt_re_dv_qp_init_attr*);
  using destroy_qp_t = int (*)(struct ibv_qp*);
  using modify_qp_t = int (*)(struct ibv_qp*, struct ibv_qp_attr*, int, int, int);
  using alloc_db_region_t = struct bnxt_re_dv_db_region_attr* (*)(struct ibv_context*);
  using free_db_region_t = int (*)(struct ibv_context*, struct bnxt_re_dv_db_region_attr*);
  using modify_qp_udp_sport_t = int (*)(struct ibv_qp*, uint16_t);
  using get_default_db_region_t = int (*)(struct ibv_context*, struct bnxt_re_dv_db_region_attr*);

  umem_reg_t umem_reg = nullptr;
  umem_dereg_t umem_dereg = nullptr;
  create_cq_t create_cq = nullptr;
  destroy_cq_t destroy_cq = nullptr;
  init_obj_t init_obj = nullptr;
  create_qp_t create_qp = nullptr;
  destroy_qp_t destroy_qp = nullptr;
  modify_qp_t modify_qp = nullptr;
  alloc_db_region_t alloc_db_region = nullptr;
  free_db_region_t free_db_region = nullptr;
  modify_qp_udp_sport_t modify_qp_udp_sport = nullptr;
  get_default_db_region_t get_default_db_region = nullptr;

  void* handle = nullptr;

  bool Load() {
    handle = DvLoadLibrary("libbnxt_re-rdmav34.so");
    if (!handle) {
      handle = DvLoadLibrary("libbnxt_re.so");
    }
    if (!handle) return false;

    umem_reg = (umem_reg_t)DvLoadSymbol(handle, "bnxt_re_dv_umem_reg");
    umem_dereg = (umem_dereg_t)DvLoadSymbol(handle, "bnxt_re_dv_umem_dereg");
    create_cq = (create_cq_t)DvLoadSymbol(handle, "bnxt_re_dv_create_cq");
    destroy_cq = (destroy_cq_t)DvLoadSymbol(handle, "bnxt_re_dv_destroy_cq");
    init_obj = (init_obj_t)DvLoadSymbol(handle, "bnxt_re_dv_init_obj");
    create_qp = (create_qp_t)DvLoadSymbol(handle, "bnxt_re_dv_create_qp");
    destroy_qp = (destroy_qp_t)DvLoadSymbol(handle, "bnxt_re_dv_destroy_qp");
    modify_qp = (modify_qp_t)DvLoadSymbol(handle, "bnxt_re_dv_modify_qp");
    alloc_db_region = (alloc_db_region_t)DvLoadSymbol(handle, "bnxt_re_dv_alloc_db_region");
    free_db_region = (free_db_region_t)DvLoadSymbol(handle, "bnxt_re_dv_free_db_region");
    modify_qp_udp_sport =
        (modify_qp_udp_sport_t)DvLoadSymbol(handle, "bnxt_re_dv_modify_qp_udp_sport");
    get_default_db_region =
        (get_default_db_region_t)DvLoadSymbol(handle, "bnxt_re_dv_get_default_db_region");

    // Required symbols for basic operation
    return umem_reg && umem_dereg && create_cq && destroy_cq && init_obj && create_qp &&
           destroy_qp && modify_qp;
  }

  static BnxtDvApi& Instance() {
    static BnxtDvApi api;
    return api;
  }

  static bool Available() {
    static bool loaded = Instance().Load();
    return loaded;
  }
};

// ============================================================================
// IONIC direct-verbs function pointers
// ============================================================================
struct IonicDvApi {
  using get_ctx_t = int (*)(struct ionic_dv_ctx*, struct ibv_context*);
  using qp_get_udma_idx_t = uint8_t (*)(struct ibv_qp*);
  using get_cq_t = int (*)(struct ionic_dv_cq*, struct ibv_cq*, uint8_t);
  using get_qp_t = int (*)(struct ionic_dv_qp*, struct ibv_qp*);
  using pd_set_sqcmb_t = int (*)(struct ibv_pd*, bool, bool, bool);
  using pd_set_rqcmb_t = int (*)(struct ibv_pd*, bool, bool, bool);
  using pd_set_udma_mask_t = int (*)(struct ibv_pd*, uint32_t);

  get_ctx_t get_ctx = nullptr;
  qp_get_udma_idx_t qp_get_udma_idx = nullptr;
  get_cq_t get_cq = nullptr;
  get_qp_t get_qp = nullptr;
  pd_set_sqcmb_t pd_set_sqcmb = nullptr;
  pd_set_rqcmb_t pd_set_rqcmb = nullptr;
  pd_set_udma_mask_t pd_set_udma_mask = nullptr;

  void* handle = nullptr;

  bool Load() {
    handle = DvLoadLibrary("libionic.so");
    if (!handle) return false;

    get_ctx = (get_ctx_t)DvLoadSymbol(handle, "ionic_dv_get_ctx");
    qp_get_udma_idx = (qp_get_udma_idx_t)DvLoadSymbol(handle, "ionic_dv_qp_get_udma_idx");
    get_cq = (get_cq_t)DvLoadSymbol(handle, "ionic_dv_get_cq");
    get_qp = (get_qp_t)DvLoadSymbol(handle, "ionic_dv_get_qp");
    pd_set_sqcmb = (pd_set_sqcmb_t)DvLoadSymbol(handle, "ionic_dv_pd_set_sqcmb");
    pd_set_rqcmb = (pd_set_rqcmb_t)DvLoadSymbol(handle, "ionic_dv_pd_set_rqcmb");
    pd_set_udma_mask = (pd_set_udma_mask_t)DvLoadSymbol(handle, "ionic_dv_pd_set_udma_mask");

    return get_ctx && qp_get_udma_idx && get_cq && get_qp && pd_set_sqcmb && pd_set_rqcmb &&
           pd_set_udma_mask;
  }

  static IonicDvApi& Instance() {
    static IonicDvApi api;
    return api;
  }

  static bool Available() {
    static bool loaded = Instance().Load();
    return loaded;
  }
};

namespace mori {
namespace application {
using ::BnxtDvApi;
using ::IonicDvApi;
using ::Mlx5DvApi;
}  // namespace application
}  // namespace mori
