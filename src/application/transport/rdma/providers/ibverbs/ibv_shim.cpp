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

// Runtime shim for the core libibverbs API.
//
// Historically mori linked libibverbs at build time (-libverbs). This file
// removes that link-time dependency: it defines every core ibv_* symbol that
// mori references and forwards each call to libibverbs.so.1, which is dlopen()ed
// lazily on first use. This mirrors the dlopen approach already used for the
// vendor direct-verbs libraries (libmlx5 / libbnxt_re / libionic) in
// dv_loader.cpp.
//
// Why this works without touching any call site:
//   * The plain ibv_* functions mori calls (ibv_alloc_pd, ibv_create_qp, ...)
//     resolve to the definitions below.
//   * The static-inline / macro wrappers in <infiniband/verbs.h>
//     (ibv_reg_mr, ibv_query_port, ibv_query_gid_ex, ibv_create_qp_ex,
//     ibv_query_device_ex, ...) ultimately call a handful of real symbols
//     (ibv_reg_mr, ibv_reg_mr_iova2, ibv_query_port, _ibv_query_gid_ex,
//     ibv_create_qp, ibv_query_device); those are provided here too.
//   * The remaining inline helpers (ibv_post_send, ibv_poll_cq, ...) dispatch
//     through context/qp op tables and need no external symbol at all.
//
// All symbols defined here are compiled with hidden visibility (see CMake) so
// each mori shared object carries its own private copy and we never interpose a
// real libibverbs that may also be present in the process (e.g. via RCCL).

// dlvsym() (used to pin the libibverbs ABI version, see IbvSym) is a GNU
// extension and is only declared by <dlfcn.h> when _GNU_SOURCE is set.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdlib>

#include "mori/utils/mori_log.hpp"

// <infiniband/verbs.h> defines ibv_reg_mr and ibv_query_port as function-like
// macros. Undefine them so we can define the underlying real symbols that those
// macros (and the header's inline wrappers) expand to.
#undef ibv_reg_mr
#undef ibv_query_port

namespace {

// Lazily dlopen libibverbs once. Order (mirrors NCCL): MORI_IBVERBS_LIB override,
// then the unversioned and versioned sonames. Returns nullptr if none is found,
// in which case every shim degrades to a failure return so that RDMA discovery /
// setup fails gracefully instead of crashing on a host without RDMA.
void* IbvHandle() {
  static void* handle = [] {
    const char* libs[] = {std::getenv("MORI_IBVERBS_LIB"), "libibverbs.so", "libibverbs.so.1"};
    for (const char* lib : libs) {
      if (!lib || !*lib) continue;
      void* h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
      if (h) {
        MORI_APP_TRACE("dlopen({}) succeeded", lib);
        return h;
      }
      MORI_APP_TRACE("dlopen({}) failed: {}", lib, dlerror());
    }
    MORI_APP_WARN("failed to dlopen libibverbs (set MORI_IBVERBS_LIB to override)");
    return static_cast<void*>(nullptr);
  }();
  return handle;
}

// Resolve a symbol, preferring the exact ABI version (like NCCL's dlvsym usage)
// for determinism across libibverbs revisions, and falling back to the default /
// unversioned symbol so we never regress when a version string does not match
// (e.g. ibv_create_comp_channel only exists as IBVERBS_1.0 on some builds).
void* IbvSym(const char* name, const char* version) {
  void* h = IbvHandle();
  if (!h) return nullptr;
  dlerror();  // clear any stale error before the lookups (dl* API contract)
  void* sym = dlvsym(h, name, version);
  if (!sym) sym = dlsym(h, name);
  if (!sym) {
    const char* err = dlerror();
    MORI_APP_WARN("failed to resolve {}: {}", name, err ? err : "symbol not found");
  }
  return sym;
}

// Failure value for a shim whose symbol could not be resolved. Set errno so
// callers that log it (e.g. RDMA MR registration paths) get an explicit reason
// instead of a stale/irrelevant value, mirroring how libibverbs reports errors.
template <typename T>
T IbvUnavailable(T fail) {
  errno = ENOSYS;
  return fail;
}

}  // namespace

// Resolve a symbol once per shim (function-local static init is thread-safe) and
// reuse the cached pointer. decltype(&name) yields the exact function-pointer
// type from the verbs.h declaration, so the forwarding stays type-checked. The
// version string is the symbol's libibverbs ABI version (see IbvSym).
#define MORI_IBV_RESOLVE(name, version) \
  static auto fn = reinterpret_cast<decltype(&name)>(IbvSym(#name, version))

extern "C" {

// ---- device enumeration -----------------------------------------------------
struct ibv_device** ibv_get_device_list(int* num_devices) {
  MORI_IBV_RESOLVE(ibv_get_device_list, "IBVERBS_1.1");
  if (!fn) {
    // Match libibverbs: report zero devices so callers reading the out-param
    // (e.g. RdmaContext::nums_device) don't observe an uninitialized count.
    if (num_devices) *num_devices = 0;
    return IbvUnavailable<struct ibv_device**>(nullptr);
  }
  return fn(num_devices);
}
void ibv_free_device_list(struct ibv_device** list) {
  MORI_IBV_RESOLVE(ibv_free_device_list, "IBVERBS_1.1");
  if (fn) fn(list);
}
struct ibv_context* ibv_open_device(struct ibv_device* device) {
  MORI_IBV_RESOLVE(ibv_open_device, "IBVERBS_1.1");
  return fn ? fn(device) : IbvUnavailable<struct ibv_context*>(nullptr);
}
int ibv_close_device(struct ibv_context* context) {
  MORI_IBV_RESOLVE(ibv_close_device, "IBVERBS_1.1");
  return fn ? fn(context) : IbvUnavailable(-1);
}

// ---- protection domains -----------------------------------------------------
struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) {
  MORI_IBV_RESOLVE(ibv_alloc_pd, "IBVERBS_1.1");
  return fn ? fn(context) : IbvUnavailable<struct ibv_pd*>(nullptr);
}
int ibv_dealloc_pd(struct ibv_pd* pd) {
  MORI_IBV_RESOLVE(ibv_dealloc_pd, "IBVERBS_1.1");
  return fn ? fn(pd) : IbvUnavailable(-1);
}

// ---- memory regions ---------------------------------------------------------
struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  MORI_IBV_RESOLVE(ibv_reg_mr, "IBVERBS_1.1");
  return fn ? fn(pd, addr, length, access) : IbvUnavailable<struct ibv_mr*>(nullptr);
}
struct ibv_mr* ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                unsigned int access) {
  MORI_IBV_RESOLVE(ibv_reg_mr_iova2, "IBVERBS_1.8");
  return fn ? fn(pd, addr, length, iova, access) : IbvUnavailable<struct ibv_mr*>(nullptr);
}
struct ibv_mr* ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                 int fd, int access) {
  MORI_IBV_RESOLVE(ibv_reg_dmabuf_mr, "IBVERBS_1.12");
  return fn ? fn(pd, offset, length, iova, fd, access) : IbvUnavailable<struct ibv_mr*>(nullptr);
}
int ibv_dereg_mr(struct ibv_mr* mr) {
  MORI_IBV_RESOLVE(ibv_dereg_mr, "IBVERBS_1.1");
  return fn ? fn(mr) : IbvUnavailable(-1);
}

// ---- completion channels / queues -------------------------------------------
// NB: ibv_create/destroy_comp_channel are only versioned IBVERBS_1.0.
struct ibv_comp_channel* ibv_create_comp_channel(struct ibv_context* context) {
  MORI_IBV_RESOLVE(ibv_create_comp_channel, "IBVERBS_1.0");
  return fn ? fn(context) : IbvUnavailable<struct ibv_comp_channel*>(nullptr);
}
int ibv_destroy_comp_channel(struct ibv_comp_channel* channel) {
  MORI_IBV_RESOLVE(ibv_destroy_comp_channel, "IBVERBS_1.0");
  return fn ? fn(channel) : IbvUnavailable(-1);
}
struct ibv_cq* ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context,
                             struct ibv_comp_channel* channel, int comp_vector) {
  MORI_IBV_RESOLVE(ibv_create_cq, "IBVERBS_1.1");
  return fn ? fn(context, cqe, cq_context, channel, comp_vector)
            : IbvUnavailable<struct ibv_cq*>(nullptr);
}
int ibv_destroy_cq(struct ibv_cq* cq) {
  MORI_IBV_RESOLVE(ibv_destroy_cq, "IBVERBS_1.1");
  return fn ? fn(cq) : IbvUnavailable(-1);
}
int ibv_get_cq_event(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context) {
  MORI_IBV_RESOLVE(ibv_get_cq_event, "IBVERBS_1.1");
  return fn ? fn(channel, cq, cq_context) : IbvUnavailable(-1);
}
void ibv_ack_cq_events(struct ibv_cq* cq, unsigned int nevents) {
  MORI_IBV_RESOLVE(ibv_ack_cq_events, "IBVERBS_1.1");
  if (fn) fn(cq, nevents);
}

// ---- queue pairs / shared receive queues ------------------------------------
struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
  MORI_IBV_RESOLVE(ibv_create_qp, "IBVERBS_1.1");
  return fn ? fn(pd, qp_init_attr) : IbvUnavailable<struct ibv_qp*>(nullptr);
}
int ibv_destroy_qp(struct ibv_qp* qp) {
  MORI_IBV_RESOLVE(ibv_destroy_qp, "IBVERBS_1.1");
  return fn ? fn(qp) : IbvUnavailable(-1);
}
int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  MORI_IBV_RESOLVE(ibv_modify_qp, "IBVERBS_1.1");
  return fn ? fn(qp, attr, attr_mask) : IbvUnavailable(-1);
}
struct ibv_srq* ibv_create_srq(struct ibv_pd* pd, struct ibv_srq_init_attr* srq_init_attr) {
  MORI_IBV_RESOLVE(ibv_create_srq, "IBVERBS_1.1");
  return fn ? fn(pd, srq_init_attr) : IbvUnavailable<struct ibv_srq*>(nullptr);
}
int ibv_destroy_srq(struct ibv_srq* srq) {
  MORI_IBV_RESOLVE(ibv_destroy_srq, "IBVERBS_1.1");
  return fn ? fn(srq) : IbvUnavailable(-1);
}

// ---- device / port / gid queries --------------------------------------------
int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
  MORI_IBV_RESOLVE(ibv_query_device, "IBVERBS_1.1");
  return fn ? fn(context, device_attr) : IbvUnavailable(-1);
}
int ibv_query_port(struct ibv_context* context, uint8_t port_num,
                   struct _compat_ibv_port_attr* port_attr) {
  MORI_IBV_RESOLVE(ibv_query_port, "IBVERBS_1.1");
  return fn ? fn(context, port_num, port_attr) : IbvUnavailable(-1);
}
int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
  MORI_IBV_RESOLVE(ibv_query_gid, "IBVERBS_1.1");
  return fn ? fn(context, port_num, index, gid) : IbvUnavailable(-1);
}
int _ibv_query_gid_ex(struct ibv_context* context, uint32_t port_num, uint32_t gid_index,
                      struct ibv_gid_entry* entry, uint32_t flags, size_t entry_size) {
  MORI_IBV_RESOLVE(_ibv_query_gid_ex, "IBVERBS_1.11");
  return fn ? fn(context, port_num, gid_index, entry, flags, entry_size) : IbvUnavailable(-1);
}

// ---- misc -------------------------------------------------------------------
const char* ibv_wc_status_str(enum ibv_wc_status status) {
  MORI_IBV_RESOLVE(ibv_wc_status_str, "IBVERBS_1.1");
  return fn ? fn(status) : IbvUnavailable<const char*>("unknown (libibverbs unavailable)");
}

}  // extern "C"
