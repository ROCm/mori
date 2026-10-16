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
// SPDX-FileCopyrightText: © Hewlett Packard Enterprise Development LP
// MIT License — see root LICENSE file.
//
// Runtime shim for the core libfabric API.
//
// mori requests the OFI backend at build time only for libfabric's headers (the
// include path). This file removes the link-time dependency on libfabric.so so
// libmori_io.so carries no DT_NEEDED on it: every libfabric symbol mori needs is
// defined here and forwarded to libfabric.so.1, which is dlopen()ed lazily on
// first use. This mirrors the dlopen shim already used for libibverbs
// (ibv_shim.cpp / mori_ibv_shim).
//
// Why only five symbols: almost every fi_* call mori makes is a static inline in
// the libfabric headers that dispatches through a fid's ops table (fi_domain,
// fi_endpoint, fi_cq_open, fi_read, fi_writemsg, fi_mr_regattr, fi_close, …) and
// needs no external symbol at all. fi_allocinfo is a static inline over
// fi_dupinfo. The only real exported symbols that remain are the ones below.
//
// All symbols here are compiled with hidden visibility (see CMake) so each mori
// shared object carries its own private copy and we never interpose a real
// libfabric that may also be present in the process.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <rdma/fabric.h>
#include <rdma/fi_errno.h>

#include <cerrno>

#include "mori/utils/mori_log.hpp"

namespace {

// Lazily dlopen libfabric once, trying the unversioned then the versioned
// soname. Returns nullptr if neither is found, in which case every shim degrades
// to a failure return so OFI discovery fails gracefully instead of crashing on a
// host without libfabric. Point LD_LIBRARY_PATH at an out-of-tree libfabric.
void* OfiHandle() {
  static void* handle = [] {
    const char* libs[] = {"libfabric.so", "libfabric.so.1"};
    for (const char* lib : libs) {
      void* h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
      if (h) {
        MORI_APP_TRACE("dlopen({}) succeeded", lib);
        return h;
      }
      MORI_APP_TRACE("dlopen({}) failed: {}", lib, dlerror());
    }
    MORI_APP_WARN("failed to dlopen libfabric (set LD_LIBRARY_PATH if it is out-of-tree)");
    return static_cast<void*>(nullptr);
  }();
  return handle;
}

void* OfiSym(const char* name) {
  void* h = OfiHandle();
  if (!h) return nullptr;
  dlerror();  // clear any stale error before the lookup (dl* API contract)
  void* sym = dlsym(h, name);
  if (!sym) {
    const char* err = dlerror();
    MORI_APP_WARN("failed to resolve {}: {}", name, err ? err : "symbol not found");
  }
  return sym;
}

template <typename T>
T OfiUnavailable(T fail) {
  errno = ENOSYS;
  return fail;
}

}  // namespace

// Resolve a symbol once per shim (function-local static init is thread-safe) and
// reuse the cached pointer. decltype(&name) yields the exact function-pointer
// type from the libfabric header, so the forwarding stays type-checked.
#define MORI_OFI_RESOLVE(name) static auto fn = reinterpret_cast<decltype(&name)>(OfiSym(#name))

extern "C" {

int fi_getinfo(uint32_t version, const char* node, const char* service, uint64_t flags,
               const struct fi_info* hints, struct fi_info** info) {
  MORI_OFI_RESOLVE(fi_getinfo);
  return fn ? fn(version, node, service, flags, hints, info) : OfiUnavailable(-FI_ENOSYS);
}

void fi_freeinfo(struct fi_info* info) {
  MORI_OFI_RESOLVE(fi_freeinfo);
  if (fn) fn(info);
}

struct fi_info* fi_dupinfo(const struct fi_info* info) {
  MORI_OFI_RESOLVE(fi_dupinfo);
  return fn ? fn(info) : OfiUnavailable<struct fi_info*>(nullptr);
}

int fi_fabric(struct fi_fabric_attr* attr, struct fid_fabric** fabric, void* context) {
  MORI_OFI_RESOLVE(fi_fabric);
  return fn ? fn(attr, fabric, context) : OfiUnavailable(-FI_ENOSYS);
}

const char* fi_strerror(int errnum) {
  MORI_OFI_RESOLVE(fi_strerror);
  return fn ? fn(errnum) : "unknown (libfabric unavailable)";
}

}  // extern "C"
