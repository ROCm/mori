// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/xshmem/xshmem_types.hpp"

namespace mori {
namespace xshmem {

// Forward-declare XshmemComm so the API header compiles under __HIPCC__.
// Full definition is host-only (guarded in xshmem_types.hpp).
#if defined(__HIPCC__) || defined(__CUDACC__)
struct XshmemComm;
#endif

// ── Phase 1: Communicator ──
int XshmemCommCreate(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                     XshmemComm** comm);
int XshmemCommDestroy(XshmemComm* comm);

// ── Phase 1.5 (optional): VMM allocation + P2P flat-space mapping ──
int XshmemMemAlloc(XshmemComm* comm, size_t size, void** ptr);
int XshmemMemFree(XshmemComm* comm, void* ptr);

// ── Phase 2: Window registration (P2P mapping + RDMA MR + SDMA signals + GPU structs) ──
// Collective: all ranks must call in the same order with the same size.
// Overload A: internal allocation (= MemAlloc + WindowRegister(ptr))
int XshmemWindowRegister(XshmemComm* comm, size_t size, XshmemWindow_t* win, void** localPtr);
// Overload B: register pre-allocated ptr from XshmemMemAlloc
int XshmemWindowRegister(XshmemComm* comm, void* ptr, size_t size, XshmemWindow_t* win);
// Teardown order: WindowDeregister → MemFree (if using separate alloc)
int XshmemWindowDeregister(XshmemComm* comm, XshmemWindow_t win);

// ── Phase 3: Device communicator ──
int XshmemDevCommCreate(XshmemComm* comm, XshmemDevComm** devComm);
int XshmemDevCommDestroy(XshmemDevComm* devComm);

// ── Host barrier ──
int XshmemBarrierAll(XshmemComm* comm);

}  // namespace xshmem
}  // namespace mori
