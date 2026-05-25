// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include "mori/application/bootstrap/bootstrap.hpp"
#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

// Forward-declare CcoComm so the API header compiles under __HIPCC__.
// Full definition is host-only (guarded in cco_types.hpp).
#if defined(__HIPCC__) || defined(__CUDACC__)
struct CcoComm;
#endif

// ── Phase 1: Communicator ──
int CcoCommCreate(application::BootstrapNetwork* bootNet, size_t perRankVmmSize,
                     CcoComm** comm);
int CcoCommDestroy(CcoComm* comm);

// ── Phase 1.5 (optional): VMM allocation + P2P flat-space mapping ──
int CcoMemAlloc(CcoComm* comm, size_t size, void** ptr);
int CcoMemFree(CcoComm* comm, void* ptr);

// ── Phase 2: Window registration (P2P mapping + RDMA MR + SDMA signals + GPU structs) ──
// Collective: all ranks must call in the same order with the same size.
// Overload A: internal allocation (= MemAlloc + WindowRegister(ptr))
int CcoWindowRegister(CcoComm* comm, size_t size, CcoWindow_t* win, void** localPtr);
// Overload B: register pre-allocated ptr from CcoMemAlloc
int CcoWindowRegister(CcoComm* comm, void* ptr, size_t size, CcoWindow_t* win);
// Teardown order: WindowDeregister → MemFree (if using separate alloc)
int CcoWindowDeregister(CcoComm* comm, CcoWindow_t win);

// ── Phase 3: Device communicator ──
int CcoDevCommCreate(CcoComm* comm, CcoDevComm** devComm);
int CcoDevCommDestroy(CcoDevComm* devComm);

// ── Host barrier ──
int CcoBarrierAll(CcoComm* comm);

}  // namespace cco
}  // namespace mori
