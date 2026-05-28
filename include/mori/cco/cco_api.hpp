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
int CcoCommCreate(application::BootstrapNetwork* bootNet, size_t perRankVmmSize, CcoComm** comm);
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
//
// Initialize `reqs` via CCO_DEV_COMM_REQUIREMENTS_INITIALIZER and override
// per-DevComm settings (gdaSignalCount, gdaConnectionType, ...) as needed.
// `reqs` must not be NULL; passing NULL or a struct without the magic/version
// triplet results in an error return (binary forward-compat check).
int CcoDevCommCreate(CcoComm* comm,
                     const CcoDevCommRequirements* reqs,
                     CcoDevComm** devComm);
int CcoDevCommDestroy(CcoComm* comm, CcoDevComm* devComm);

// ── Host barrier ──
int CcoBarrierAll(CcoComm* comm);

}  // namespace cco
}  // namespace mori
