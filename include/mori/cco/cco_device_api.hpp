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
//
// CCO Device API — common helpers shared by all backends (GDA, LSA, SDMA).
//
// Per-backend session classes live in their own headers:
//   gda/  : CcoGda             (RDMA via NIC GPU-direct)
//   lsa/  : CcoLsa             (intra-node P2P direct store, planned)
//   sdma/ : CcoSdma            (intra-node SDMA copy engine, planned)
//
// Naming convention (Plan C):
//   * Free helpers in mori::cco namespace use lowercase camelCase WITHOUT
//     the `Cco` prefix (namespace already disambiguates).
//   * Public types/handles still carry the `Cco` prefix so they read well
//     when fully qualified or imported via `using` declarations.
//
// See cco.md "Naming Convention" for full rules.
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

// ─────────────────────────────────────────────────────────────────────────────
// Window lookup (analogous to ncclFindWindow): scan the per-comm window table
// to map a local pointer back to its registered CcoWindowDevice.
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline CcoWindow_t findWindow(CcoDevComm* comm, const void* ptr) {
  uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
  CcoWindowTableNode* node = comm->windowTable;
  while (node) {
    for (int i = 0; i < CCO_WINDOW_TABLE_SIZE; i++) {
      auto& e = node->entries[i];
      if (e.base != 0 && e.size != 0 && e.window != nullptr) {
        if (uptr >= e.base && uptr < e.base + e.size) {
          return e.window;
        }
      }
    }
    node = node->next;
  }
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat-VA address helpers (NCCL-compatible layout):
//   peer  VA = winBase + ((uint64_t)pe   * stride4G << 32) + offset
//   local VA = winBase + ((uint64_t)rank * stride4G << 32) + offset
// Used by the LSA (P2P direct store) and SDMA backends to resolve a peer's
// physical address within the per-comm flat VA space. The GDA (RDMA) backend
// uses iova=0 and does NOT need these helpers.
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline void* getPeerPtr(CcoWindow_t win, int pe, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(pe) * win->stride4G) << 32) + offset;
}

__device__ inline void* getLocalPtr(CcoWindow_t win, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(win->rank) * win->stride4G) << 32) + offset;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend session classes (Phase 2 — placeholders).
//
//   CcoGda             — implemented under gda/gda_device_api.hpp
//   CcoLsa             — TODO: intra-node P2P direct store
//                          struct CcoLsa {
//                            __device__ void put(int peer, CcoWindow_t dst,
//                                                size_t dstOff, CcoWindow_t src,
//                                                size_t srcOff, size_t bytes);
//                            __device__ void putValue<T>(int peer, ..., T value);
//                          };
//   CcoSdma            — TODO: intra-node SDMA copy
//                          struct CcoSdma {
//                            __device__ void put(int peer, ...);
//                            __device__ void quiet(int peer);
//                          };
//   CcoLsaBarrierSession — TODO: collective barrier
//                          struct CcoLsaBarrierSession {
//                            __device__ void arrive(Coop, cuda::memory_order);
//                            __device__ void wait(Coop, cuda::memory_order);
//                            __device__ void sync(Coop, cuda::memory_order);
//                          };
// ─────────────────────────────────────────────────────────────────────────────

}  // namespace cco
}  // namespace mori
