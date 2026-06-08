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
// CCO Device API — single include for device-side (kernel) code.
//
// Include this one header from device/kernel sources; host control-plane code
// includes cco_api.hpp instead. This header aggregates every device-side
// facility: the common window helpers defined below, cooperative groups,
// teams, and the per-backend session classes (LSA, GDA).
#pragma once

#include "mori/cco/cco_types.hpp"

// Cooperative groups + teams used across all device sessions.
#include "mori/cco/cco_coop.hpp"
#include "mori/cco/cco_team.hpp"

// clang-format off
#include "mori/cco/cco_lsa_types.hpp"
#include "mori/cco/cco_lsa_impl.hpp"
// clang-format off

#include "mori/cco/gda/gda_device.hpp"

namespace mori {
namespace cco {

// Look up a registered window by a local pointer that lies within it.
__device__ inline ccoWindow_t findWindow(ccoDevComm* comm, const void* ptr) {
  uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
  ccoWindowTableNode* node = comm->windowTable;
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

// Flat-VA helpers — intra-node addressing only. The flat VA covers the LSA
// team, so peer indexing is by LSA rank. Cross-node access goes through the
// GDA backend with iova=0 + offset and doesn't need these.
__device__ inline void* getLsaPeerPtr(ccoWindow_t win, int peerLsaRank, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(peerLsaRank) * win->stride4G) << 32) + offset;
}

__device__ inline void* getLocalPtr(ccoWindow_t win, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(win->lsaRank) * win->stride4G) << 32) + offset;
}

}  // namespace cco
}  // namespace mori
