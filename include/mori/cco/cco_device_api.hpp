// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Device API — common helpers shared by all backends.
// Per-backend session classes live under gda/, lsa/, sdma/.
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

// Look up a registered window by a local pointer that lies within it.
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

// Flat-VA helpers — intra-node addressing only. The flat VA covers the LSA
// team, so peer indexing is by LSA rank. Cross-node access goes through the
// GDA backend with iova=0 + offset and doesn't need these.
__device__ inline void* getLsaPeerPtr(CcoWindow_t win, int peerLsaRank, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(peerLsaRank) * win->stride4G) << 32) + offset;
}

__device__ inline void* getLocalPtr(CcoWindow_t win, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(win->lsaRank) * win->stride4G) << 32) + offset;
}

}  // namespace cco
}  // namespace mori
