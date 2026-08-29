// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

#ifdef __HIPCC__

namespace mori {
namespace core {

// Returns sequence number (monotonically increasing). Mask with PROXY_RING_MASK for slot index.
inline __device__ uint32_t ProxyReserveSlot(volatile ProxyRing* ring) {
  return __hip_atomic_fetch_add(
      (uint32_t*)&ring->gpu_head, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

inline __device__ void ProxyWaitSlotFree(volatile ProxyRing* ring, uint32_t slot) {
  int spins = 0;
  while (true) {
    uint32_t st = __hip_atomic_load(
        (uint32_t*)&ring->cmds[slot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (st == PROXY_FREE || st == PROXY_COMPLETED) break;
    if (++spins % 100000 == 0) __builtin_amdgcn_s_sleep(1);
  }
}

inline __device__ void ProxyWaitSlotCompleted(volatile ProxyRing* ring, uint32_t slot) {
  while (true) {
    uint32_t st = __hip_atomic_load(
        (uint32_t*)&ring->cmds[slot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (st == PROXY_COMPLETED || st == PROXY_ERROR) break;
    __builtin_amdgcn_s_sleep(1);
  }
}

inline __device__ uint32_t ProxyPostWrite(
    volatile ProxyRing* ring, uint32_t qp_idx,
    uint64_t src_addr, uint32_t lkey,
    uint64_t dst_addr, uint32_t rkey,
    uint32_t length) {
  uint32_t seq = ProxyReserveSlot(ring);
  uint32_t slot = seq & PROXY_RING_MASK;
  ProxyWaitSlotFree(ring, slot);

  ring->cmds[slot].op = PROXY_RDMA_WRITE;
  ring->cmds[slot].qp_idx = qp_idx;
  ring->cmds[slot].src_addr = src_addr;
  ring->cmds[slot].dst_addr = dst_addr;
  ring->cmds[slot].length = length;
  ring->cmds[slot].lkey = lkey;
  ring->cmds[slot].rkey = rkey;
  ring->cmds[slot].flags = 1;

  __threadfence_system();
  ring->cmds[slot].status = PROXY_PENDING;
  return seq;
}

inline __device__ uint32_t ProxyPostWriteInline(
    volatile ProxyRing* ring, uint32_t qp_idx,
    const void* src, uint32_t lkey,
    uint64_t dst_addr, uint32_t rkey,
    uint32_t length) {
  uint32_t seq = ProxyReserveSlot(ring);
  uint32_t slot = seq & PROXY_RING_MASK;
  ProxyWaitSlotFree(ring, slot);

  ring->cmds[slot].op = PROXY_RDMA_WRITE_INLINE;
  ring->cmds[slot].qp_idx = qp_idx;
  ring->cmds[slot].dst_addr = dst_addr;
  ring->cmds[slot].length = length;
  ring->cmds[slot].lkey = lkey;
  ring->cmds[slot].rkey = rkey;
  ring->cmds[slot].flags = 1;

  if (src != nullptr && length > 0 && length <= PROXY_MAX_INLINE_DATA) {
    const uint64_t* s64 = reinterpret_cast<const uint64_t*>(src);
    volatile uint64_t* d64 = reinterpret_cast<volatile uint64_t*>(ring->cmds[slot].inline_data);
    for (uint32_t i = 0; i < (length + 7) / 8; i++) d64[i] = s64[i];
    ring->cmds[slot].inline_tag = PROXY_INLINE_SCALAR_WRITE;
    ring->cmds[slot].inline_len = length;
  } else {
    ring->cmds[slot].inline_tag = PROXY_INLINE_NONE;
    ring->cmds[slot].inline_len = 0;
  }

  __threadfence_system();
  ring->cmds[slot].status = PROXY_PENDING;
  return seq;
}

inline __device__ uint32_t ProxyPostAtomicNonFetch(
    volatile ProxyRing* ring, uint32_t qp_idx,
    uint64_t dst_addr, uint32_t rkey,
    uint64_t add_value, uint32_t lkey,
    uint64_t ibuf_addr) {
  uint32_t seq = ProxyReserveSlot(ring);
  uint32_t slot = seq & PROXY_RING_MASK;
  ProxyWaitSlotFree(ring, slot);

  ring->cmds[slot].op = PROXY_ATOMIC_FETCH_ADD;
  ring->cmds[slot].qp_idx = qp_idx;
  ring->cmds[slot].src_addr = ibuf_addr;
  ring->cmds[slot].dst_addr = dst_addr;
  ring->cmds[slot].length = 8;
  ring->cmds[slot].lkey = lkey;
  ring->cmds[slot].rkey = rkey;
  ring->cmds[slot].atomic_arg = add_value;
  ring->cmds[slot].flags = PROXY_FLAGS_DEFAULT;

  __threadfence_system();
  ring->cmds[slot].status = PROXY_PENDING;
  return seq;
}

// Signal write: RDMA_WRITE of value to remote addr on the SAME NIC path
// as the preceding data write. Used for signals paired with data
// (ShmemPutMemNbiSignalThread) to ensure PCIe write ordering.
inline __device__ uint32_t ProxyPostSignalWrite(
    volatile ProxyRing* ring, uint32_t qp_idx,
    uint64_t dst_addr, uint32_t rkey,
    uint64_t value, uint32_t lkey,
    uint64_t ibuf_addr) {
  uint32_t seq = ProxyReserveSlot(ring);
  uint32_t slot = seq & PROXY_RING_MASK;
  ProxyWaitSlotFree(ring, slot);

  ring->cmds[slot].op = PROXY_SIGNAL_WRITE;
  ring->cmds[slot].qp_idx = qp_idx;
  ring->cmds[slot].src_addr = ibuf_addr;
  ring->cmds[slot].dst_addr = dst_addr;
  ring->cmds[slot].length = 8;
  ring->cmds[slot].lkey = lkey;
  ring->cmds[slot].rkey = rkey;
  ring->cmds[slot].atomic_arg = value;
  ring->cmds[slot].flags = 1;

  __threadfence_system();
  ring->cmds[slot].status = PROXY_PENDING;
  return seq;
}

inline __device__ uint64_t ProxyPostAtomicFetch(
    volatile ProxyRing* ring, uint32_t qp_idx,
    uint64_t dst_addr, uint32_t rkey,
    uint64_t add_value, uint32_t lkey,
    uint64_t ibuf_addr) {
  uint32_t seq = ProxyReserveSlot(ring);
  uint32_t slot = seq & PROXY_RING_MASK;
  ProxyWaitSlotFree(ring, slot);

  ring->cmds[slot].op = PROXY_ATOMIC_FETCH_ADD;
  ring->cmds[slot].qp_idx = qp_idx;
  ring->cmds[slot].src_addr = ibuf_addr;
  ring->cmds[slot].dst_addr = dst_addr;
  ring->cmds[slot].length = 8;
  ring->cmds[slot].lkey = lkey;
  ring->cmds[slot].rkey = rkey;
  ring->cmds[slot].atomic_arg = add_value;
  ring->cmds[slot].flags = PROXY_FLAGS_FETCH_REQUIRED;
  // Store slot index in inline_data so the remote can send it back in the reply
  *reinterpret_cast<volatile uint64_t*>(&ring->cmds[slot].inline_data[0]) = static_cast<uint64_t>(slot);
  ring->cmds[slot].result = 0;

  __threadfence_system();
  ring->cmds[slot].status = PROXY_PENDING;

  ProxyWaitSlotCompleted(ring, slot);
  return ring->cmds[slot].result;
}

// Wait for all ops from [first_seq, first_seq + count) to complete.
// When count > PROXY_RING_SIZE, slots were reused during submission.
// ProxyWaitSlotFree already ensured earlier slots completed before reuse,
// so we only need to wait for the tail — the last PROXY_RING_SIZE slots.
inline __device__ void ProxyQuiet(volatile ProxyRing* ring, uint32_t first_seq, uint32_t count) {
  if (count == 0) return;
  uint32_t start = first_seq;
  if (count > PROXY_RING_SIZE) {
    start = first_seq + count - PROXY_RING_SIZE;
  }
  uint32_t end = first_seq + count;
  for (uint32_t seq = start; seq < end; seq++) {
    uint32_t slot = seq & PROXY_RING_MASK;
    ProxyWaitSlotCompleted(ring, slot);
  }
}

// Range variant for multi-warp callers that know the exact range.
inline __device__ void ProxyQuietRange(volatile ProxyRing* ring, uint32_t from, uint32_t to) {
  ProxyQuiet(ring, from, to - from);
}

}  // namespace core
}  // namespace mori

#endif  // __HIPCC__
