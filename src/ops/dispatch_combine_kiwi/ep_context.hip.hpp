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
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "all2all.hip.hpp"
#include "barrier.hpp"
#include "comm/common_descriptors.hip.hpp"
#include "comm/context.hip.hpp"
#include "ep_config.hpp"
#include "ep_gcontext.hip.hpp"
#include "memory_scope.hip.hpp"
#include "mori/application/bootstrap/base_bootstrap.hpp"
#include "utils.hpp"
#include "workspace/device_workspace.hpp"

namespace mori::kiwi::ep {

// MORI-style helper for mapping a global set of wavefronts onto a
// (num_items x dim_size) loop.  When the launch has more wavefronts than
// items, several wavefronts collaborate on one item by taking different
// in-item parts; when items dominate, each item gets one wavefront.
struct ep_multi_warp_iter_t {
  int warps_per_item = 1;
  size_t dim_per_warp = 0;
  size_t dim_size = 0;

  __host__ __device__ ep_multi_warp_iter_t() = default;

  __host__ __device__ ep_multi_warp_iter_t(int global_warp_num, int num_items, size_t dim_size_)
      : dim_size(dim_size_) {
    if (global_warp_num <= 0 || num_items <= 0) {
      warps_per_item = 1;
      dim_per_warp = dim_size;
      return;
    }
    warps_per_item = (global_warp_num + num_items - 1) / num_items;
    if (warps_per_item < 1) {
      warps_per_item = 1;
    }
    dim_per_warp =
        (dim_size + static_cast<size_t>(warps_per_item) - 1) / static_cast<size_t>(warps_per_item);
  }

  __host__ __device__ int total_parts(int num_items) const { return num_items * warps_per_item; }

  __host__ __device__ void decode(int i, int& item_id, int& in_item_part_id, size_t& dim_offset,
                                  size_t& dim_chunk) const {
    item_id = i / warps_per_item;
    in_item_part_id = i - item_id * warps_per_item;
    dim_offset = static_cast<size_t>(in_item_part_id) * dim_per_warp;
    dim_chunk =
        (dim_offset < dim_size)
            ? ((dim_size - dim_offset) < dim_per_warp ? (dim_size - dim_offset) : dim_per_warp)
            : size_t{0};
  }
};

// Cooperatively copy bytes.  Prefer the widest aligned access available, then
// finish the remaining tail with decreasing 8B/4B/2B/1B widths.  Callers that
// immediately publish the destination to an external consumer such as the NIC
// can select system-scope stores and then issue an explicit waitcnt before
// enqueueing that consumer.
template <bool SystemScopeStore = false, bool SystemScopeLoad = false>
__device__ inline void copy_bytes_block(const void* src, void* dst, int num_bytes, int thread_idx,
                                        int thread_count) {
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  int copied = 0;

  if (((src_addr | dst_addr) & 0xf) == 0) {
    const int n_vec4 = num_bytes / static_cast<int>(sizeof(uint4));
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int i = thread_idx; i < n_vec4; i += thread_count) {
      ep_store<SystemScopeStore>(&dst4[i], ep_load<SystemScopeLoad>(&src4[i]));
    }
    copied = n_vec4 * static_cast<int>(sizeof(uint4));
  }

  if ((((src_addr + copied) | (dst_addr + copied)) & 0x7) == 0) {
    const int n64 = (num_bytes - copied) / static_cast<int>(sizeof(uint64_t));
    const uint64_t* src64 =
        reinterpret_cast<const uint64_t*>(static_cast<const char*>(src) + copied);
    uint64_t* dst64 = reinterpret_cast<uint64_t*>(static_cast<char*>(dst) + copied);
    for (int i = thread_idx; i < n64; i += thread_count) {
      ep_store<SystemScopeStore>(&dst64[i], ep_load<SystemScopeLoad>(&src64[i]));
    }
    copied += n64 * static_cast<int>(sizeof(uint64_t));
  }

  if ((((src_addr + copied) | (dst_addr + copied)) & 0x3) == 0) {
    const int n32 = (num_bytes - copied) / static_cast<int>(sizeof(uint32_t));
    const uint32_t* src32 =
        reinterpret_cast<const uint32_t*>(static_cast<const char*>(src) + copied);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(static_cast<char*>(dst) + copied);
    for (int i = thread_idx; i < n32; i += thread_count) {
      ep_store<SystemScopeStore>(&dst32[i], ep_load<SystemScopeLoad>(&src32[i]));
    }
    copied += n32 * static_cast<int>(sizeof(uint32_t));
  }

  if ((((src_addr + copied) | (dst_addr + copied)) & 0x1) == 0) {
    const int n16 = (num_bytes - copied) / static_cast<int>(sizeof(uint16_t));
    const uint16_t* src16 =
        reinterpret_cast<const uint16_t*>(static_cast<const char*>(src) + copied);
    uint16_t* dst16 = reinterpret_cast<uint16_t*>(static_cast<char*>(dst) + copied);
    for (int i = thread_idx; i < n16; i += thread_count) {
      ep_store<SystemScopeStore>(&dst16[i], ep_load<SystemScopeLoad>(&src16[i]));
    }
    copied += n16 * static_cast<int>(sizeof(uint16_t));
  }

  const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src);
  uint8_t* dst8 = reinterpret_cast<uint8_t*>(dst);
  for (int b = copied + thread_idx; b < num_bytes; b += thread_count) {
    ep_store<SystemScopeStore>(&dst8[b], ep_load<SystemScopeLoad>(&src8[b]));
  }
}

template <bool SystemScopeStore = false, bool SystemScopeLoad = false>
__device__ inline void copy_hidden_payload_block(const void* src, void* dst, int hidden_bytes,
                                                 int thread_idx, int thread_count) {
  copy_bytes_block<SystemScopeStore, SystemScopeLoad>(src, dst, hidden_bytes, thread_idx,
                                                      thread_count);
}

// Cooperatively copy the non-hidden slot metadata that travels with an expert
// row.  Keep the hidden-payload helper separate because hidden rows are large
// and vectorized, while metadata is small but should not serialize the whole
// block/wave on lane 0 for every staged slot.
template <bool SystemScopeStore = false, bool SystemScopeLoad = false>
__device__ inline void copy_slot_metadata_block(const int32_t* src_indices,
                                                const float* src_weights, const void* src_scales,
                                                int32_t src_id, int topk, int scale_bytes,
                                                int32_t* dst_indices, float* dst_weights,
                                                int32_t* dst_src_id, void* dst_scales,
                                                int thread_idx, int thread_count) {
  for (int k = thread_idx; k < topk; k += thread_count) {
    ep_store<SystemScopeStore>(&dst_indices[k], ep_load<SystemScopeLoad>(&src_indices[k]));
    ep_store<SystemScopeStore>(&dst_weights[k], ep_load<SystemScopeLoad>(&src_weights[k]));
  }

  if (thread_idx == 0) {
    ep_store<SystemScopeStore>(dst_src_id, src_id);
  }

  if (src_scales != nullptr && dst_scales != nullptr && scale_bytes > 0) {
    copy_bytes_block<SystemScopeStore, SystemScopeLoad>(src_scales, dst_scales, scale_bytes,
                                                        thread_idx, thread_count);
  }
}

// Payload readiness flags are stored in the symmetric heap. Use explicit
// system-scope accesses for both raw-RDMA and HIP-IPC publication, polling,
// and resets.
__device__ inline void payload_readiness_flag_publish(uint8_t* flag) {
  ep_store_system_scope(flag, static_cast<uint8_t>(1));
}

__device__ inline void payload_readiness_flag_wait(const uint8_t* flag) {
  while (ep_load_system_scope<uint8_t>(flag) == 0) {
    __builtin_amdgcn_s_sleep(1);
  }
}

__device__ inline void payload_readiness_flag_reset(uint8_t* flag) {
  ep_store_system_scope(flag, static_cast<uint8_t>(0));
}

}  // namespace mori::kiwi::ep

#include "comm/transport.hip.hpp"

namespace mori::kiwi::ep {

__device__ inline int32_t device_atomic_load_acquire(const int32_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
  return *ptr;
#endif
}

__device__ inline uint64_t device_atomic_load_acquire(const uint64_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#else
  return *ptr;
#endif
}

struct grid_barrier_ticket_t {
  // Target generation to wait for.  A ticket is satisfied once the barrier
  // reaches this phase or any later phase.
  uint32_t phase = 0;
};

__host__ __device__ inline uint32_t grid_barrier_phase(uint64_t word) {
  return static_cast<uint32_t>(word >> 32);
}

__host__ __device__ inline bool grid_barrier_phase_reached(uint32_t observed, uint32_t target) {
  return static_cast<int32_t>(observed - target) >= 0;
}

__host__ __device__ inline grid_barrier_ticket_t grid_barrier_advance_ticket(
    grid_barrier_ticket_t ticket, uint32_t generations) {
  return grid_barrier_ticket_t{static_cast<uint32_t>(ticket.phase + generations)};
}

// Reusable grid barrier encoded in one 64-bit word:
//   low 32 bits  = arrivals in the current generation
//   high 32 bits = generation/phase
// The block's thread 0 contributes the block arrival only after a block-wide
// synchronization point.  The release update to the barrier word publishes the
// block's pre-barrier work to acquire waiters.  The last expected participant
// publishes the next generation with the arrival count reset to zero.  The
// returned ticket names that next generation, so waiting on a future ticket
// cannot be satisfied by an older phase.
__device__ inline grid_barrier_ticket_t grid_barrier_arrive(uint64_t* grid_barrier,
                                                            uint32_t expected_participants) {
  __syncthreads();

  uint64_t old = 0;
  if (threadIdx.x == 0) {
    old = __atomic_fetch_add(grid_barrier, 1ULL, __ATOMIC_RELEASE);
    const uint32_t phase = grid_barrier_phase(old);
    const uint32_t arrival_count = static_cast<uint32_t>(old & 0xffffffffu) + 1u;
    if (arrival_count == expected_participants) {
      const uint64_t next = static_cast<uint64_t>(phase + 1u) << 32;
      __atomic_store_n(grid_barrier, next, __ATOMIC_RELEASE);
    }
  }

  grid_barrier_ticket_t ticket;
  ticket.phase = static_cast<uint32_t>(grid_barrier_phase(old) + 1u);
  return ticket;
}

__device__ inline void grid_barrier_wait(const uint64_t* grid_barrier,
                                         grid_barrier_ticket_t ticket) {
  if (threadIdx.x == 0) {
    uint32_t observed_phase;
    do {
      observed_phase = grid_barrier_phase(device_atomic_load_acquire(grid_barrier));
      if (!grid_barrier_phase_reached(observed_phase, ticket.phase)) {
        __builtin_amdgcn_s_sleep(1);
      }
    } while (!grid_barrier_phase_reached(observed_phase, ticket.phase));
  }
  __syncthreads();
}

__device__ inline grid_barrier_ticket_t grid_barrier_arrive_and_wait(
    uint64_t* grid_barrier, uint32_t expected_participants) {
  grid_barrier_ticket_t ticket = grid_barrier_arrive(grid_barrier, expected_participants);
  grid_barrier_wait(grid_barrier, ticket);
  return ticket;
}

__device__ inline void flat_to_rank_slot_direct(const ep_gcontext_t& gctx, int flat_idx,
                                                int& out_src_rank, int& out_local_slot) {
  out_src_rank = gctx.device_workspace.recv_src_rank_map()[flat_idx];
  out_local_slot = gctx.device_workspace.recv_local_slot_map()[flat_idx];
}

// Endpoint 0 is always in the active prefix. Invoke this callback with every
// lane in block 0 / wave 0 before that wave uses endpoint 0 for EP traffic.
__device__ inline void publish_active_queue_count(const ep_gcontext_t& gctx,
                                                  ep_comm_handle_t& endpoint_zero_handle) {
  internal::ep_cmd_desc_active_queues_t desc;
  desc.active_count = static_cast<int>(gridDim.x) * EP_WAVES_PER_BLOCK;
  endpoint_zero_handle.post_desc(gctx.active_queue_count_cb_id, desc);
  __builtin_amdgcn_wave_barrier();
}

// Host-side context managing all MoE dispatch/combine resources.
class ep_context_t {
 public:
  // Construction is collective over `bootstrap`; config.world_size must
  // equal bootstrap->GetWorldSize().
  ep_context_t(const config_t& cfg, std::shared_ptr<application::BootstrapNetwork> bootstrap);
  ~ep_context_t();

  ep_context_t(const ep_context_t&) = delete;
  ep_context_t& operator=(const ep_context_t&) = delete;

  // Get GPU context for kernel launches
  ep_gcontext_t get_device_context();

  // Lifecycle
  void spawn_proxy_threads(int num_threads = -1);
  void join_proxy_threads();
  void all2all_callback(internal::ep_cmd_desc_all2all_t desc);

  // Reset signal/count areas between rounds
  void reset(hipStream_t stream = 0);

  // Access
  const config_t& config() const { return cfg_; }
  int rank_me() const { return comm_ctx_->rank_me(); }
  application::BootstrapNetwork& bootstrap() const { return comm_ctx_->bootstrap(); }
  const internal::ep_heap_table_t& heap_table() const { return comm_ctx_->heap_table(); }
  // Route-capability hook for native tests and diagnostics. A one denotes a
  // provisioned HIP-IPC-capable payload route for that peer; zero means no
  // IPC route was provisioned.
  const std::vector<uint8_t>& rcomp_peer_is_ipc() const { return rcomp_peer_is_ipc_; }

  // Exact-hostname locality used by hybrid setup.  Strict setup guarantees
  // every local non-self peer also has a corresponding IPC route.
  const std::vector<uint8_t>& rcomp_peer_is_local() const { return rcomp_peer_is_local_; }

  ep_device_workspace_t device_workspace() const {
    return ep_device_workspace_t{d_device_workspace_, device_workspace_layout_};
  }

  // Synchronizes and issues an explicit D2H copy, so production
  // dispatch/combine paths must not call it.
  int32_t copy_recv_count_to_host(hipStream_t stream = 0) const;

  // Host-side stream synchronization wrapper.
  void stream_sync(hipStream_t stream);

 private:
  config_t cfg_;
  ep_device_workspace_layout_t device_workspace_layout_;
  std::unique_ptr<ep_comm_context_t> comm_ctx_;
  std::unique_ptr<all2all_t> all2all_;
  // HIP-IPC payload resources are provisioned independently from the
  // backend selected for any particular launch.
  bool rcomp_ipc_resources_provisioned_ = false;
  std::vector<uint8_t> rcomp_peer_is_local_;
  std::vector<uint8_t> rcomp_peer_is_ipc_;
  std::vector<std::string> rcomp_peer_hostnames_;
  // Per-rank physical GPU PCI Bus IDs are exchanged between processes.
  // rcomp_peer_hip_devices_ stores the matching ordinal resolved inside this
  // process, so it is valid for HIP capability queries on this rank.
  std::vector<std::string> rcomp_peer_hip_pci_bus_ids_;
  std::vector<int> rcomp_peer_hip_devices_;
  std::vector<uintptr_t> rcomp_peer_heap_bases_;
  std::vector<void*> rcomp_peer_heap_ipc_mappings_;
  uintptr_t* d_rcomp_peer_heap_bases_ = nullptr;
  uint8_t* d_rcomp_peer_is_ipc_ = nullptr;

  // Symmetric heap buffer (device memory, RDMA-registered)
  char* d_heap_buf_ = nullptr;
  int heap_id_ = -1;
  // Regular device workspace for hot scratch/maps and collective control.
  // Payload readiness remains in the RDMA-visible symmetric heap.
  char* d_device_workspace_ = nullptr;

  // EP-owned invoke callback ids for launch-driven proxy polling, hybrid
  // count all-to-all, and raw payload puts.
  uint8_t active_queue_count_cb_id_ = 0;
  uint8_t ep_all2all_cb_id_ = 0;
  uint8_t post_put_cb_id_ = 0;
  uint8_t post_put_multicast_cb_id_ = 0;
  // Output tensors for recv data (device memory)
  void* d_recv_hidden_ = nullptr;
  int32_t* d_recv_indices_ = nullptr;
  float* d_recv_weights_ = nullptr;
  void* d_recv_scales_ = nullptr;
  int32_t* d_recv_src_token_pos_ = nullptr;
};

// ----- Dispatch -----

// Common asynchronous dispatch path. It enqueues one fused dispatch kernel that
// packs/sends, publishes and polls receive counts, builds direct receive
// maps/ranges, and unpacks received slots. It returns before the stream work
// completes so callers can capture it or synchronize explicitly for eager
// execution.
void dispatch_enqueue(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_tokens,
                      const int32_t* d_expert_indices, const float* d_weights, const void* d_scales,
                      int num_tokens, hipStream_t stream = 0);

// ----- Combine -----

// Combine sends expert-processed tokens back to source ranks and accumulates
// MORI-compatible valid expert rows into d_output[num_tokens][combine_hidden_elems].
//
// d_expert_output: result of expert computation on received tokens
// d_expert_scales: accepted for MORI API compatibility; combine currently
//                  returns no scale output and does not consume this tensor
// d_topk_ids: original topk expert IDs (same as dispatch input)
// d_output: accumulated output buffer
// num_tokens: number of original tokens on this rank (same as dispatch input)

void combine(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_expert_output,
             const float* d_expert_weights, const void* d_expert_scales, const int32_t* d_topk_ids,
             void* d_output, float* d_output_weights, int num_tokens, hipStream_t stream = 0);

// Enqueue the common async combine path without synchronizing. It keeps actual
// puts offloaded through the always-running host proxy threads, while waits and
// resets visible to eager execution or graph replay are stream-enqueued GPU work.
void combine_enqueue(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_expert_output,
                     const float* d_expert_weights, const void* d_expert_scales,
                     const int32_t* d_topk_ids, void* d_output, float* d_output_weights,
                     int num_tokens, hipStream_t stream = 0);

inline void combine(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_expert_output,
                    const void* d_expert_scales, const int32_t* d_topk_ids, void* d_output,
                    int num_tokens, hipStream_t stream = 0) {
  combine(ctx, gctx, d_expert_output, nullptr, d_expert_scales, d_topk_ids, d_output, nullptr,
          num_tokens, stream);
}

// ----- Convenience -----

// Dispatch using the common enqueue path, then wait for receives and unpack.
// After return, gctx.d_recv_hidden/indices/weights/scales/count are populated.
inline void dispatch(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_tokens,
                     const int32_t* d_expert_indices, const float* d_weights, const void* d_scales,
                     int num_tokens, hipStream_t stream = 0) {
  dispatch_enqueue(ctx, gctx, d_tokens, d_expert_indices, d_weights, d_scales, num_tokens, stream);
  ctx.stream_sync(stream);
}

}  // namespace mori::kiwi::ep
