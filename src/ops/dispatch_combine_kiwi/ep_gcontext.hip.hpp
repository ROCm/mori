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

#include <cstddef>
#include <cstdint>

#include "assert.hpp"
#include "comm/device_context.hip.hpp"
#include "ep_config.hpp"
#include "workspace/device_workspace.hpp"

namespace mori::kiwi::ep {

class ep_context_t;

// GPU-side context for ep operations. Passed by value to kernels.
// Owns the EP comm device context plus ep-specific buffer pointers and layout info.
struct ep_gcontext_t {
  // Host ep_context_t pointer copied into invoke descriptors.  Kernels only
  // pass this opaque value back to host callbacks; they never dereference it.
  ep_context_t* host_context;

  ep_comm_device_context_t comm;

  // Heap buffer base pointer (device-accessible)
  char* heap_base;
  int heap_id;

  // Layout constants (copied from config for device access)
  int world_size;
  int rank_me;
  int num_experts_per_rank;
  int model_hidden_dim;  // one shared logical model feature dimension
  int dispatch_hidden_elems;
  int combine_hidden_elems;
  int topk;
  int max_slots;  // total MORI-visible receive rows
  // MORI-compatible source-token capacity; per-source receive slot count.
  int max_num_inp_token_per_rank;
  int dispatch_scale_dim;
  int dispatch_scale_type_size;
  int combine_scale_dim;
  int combine_scale_type_size;
  quant_format_t dispatch_quant_format;
  quant_format_t combine_quant_format;
  // Wire-format + heap-region layouts.  These mirror DeepEP's separation of
  // per-token slot field layout from buffer/heap placement: dispatch and
  // combine keep independent instances for their different element sizes and
  // heap regions while sharing the same accessors below.
  slot_buffer_layout_t dispatch_layout;
  slot_buffer_layout_t combine_layout;

  // Output tensors (device memory, NOT in RDMA heap)
  // Populated by the dispatch receive/unpack kernels after dispatch completes.
  void* d_recv_hidden;            // [max_slots][dispatch_hidden_elems] in dispatch_elem_size
  int32_t* d_recv_indices;        // [max_slots][topk]
  float* d_recv_weights;          // [max_slots][topk]
  void* d_recv_scales;            // [max_slots][dispatch_scale_dim] in dispatch_scale_type_size
  int32_t* d_recv_src_token_pos;  // [max_slots] MORI-encoded as src_rank * (world_size *
                                  // max_num_inp_token_per_rank) + src_token_id

  // Device workspace plus callback ids for device-initiated collectives.
  ep_device_workspace_t device_workspace;
  int count_alltoall_local_size;
  int count_alltoall_num_nodes;
  int count_alltoall_node_idx;
  int count_alltoall_lane_idx;
  const uintptr_t* count_alltoall_local_peer_heap_bases;
  // Persistent dispatch/combine round epoch. It lives outside resettable
  // device workspace so stale ingress cannot match after ep_context::reset().
  size_t protocol_epoch_offset;
  size_t count_alltoall_egress_offset;
  size_t count_alltoall_ingress_offset;
  size_t barrier_alltoall_egress_offset;
  size_t barrier_alltoall_ingress_offset;
  uint8_t active_queue_count_cb_id;
  uint8_t post_put_cb_id;
  uint8_t post_put_multicast_cb_id;
  // Provisioned IPC payload-routing metadata. peer_heap_bases and
  // peer_is_ipc are device allocations populated during context setup:
  // exact-hostname-local non-self peers use an imported HIP IPC heap, while
  // self remains direct and remote peers use RDMA.
  const uintptr_t* rcomp_peer_heap_bases;
  const uint8_t* rcomp_peer_is_ipc;
  size_t dispatch_readiness_flags_offset;
  size_t combine_readiness_flags_offset;
  // ----- Device helpers -----

  __host__ __device__ size_t token_route_index(int token_idx, int k) const {
    return static_cast<size_t>(token_idx) * topk + k;
  }

  __host__ __device__ int32_t* token_to_producer_rank_ptr(int token_idx, int k) const {
    return device_workspace.token_to_producer_rank() + token_route_index(token_idx, k);
  }

  __host__ __device__ int32_t* token_to_dispatch_recv_slot_ptr(int token_idx, int k) const {
    return device_workspace.token_to_dispatch_recv_slot() + token_route_index(token_idx, k);
  }

  __host__ __device__ int32_t token_to_producer_rank(int token_idx, int k) const {
    return *token_to_producer_rank_ptr(token_idx, k);
  }

  __host__ __device__ int32_t token_to_dispatch_recv_slot(int token_idx, int k) const {
    return *token_to_dispatch_recv_slot_ptr(token_idx, k);
  }

  __host__ __device__ char* dispatch_staging_slot(int slot) const {
    return dispatch_layout.staging_slot(heap_base, slot);
  }

  __host__ __device__ char* combine_staging_slot(int slot) const {
    return combine_layout.staging_slot(heap_base, slot);
  }

  __host__ __device__ char* dispatch_recv_slot(int source_rank,
                                               int destination_local_dispatch_slot) const {
    return dispatch_layout.recv_slot(heap_base, source_rank, destination_local_dispatch_slot);
  }

  // Combine receive slots are keyed by the original producer rank and source
  // token id:
  //   combine_recv_slot(producer_rank, source_token_id)
  // This is intentionally distinct from dispatch receive slots:
  //   dispatch_recv_slot(source_rank, destination_local_dispatch_slot)
  // Combine must not use dispatch_recv_slot/source-local dispatch slot
  // metadata for indexing returned expert rows.
  __host__ __device__ char* combine_recv_slot(int producer_rank, int source_token_id) const {
    return combine_layout.recv_slot(heap_base, producer_rank, source_token_id);
  }

  __device__ uint64_t* protocol_epoch() const {
    return reinterpret_cast<uint64_t*>(heap_base + protocol_epoch_offset);
  }

  __device__ uint64_t* count_alltoall_egress() const {
    return reinterpret_cast<uint64_t*>(heap_base + count_alltoall_egress_offset);
  }

  __device__ uint64_t* count_alltoall_ingress() const {
    return reinterpret_cast<uint64_t*>(heap_base + count_alltoall_ingress_offset);
  }

  __device__ uint64_t* count_alltoall_peer_ingress(int source_lane) const {
    return reinterpret_cast<uint64_t*>(count_alltoall_local_peer_heap_bases[source_lane] +
                                       count_alltoall_ingress_offset);
  }

  __host__ __device__ size_t barrier_alltoall_bank_bytes() const {
    return static_cast<size_t>(world_size) * sizeof(uint64_t);
  }

  __host__ __device__ size_t barrier_alltoall_bank_offset(uint32_t protocol_epoch) const {
    return static_cast<size_t>(protocol_epoch & 1u) * barrier_alltoall_bank_bytes();
  }

  __device__ uint64_t* barrier_alltoall_egress(uint32_t protocol_epoch) const {
    return reinterpret_cast<uint64_t*>(heap_base + barrier_alltoall_egress_offset +
                                       barrier_alltoall_bank_offset(protocol_epoch));
  }

  __device__ uint64_t* barrier_alltoall_ingress(uint32_t protocol_epoch) const {
    return reinterpret_cast<uint64_t*>(heap_base + barrier_alltoall_ingress_offset +
                                       barrier_alltoall_bank_offset(protocol_epoch));
  }

  __host__ __device__ size_t barrier_alltoall_ingress_epoch_offset(uint32_t protocol_epoch) const {
    return barrier_alltoall_ingress_offset + barrier_alltoall_bank_offset(protocol_epoch);
  }

  __device__ uint64_t* alltoall_peer_ingress(int source_lane, size_t ingress_offset) const {
    return reinterpret_cast<uint64_t*>(count_alltoall_local_peer_heap_bases[source_lane] +
                                       ingress_offset);
  }

  // Remote displacement for a slot in dest's dispatch recv region (from this rank)
  __host__ __device__ uintptr_t dispatch_remote_disp(int recv_slot_idx) const {
    return dispatch_layout.remote_disp(rank_me, recv_slot_idx);
  }

  __host__ __device__ uintptr_t combine_remote_disp(int recv_slot_idx) const {
    return combine_layout.remote_disp(rank_me, recv_slot_idx);
  }

  __host__ __device__ size_t payload_readiness_flag_index(int source_rank,
                                                          int source_local_slot) const {
    return kiwi::ep::payload_readiness_flag_index(source_rank, source_local_slot,
                                                  max_num_inp_token_per_rank);
  }

  __device__ bool rcomp_peer_uses_ipc(int peer_rank) const {
    return rcomp_peer_is_ipc != nullptr && peer_rank >= 0 && peer_rank < world_size &&
           rcomp_peer_is_ipc[peer_rank] != 0;
  }

  __device__ char* rcomp_peer_heap_base(int peer_rank) const {
    return reinterpret_cast<char*>(rcomp_peer_heap_bases[peer_rank]);
  }

  __device__ char* rcomp_peer_dispatch_recv_slot(int peer_rank, int recv_slot_idx) const {
    return rcomp_peer_heap_base(peer_rank) + dispatch_remote_disp(recv_slot_idx);
  }

  __device__ char* rcomp_peer_combine_recv_slot(int peer_rank, int recv_slot_idx) const {
    return rcomp_peer_heap_base(peer_rank) + combine_remote_disp(recv_slot_idx);
  }

  __device__ uint8_t* dispatch_readiness_flags() const {
    return reinterpret_cast<uint8_t*>(heap_base + dispatch_readiness_flags_offset);
  }

  __device__ uint8_t* combine_readiness_flags() const {
    return reinterpret_cast<uint8_t*>(heap_base + combine_readiness_flags_offset);
  }

  __device__ uint8_t* peer_dispatch_readiness_flags(int peer_rank) const {
    return reinterpret_cast<uint8_t*>(rcomp_peer_heap_base(peer_rank) +
                                      dispatch_readiness_flags_offset);
  }

  __device__ uint8_t* peer_combine_readiness_flags(int peer_rank) const {
    return reinterpret_cast<uint8_t*>(rcomp_peer_heap_base(peer_rank) +
                                      combine_readiness_flags_offset);
  }
};

// Begin one dispatch/combine protocol round. Strict rank alternation keeps
// independently incremented per-rank values equal across the collective.
__device__ inline uint32_t ep_protocol_epoch_begin(const ep_gcontext_t& gctx) {
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  uint32_t epoch = 0;
  if (lane == 0) {
    const uint64_t current_epoch = __atomic_load_n(gctx.protocol_epoch(), __ATOMIC_ACQUIRE);
    const uint64_t completed_barrier_epoch =
        __atomic_load_n(gctx.device_workspace.barrier_counter(), __ATOMIC_ACQUIRE);
    KIWI_EP_ASSERT(current_epoch == completed_barrier_epoch,
                   "protocol epoch and barrier completion counter diverged");
    const uint64_t next_epoch = __atomic_add_fetch(gctx.protocol_epoch(), 1ULL, __ATOMIC_RELAXED);
    KIWI_EP_ASSERT(next_epoch <= UINT32_MAX, "uint32 protocol epoch space exhausted");
    epoch = static_cast<uint32_t>(next_epoch);
  }
  return __shfl(epoch, 0);
}

__device__ inline uint32_t ep_protocol_epoch_current(const ep_gcontext_t& gctx) {
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  uint32_t epoch = 0;
  if (lane == 0) {
    const uint64_t current_epoch = __atomic_load_n(gctx.protocol_epoch(), __ATOMIC_ACQUIRE);
    KIWI_EP_ASSERT(current_epoch != 0 && current_epoch <= UINT32_MAX,
                   "current protocol epoch is invalid");
    epoch = static_cast<uint32_t>(current_epoch);
  }
  return __shfl(epoch, 0);
}

}  // namespace mori::kiwi::ep
