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
#include <stdexcept>
#include <string>

#ifndef KIWI_EP_HOST_DEVICE
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__CUDACC__) || \
    (defined(__host__) && defined(__device__))
#define KIWI_EP_HOST_DEVICE __host__ __device__
#else
#define KIWI_EP_HOST_DEVICE
#endif
#endif

namespace mori::kiwi::ep {

inline constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Register-array helpers in dispatch/combine assume topk stays small.
// Keep one shared bound across strategies so dispatch and combine agree.
inline constexpr int MAX_TOPK_SUPPORTED = 16;
inline constexpr int EP_WAVE_SIZE = 64;
inline constexpr int EP_WAVES_PER_BLOCK = 16;
inline constexpr int EP_THREADS_PER_BLOCK = EP_WAVE_SIZE * EP_WAVES_PER_BLOCK;
inline constexpr size_t EP_SLOT_ALIGNMENT = 64;

enum class quant_format_t : int32_t {
  NONE = 0,
  FP8 = 1,
  FP4 = 2,
};

// RCOMP payload backend. RDMA_ONLY uses RDMA payloads; HYBRID writes
// exact-hostname-local peer payloads through HIP IPC and retains RDMA payloads
// plus receiver-owned readiness signals for all other peers. AUTO resolves the
// concrete backend independently for each active-token launch.
enum class rcomp_backend_t : int32_t {
  RDMA_ONLY = 0,
  HYBRID = 1,
  AUTO = 2,
};

KIWI_EP_HOST_DEVICE inline size_t payload_readiness_flag_index(int source_rank,
                                                               int source_local_slot,
                                                               int slots_per_rank) {
  return static_cast<size_t>(source_rank) * static_cast<size_t>(slots_per_rank) +
         static_cast<size_t>(source_local_slot);
}

struct runtime_defaults_t {
  int num_blocks;
  rcomp_backend_t rcomp_backend;
};

// Return the measured runtime defaults for a positive per-rank token count and
// world size. Context construction uses the maximum configured capacity to
// allocate enough queues and optional hybrid resources. Automatic dispatch
// and combine policy can then select a smaller launch from the active token
// count without changing those allocations.
//
// The common block bands combine a BF16 single-node (MI300X) one-token result,
// 1/2/4/8-node small-token sweeps, and the existing high-token bands. ws8 uses its measured backend
// threshold; every other topology uses the ws16-derived fallback threshold.
inline runtime_defaults_t recommended_runtime_defaults(int num_tokens, int world_size) {
  if (num_tokens <= 0) {
    throw std::invalid_argument(
        "kiwi::ep::recommended_runtime_defaults: "
        "num_tokens must be positive");
  }

  const int num_blocks = num_tokens == 1    ? 12
                         : num_tokens == 2  ? 20
                         : num_tokens <= 32 ? 32
                         : num_tokens <= 64 ? 64
                                            : 256;

  const int hybrid_min_tokens = world_size == 8 ? 8 : 4;
  return {
      num_blocks,
      num_tokens >= hybrid_min_tokens ? rcomp_backend_t::HYBRID : rcomp_backend_t::RDMA_ONLY,
  };
}

inline int recommended_num_blocks(int num_tokens, int world_size) {
  return recommended_runtime_defaults(num_tokens, world_size).num_blocks;
}

inline const char* rcomp_backend_name(rcomp_backend_t backend) {
  switch (backend) {
    case rcomp_backend_t::RDMA_ONLY:
      return "rdma-only";
    case rcomp_backend_t::HYBRID:
      return "hybrid";
    case rcomp_backend_t::AUTO:
      return "auto";
  }
  return "unknown";
}

// Wire-format layout for one MORI-style communication slot.  Dispatch and
// combine may carry different hidden element types (for example FP8 dispatch
// rows and BF16 combine rows), so keep the two slot layouts independent even
// when they share one symmetric heap allocation.
struct slot_layout_t {
  int elem_size = 4;
  size_t hidden_offset = 0;
  size_t indices_offset = 0;
  size_t weights_offset = 0;
  size_t scales_offset = 0;
  size_t src_id_offset = 0;
  size_t payload_size = 0;
  size_t stride = 0;

  KIWI_EP_HOST_DEVICE size_t hidden_size(int hidden_dim) const {
    return static_cast<size_t>(hidden_dim) * elem_size;
  }

  KIWI_EP_HOST_DEVICE void* hidden(char* slot_ptr) const { return slot_ptr + hidden_offset; }
  KIWI_EP_HOST_DEVICE const void* hidden(const char* slot_ptr) const {
    return slot_ptr + hidden_offset;
  }
  KIWI_EP_HOST_DEVICE int32_t* indices(char* slot_ptr) const {
    return reinterpret_cast<int32_t*>(slot_ptr + indices_offset);
  }
  KIWI_EP_HOST_DEVICE const int32_t* indices(const char* slot_ptr) const {
    return reinterpret_cast<const int32_t*>(slot_ptr + indices_offset);
  }
  KIWI_EP_HOST_DEVICE float* weights(char* slot_ptr) const {
    return reinterpret_cast<float*>(slot_ptr + weights_offset);
  }
  KIWI_EP_HOST_DEVICE const float* weights(const char* slot_ptr) const {
    return reinterpret_cast<const float*>(slot_ptr + weights_offset);
  }
  KIWI_EP_HOST_DEVICE void* scales(char* slot_ptr) const { return slot_ptr + scales_offset; }
  KIWI_EP_HOST_DEVICE const void* scales(const char* slot_ptr) const {
    return slot_ptr + scales_offset;
  }
  KIWI_EP_HOST_DEVICE int32_t* src_id(char* slot_ptr) const {
    return reinterpret_cast<int32_t*>(slot_ptr + src_id_offset);
  }
  KIWI_EP_HOST_DEVICE const int32_t* src_id(const char* slot_ptr) const {
    return reinterpret_cast<const int32_t*>(slot_ptr + src_id_offset);
  }
};

// Layout for one wire format's heap regions.  Dispatch and combine own
// separate instances because their slot element sizes and heap regions can
// differ, but their address arithmetic stays shared.
struct slot_buffer_layout_t {
  slot_layout_t slot;
  size_t staging_offset = 0;
  size_t recv_offset = 0;
  size_t recv_region_size = 0;

  KIWI_EP_HOST_DEVICE char* staging_slot(char* heap_base, int slot_idx) const {
    return heap_base + staging_offset + slot_idx * slot.stride;
  }

  KIWI_EP_HOST_DEVICE char* recv_slot(char* heap_base, int src_rank, int slot_idx) const {
    return heap_base + recv_offset + src_rank * recv_region_size + slot_idx * slot.stride;
  }

  KIWI_EP_HOST_DEVICE uintptr_t remote_disp(int rank_me, int recv_slot_idx) const {
    return recv_offset + static_cast<size_t>(rank_me) * recv_region_size +
           recv_slot_idx * slot.stride;
  }
};

// MoE configuration
struct config_t {
  int world_size = 2;
  int num_experts = 8;  // total experts across all ranks
  // One logical model dimension and two physical hidden-element counts.
  // FP4 dispatch stores two logical values per byte, so dispatch_hidden_elems can be
  // model_hidden_dim/2 while combine_hidden_elems remains the downstream activation
  // hidden-element count.  These are configured shapes; runtime tensors are expected
  // to match the configured hidden elements, not a smaller/larger capacity model.
  int model_hidden_dim = 128;
  int dispatch_hidden_elems = 0;  // <=0 means model_hidden_dim
  int combine_hidden_elems = 0;   // <=0 means model_hidden_dim
  int topk = 2;                   // experts per token
  // MORI-compatible maximum input/source tokens per rank.  Receive and
  // staging capacities are derived from this value below.
  int max_num_inp_token_per_rank = 256;
  // Explicit dispatch/combine grid block count. Set to zero to select the
  // per-call launch count from active tokens. Automatic contexts provision
  // their endpoint capacity from max_num_inp_token_per_rank.
  int num_blocks = 4;
  int num_proxy_threads = 4;  // CPU proxy threads for invoke→RDMA
  // RDMA devices used by proxies. Not an independent tunable for EP: each proxy
  // thread owns exactly one device (concurrent progress on a shared device
  // races), so ep_context_t always overrides this to num_proxy_threads.
  int num_devices = 4;
  // Per-queue ring size in 64-byte lines (power of two). Messages are
  // line-aligned and a line carries up to 32 B of payload: a put descriptor
  // (56 B) uses 2 lines and a multicast put descriptor (168 B) 6 lines.
  // Each queue costs queue_capacity * 64 bytes of pinned coherent host
  // memory, so total pinned cost scales with num_queues() * queue_capacity.
  // EP drains puts continuously and needs only a shallow queue; keep this
  // small so the kiwi/queue/wavefront count can scale cheaply.
  int queue_capacity = 2048;
  // Optional RDMA HCA/NIC name. Empty selects MORI_KIWI_DEVICE_NAME, then
  // the NIC nearest the current HIP device.
  std::string device_name;
  int dispatch_elem_size = 4;  // bytes per dispatch hidden element
  int combine_elem_size = 4;   // bytes per combine hidden element
  quant_format_t dispatch_quant_format = quant_format_t::NONE;
  quant_format_t combine_quant_format = quant_format_t::NONE;
  int dispatch_scale_dim = 0;        // dispatch scale vector dimension per token (0 = no scales)
  int dispatch_scale_type_size = 0;  // bytes per dispatch scale element
  // Retained for MORI configuration compatibility. The current combine API
  // has no scale output, so expert scales are not part of the combine wire
  // payload and these values do not affect combine slot sizing.
  int combine_scale_dim = 0;
  int combine_scale_type_size = 0;
  // Native config defaults to an explicit RDMA-only payload backend. AUTO
  // resolves the measured backend policy from each launch's active tokens.
  rcomp_backend_t rcomp_backend = rcomp_backend_t::RDMA_ONLY;
  int num_experts_per_rank() const { return num_experts / world_size; }
  int rank_of_expert(int expert_id) const { return expert_id / num_experts_per_rank(); }
  bool uses_automatic_num_blocks() const { return num_blocks == 0; }
  // Number of blocks provisioned at construction. Automatic mode derives
  // this directly from the configured maximum input capacity; explicit mode
  // keeps its fixed launch count.
  int provisioned_num_blocks() const {
    return uses_automatic_num_blocks()
               ? recommended_num_blocks(max_num_inp_token_per_rank, world_size)
               : num_blocks;
  }
  // Number of invoke SPSC queues: one per producer wavefront. Derived
  // from provisioned_num_blocks(), never tuned directly.
  int num_queues() const { return provisioned_num_blocks() * EP_WAVES_PER_BLOCK; }
  int dispatch_hidden_elems_effective() const {
    return dispatch_hidden_elems > 0 ? dispatch_hidden_elems : model_hidden_dim;
  }
  int combine_hidden_elems_effective() const {
    return combine_hidden_elems > 0 ? combine_hidden_elems : model_hidden_dim;
  }
  runtime_defaults_t launch_policy_for(int active_num_tokens) const {
    runtime_defaults_t policy =
        recommended_runtime_defaults(active_num_tokens > 0 ? active_num_tokens : 1, world_size);
    if (!uses_automatic_num_blocks()) {
      policy.num_blocks = num_blocks;
    }
    if (rcomp_backend != rcomp_backend_t::AUTO) {
      policy.rcomp_backend = rcomp_backend;
    }
    return policy;
  }
  // Automatic payload routing can select either backend across token
  // buckets. Provision HIP-IPC payload resources whenever any bucket up to
  // capacity selects hybrid; explicit backend selection remains fixed.
  bool rcomp_ipc_resources_required() const {
    if (rcomp_backend != rcomp_backend_t::AUTO) {
      return rcomp_backend == rcomp_backend_t::HYBRID;
    }
    return recommended_runtime_defaults(max_num_inp_token_per_rank, world_size).rcomp_backend ==
           rcomp_backend_t::HYBRID;
  }

  // Derived capacities from MORI's max_num_inp_token_per_rank.  This value
  // is the maximum number of input/source tokens owned by each rank; it is
  // not multiplied by topk for source-token-addressed buffers.
  //
  // Dispatch payload staging is keyed by the local source token id, because
  // one staged payload can be multicast to every unique remote destination
  // selected by that token.  Top-k only affects route metadata and the
  // per-destination dispatch fanout counts.
  int max_dispatch_staging_slots() const { return max_num_inp_token_per_rank; }

  // Combine stages every dispatch row this rank received before returning
  // rows to their source ranks.  There can be one received row per source
  // token per source rank, so combine staging keeps the full cross-rank
  // source-token capacity.
  int max_combine_staging_slots() const { return world_size * max_num_inp_token_per_rank; }

  // MORI exposes/allocates one received row per source token per source
  // rank. Multiple experts on the same destination rank are deduplicated
  // into that row.
  int max_recv_slots_per_rank() const { return max_num_inp_token_per_rank; }
  int max_recv_slots_total() const { return world_size * max_num_inp_token_per_rank; }

  // Backward-compatible alias for tensors exposed by this operator.
  int max_slots() const { return max_recv_slots_total(); }

  size_t payload_readiness_flag_count() const {
    return static_cast<size_t>(world_size) * static_cast<size_t>(max_num_inp_token_per_rank);
  }

  KIWI_EP_HOST_DEVICE size_t payload_readiness_flag_index(int source_rank,
                                                          int source_local_slot) const {
    return kiwi::ep::payload_readiness_flag_index(source_rank, source_local_slot,
                                                  max_num_inp_token_per_rank);
  }

  slot_layout_t make_slot_layout(int physical_hidden_elems_for_layout, int hidden_elem_size,
                                 int scale_dim_for_layout, int scale_type_size_for_layout) const {
    slot_layout_t layout{};
    layout.elem_size = hidden_elem_size;
    layout.hidden_offset = 0;
    layout.indices_offset =
        static_cast<size_t>(physical_hidden_elems_for_layout) * hidden_elem_size;
    layout.weights_offset = layout.indices_offset + static_cast<size_t>(topk) * sizeof(int32_t);
    layout.scales_offset = layout.weights_offset + static_cast<size_t>(topk) * sizeof(float);
    layout.src_id_offset = layout.scales_offset +
                           static_cast<size_t>(scale_dim_for_layout) * scale_type_size_for_layout;
    layout.payload_size = layout.src_id_offset + sizeof(int32_t);
    layout.stride = align_up(layout.payload_size, EP_SLOT_ALIGNMENT);
    return layout;
  }

  // Dispatch wire layout:
  // [hidden: physical_hidden_elems*elem_size | indices: topk*4 | weights: topk*4 |
  //  scales: explicit scale_dim*scale_type_size | src_token_id: 4]
  slot_layout_t dispatch_slot_layout() const {
    return make_slot_layout(dispatch_hidden_elems_effective(), dispatch_elem_size,
                            dispatch_scale_dim, dispatch_scale_type_size);
  }

  // Combine destinations are source-token-addressed, and dispatch retains
  // the producer-rank routing needed for accumulation.  Therefore combine
  // retransmits only the expert output and the weight vector required by
  // the optional output-weights API.  Indices, scales, and source token ID
  // are not consumed from combine receive slots.
  slot_layout_t combine_slot_layout() const {
    slot_layout_t layout{};
    layout.elem_size = combine_elem_size;
    layout.hidden_offset = 0;
    layout.weights_offset =
        static_cast<size_t>(combine_hidden_elems_effective()) * combine_elem_size;
    layout.payload_size = layout.weights_offset + static_cast<size_t>(topk) * sizeof(float);
    layout.stride = align_up(layout.payload_size, EP_SLOT_ALIGNMENT);
    // Unused combine fields point at the end of the payload rather than
    // reserving wire space for data that dispatch state already retains.
    layout.indices_offset = layout.payload_size;
    layout.scales_offset = layout.payload_size;
    layout.src_id_offset = layout.payload_size;
    return layout;
  }

  slot_buffer_layout_t dispatch_buffer_layout() const {
    slot_buffer_layout_t layout{};
    layout.slot = dispatch_slot_layout();
    layout.staging_offset = dispatch_staging_offset();
    layout.recv_offset = dispatch_recv_offset();
    layout.recv_region_size = dispatch_recv_region_size();
    return layout;
  }

  slot_buffer_layout_t combine_buffer_layout() const {
    slot_buffer_layout_t layout{};
    layout.slot = combine_slot_layout();
    layout.staging_offset = combine_staging_offset();
    layout.recv_offset = combine_recv_offset();
    layout.recv_region_size = combine_recv_region_size();
    return layout;
  }

  size_t dispatch_slot_hidden_offset() const { return dispatch_slot_layout().hidden_offset; }
  size_t dispatch_slot_indices_offset() const { return dispatch_slot_layout().indices_offset; }
  size_t dispatch_slot_weights_offset() const { return dispatch_slot_layout().weights_offset; }
  size_t dispatch_slot_scales_offset() const { return dispatch_slot_layout().scales_offset; }
  size_t dispatch_slot_src_id_offset() const { return dispatch_slot_layout().src_id_offset; }
  size_t dispatch_slot_payload_size() const { return dispatch_slot_layout().payload_size; }
  size_t dispatch_slot_stride() const { return dispatch_slot_layout().stride; }

  size_t combine_slot_hidden_offset() const { return combine_slot_layout().hidden_offset; }
  size_t combine_slot_indices_offset() const { return combine_slot_layout().indices_offset; }
  size_t combine_slot_weights_offset() const { return combine_slot_layout().weights_offset; }
  size_t combine_slot_scales_offset() const { return combine_slot_layout().scales_offset; }
  size_t combine_slot_src_id_offset() const { return combine_slot_layout().src_id_offset; }
  size_t combine_slot_payload_size() const { return combine_slot_layout().payload_size; }
  size_t combine_slot_stride() const { return combine_slot_layout().stride; }

  // Buffer section offsets within the single symmetric heap
  size_t dispatch_staging_offset() const { return 0; }
  size_t dispatch_staging_size() const {
    return static_cast<size_t>(max_dispatch_staging_slots()) * dispatch_slot_stride();
  }

  size_t combine_staging_offset() const { return dispatch_staging_size(); }
  size_t combine_staging_size() const {
    return static_cast<size_t>(max_combine_staging_slots()) * combine_slot_stride();
  }

  size_t dispatch_recv_offset() const { return combine_staging_offset() + combine_staging_size(); }
  size_t dispatch_recv_region_size() const {
    return static_cast<size_t>(max_recv_slots_per_rank()) * dispatch_slot_stride();
  }
  size_t dispatch_recv_size() const { return world_size * dispatch_recv_region_size(); }

  size_t combine_recv_offset() const { return dispatch_recv_offset() + dispatch_recv_size(); }
  size_t combine_recv_region_size() const {
    return static_cast<size_t>(max_recv_slots_per_rank()) * combine_slot_stride();
  }
  size_t combine_recv_size() const { return world_size * combine_recv_region_size(); }

  // Receiver-owned readiness flags live in the symmetric heap alongside
  // payload slots. Both raw RDMA signals and HIP IPC writers address this
  // common layout, so imported peer heaps expose payload and readiness state
  // with identical offsets.
  size_t dispatch_readiness_flags_offset() const {
    return align_up(combine_recv_offset() + combine_recv_size(), alignof(uint8_t));
  }

  size_t combine_readiness_flags_offset() const {
    return dispatch_readiness_flags_offset() + payload_readiness_flag_count() * sizeof(uint8_t);
  }

  size_t payload_heap_size() const {
    return combine_readiness_flags_offset() + payload_readiness_flag_count() * sizeof(uint8_t);
  }

  size_t protocol_epoch_offset() const { return align_up(payload_heap_size(), alignof(uint64_t)); }

  size_t count_alltoall_egress_offset() const { return protocol_epoch_offset() + sizeof(uint64_t); }

  size_t count_alltoall_region_size() const {
    return static_cast<size_t>(world_size) * sizeof(uint64_t);
  }

  size_t count_alltoall_ingress_offset() const {
    return count_alltoall_egress_offset() + count_alltoall_region_size();
  }

  size_t barrier_alltoall_egress_offset() const {
    return count_alltoall_ingress_offset() + count_alltoall_region_size();
  }

  size_t barrier_alltoall_region_size() const { return 2 * count_alltoall_region_size(); }

  size_t barrier_alltoall_ingress_offset() const {
    return barrier_alltoall_egress_offset() + barrier_alltoall_region_size();
  }

  size_t total_heap_size() const {
    return barrier_alltoall_ingress_offset() + barrier_alltoall_region_size();
  }

  size_t dispatch_recv_from_rank_offset(int src_rank) const {
    return dispatch_recv_offset() + src_rank * dispatch_recv_region_size();
  }

  size_t combine_recv_from_rank_offset(int src_rank) const {
    return combine_recv_offset() + src_rank * combine_recv_region_size();
  }

  // Per token-expert route metadata.  Keep the two concepts in separate
  // arrays so combine cannot accidentally depend on a paired dispatch-side
  // field layout.
  //
  // token_to_producer_rank[token, k]:
  //   rank that produced this expert row; consumed by combine accumulation.
  //   Duplicate experts targeting an already-seen destination use an
  //   out-of-range producer_rank sentinel. Reset leaves unused entries at
  //   producer_rank == -1.
  //
  // token_to_dispatch_recv_slot[token, k]:
  //   destination rank's dispatch receive-slot metadata for dispatch/debug
  //   tests only. Combine slot identity is instead
  //   combine_recv_slot(producer_rank, source_token_id).
  size_t route_metadata_entries() const {
    return static_cast<size_t>(max_num_inp_token_per_rank) * topk;
  }
  size_t token_to_producer_rank_size() const { return route_metadata_entries() * sizeof(int32_t); }
  size_t token_to_dispatch_recv_slot_size() const {
    return route_metadata_entries() * sizeof(int32_t);
  }
};

}  // namespace mori::kiwi::ep
