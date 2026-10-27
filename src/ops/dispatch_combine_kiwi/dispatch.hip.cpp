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
#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif
#include <hip/hip_runtime.h>

#include "ep_context.hip.hpp"
#include "hip_check.hpp"
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_warp_sync_functions.h>
#endif
#include <cstdio>

// Diagnostic-only compile-time toggles for issue #10 payload-bandwidth
// experiments.  The default build keeps normal correctness-preserving behavior.
// Enabling either toggle is intentionally timing-only: hidden outputs become
// invalid, but slot metadata is still transferred so the benchmark can keep
// exercising the routing/control path without crashing.
#ifndef KIWI_EP_DIAG_SKIP_HIDDEN_COPY
#define KIWI_EP_DIAG_SKIP_HIDDEN_COPY 0
#endif

#ifndef KIWI_EP_DIAG_METADATA_ONLY_PUT
#define KIWI_EP_DIAG_METADATA_ONLY_PUT 0
#endif

namespace mori::kiwi::ep {

inline constexpr int DISPATCH_PRESCAN_LANES_PER_TOKEN = 8;

namespace dispatch_debug_slot {

inline constexpr int entry = 0;
inline constexpr int count_prescan_done = 1;
inline constexpr int count_all2all_started = 2;
inline constexpr int rcomp_rdma_posted = 3;
inline constexpr int rcomp_ipc_copies_complete = 4;
inline constexpr int incoming_count_all2all_wait_start = 5;
inline constexpr int incoming_counts_observed = 6;
inline constexpr int rank_slot_maps_built = 7;
inline constexpr int unpack_barrier_released = 8;
inline constexpr int unpack_done = 9;

}  // namespace dispatch_debug_slot

__device__ inline char* dispatch_put_src_ptr(const ep_gcontext_t& gctx, char* slot) {
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  return slot + gctx.dispatch_layout.slot.indices_offset;
#else
  return slot;
#endif
}

__device__ inline uintptr_t dispatch_put_remote_disp(const ep_gcontext_t& gctx, int recv_slot) {
  uintptr_t remote_disp = gctx.dispatch_remote_disp(recv_slot);
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  remote_disp += gctx.dispatch_layout.slot.indices_offset;
#endif
  return remote_disp;
}

__device__ inline size_t dispatch_put_bytes(const ep_gcontext_t& gctx) {
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  return gctx.dispatch_layout.slot.payload_size - gctx.dispatch_layout.slot.indices_offset;
#else
  return gctx.dispatch_layout.slot.stride;
#endif
}

__device__ inline uint8_t* dispatch_payload_readiness_flag(const ep_gcontext_t& gctx,
                                                           int source_rank, int source_local_slot) {
  return gctx.dispatch_readiness_flags() +
         gctx.payload_readiness_flag_index(source_rank, source_local_slot);
}

template <bool SystemScopeStore>
__device__ inline void dispatch_pack_slot_part(const ep_gcontext_t& gctx, const void* src_hidden,
                                               const int32_t* src_indices, const float* src_weights,
                                               const void* src_scales, int32_t src_token_id,
                                               int topk, size_t hidden_offset, int hidden_bytes,
                                               int scale_bytes, char* dst_slot, bool copy_metadata,
                                               int lane) {
  if (!KIWI_EP_DIAG_SKIP_HIDDEN_COPY) {
    void* dst_hidden =
        static_cast<char*>(gctx.dispatch_layout.slot.hidden(dst_slot)) + hidden_offset;
    copy_hidden_payload_block<SystemScopeStore>(src_hidden, dst_hidden, hidden_bytes, lane,
                                                EP_WAVE_SIZE);
  }
  if (copy_metadata) {
    copy_slot_metadata_block<SystemScopeStore>(
        src_indices, src_weights, src_scales, src_token_id, topk, scale_bytes,
        gctx.dispatch_layout.slot.indices(dst_slot), gctx.dispatch_layout.slot.weights(dst_slot),
        gctx.dispatch_layout.slot.src_id(dst_slot), gctx.dispatch_layout.slot.scales(dst_slot),
        lane, EP_WAVE_SIZE);
  }
}

__device__ inline void dispatch_debug_store_time(uint64_t* debug_time_buf, int idx) {
  if (debug_time_buf != nullptr) {
    debug_time_buf[idx] = static_cast<uint64_t>(wall_clock64());
  }
}

__device__ inline void dispatch_debug_store_meta(int32_t* debug_time_meta, int idx, int32_t value) {
  if (debug_time_meta != nullptr) {
    debug_time_meta[idx] = value;
  }
}

// First-wave lanes route one token's top-k experts in parallel.  One lane owns
// one top-k entry; __match_any_sync groups equal destination ranks and the
// lowest lane in each group remains the canonical first top-k occurrence.
// Leader lanes allocate receive slots and populate the compact destination list
// consumed by the copy/post phase.  Duplicate same-rank experts are marked with
// the out-of-range producer-rank sentinel in token_to_producer_rank.
__device__ inline void route_topk_parallel_block(const ep_gcontext_t& gctx,
                                                 const int32_t* token_expert_indices, int token_idx,
                                                 bool route_owner, int* s_num_unique,
                                                 int* s_unique_dests, int* s_unique_recv_slots) {
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  const int wf_in_blk = threadIdx.x / EP_WAVE_SIZE;
  const int topk = gctx.topk;
  const int world_size = gctx.world_size;
  const int num_experts_per_rank = gctx.num_experts_per_rank;
  const ep_device_workspace_t& device_workspace = gctx.device_workspace;

  if (route_owner && lane == 0) {
    *s_num_unique = 0;
  }

  if (route_owner && token_expert_indices != nullptr && lane < topk) {
    const int k = lane;
    const int expert_id = token_expert_indices[k];
    const int dest_rank = (expert_id >= 0) ? (expert_id / num_experts_per_rank) : -1;
    const bool valid_dest = (dest_rank >= 0 && dest_rank < world_size);
    const unsigned long long active_topk_mask =
        (topk >= EP_WAVE_SIZE) ? ~0ull : ((1ull << topk) - 1ull);
    const unsigned long long match_mask = __match_any_sync(active_topk_mask, dest_rank);
    const int leader_lane = __builtin_ctzll(match_mask);
    const bool first_for_dest = valid_dest && (leader_lane == lane);
    const unsigned long long leader_mask = __ballot(first_for_dest) & active_topk_mask;
    const unsigned long long lower_mask = (lane == 0) ? 0ull : ((1ull << lane) - 1ull);
    const int unique_idx = __popcll(leader_mask & lower_mask);

    int32_t* producer_rank_ptr = gctx.token_to_producer_rank_ptr(token_idx, k);
    int32_t* dispatch_recv_slot_ptr = gctx.token_to_dispatch_recv_slot_ptr(token_idx, k);
    if (first_for_dest) {
      const int recv_slot =
          __atomic_fetch_add(&device_workspace.dest_token_counts()[dest_rank], 1, __ATOMIC_RELAXED);
      s_unique_dests[unique_idx] = dest_rank;
      s_unique_recv_slots[unique_idx] = recv_slot;
      *producer_rank_ptr = dest_rank;
      *dispatch_recv_slot_ptr = recv_slot;
    } else {
      *producer_rank_ptr = world_size;
      *dispatch_recv_slot_ptr = 0;
    }

    if (lane == 0) {
      *s_num_unique = __popcll(leader_mask);
    }
  }
  __syncthreads();
}

// Count the same canonical unique destination ranks that
// route_topk_parallel_block will later materialize for slot assignment.  This
// prescan deliberately does not touch device_workspace.dest_token_counts(): that array
// remains the per-destination slot allocator for the pack/post phase.
//
// The low-token dispatch path is latency-sensitive, so avoid assigning one
// full lane to one token and serializing that token's top-k scan in the lane.
// Instead, each 8-lane subgroup owns one token.  For topk <= 8 this maps one
// top-k entry to one lane and uses an intra-subgroup match to deduplicate
// same-destination experts.  Larger supported top-k values are handled in
// 8-entry chunks; only those later chunks need a short scan over prior chunks.
__device__ inline void dispatch_prescan_unique_dest_counts(const ep_gcontext_t& gctx,
                                                           const int32_t* token_expert_indices,
                                                           int num_tokens, int32_t* send_counts) {
  static_assert(EP_WAVE_SIZE % DISPATCH_PRESCAN_LANES_PER_TOKEN == 0,
                "prescan token subgroups must not cross wavefronts");
  static_assert(EP_THREADS_PER_BLOCK % DISPATCH_PRESCAN_LANES_PER_TOKEN == 0,
                "prescan token subgroups must evenly tile a block");

  const int lane = threadIdx.x % EP_WAVE_SIZE;
  const int prescan_lane = lane % DISPATCH_PRESCAN_LANES_PER_TOKEN;
  const int prescan_group = threadIdx.x / DISPATCH_PRESCAN_LANES_PER_TOKEN;
  const int prescan_groups_per_block = blockDim.x / DISPATCH_PRESCAN_LANES_PER_TOKEN;
  const int topk = gctx.topk;
  const int world_size = gctx.world_size;
  const int num_experts_per_rank = gctx.num_experts_per_rank;

  for (int r = threadIdx.x; r < gctx.world_size; r += blockDim.x) {
    send_counts[r] = 0;
  }
  __syncthreads();

  for (int token_idx = prescan_group; token_idx < num_tokens;
       token_idx += prescan_groups_per_block) {
    const int32_t* token_indices = token_expert_indices + static_cast<size_t>(token_idx) * topk;

    for (int k_base = 0; k_base < topk; k_base += DISPATCH_PRESCAN_LANES_PER_TOKEN) {
      const int entries_remaining = topk - k_base;
      const int active_lanes = entries_remaining < DISPATCH_PRESCAN_LANES_PER_TOKEN
                                   ? entries_remaining
                                   : DISPATCH_PRESCAN_LANES_PER_TOKEN;
      const int k = k_base + prescan_lane;

      if (prescan_lane >= active_lanes) {
        continue;
      }

      const int expert_id = token_indices[k];
      const int dest_rank = (expert_id >= 0) ? (expert_id / num_experts_per_rank) : -1;
      const bool valid_dest = dest_rank >= 0 && dest_rank < world_size;

      const int prescan_lane_base = lane - prescan_lane;
      const unsigned long long active_mask = ((1ull << active_lanes) - 1ull) << prescan_lane_base;
      const unsigned long long match_mask = __match_any_sync(active_mask, dest_rank);
      const int leader_lane = (match_mask != 0ull) ? __builtin_ctzll(match_mask) : EP_WAVE_SIZE;
      const bool first_in_chunk = leader_lane == lane;

      bool seen_in_prior_chunk = false;
      if (valid_dest) {
        for (int prev = 0; prev < k_base; prev++) {
          const int prev_expert_id = token_indices[prev];
          const int prev_dest =
              (prev_expert_id >= 0) ? (prev_expert_id / num_experts_per_rank) : -1;
          if (prev_dest == dest_rank) {
            seen_in_prior_chunk = true;
            break;
          }
        }
      }

      if (valid_dest && first_in_chunk && !seen_in_prior_chunk) {
        __atomic_fetch_add(&send_counts[dest_rank], 1, __ATOMIC_RELAXED);
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Canonical single-kernel dispatch mirroring mori-ep's GPU-driven flow.
//
// Grid: num_blocks blocks × EP_THREADS_PER_BLOCK threads.  Each block contains EP_WAVES_PER_BLOCK
// wavefronts, and each wavefront owns a dedicated EP comm endpoint.  The EP comm context is
// provisioned with provisioned_num_blocks * EP_WAVES_PER_BLOCK queues while each launch selects its
// active prefix independently. Threads are organized as EP_WAVES_PER_BLOCK wavefronts/WG.  Each
// block dynamically assigns its wavefronts to a block-local token group: the owner wavefront
// handles a token's routing/dedup/control/post work, and all wavefronts assigned to that token
// cooperatively copy disjoint hidden payload ranges.
//
// Phase 0 (block 0): Prescan top-k maps to count each token's unique valid
//   destination ranks and publish the dispatch send-count vector before payload
//   packing.
//
// Phase 1 (all blocks): Strided token packing with per-token
//   deduplication.  Tokens with remote destinations stage one reusable payload
//   and submit one multicast descriptor covering all unique remote ranks.
//   Additionally populates
//   token_to_producer_rank for later combine accumulation and
//   token_to_dispatch_recv_slot for dispatch/debug metadata.
//
// Grid barrier: all blocks arrive after phase 1, then wait while block 0
//   completes receive setup.  The reusable generation-counted barrier lets the
//   same kernel continue into multi-block unpack without a host reset.
//
// Phase 2 (block 0 only): Observes the count all2all before building maps.
//   Each remote payload is made visible through its own readiness flag.
//
// Phase 3: Block 0 waits for the EP all2all per-source receive-count vector
//   and builds per-rank receive counts/prefixes/direct flat-row maps. Then all
//   dispatch blocks unpack received slots into the public receive tensors using
//   the same fixed launch shape as phase 1.
// ---------------------------------------------------------------------------

// This template specializes the RCOMP payload backend: RDMA-only or hybrid
// (IPC for same-host peers and RDMA for remote peers).
template <bool UseHybridRcomp>
__global__ void dispatch(ep_gcontext_t gctx, const void* __restrict__ d_tokens,
                         const int32_t* __restrict__ d_expert_indices,
                         const float* __restrict__ d_weights, const void* __restrict__ d_scales,
                         int num_tokens, uint64_t* debug_time_buf, int32_t* debug_time_meta) {
  const ep_device_workspace_t& device_workspace = gctx.device_workspace;
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  const int wf_in_blk = threadIdx.x / EP_WAVE_SIZE;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;
  const int rank_me = gctx.rank_me;
  const int topk = gctx.topk;
  const int dispatch_hidden_elems = gctx.dispatch_hidden_elems;
  const int dispatch_elem_size = gctx.dispatch_layout.slot.elem_size;
  const int dispatch_scale_dim = gctx.dispatch_scale_dim;
  const int dispatch_scale_type_size = gctx.dispatch_scale_type_size;
  const int heap_id = gctx.heap_id;
  const int world_size = gctx.world_size;
  int32_t* dispatch_dest_token_counts = device_workspace.dispatch_dest_token_counts_snapshot();
  int32_t* recv_counts = device_workspace.dispatch_recv_counts();

  const int dispatch_hidden_bytes = dispatch_hidden_elems * dispatch_elem_size;
  const int scale_bytes = dispatch_scale_dim * dispatch_scale_type_size;

  constexpr int kDispatchWavesPerBlock = EP_WAVES_PER_BLOCK;
  static_assert(EP_THREADS_PER_BLOCK == EP_WAVE_SIZE * EP_WAVES_PER_BLOCK,
                "EP launch constants are inconsistent");
  const int wave_count = kDispatchWavesPerBlock;
  // Acquire each wave's endpoint once. In block 0 / wave 0 this is endpoint
  // 0, reused for the active-count control callback and count all-to-all.
  const size_t dispatch_endpoint_id = static_cast<size_t>(bid * kDispatchWavesPerBlock + wf_in_blk);
  auto dispatch_handle = gctx.comm.get_handle(dispatch_endpoint_id);

  // invoke invoke is wave-cooperative. Block 0 / wave 0 publishes the
  // active endpoint prefix through endpoint 0 before any EP traffic uses it.
  if (bid == 0 && wf_in_blk == 0) {
    publish_active_queue_count(gctx, dispatch_handle);
  }

  if (bid == 0 && threadIdx.x == 0) {
    dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::entry);
    dispatch_debug_store_meta(debug_time_meta, 0, rank_me);
    dispatch_debug_store_meta(debug_time_meta, 1, world_size);
    dispatch_debug_store_meta(debug_time_meta, 2, num_tokens);
    dispatch_debug_store_meta(debug_time_meta, 3,
                              gctx.max_num_inp_token_per_rank);  // dispatch staging slots
    dispatch_debug_store_meta(debug_time_meta, 4, 0);            // total recv; filled later
    dispatch_debug_store_meta(debug_time_meta, 5, num_blocks);
    dispatch_debug_store_meta(debug_time_meta, 6, 0);  // remote recv; filled later
    dispatch_debug_store_meta(debug_time_meta, 7, 0);  // self recv; filled later
  }

  __shared__ int s_num_unique[kDispatchWavesPerBlock];  // number of unique destinations for each
                                                        // per-token slot
  __shared__ int
      s_num_post[kDispatchWavesPerBlock];  // number of compact route-list destinations per token
                                           // slot that need a posted communication operation
  __shared__ int s_unique_dests[kDispatchWavesPerBlock]
                               [MAX_TOPK_SUPPORTED];  // compact route-list destinations for each
                                                      // per-token slot
  __shared__ int s_unique_recv_slots[kDispatchWavesPerBlock]
                                    [MAX_TOPK_SUPPORTED];  // compact route-list receive slots for
                                                           // each per-token slot
  // Per-wavefront staging for one multicast put descriptor. Padded to whole
  // 8-byte words because the shared push reads whole words.
  constexpr size_t kMcStageWords = kiwi::ep::EP_MULTICAST_DESC_STAGING_BYTES / sizeof(uint64_t);
  static_assert(kiwi::ep::EP_MULTICAST_DESC_STAGING_BYTES % sizeof(uint64_t) == 0,
                "multicast descriptor staging must be 8-byte aligned");
  __shared__ uint64_t s_mc_stage[kDispatchWavesPerBlock][kMcStageWords];

  const int raw_stage_tokens_per_block = (num_tokens + num_blocks - 1) / num_blocks;
  int stage_tokens_per_block = raw_stage_tokens_per_block;
  if (stage_tokens_per_block < 1) {
    stage_tokens_per_block = 1;
  }
  if (stage_tokens_per_block > wave_count) {
    stage_tokens_per_block = wave_count;
  }
  const int waves_per_token_candidate = wave_count / stage_tokens_per_block;
  const int waves_per_token = waves_per_token_candidate > 0 ? waves_per_token_candidate : 1;

  ep_all2all_ticket_t count_all2all_ticket{};

  // =====================================================================
  // Phase 0: Prescan routing and start count all2all (block 0)
  // =====================================================================
  if (bid == 0) {
    dispatch_prescan_unique_dest_counts(gctx, d_expert_indices, num_tokens,
                                        dispatch_dest_token_counts);
    if (tid == 0) {
      dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::count_prescan_done);
    }

    if (wf_in_blk == 0) {
      __builtin_amdgcn_s_waitcnt(0);
      count_all2all_ticket = post_all2all(
          gctx, dispatch_handle, reinterpret_cast<const uint32_t*>(dispatch_dest_token_counts));
      if (lane == 0) {
        dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::count_all2all_started);
      }
    }
  }

  // =====================================================================
  // Phase 1: Pack and send slots (all blocks)
  // =====================================================================
  for (int base_token = bid * stage_tokens_per_block; base_token < num_tokens;
       base_token += num_blocks * stage_tokens_per_block) {
    const int remaining = num_tokens - base_token;
    const int items_in_block =
        remaining < stage_tokens_per_block ? remaining : stage_tokens_per_block;
    const int item_in_block = wf_in_blk / waves_per_token;
    const int part_in_token = wf_in_blk % waves_per_token;
    const bool valid_token = wf_in_blk < kDispatchWavesPerBlock && item_in_block < items_in_block;
    const int t = base_token + item_in_block;
    const bool route_owner = valid_token && part_in_token == 0;
    const int shared_item = valid_token ? item_in_block : 0;
    int* s_item_num_unique = &s_num_unique[shared_item];
    int* s_item_num_post = &s_num_post[shared_item];
    int* s_item_unique_dests = s_unique_dests[shared_item];
    int* s_item_unique_recv_slots = s_unique_recv_slots[shared_item];
    const size_t dispatch_col_begin = (static_cast<size_t>(dispatch_hidden_elems) * part_in_token) /
                                      static_cast<size_t>(waves_per_token);
    const size_t dispatch_col_end =
        (static_cast<size_t>(dispatch_hidden_elems) * (part_in_token + 1)) /
        static_cast<size_t>(waves_per_token);
    const size_t dispatch_col_count = dispatch_col_end - dispatch_col_begin;

    route_topk_parallel_block(gctx, route_owner ? (d_expert_indices + t * topk) : nullptr, t,
                              route_owner, s_item_num_unique, s_item_unique_dests,
                              s_item_unique_recv_slots);

    // The hybrid path classifies the route owner's unique pairs once into
    // one compact bit mask. Its set-bit ordinal is the flat IPC route
    // index, so token waves can reuse the route list without compaction or
    // per-wave run records.
    int num_post = 0;
    int self_recv_slot = -1;
    static_assert(MAX_TOPK_SUPPORTED <= 32, "IPC route bitmask requires at most 32 top-k entries");
    unsigned int ipc_route_mask = 0u;
    if constexpr (UseHybridRcomp) {
      const int owner_wave = item_in_block * waves_per_token;
      if (route_owner) {
        int raw_dest = -1;
        int raw_recv_slot = 0;
        if (lane < *s_item_num_unique) {
          raw_dest = s_item_unique_dests[lane];
          raw_recv_slot = s_item_unique_recv_slots[lane];
        }

        const bool is_ipc =
            lane < *s_item_num_unique && raw_dest != rank_me && gctx.rcomp_peer_uses_ipc(raw_dest);
        const bool is_direct_self = lane < *s_item_num_unique && raw_dest == rank_me;
        const bool is_post = lane < *s_item_num_unique && !is_ipc && !is_direct_self;
        const unsigned long long owner_ipc_mask = __ballot(is_ipc);
        const unsigned long long post_mask = __ballot(is_post);
        const unsigned long long self_mask = __ballot(is_direct_self);
        int owner_self_recv_slot = -1;
        if (self_mask != 0ull) {
          owner_self_recv_slot = __shfl(raw_recv_slot, __builtin_ctzll(self_mask));
        }

        if (lane == 0) {
          // MAX_TOPK_SUPPORTED is 16, so the low 32 bits hold the
          // raw-unique IPC mask. The upper 32 bits hold the direct
          // self receive slot. This staging row becomes a multicast
          // descriptor before the IPC copy phase.
          const uint64_t packed_mask_and_self = (owner_ipc_mask & 0xffffffffull) |
                                                (static_cast<uint64_t>(owner_self_recv_slot) << 32);
          s_mc_stage[owner_wave][0] = packed_mask_and_self;
          *s_item_num_post = __popcll(post_mask);
        }
      }
      __syncthreads();

      if (valid_token) {
        const uint64_t packed_mask_and_self = s_mc_stage[owner_wave][0];
        ipc_route_mask = static_cast<unsigned int>(packed_mask_and_self & 0xffffffffull);
        self_recv_slot = static_cast<int>(static_cast<int32_t>(packed_mask_and_self >> 32));
        num_post = *s_item_num_post;
      }
    } else {
      if (valid_token) {
        for (int u = 0; u < *s_item_num_unique; ++u) {
          const int dest = s_item_unique_dests[u];
          if (dest == rank_me) {
            self_recv_slot = s_item_unique_recv_slots[u];
            continue;
          }
          bool uses_ipc = false;
          if constexpr (UseHybridRcomp) {
            uses_ipc = gctx.rcomp_peer_uses_ipc(dest);
          }
          if (!uses_ipc) {
            ++num_post;
          }
        }
      }
    }

    char* staging = num_post > 0 ? gctx.dispatch_staging_slot(t) : nullptr;

    if (num_post > 0) {
      const size_t hidden_offset = dispatch_col_begin * static_cast<size_t>(dispatch_elem_size);
      const char* token_hidden =
          static_cast<const char*>(d_tokens) + static_cast<size_t>(t) * dispatch_hidden_bytes;
      const void* src_hidden = token_hidden + hidden_offset;
      const int hidden_copy_bytes =
          static_cast<int>(dispatch_col_count * static_cast<size_t>(dispatch_elem_size));
      const void* src_scales = (d_scales != nullptr && dispatch_scale_dim > 0)
                                   ? static_cast<const void*>(static_cast<const char*>(d_scales) +
                                                              static_cast<size_t>(t) * scale_bytes)
                                   : nullptr;
      const int32_t* src_indices = d_expert_indices + t * topk;
      const float* src_weights = d_weights + t * topk;

      dispatch_pack_slot_part<true>(gctx, src_hidden, src_indices, src_weights, src_scales, t, topk,
                                    hidden_offset, hidden_copy_bytes, scale_bytes, staging,
                                    route_owner, lane);
    }

    if constexpr (UseHybridRcomp) {
      // The RDMA staging payload is shared by all token waves. Drain
      // each wave's stores and synchronize the block before the
      // route-owner wave exposes that staging slot to the proxy.
      __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    // Post the complete RDMA multicast before the independent direct-self
    // and IPC copies. Do not add a block barrier after this submission:
    // non-owner waves must be able to begin local copies while the proxy
    // advances the remote puts.
    if (route_owner && num_post > 0) {
      const int num_unique = *s_item_num_unique;
      bool lane_has_post = lane < num_unique && s_item_unique_dests[lane] != rank_me;
      if constexpr (UseHybridRcomp) {
        lane_has_post = lane_has_post && (ipc_route_mask & (1u << lane)) == 0u;
      }
      const unsigned long long post_mask = __ballot(lane_has_post);

      auto* payload_mc = ep_multicast_desc_from_stage(&s_mc_stage[wf_in_blk][0]);
      const unsigned long long lower_mask = (lane == 0) ? 0ull : ((1ull << lane) - 1ull);
      const int out_idx = lane_has_post ? __popcll(post_mask & lower_mask) : -1;

      for (int base = 0; base < num_post; base += internal::EP_COMM_MAX_DESTS) {
        const int chunk_n = (num_post - base) < internal::EP_COMM_MAX_DESTS
                                ? (num_post - base)
                                : internal::EP_COMM_MAX_DESTS;
        if (out_idx >= base && out_idx < base + chunk_n) {
          ep_multicast_desc_set_rank_disp(
              payload_mc, out_idx - base, s_item_unique_dests[lane],
              dispatch_put_remote_disp(gctx, s_item_unique_recv_slots[lane]),
              s_item_unique_recv_slots[lane]);
        }
        if (lane == 0) {
          ep_multicast_desc_set_scalars(gctx, payload_mc, dispatch_put_src_ptr(gctx, staging),
                                        dispatch_put_bytes(gctx), heap_id, chunk_n,
                                        gctx.dispatch_readiness_flags_offset);
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        post_put_multicast(gctx, dispatch_handle, payload_mc);
        __builtin_amdgcn_wave_barrier();
      }

      if constexpr (UseHybridRcomp) {
        if (bid == 0 && wf_in_blk == 0 && t == 0 && lane == 0) {
          dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::rcomp_rdma_posted);
        }
      }
    }

    if (valid_token && *s_item_num_unique > 0) {
      const size_t hidden_offset = dispatch_col_begin * static_cast<size_t>(dispatch_elem_size);
      const char* token_hidden =
          static_cast<const char*>(d_tokens) + static_cast<size_t>(t) * dispatch_hidden_bytes;
      const void* src_hidden = token_hidden + hidden_offset;
      const int hidden_copy_bytes =
          static_cast<int>(dispatch_col_count * static_cast<size_t>(dispatch_elem_size));
      const void* src_scales = (d_scales != nullptr && dispatch_scale_dim > 0)
                                   ? static_cast<const void*>(static_cast<const char*>(d_scales) +
                                                              static_cast<size_t>(t) * scale_bytes)
                                   : nullptr;
      const int32_t* src_indices = d_expert_indices + t * topk;
      const float* src_weights = d_weights + t * topk;

      if (self_recv_slot >= 0) {
        dispatch_pack_slot_part<false>(gctx, src_hidden, src_indices, src_weights, src_scales, t,
                                       topk, hidden_offset, hidden_copy_bytes, scale_bytes,
                                       gctx.dispatch_recv_slot(rank_me, self_recv_slot),
                                       route_owner, lane);
      }

      if constexpr (UseHybridRcomp) {
        // Each wave copies its legacy contiguous hidden range for
        // every IPC destination. Rotate the raw unique-destination
        // order so waves do not begin at the same destination.
        const int num_unique = *s_item_num_unique;
        const int ipc_start = (part_in_token * num_unique) / waves_per_token;
        for (int u = ipc_start; u < num_unique; ++u) {
          if ((ipc_route_mask & (1u << u)) != 0u) {
            dispatch_pack_slot_part<true>(gctx, src_hidden, src_indices, src_weights, src_scales, t,
                                          topk, hidden_offset, hidden_copy_bytes, scale_bytes,
                                          gctx.rcomp_peer_dispatch_recv_slot(
                                              s_item_unique_dests[u], s_item_unique_recv_slots[u]),
                                          route_owner, lane);
          }
        }
        for (int u = 0; u < ipc_start; ++u) {
          if ((ipc_route_mask & (1u << u)) != 0u) {
            dispatch_pack_slot_part<true>(gctx, src_hidden, src_indices, src_weights, src_scales, t,
                                          topk, hidden_offset, hidden_copy_bytes, scale_bytes,
                                          gctx.rcomp_peer_dispatch_recv_slot(
                                              s_item_unique_dests[u], s_item_unique_recv_slots[u]),
                                          route_owner, lane);
          }
        }
      }
    }
    if constexpr (UseHybridRcomp) {
      // Every IPC writer drains its system-scope stores before this
      // unconditional block synchronization. The route owner publishes
      // each receiver-owned readiness flag only after the barrier.
      __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    if constexpr (UseHybridRcomp) {
      if (bid == 0 && t == 0 && tid == 0) {
        dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::rcomp_ipc_copies_complete);
      }
      if (route_owner) {
        for (int u = lane; u < *s_item_num_unique; u += EP_WAVE_SIZE) {
          if ((ipc_route_mask & (1u << u)) != 0u) {
            const size_t flag_index =
                gctx.payload_readiness_flag_index(rank_me, s_item_unique_recv_slots[u]);
            payload_readiness_flag_publish(
                gctx.peer_dispatch_readiness_flags(s_item_unique_dests[u]) + flag_index);
          }
        }
      }
    }

    __syncthreads();
  }

  // =====================================================================
  // Single pre-unpack grid barrier. Nonzero blocks arrive after local
  // pack/post work. Block 0 arrives after the count-all2all wait and receive
  // map construction. RCOMP unpack polls each remote slot's own readiness
  // flag.
  // =====================================================================
  if (bid == 0) {
    if (wf_in_blk == 0) {
      // Fused receive-count all2all and map building.  Wave 0 performs
      // the dynamic collective-control work once; all block-0 threads join
      // below to parallelize the row-map construction.
      if (lane == 0) {
        dispatch_debug_store_time(debug_time_buf,
                                  dispatch_debug_slot::incoming_count_all2all_wait_start);
      }
      volatile uint64_t* all2all_recv =
          reinterpret_cast<volatile uint64_t*>(device_workspace.all2all_counts_recv());
      wait_all2all(gctx, count_all2all_ticket, device_workspace.all2all_counts_recv());
      // all2all_recv[r] is the number of dispatch rows this rank will
      // receive from producer rank r.  Load the per-source vector across
      // wave lanes, publish recv_counts[r] for map construction, and
      // reduce the same values to the total and peer-only incoming row
      // counts in lane 0.
      int32_t lane_total = 0;
      int32_t lane_remote = 0;
      for (int r = lane; r < world_size; r += EP_WAVE_SIZE) {
        const int32_t c = static_cast<int32_t>(all2all_recv[r]);
        recv_counts[r] = c;
        lane_total += c;
        if (r != rank_me) {
          lane_remote += c;
        }
      }
      for (int offset = EP_WAVE_SIZE / 2; offset > 0; offset >>= 1) {
        lane_total += __shfl_down(lane_total, offset);
        lane_remote += __shfl_down(lane_remote, offset);
      }
      // The wave barrier only synchronizes control flow.  Drain the lanes'
      // recv_counts[] stores before lane 0 reads that vector below.
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_wave_barrier();

      if (lane == 0) {
        const int32_t total = lane_total;
        const int32_t remote_recv = lane_remote;
        dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::incoming_counts_observed);

        recv_counts[world_size] = total;
        *device_workspace.recv_count() = total;
        dispatch_debug_store_meta(debug_time_meta, 4, total);
        dispatch_debug_store_meta(debug_time_meta, 6, remote_recv);
        dispatch_debug_store_meta(debug_time_meta, 7, recv_counts[gctx.rank_me]);

        device_workspace.recv_begin()[0] = 0;
      }

      // Build d_recv_begin as an exclusive prefix boundary array.  Each
      // wave chunk scans up to EP_WAVE_SIZE producer counts in parallel;
      // running_begin carries the sum of earlier chunks so this remains
      // correct even if world_size exceeds one wave.
      int running_begin = 0;
      for (int base = 0; base < world_size; base += EP_WAVE_SIZE) {
        const int active = (world_size - base < EP_WAVE_SIZE) ? (world_size - base) : EP_WAVE_SIZE;
        const int r = base + lane;
        int prefix = ep_wave_inclusive_prefix_sum((lane < active) ? recv_counts[r] : 0);
        if (lane < active) {
          device_workspace.recv_begin()[r + 1] = running_begin + prefix;
        }
        running_begin += __shfl(prefix, active - 1);
      }
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_wave_barrier();
    }
    __syncthreads();

    // Build direct flat-row metadata once so unpack and later combine paths
    // avoid rescanning recv_counts for each row.
    for (int r = wf_in_blk; r < world_size; r += kDispatchWavesPerBlock) {
      const int begin = device_workspace.recv_begin()[r];
      const int count = recv_counts[r];
      for (int i = lane; i < count; i += EP_WAVE_SIZE) {
        const int flat = begin + i;
        device_workspace.recv_src_rank_map()[flat] = r;
        device_workspace.recv_local_slot_map()[flat] = i;
      }
    }
    __syncthreads();
    if (tid == 0) {
      dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::rank_slot_maps_built);
    }
  }

  grid_barrier_ticket_t unpack_ready_ticket =
      grid_barrier_arrive(device_workspace.grid_barrier(), static_cast<uint32_t>(num_blocks));
  grid_barrier_wait(device_workspace.grid_barrier(), unpack_ready_ticket);

  // No more dispatch receive-slot allocations can happen in this dispatch
  // once the single pre-unpack barrier has released.  Reset the per-dest
  // slot allocators for the next dispatch; current unpack uses the
  // snapshotted receive counts/maps, not these allocators.
  if (bid == 0) {
    for (int r = tid; r < world_size; r += blockDim.x) {
      device_workspace.dest_token_counts()[r] = 0;
    }
    if (tid == 0) {
      dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::unpack_barrier_released);
    }
  }

  // Multi-block fused unpack.  The launch shape stays fixed for graph
  // replay.  Use the MORI-style multi-warp iterator only for this
  // receive-side copy: it cannot change per-token posting counts, but it
  // lets small-token cases split each hidden row across otherwise idle
  // wavefronts.  When rows dominate resident wavefronts, this collapses
  // to the existing one-wavefront-per-row mapping.
  const int total_recv = recv_counts[world_size];
  const int global_wave_count = num_blocks * wave_count;
  ep_multi_warp_iter_t unpack_iter(global_wave_count, total_recv,
                                   static_cast<size_t>(dispatch_hidden_elems));
  const int total_parts = unpack_iter.total_parts(total_recv);
  for (int base_part = bid * wave_count; base_part < total_parts;
       base_part += num_blocks * wave_count) {
    const int part_idx = base_part + wf_in_blk;
    const bool valid_part = wf_in_blk < kDispatchWavesPerBlock && part_idx < total_parts;

    int flat = 0;
    int in_flat_part = 0;
    size_t dispatch_col_offset = 0;
    size_t dispatch_col_count = 0;
    if (valid_part) {
      unpack_iter.decode(part_idx, flat, in_flat_part, dispatch_col_offset, dispatch_col_count);
    }
    const bool valid_flat = valid_part && flat < total_recv;
    if (!valid_flat) {
      continue;
    }

    int src_rank, local_slot;
    flat_to_rank_slot_direct(gctx, flat, src_rank, local_slot);
    char* slot = gctx.dispatch_recv_slot(src_rank, local_slot);

    if (src_rank != rank_me && lane == 0) {
      payload_readiness_flag_wait(dispatch_payload_readiness_flag(gctx, src_rank, local_slot));
    }
    __builtin_amdgcn_wave_barrier();

    if (in_flat_part == 0 && lane == 0) {
      const int32_t* src_token_id_ptr = gctx.dispatch_layout.slot.src_id(slot);
      const int32_t src_token_id =
          (src_rank != rank_me) ? ep_load<true>(src_token_id_ptr) : *src_token_id_ptr;
      gctx.d_recv_src_token_pos[flat] =
          src_rank * (gctx.world_size * gctx.max_num_inp_token_per_rank) + src_token_id;
    }

    const size_t dispatch_byte_offset =
        dispatch_col_offset * static_cast<size_t>(dispatch_elem_size);
    const void* src_hidden =
        static_cast<const char*>(gctx.dispatch_layout.slot.hidden(slot)) + dispatch_byte_offset;
    void* dst_hidden = reinterpret_cast<char*>(gctx.d_recv_hidden) +
                       static_cast<size_t>(flat) * dispatch_hidden_bytes + dispatch_byte_offset;
    const int hidden_copy_bytes =
        static_cast<int>(dispatch_col_count * static_cast<size_t>(dispatch_elem_size));
    if (src_rank != rank_me) {
      copy_hidden_payload_block<false, true>(src_hidden, dst_hidden, hidden_copy_bytes, lane,
                                             EP_WAVE_SIZE);
    } else {
      copy_hidden_payload_block(src_hidden, dst_hidden, hidden_copy_bytes, lane, EP_WAVE_SIZE);
    }

    if (in_flat_part == 0) {
      const int32_t* src_idx = gctx.dispatch_layout.slot.indices(slot);
      int32_t* dst_idx = gctx.d_recv_indices + flat * topk;
      const float* src_w = gctx.dispatch_layout.slot.weights(slot);
      float* dst_w = gctx.d_recv_weights + flat * topk;
      for (int k = lane; k < topk; k += EP_WAVE_SIZE) {
        if (src_rank != rank_me) {
          dst_idx[k] = ep_load<true>(&src_idx[k]);
          dst_w[k] = ep_load<true>(&src_w[k]);
        } else {
          dst_idx[k] = src_idx[k];
          dst_w[k] = src_w[k];
        }
      }
    }

    if (in_flat_part == 0 && gctx.dispatch_scale_dim > 0 && gctx.d_recv_scales != nullptr) {
      const int scale_bytes = gctx.dispatch_scale_dim * gctx.dispatch_scale_type_size;
      const void* src_s = gctx.dispatch_layout.slot.scales(slot);
      void* dst_s =
          reinterpret_cast<char*>(gctx.d_recv_scales) + static_cast<size_t>(flat) * scale_bytes;
      if (src_rank != rank_me) {
        copy_bytes_block<false, true>(src_s, dst_s, scale_bytes, lane, EP_WAVE_SIZE);
      } else {
        copy_bytes_block(src_s, dst_s, scale_bytes, lane, EP_WAVE_SIZE);
      }
    }
  }

  __syncthreads();

  if (bid == 0 && threadIdx.x == 0) {
    dispatch_debug_store_time(debug_time_buf, dispatch_debug_slot::unpack_done);
  }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

static void launch_dispatch_round(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_tokens,
                                  const int32_t* d_expert_indices, const float* d_weights,
                                  const void* d_scales, int num_tokens, hipStream_t stream) {
  if (num_tokens < 0 || num_tokens > ctx.config().max_num_inp_token_per_rank) {
    fprintf(stderr,
            "kiwi::ep::dispatch_enqueue: num_tokens=%d exceeds "
            "max_num_inp_token_per_rank=%d\n",
            num_tokens, ctx.config().max_num_inp_token_per_rank);
    abort();
  }

  // Even empty dispatches must run the signal phase so peer ranks do not
  // block waiting for MORI's count completion marker.
  const runtime_defaults_t launch_policy = ctx.config().launch_policy_for(num_tokens);
  const int launch_num_blocks = launch_policy.num_blocks;
  const int provisioned_num_blocks = ctx.config().provisioned_num_blocks();
  if (launch_num_blocks <= 0 || launch_num_blocks > provisioned_num_blocks) {
    fprintf(stderr,
            "kiwi::ep::dispatch_enqueue: launch_num_blocks=%d exceeds "
            "provisioned_num_blocks=%d\n",
            launch_num_blocks, provisioned_num_blocks);
    abort();
  }

  if (launch_policy.rcomp_backend == rcomp_backend_t::HYBRID) {
    dispatch<true><<<launch_num_blocks, EP_THREADS_PER_BLOCK, 0, stream>>>(
        gctx, d_tokens, d_expert_indices, d_weights, d_scales, num_tokens,
        ctx.debug_timing_enabled() ? ctx.debug_time_buf() : nullptr,
        ctx.debug_timing_enabled() ? ctx.debug_time_meta() : nullptr);
  } else {
    dispatch<false><<<launch_num_blocks, EP_THREADS_PER_BLOCK, 0, stream>>>(
        gctx, d_tokens, d_expert_indices, d_weights, d_scales, num_tokens,
        ctx.debug_timing_enabled() ? ctx.debug_time_buf() : nullptr,
        ctx.debug_timing_enabled() ? ctx.debug_time_meta() : nullptr);
  }
  HIP_CHECK(hipGetLastError());
}

void dispatch_enqueue(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_tokens,
                      const int32_t* d_expert_indices, const float* d_weights, const void* d_scales,
                      int num_tokens, hipStream_t stream) {
  launch_dispatch_round(ctx, gctx, d_tokens, d_expert_indices, d_weights, d_scales, num_tokens,
                        stream);
}

}  // namespace mori::kiwi::ep
