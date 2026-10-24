#pragma once

#include <cstddef>
#include "comm/transport_descriptors.hip.hpp"
#include "comm/context.hip.hpp"
#include "mori/kiwi/invoke/invoke.hpp"

namespace mori::kiwi::ep {

class ep_gcontext_t;

using ep_multicast_cmd_desc_t = internal::post_put_cmd_desc_multicast_t;
// invoke_shared() reads whole 8-byte words.
inline constexpr size_t EP_MULTICAST_DESC_STAGING_BYTES =
    (sizeof(ep_multicast_cmd_desc_t) + 7) / 8 * 8;

namespace internal {
void register_post_put_callbacks(ep_comm_context_t& comm_ctx,
                                   uint8_t& post_put_cb_id,
                                   uint8_t& post_put_multicast_cb_id);
} // namespace internal

__device__ inline void post_put(
        const ep_gcontext_t& gctx, ep_comm_handle_t& handle,
        int heap_id, int rank, void* local_buffer, size_t size,
        uintptr_t remote_disp, uintptr_t readiness_flags_offset,
        int readiness_slot) {
    internal::post_put_cmd_desc_t desc;
    desc.ep_context = gctx.host_context;
    desc.local_buffer = reinterpret_cast<uintptr_t>(local_buffer);
    desc.size = size;
    desc.remote_disp = remote_disp;
    desc.readiness_flags_offset = readiness_flags_offset;
    desc.rank = rank;
    desc.heap_id = heap_id;
    desc.readiness_slot = readiness_slot;
    handle.post_desc(gctx.post_put_cb_id, desc);
}

__device__ inline void post_put_multicast(
        const ep_gcontext_t& gctx, ep_comm_handle_t& handle,
        const internal::post_put_cmd_desc_multicast_t* desc) {
    handle.post_desc_shared(gctx.post_put_multicast_cb_id,
                            desc,
                            sizeof(internal::post_put_cmd_desc_multicast_t));
}

__device__ inline ep_multicast_cmd_desc_t*
ep_multicast_desc_from_stage(uint64_t* stage) {
    return reinterpret_cast<ep_multicast_cmd_desc_t*>(stage);
}

__device__ inline void ep_multicast_desc_set_rank_disp(
        ep_multicast_cmd_desc_t* desc, int idx, int rank,
        uintptr_t remote_disp, int readiness_slot) {
    desc->ranks[idx] = rank;
    desc->remote_disps[idx] = remote_disp;
    desc->readiness_slots[idx] = readiness_slot;
}

__device__ inline void ep_multicast_desc_set_scalars(
        const ep_gcontext_t& gctx, ep_multicast_cmd_desc_t* desc,
        void* local_buffer, size_t size, int heap_id, int n,
        uintptr_t readiness_flags_offset) {
    desc->ep_context = gctx.host_context;
    desc->local_buffer = reinterpret_cast<uintptr_t>(local_buffer);
    desc->size = size;
    desc->readiness_flags_offset = readiness_flags_offset;
    desc->heap_id = heap_id;
    desc->n = n;
}

__device__ inline void ep_post_put(
        const ep_gcontext_t& gctx, ep_comm_handle_t& handle,
        int heap_id, int rank, void* local_buffer, size_t size,
        uintptr_t remote_disp, uintptr_t readiness_flags_offset,
        int readiness_slot) {
    post_put(gctx, handle, heap_id, rank,
                        local_buffer, size, remote_disp,
                        readiness_flags_offset, readiness_slot);
}

} // namespace kiwi::ep
