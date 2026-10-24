#include "comm/transport_descriptors.hip.hpp"
#include "comm/context.hip.hpp"
#include "comm/proxy_network.hip.hpp"
#include "ep_context.hip.hpp"

namespace mori::kiwi::ep {
namespace internal {
namespace {

void post_raw_payload_and_readiness(
        int rank, void* local_buffer, size_t size, const rdma::mr_t& mr,
        uintptr_t remote_disp, uintptr_t readiness_disp,
        const rdma::rmr_t& rmr) {
    // Both PUTs use this proxy device's QP to `rank`. Do not leave this
    // callback until the payload is posted and its later readiness signal is
    // posted, preserving payload-before-signal ordering.
    internal::proxy_network_put(rank, local_buffer, size, mr, remote_disp, rmr);

    const uint8_t ready = 1;
    internal::proxy_network_put_inline(rank, &ready, sizeof(ready),
                                       readiness_disp, rmr);
}

void post_put_callback(post_put_cmd_desc_t desc) {
    const auto& heap = desc.ep_context->heap_table()[desc.heap_id];
    const rdma::rmr_t& rmr = heap.rmrs[desc.rank];

    const size_t readiness_index =
        desc.ep_context->config().payload_readiness_flag_index(
            desc.ep_context->rank_me(), desc.readiness_slot);
    const uintptr_t readiness_disp =
        desc.readiness_flags_offset + readiness_index * sizeof(uint8_t);
    post_raw_payload_and_readiness(
        desc.rank, reinterpret_cast<void*>(desc.local_buffer), desc.size,
        heap.mr, desc.remote_disp, readiness_disp, rmr);
}

void post_put_multicast_callback(post_put_cmd_desc_multicast_t desc) {
    const auto& heap = desc.ep_context->heap_table()[desc.heap_id];
    const int rank_me = desc.ep_context->rank_me();

    for (int i = 0; i < desc.n; i++) {
        const rdma::rmr_t& rmr = heap.rmrs[desc.ranks[i]];
        const size_t readiness_index =
            desc.ep_context->config().payload_readiness_flag_index(
                rank_me, desc.readiness_slots[i]);
        const uintptr_t readiness_disp =
            desc.readiness_flags_offset + readiness_index * sizeof(uint8_t);
        post_raw_payload_and_readiness(
            desc.ranks[i], reinterpret_cast<void*>(desc.local_buffer),
            desc.size, heap.mr, desc.remote_disps[i], readiness_disp, rmr);
    }
}

} // namespace

void register_post_put_callbacks(ep_comm_context_t& comm_ctx,
                                   uint8_t& post_put_cb_id,
                                   uint8_t& post_put_multicast_cb_id) {
    post_put_cb_id = comm_ctx.register_callback<post_put_cmd_desc_t>(
        post_put_callback);

    post_put_multicast_cb_id =
        comm_ctx.register_callback<post_put_cmd_desc_multicast_t>(
        post_put_multicast_callback);
}

} // namespace internal
} // namespace kiwi::ep
