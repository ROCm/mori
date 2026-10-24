#pragma once

#include <cstdint>

#include "comm/common_descriptors.hip.hpp"

namespace mori::kiwi::ep {
class ep_context_t;

namespace internal {

struct post_put_cmd_desc_t {
    ep_context_t* ep_context;
    uintptr_t local_buffer;
    uint64_t size;
    uintptr_t remote_disp;
    uintptr_t readiness_flags_offset;
    int rank;
    int heap_id;
    int readiness_slot;
};

struct post_put_cmd_desc_multicast_t {
    ep_context_t* ep_context;
    uintptr_t local_buffer;
    uint64_t size;
    uintptr_t remote_disps[EP_COMM_MAX_DESTS];
    uintptr_t readiness_flags_offset;
    int heap_id;
    int n;
    int readiness_slots[EP_COMM_MAX_DESTS];
    int ranks[EP_COMM_MAX_DESTS];
};

static_assert(sizeof(post_put_cmd_desc_t) <= kiwi::queue::kMaxMessageBytes,
              "post_put_cmd_desc_t must fit in one invoke message");
static_assert(sizeof(post_put_cmd_desc_multicast_t) <= kiwi::queue::kMaxMessageBytes,
              "post_put_cmd_desc_multicast_t must fit in one invoke message");

} // namespace internal

} // namespace kiwi::ep
