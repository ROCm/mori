#pragma once

#include <cstdint>

#include <mori/kiwi/queue/queue.hpp>

namespace mori::kiwi::ep {
class ep_context_t;

namespace internal {

// Shared fanout cap for EP transport multicast descriptors.
inline constexpr int EP_COMM_MAX_DESTS = 8;

struct ep_cmd_desc_all2all_t {
    ep_context_t* ep_context;
    uintptr_t send_buf;
    uintptr_t remote_disp;
};

static_assert(sizeof(ep_cmd_desc_all2all_t) <= kiwi::queue::kMaxMessageBytes,
              "ep_cmd_desc_all2all_t must fit in one invoke message");

struct ep_cmd_desc_active_queues_t {
    int active_count;
};

static_assert(sizeof(ep_cmd_desc_active_queues_t) <= kiwi::queue::kMaxMessageBytes,
              "ep_cmd_desc_active_queues_t must fit in one invoke message");

} // namespace internal
} // namespace kiwi::ep
