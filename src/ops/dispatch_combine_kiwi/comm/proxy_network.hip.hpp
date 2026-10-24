#pragma once

#include <cstddef>
#include <cstdint>

#include "mori/kiwi/rdma/ibverbs.hpp"

namespace mori::kiwi::ep::internal {

void proxy_network_progress();

// Every raw WRITE is signaled; record it so teardown drains its completion.
void proxy_network_record_posted_write();

// Posts a signaled RDMA WRITE on this proxy thread's device, progressing the
// CQ while the target QP's send queue is full.
void proxy_network_put(int rank, const void* local, size_t size,
                       const rdma::mr_t& mr, uint64_t remote_disp,
                       const rdma::rmr_t& rmr);

// Inline variant: `local` may be reused as soon as this returns.
void proxy_network_put_inline(int rank, const void* local, size_t size,
                              uint64_t remote_disp, const rdma::rmr_t& rmr);

} // namespace mori::kiwi::ep::internal
