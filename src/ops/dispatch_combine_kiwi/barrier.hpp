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

#include <cstdint>

#include "all2all.hip.hpp"
#include "comm/context.hip.hpp"
#include "ep_gcontext.hip.hpp"
#include "workspace/device_workspace.hpp"

namespace mori::kiwi::ep {

struct ep_barrier_ticket_t {
  uint32_t protocol_epoch = 0;
};
static_assert(sizeof(ep_barrier_ticket_t) == sizeof(uint32_t),
              "barrier ticket must contain only its protocol epoch");

// Start the one EP cross-GPU execution barrier that closes the current
// protocol epoch. Payload readiness and memory completion remain the caller's
// responsibility.
template <typename GContext>
__device__ inline ep_barrier_ticket_t post_barrier(const GContext& gctx, ep_comm_handle_t& handle) {
  const uint32_t target = ep_protocol_epoch_current(gctx);

  const size_t ingress_offset = gctx.barrier_alltoall_ingress_epoch_offset(target);
  post_all2all_exchange(gctx, handle, nullptr, target, gctx.barrier_alltoall_egress(target),
                        gctx.barrier_alltoall_ingress(target), ingress_offset);
  return ep_barrier_ticket_t{target};
}

template <typename GContext>
__device__ inline void wait_barrier(const GContext& gctx, ep_barrier_ticket_t ticket) {
  wait_all2all_exchange(gctx, ticket.protocol_epoch,
                        gctx.barrier_alltoall_ingress_epoch_offset(ticket.protocol_epoch), nullptr);
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  if (lane == 0) {
    __atomic_store_n(gctx.device_workspace.barrier_counter(),
                     static_cast<uint64_t>(ticket.protocol_epoch), __ATOMIC_RELAXED);
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();
}

}  // namespace mori::kiwi::ep
