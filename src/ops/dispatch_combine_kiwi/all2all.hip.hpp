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
#include <vector>

#include "assert.hpp"
#include "comm/common_descriptors.hip.hpp"
#include "comm/context.hip.hpp"
#include "ep_config.hpp"
#include "ep_gcontext.hip.hpp"
#include "memory_scope.hip.hpp"

namespace mori::kiwi::ep {

// Host/proxy-side callback for the EP hybrid all-to-all.
void ep_all2all_callback(internal::ep_cmd_desc_all2all_t desc);

struct ep_all2all_ticket_t {
  uint32_t protocol_epoch = 0;
};
static_assert(sizeof(ep_all2all_ticket_t) == sizeof(uint32_t),
              "all-to-all ticket must contain only its protocol epoch");

__device__ inline void post_all2all_exchange(const ep_gcontext_t& gctx, ep_comm_handle_t& handle,
                                             const uint32_t* send_values, uint32_t protocol_epoch,
                                             uint64_t* egress, uint64_t* own_ingress,
                                             size_t ingress_offset) {
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  KIWI_EP_ASSERT(protocol_epoch != 0, "hybrid all-to-all protocol epoch is zero");

  for (int dest = lane; dest < gctx.world_size; dest += EP_WAVE_SIZE) {
    const uint32_t value = send_values != nullptr ? send_values[dest] : 0;
    const uint64_t packed = (static_cast<uint64_t>(protocol_epoch) << 32) | value;
    const int target_node = dest / gctx.count_alltoall_local_size;
    if (target_node == gctx.count_alltoall_node_idx) {
      const int dest_lane = dest % gctx.count_alltoall_local_size;
      const size_t cell =
          static_cast<size_t>(gctx.count_alltoall_node_idx) * gctx.count_alltoall_local_size +
          dest_lane;
      ep_store_system_scope(&own_ingress[cell], packed);
    } else {
      ep_store_system_scope(&egress[dest], packed);
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();

  if (gctx.count_alltoall_num_nodes > 1) {
    const size_t ingress_cell =
        static_cast<size_t>(gctx.count_alltoall_node_idx) * gctx.count_alltoall_local_size;
    internal::ep_cmd_desc_all2all_t desc;
    desc.ep_context = gctx.host_context;
    desc.send_buf = reinterpret_cast<uintptr_t>(egress);
    desc.remote_disp = ingress_offset + ingress_cell * sizeof(uint64_t);
    handle.post_desc(gctx.comm.all2all_cb_id(), desc);
  }
}

__device__ inline void wait_all2all_exchange(const ep_gcontext_t& gctx, uint32_t protocol_epoch,
                                             size_t ingress_offset, uint64_t* recv_values) {
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  for (int src_rank = lane; src_rank < gctx.world_size; src_rank += EP_WAVE_SIZE) {
    const int source_node = src_rank / gctx.count_alltoall_local_size;
    const int source_lane = src_rank % gctx.count_alltoall_local_size;
    const size_t cell = static_cast<size_t>(source_node) * gctx.count_alltoall_local_size +
                        gctx.count_alltoall_lane_idx;
    const uint64_t* ingress = gctx.alltoall_peer_ingress(source_lane, ingress_offset);
    uint64_t packed = 0;
    do {
      packed = ep_load_system_scope<uint64_t>(&ingress[cell]);
      const uint32_t observed_epoch = static_cast<uint32_t>(packed >> 32);
      if (observed_epoch != protocol_epoch) {
        __builtin_amdgcn_s_sleep(1);
      }
    } while (static_cast<uint32_t>(packed >> 32) != protocol_epoch);
    if (recv_values != nullptr) {
      recv_values[src_rank] = static_cast<uint32_t>(packed);
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();
}

__device__ inline ep_all2all_ticket_t post_all2all(const ep_gcontext_t& gctx,
                                                   ep_comm_handle_t& handle,
                                                   const uint32_t* send_counts) {
  const uint32_t protocol_epoch = ep_protocol_epoch_begin(gctx);
  post_all2all_exchange(gctx, handle, send_counts, protocol_epoch, gctx.count_alltoall_egress(),
                        gctx.count_alltoall_ingress(), gctx.count_alltoall_ingress_offset);
  return ep_all2all_ticket_t{protocol_epoch};
}

__device__ inline void wait_all2all(const ep_gcontext_t& gctx, ep_all2all_ticket_t ticket,
                                    uint64_t* recv_counts) {
  wait_all2all_exchange(gctx, ticket.protocol_epoch, gctx.count_alltoall_ingress_offset,
                        recv_counts);
}

// Host-side callback state plus same-node GPU heap mappings for the EP
// all-to-all. The receive path itself is entirely GPU-side.
class all2all_t {
 public:
  all2all_t(const config_t& cfg, ep_comm_context_t& comm_ctx, char* heap_base, int heap_id,
            const std::vector<uintptr_t>* reusable_peer_heap_bases);
  ~all2all_t();

  all2all_t(const all2all_t&) = delete;
  all2all_t& operator=(const all2all_t&) = delete;

  void start(internal::ep_cmd_desc_all2all_t desc);

  int local_size() const { return local_size_; }
  int num_nodes() const { return num_nodes_; }
  int node_idx() const { return node_idx_; }
  int lane_idx() const { return lane_idx_; }
  const uintptr_t* device_local_peer_heap_bases() const { return d_local_peer_heap_bases_; }

 private:
  void initialize(char* heap_base, const std::vector<uintptr_t>* reusable_peer_heap_bases);
  void finalize();

  const config_t& cfg_;
  ep_comm_context_t& comm_ctx_;
  int heap_id_ = -1;
  std::vector<void*> owned_peer_heap_ipc_mappings_;
  uintptr_t* d_local_peer_heap_bases_ = nullptr;
  int local_size_ = 0;
  int num_nodes_ = 0;
  int node_idx_ = -1;
  int lane_idx_ = -1;
};

}  // namespace mori::kiwi::ep
