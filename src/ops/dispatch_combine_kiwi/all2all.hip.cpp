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
#include "all2all.hip.hpp"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "assert.hpp"
#include "comm/proxy_network.hip.hpp"
#include "hip_check.hpp"

namespace mori::kiwi::ep {

namespace {

constexpr size_t EP_HOSTNAME_BYTES = 256;

}  // namespace

all2all_t::all2all_t(const config_t& cfg, ep_comm_context_t& comm_ctx, char* heap_base, int heap_id,
                     const std::vector<uintptr_t>* reusable_peer_heap_bases)
    : cfg_(cfg), comm_ctx_(comm_ctx), heap_id_(heap_id) {
  initialize(heap_base, reusable_peer_heap_bases);
}

all2all_t::~all2all_t() { finalize(); }

void all2all_t::initialize(char* heap_base,
                           const std::vector<uintptr_t>* reusable_peer_heap_bases) {
  const int rank_n = comm_ctx_.rank_n();
  const int rank_me = comm_ctx_.rank_me();
  KIWI_EP_ASSERT(rank_n == cfg_.world_size,
                 "config.world_size does not match bootstrap rank count");
  KIWI_EP_ASSERT(heap_base != nullptr && heap_id_ >= 0,
                 "hybrid count all-to-all requires a symmetric GPU heap");

  std::array<char, EP_HOSTNAME_BYTES> local_hostname{};
  KIWI_EP_ASSERT(gethostname(local_hostname.data(), local_hostname.size()) == 0,
                 "gethostname failed during hybrid count all-to-all initialization");
  local_hostname.back() = '\0';

  std::vector<char> all_hostnames(static_cast<size_t>(rank_n) * EP_HOSTNAME_BYTES, '\0');
  comm_ctx_.bootstrap().Allgather(local_hostname.data(), all_hostnames.data(), EP_HOSTNAME_BYTES);

  std::vector<std::string> hostnames(static_cast<size_t>(rank_n));
  for (int rank = 0; rank < rank_n; ++rank) {
    const char* hostname = all_hostnames.data() + static_cast<size_t>(rank) * EP_HOSTNAME_BYTES;
    hostnames[static_cast<size_t>(rank)] =
        std::string(hostname, strnlen(hostname, EP_HOSTNAME_BYTES));
  }

  int group_size = 0;
  int node_count = 0;
  for (int begin = 0; begin < rank_n;) {
    int end = begin + 1;
    while (end < rank_n &&
           hostnames[static_cast<size_t>(end)] == hostnames[static_cast<size_t>(begin)]) {
      ++end;
    }
    const int size = end - begin;
    if (group_size == 0) {
      group_size = size;
    } else {
      KIWI_EP_ASSERT(size == group_size, "ranks must form equal-size contiguous hostname groups");
    }
    for (int prior = 0; prior < begin; ++prior) {
      KIWI_EP_ASSERT(hostnames[static_cast<size_t>(prior)] != hostnames[static_cast<size_t>(begin)],
                     "ranks for one hostname are not contiguous");
    }
    ++node_count;
    begin = end;
  }
  KIWI_EP_ASSERT(group_size > 0 && node_count * group_size == rank_n,
                 "invalid contiguous node topology");

  local_size_ = group_size;
  num_nodes_ = node_count;
  node_idx_ = rank_me / group_size;
  lane_idx_ = rank_me % group_size;
  const int local_begin = node_idx_ * group_size;

  std::vector<uintptr_t> local_peer_heap_bases(static_cast<size_t>(local_size_), uintptr_t{0});
  owned_peer_heap_ipc_mappings_.assign(static_cast<size_t>(local_size_), nullptr);
  local_peer_heap_bases[static_cast<size_t>(lane_idx_)] = reinterpret_cast<uintptr_t>(heap_base);

  if (reusable_peer_heap_bases != nullptr) {
    KIWI_EP_ASSERT(reusable_peer_heap_bases->size() == static_cast<size_t>(rank_n),
                   "reusable peer heap table has the wrong size");
    for (int lane = 0; lane < local_size_; ++lane) {
      if (lane == lane_idx_) {
        continue;
      }
      const int peer_rank = local_begin + lane;
      const uintptr_t peer_base = (*reusable_peer_heap_bases)[static_cast<size_t>(peer_rank)];
      KIWI_EP_ASSERT(peer_base != 0, "reusable peer heap table is missing a local peer mapping");
      local_peer_heap_bases[static_cast<size_t>(lane)] = peer_base;
    }
  } else {
    hipIpcMemHandle_t local_heap_handle{};
    HIP_CHECK(hipIpcGetMemHandle(&local_heap_handle, heap_base));
    std::vector<hipIpcMemHandle_t> peer_heap_handles(static_cast<size_t>(rank_n));
    comm_ctx_.bootstrap().Allgather(&local_heap_handle, peer_heap_handles.data(),
                                    sizeof(hipIpcMemHandle_t));

    for (int lane = 0; lane < local_size_; ++lane) {
      if (lane == lane_idx_) {
        continue;
      }
      const int peer_rank = local_begin + lane;
      void* peer_heap = nullptr;
      HIP_CHECK(hipIpcOpenMemHandle(&peer_heap, peer_heap_handles[static_cast<size_t>(peer_rank)],
                                    hipIpcMemLazyEnablePeerAccess));
      KIWI_EP_ASSERT(peer_heap != nullptr, "hipIpcOpenMemHandle returned a null peer heap");
      local_peer_heap_bases[static_cast<size_t>(lane)] = reinterpret_cast<uintptr_t>(peer_heap);
      owned_peer_heap_ipc_mappings_[static_cast<size_t>(lane)] = peer_heap;
    }
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_local_peer_heap_bases_),
                      static_cast<size_t>(local_size_) * sizeof(uintptr_t)));
  HIP_CHECK(hipMemcpy(d_local_peer_heap_bases_, local_peer_heap_bases.data(),
                      static_cast<size_t>(local_size_) * sizeof(uintptr_t), hipMemcpyHostToDevice));

  if (rank_me == 0) {
    std::fprintf(stderr,
                 "kiwi_ep hybrid count all-to-all nodes=%d local_size=%d "
                 "storage=gpu-symmetric-heap transpose=receive-side "
                 "lane_mapping=rank%%local_size\n",
                 num_nodes_, local_size_);
  }
}

void all2all_t::finalize() {
  if (d_local_peer_heap_bases_ != nullptr) {
    (void)hipFree(d_local_peer_heap_bases_);
    d_local_peer_heap_bases_ = nullptr;
  }
  for (void* mapping : owned_peer_heap_ipc_mappings_) {
    if (mapping != nullptr) {
      (void)hipIpcCloseMemHandle(mapping);
    }
  }
  owned_peer_heap_ipc_mappings_.clear();
}

void all2all_t::start(internal::ep_cmd_desc_all2all_t desc) {
  KIWI_EP_ASSERT(ep_current_proxy_thread_index() >= 0 && internal::tl_device != nullptr,
                 "hybrid callback ran outside an EP proxy thread");
  KIWI_EP_ASSERT(desc.send_buf != 0, "hybrid all-to-all egress buffer is null");
  KIWI_EP_ASSERT(desc.remote_disp != 0, "hybrid all-to-all ingress displacement is zero");

  const auto& heap = comm_ctx_.heap_table()[static_cast<size_t>(heap_id_)];
  const size_t block_bytes = static_cast<size_t>(local_size_) * sizeof(uint64_t);
  for (int target_node = 0; target_node < num_nodes_; ++target_node) {
    if (target_node == node_idx_) {
      continue;
    }
    const int target_rank = target_node * local_size_ + lane_idx_;
    void* source =
        reinterpret_cast<uint64_t*>(desc.send_buf) + static_cast<size_t>(target_node) * local_size_;
    internal::proxy_network_put(target_rank, source, block_bytes, heap.mr, desc.remote_disp,
                                heap.rmrs[static_cast<size_t>(target_rank)]);
  }
}

}  // namespace mori::kiwi::ep
