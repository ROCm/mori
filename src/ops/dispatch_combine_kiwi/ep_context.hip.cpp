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
#include "ep_context.hip.hpp"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "all2all.hip.hpp"
#include "assert.hpp"
#include "hip_check.hpp"

namespace mori::kiwi::ep {

namespace {

constexpr size_t EP_HOSTNAME_BYTES = 256;
constexpr size_t EP_PCI_BUS_ID_BYTES = 32;

std::string string_from_fixed_buffer(const char* data, size_t capacity) {
  size_t len = 0;
  while (len < capacity && data[len] != '\0') {
    ++len;
  }
  return std::string(data, len);
}

std::string hostname_from_fixed_buffer(const char* data) {
  return string_from_fixed_buffer(data, EP_HOSTNAME_BYTES);
}

[[noreturn]] void rcomp_hybrid_init_abort(int rank, int peer_rank,
                                          const std::string& local_hostname,
                                          const std::string& peer_hostname, int local_hip_device,
                                          const std::string& local_pci_bus_id, int peer_hip_device,
                                          const std::string& peer_pci_bus_id, const char* step,
                                          hipError_t hip_error) {
  if (peer_rank >= 0) {
    if (hip_error == hipSuccess) {
      fprintf(stderr,
              "mori::kiwi::ep::ep_context_t: rcomp_backend=hybrid rank %d "
              "cannot initialize same-host peer rank %d "
              "(hostnames '%s'/'%s', local HIP device %d PCI '%s', "
              "peer PCI '%s' resolved to local HIP device %d): %s\n",
              rank, peer_rank, local_hostname.c_str(), peer_hostname.c_str(), local_hip_device,
              local_pci_bus_id.c_str(), peer_pci_bus_id.c_str(), peer_hip_device, step);
    } else {
      fprintf(stderr,
              "mori::kiwi::ep::ep_context_t: rcomp_backend=hybrid rank %d "
              "cannot initialize same-host peer rank %d "
              "(hostnames '%s'/'%s', local HIP device %d PCI '%s', "
              "peer PCI '%s' resolved to local HIP device %d): %s (%s)\n",
              rank, peer_rank, local_hostname.c_str(), peer_hostname.c_str(), local_hip_device,
              local_pci_bus_id.c_str(), peer_pci_bus_id.c_str(), peer_hip_device, step,
              hipGetErrorString(hip_error));
    }
  } else {
    if (hip_error == hipSuccess) {
      fprintf(stderr,
              "mori::kiwi::ep::ep_context_t: rcomp_backend=hybrid rank %d "
              "cannot initialize HIP IPC on hostname '%s' local HIP "
              "device %d PCI '%s': %s\n",
              rank, local_hostname.c_str(), local_hip_device, local_pci_bus_id.c_str(), step);
    } else {
      fprintf(stderr,
              "mori::kiwi::ep::ep_context_t: rcomp_backend=hybrid rank %d "
              "cannot initialize HIP IPC on hostname '%s' local HIP "
              "device %d PCI '%s': %s (%s)\n",
              rank, local_hostname.c_str(), local_hip_device, local_pci_bus_id.c_str(), step,
              hipGetErrorString(hip_error));
    }
  }
  abort();
}

}  // namespace

ep_context_t::ep_context_t(const config_t& cfg, std::shared_ptr<rdma::bootstrap_t> bootstrap)
    : cfg_(cfg), device_workspace_layout_(cfg) {
  if (cfg_.topk > MAX_TOPK_SUPPORTED) {
    fprintf(stderr, "mori::kiwi::ep::ep_context_t: topk=%d exceeds MAX_TOPK_SUPPORTED=%d\n",
            cfg_.topk, MAX_TOPK_SUPPORTED);
    abort();
  }
  assert(cfg_.topk <= MAX_TOPK_SUPPORTED);
  if (cfg_.queue_capacity <= 0 || (cfg_.queue_capacity & (cfg_.queue_capacity - 1)) != 0) {
    throw std::invalid_argument(
        "mori::kiwi::ep::ep_context_t: queue_capacity=" + std::to_string(cfg_.queue_capacity) +
        " is not a non-zero power of two");
  }
  if (cfg_.num_proxy_threads <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_context_t: num_proxy_threads=" +
                                std::to_string(cfg_.num_proxy_threads) + " must be positive");
  }
  if (cfg_.max_num_inp_token_per_rank <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_context_t: max_num_inp_token_per_rank=" +
                                std::to_string(cfg_.max_num_inp_token_per_rank) +
                                " must be positive");
  }
  const int provisioned_num_blocks = cfg_.provisioned_num_blocks();
  if (provisioned_num_blocks <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_context_t: provisioned_num_blocks=" +
                                std::to_string(provisioned_num_blocks) + " must be positive");
  }
  rcomp_ipc_resources_provisioned_ = cfg_.rcomp_ipc_resources_required();
  // Each proxy thread owns exactly one RDMA device/CQ and polls its raw
  // network completions, so concurrent ownership would race. num_devices is not an independent
  // tunable: it must equal num_proxy_threads.
  cfg_.num_devices = cfg_.num_proxy_threads;
  // Dispatch/combine use dynamic wavefront assignment inside each
  // EP_THREADS_PER_BLOCK-thread workgroup. Give each compile-time wavefront
  // a dedicated invoke endpoint (one SPSC queue per producer wavefront);
  // the consumer side then drains them across num_proxy_threads devices.
  comm_ctx_ = std::make_unique<ep_comm_context_t>(std::move(bootstrap), cfg_.num_queues(),
                                                  cfg_.num_devices, cfg_.device_name,
                                                  static_cast<size_t>(cfg_.queue_capacity));

  if (rcomp_ipc_resources_provisioned_) {
    const int rank_n = comm_ctx_->rank_n();
    KIWI_EP_ASSERT(rank_n == cfg_.world_size,
                   "mori::kiwi::ep::ep_context_t: config.world_size does "
                   "not match bootstrap rank count");

    std::array<char, EP_HOSTNAME_BYTES> local_hostname{};
    KIWI_EP_ASSERT(gethostname(local_hostname.data(), local_hostname.size()) == 0,
                   "mori::kiwi::ep::ep_context_t: gethostname failed");
    local_hostname.back() = '\0';
    const std::string local_hostname_string = hostname_from_fixed_buffer(local_hostname.data());

    std::vector<char> all_hostnames(static_cast<size_t>(rank_n) * EP_HOSTNAME_BYTES, '\0');
    comm_ctx_->bootstrap().allgather(local_hostname.data(), all_hostnames.data(),
                                     EP_HOSTNAME_BYTES);

    rcomp_peer_is_local_.assign(static_cast<size_t>(rank_n), 0);
    rcomp_peer_hostnames_.resize(static_cast<size_t>(rank_n));
    for (int rank = 0; rank < rank_n; ++rank) {
      rcomp_peer_hostnames_[static_cast<size_t>(rank)] = hostname_from_fixed_buffer(
          all_hostnames.data() + static_cast<size_t>(rank) * EP_HOSTNAME_BYTES);
      rcomp_peer_is_local_[static_cast<size_t>(rank)] =
          (rcomp_peer_hostnames_[static_cast<size_t>(rank)] == local_hostname_string) ? 1 : 0;
    }

    const int rank_me = comm_ctx_->rank_me();
    int local_hip_device = 0;
    HIP_CHECK(hipGetDevice(&local_hip_device));
    std::array<char, EP_PCI_BUS_ID_BYTES> local_hip_pci_bus_id{};
    hipError_t err =
        hipDeviceGetPCIBusId(local_hip_pci_bus_id.data(),
                             static_cast<int>(local_hip_pci_bus_id.size()), local_hip_device);
    if (err != hipSuccess) {
      rcomp_hybrid_init_abort(rank_me, -1, local_hostname_string, local_hostname_string,
                              local_hip_device, "<unavailable>", local_hip_device, "<unavailable>",
                              "hipDeviceGetPCIBusId for the local HIP device failed", err);
    }
    const std::string local_hip_pci_bus_id_string =
        string_from_fixed_buffer(local_hip_pci_bus_id.data(), local_hip_pci_bus_id.size());
    if (local_hip_pci_bus_id_string.empty()) {
      rcomp_hybrid_init_abort(rank_me, -1, local_hostname_string, local_hostname_string,
                              local_hip_device, "<empty>", local_hip_device, "<empty>",
                              "hipDeviceGetPCIBusId returned an empty local PCI Bus ID",
                              hipSuccess);
    }

    std::vector<char> all_hip_pci_bus_ids(static_cast<size_t>(rank_n) * EP_PCI_BUS_ID_BYTES, '\0');
    comm_ctx_->bootstrap().allgather(local_hip_pci_bus_id.data(), all_hip_pci_bus_ids.data(),
                                     EP_PCI_BUS_ID_BYTES);

    rcomp_peer_hip_pci_bus_ids_.resize(static_cast<size_t>(rank_n));
    for (int rank = 0; rank < rank_n; ++rank) {
      rcomp_peer_hip_pci_bus_ids_[static_cast<size_t>(rank)] = string_from_fixed_buffer(
          all_hip_pci_bus_ids.data() + static_cast<size_t>(rank) * EP_PCI_BUS_ID_BYTES,
          EP_PCI_BUS_ID_BYTES);
    }

    int visible_hip_device_count = 0;
    err = hipGetDeviceCount(&visible_hip_device_count);
    if (err != hipSuccess || visible_hip_device_count <= 0) {
      rcomp_hybrid_init_abort(rank_me, -1, local_hostname_string, local_hostname_string,
                              local_hip_device, local_hip_pci_bus_id_string, local_hip_device,
                              local_hip_pci_bus_id_string,
                              "hipGetDeviceCount could not enumerate local HIP devices", err);
    }

    rcomp_peer_is_ipc_.assign(static_cast<size_t>(rank_n), 0);
    rcomp_peer_hip_devices_.assign(static_cast<size_t>(rank_n), -1);
    for (int rank = 0; rank < rank_n; ++rank) {
      if (rank == rank_me || rcomp_peer_is_local_[static_cast<size_t>(rank)] == 0) {
        continue;
      }

      const std::string& peer_pci_bus_id = rcomp_peer_hip_pci_bus_ids_[static_cast<size_t>(rank)];
      if (peer_pci_bus_id.empty()) {
        rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                rcomp_peer_hostnames_[static_cast<size_t>(rank)], local_hip_device,
                                local_hip_pci_bus_id_string, -1, "<empty>",
                                "same-host peer published an empty PCI Bus ID", hipSuccess);
      }

      int peer_hip_device = -1;
      for (int device = 0; device < visible_hip_device_count; ++device) {
        std::array<char, EP_PCI_BUS_ID_BYTES> candidate_pci_bus_id{};
        err = hipDeviceGetPCIBusId(candidate_pci_bus_id.data(),
                                   static_cast<int>(candidate_pci_bus_id.size()), device);
        if (err != hipSuccess) {
          rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                  rcomp_peer_hostnames_[static_cast<size_t>(rank)],
                                  local_hip_device, local_hip_pci_bus_id_string, device,
                                  peer_pci_bus_id,
                                  "hipDeviceGetPCIBusId failed while resolving the "
                                  "peer's physical GPU identity",
                                  err);
        }
        if (string_from_fixed_buffer(candidate_pci_bus_id.data(), candidate_pci_bus_id.size()) ==
            peer_pci_bus_id) {
          if (peer_hip_device >= 0) {
            rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                    rcomp_peer_hostnames_[static_cast<size_t>(rank)],
                                    local_hip_device, local_hip_pci_bus_id_string, device,
                                    peer_pci_bus_id,
                                    "peer PCI Bus ID resolved to multiple local HIP "
                                    "device ordinals",
                                    hipSuccess);
          }
          peer_hip_device = device;
        }
      }
      if (peer_hip_device < 0) {
        rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                rcomp_peer_hostnames_[static_cast<size_t>(rank)], local_hip_device,
                                local_hip_pci_bus_id_string, -1, peer_pci_bus_id,
                                "same-host peer PCI Bus ID could not be resolved to a "
                                "locally visible HIP device",
                                hipSuccess);
      }
      rcomp_peer_hip_devices_[static_cast<size_t>(rank)] = peer_hip_device;

      int peer_access = 0;
      err = hipDeviceCanAccessPeer(&peer_access, local_hip_device, peer_hip_device);
      if (err != hipSuccess || peer_access == 0) {
        rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                rcomp_peer_hostnames_[static_cast<size_t>(rank)], local_hip_device,
                                local_hip_pci_bus_id_string, peer_hip_device, peer_pci_bus_id,
                                "hipDeviceCanAccessPeer reports no peer access", err);
      }

      int native_atomics = 0;
      err = hipDeviceGetP2PAttribute(&native_atomics, hipDevP2PAttrNativeAtomicSupported,
                                     local_hip_device, peer_hip_device);
      if (err != hipSuccess || native_atomics == 0) {
        rcomp_hybrid_init_abort(rank_me, rank, local_hostname_string,
                                rcomp_peer_hostnames_[static_cast<size_t>(rank)], local_hip_device,
                                local_hip_pci_bus_id_string, peer_hip_device, peer_pci_bus_id,
                                "hipDevP2PAttrNativeAtomicSupported is unavailable", err);
      }
      rcomp_peer_is_ipc_[static_cast<size_t>(rank)] = 1;
    }
  }

  internal::register_post_put_callbacks(*comm_ctx_, post_put_cb_id_, post_put_multicast_cb_id_);

  // GPU-local scratch/maps and collective control state. Payload
  // readiness and hybrid count-all-to-all transport state live in the
  // RDMA-visible symmetric GPU heap below.
  HIP_CHECK(hipMalloc(&d_device_workspace_, device_workspace_layout_.get_num_bytes()));
  HIP_CHECK(hipMemset(d_device_workspace_, 0, device_workspace_layout_.get_num_bytes()));

  active_queue_count_cb_id_ = comm_ctx_->register_callback<internal::ep_cmd_desc_active_queues_t>(
      [this](internal::ep_cmd_desc_active_queues_t desc) {
        comm_ctx_->publish_active_queue_count(desc.active_count);
      });

  ep_all2all_cb_id_ =
      comm_ctx_->register_callback<internal::ep_cmd_desc_all2all_t>(ep_all2all_callback);

  // Allocate symmetric heap (device memory, RDMA-registered)
  size_t heap_size = cfg_.total_heap_size();
  HIP_CHECK(hipMalloc(&d_heap_buf_, heap_size));
  HIP_CHECK(hipMemset(d_heap_buf_, 0, heap_size));
  heap_id_ = comm_ctx_->alloc_symmetric(d_heap_buf_, heap_size);
  if (rcomp_ipc_resources_provisioned_) {
    const int rank_me = comm_ctx_->rank_me();
    int local_hip_device = 0;
    HIP_CHECK(hipGetDevice(&local_hip_device));
    char hostname[EP_HOSTNAME_BYTES] = {};
    KIWI_EP_ASSERT(gethostname(hostname, sizeof(hostname)) == 0,
                   "mori::kiwi::ep::ep_context_t: gethostname failed");
    hostname[sizeof(hostname) - 1] = '\0';
    const std::string local_hostname = hostname_from_fixed_buffer(hostname);
    std::array<char, EP_PCI_BUS_ID_BYTES> local_hip_pci_bus_id{};
    hipError_t err =
        hipDeviceGetPCIBusId(local_hip_pci_bus_id.data(),
                             static_cast<int>(local_hip_pci_bus_id.size()), local_hip_device);
    if (err != hipSuccess) {
      rcomp_hybrid_init_abort(rank_me, -1, local_hostname, local_hostname, local_hip_device,
                              "<unavailable>", local_hip_device, "<unavailable>",
                              "hipDeviceGetPCIBusId for the local HIP device failed", err);
    }
    const std::string local_hip_pci_bus_id_string =
        string_from_fixed_buffer(local_hip_pci_bus_id.data(), local_hip_pci_bus_id.size());

    hipIpcMemHandle_t local_heap_ipc_handle{};
    err = hipIpcGetMemHandle(&local_heap_ipc_handle, d_heap_buf_);
    if (err != hipSuccess) {
      rcomp_hybrid_init_abort(rank_me, -1, local_hostname, local_hostname, local_hip_device,
                              local_hip_pci_bus_id_string, local_hip_device,
                              local_hip_pci_bus_id_string,
                              "hipIpcGetMemHandle for the symmetric heap failed", err);
    }

    std::vector<hipIpcMemHandle_t> peer_ipc_handles(static_cast<size_t>(cfg_.world_size));
    // This is intentionally a world communicator collective so every rank
    // keeps the same setup ordering.  Only exact-hostname peers are opened.
    comm_ctx_->bootstrap().allgather(&local_heap_ipc_handle, peer_ipc_handles.data(),
                                     sizeof(hipIpcMemHandle_t));

    rcomp_peer_heap_bases_.assign(static_cast<size_t>(cfg_.world_size), uintptr_t{0});
    rcomp_peer_heap_ipc_mappings_.assign(static_cast<size_t>(cfg_.world_size), nullptr);
    for (int rank = 0; rank < cfg_.world_size; ++rank) {
      if (rcomp_peer_is_ipc_[static_cast<size_t>(rank)] == 0) {
        continue;
      }
      void* mapped_peer_heap = nullptr;
      err = hipIpcOpenMemHandle(&mapped_peer_heap, peer_ipc_handles[static_cast<size_t>(rank)],
                                hipIpcMemLazyEnablePeerAccess);
      if (err != hipSuccess || mapped_peer_heap == nullptr) {
        rcomp_hybrid_init_abort(rank_me, rank, local_hostname,
                                rcomp_peer_hostnames_[static_cast<size_t>(rank)], local_hip_device,
                                local_hip_pci_bus_id_string,
                                rcomp_peer_hip_devices_[static_cast<size_t>(rank)],
                                rcomp_peer_hip_pci_bus_ids_[static_cast<size_t>(rank)],
                                "hipIpcOpenMemHandle for the peer symmetric heap failed", err);
      }
      rcomp_peer_heap_bases_[static_cast<size_t>(rank)] =
          reinterpret_cast<uintptr_t>(mapped_peer_heap);
      rcomp_peer_heap_ipc_mappings_[static_cast<size_t>(rank)] = mapped_peer_heap;
    }

    HIP_CHECK(hipMalloc(&d_rcomp_peer_heap_bases_,
                        static_cast<size_t>(cfg_.world_size) * sizeof(uintptr_t)));
    HIP_CHECK(
        hipMalloc(&d_rcomp_peer_is_ipc_, static_cast<size_t>(cfg_.world_size) * sizeof(uint8_t)));
    HIP_CHECK(hipMemcpy(d_rcomp_peer_heap_bases_, rcomp_peer_heap_bases_.data(),
                        static_cast<size_t>(cfg_.world_size) * sizeof(uintptr_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rcomp_peer_is_ipc_, rcomp_peer_is_ipc_.data(),
                        static_cast<size_t>(cfg_.world_size) * sizeof(uint8_t),
                        hipMemcpyHostToDevice));
  }

  const std::vector<uintptr_t>* reusable_count_peer_heap_bases = nullptr;
  if (rcomp_ipc_resources_provisioned_) {
    reusable_count_peer_heap_bases = &rcomp_peer_heap_bases_;
  }
  all2all_ = std::make_unique<all2all_t>(cfg_, *comm_ctx_, d_heap_buf_, heap_id_,
                                         reusable_count_peer_heap_bases);

  // Output tensors for recv data (device memory)
  int ms = cfg_.max_slots();
  const int dispatch_hidden_elems = cfg_.dispatch_hidden_elems_effective();
  HIP_CHECK(
      hipMalloc(&d_recv_hidden_, (size_t)ms * dispatch_hidden_elems * cfg_.dispatch_elem_size));
  HIP_CHECK(hipMalloc(&d_recv_indices_, (size_t)ms * cfg_.topk * sizeof(int32_t)));
  HIP_CHECK(hipMalloc(&d_recv_weights_, (size_t)ms * cfg_.topk * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_recv_src_token_pos_, (size_t)ms * sizeof(int32_t)));
  if (cfg_.dispatch_scale_dim > 0 && cfg_.dispatch_scale_type_size > 0) {
    HIP_CHECK(hipMalloc(&d_recv_scales_,
                        (size_t)ms * cfg_.dispatch_scale_dim * cfg_.dispatch_scale_type_size));
  }
}

ep_context_t::~ep_context_t() {
  all2all_.reset();
  if (comm_ctx_ && heap_id_ >= 0) {
    comm_ctx_->deregister_symmetric(heap_id_);
  }
  for (void* ptr : rcomp_peer_heap_ipc_mappings_) {
    if (ptr != nullptr) {
      (void)hipIpcCloseMemHandle(ptr);
    }
  }
  rcomp_peer_heap_ipc_mappings_.clear();
  if (d_rcomp_peer_heap_bases_) {
    (void)hipFree(d_rcomp_peer_heap_bases_);
    d_rcomp_peer_heap_bases_ = nullptr;
  }
  if (d_rcomp_peer_is_ipc_) {
    (void)hipFree(d_rcomp_peer_is_ipc_);
    d_rcomp_peer_is_ipc_ = nullptr;
  }
  if (d_heap_buf_) (void)hipFree(d_heap_buf_);
  if (d_recv_hidden_) (void)hipFree(d_recv_hidden_);
  if (d_recv_indices_) (void)hipFree(d_recv_indices_);
  if (d_recv_weights_) (void)hipFree(d_recv_weights_);
  if (d_recv_src_token_pos_) (void)hipFree(d_recv_src_token_pos_);
  if (d_recv_scales_) (void)hipFree(d_recv_scales_);
  if (debug_time_buf_) (void)hipFree(debug_time_buf_);
  if (debug_time_meta_) (void)hipFree(debug_time_meta_);
  if (d_device_workspace_) {
    (void)hipFree(d_device_workspace_);
    d_device_workspace_ = nullptr;
  }
}

ep_gcontext_t ep_context_t::get_device_context() {
  ep_gcontext_t gctx{};
  gctx.host_context = this;
  gctx.comm = comm_ctx_->get_device_context(ep_all2all_cb_id_);
  gctx.heap_base = d_heap_buf_;
  gctx.heap_id = heap_id_;
  gctx.device_workspace = device_workspace();
  gctx.count_alltoall_local_size = all2all_->local_size();
  gctx.count_alltoall_num_nodes = all2all_->num_nodes();
  gctx.count_alltoall_node_idx = all2all_->node_idx();
  gctx.count_alltoall_lane_idx = all2all_->lane_idx();
  gctx.count_alltoall_local_peer_heap_bases = all2all_->device_local_peer_heap_bases();
  gctx.protocol_epoch_offset = cfg_.protocol_epoch_offset();
  gctx.count_alltoall_egress_offset = cfg_.count_alltoall_egress_offset();
  gctx.count_alltoall_ingress_offset = cfg_.count_alltoall_ingress_offset();
  gctx.barrier_alltoall_egress_offset = cfg_.barrier_alltoall_egress_offset();
  gctx.barrier_alltoall_ingress_offset = cfg_.barrier_alltoall_ingress_offset();
  gctx.active_queue_count_cb_id = active_queue_count_cb_id_;
  gctx.post_put_cb_id = post_put_cb_id_;
  gctx.post_put_multicast_cb_id = post_put_multicast_cb_id_;
  gctx.rcomp_peer_heap_bases = d_rcomp_peer_heap_bases_;
  gctx.rcomp_peer_is_ipc = d_rcomp_peer_is_ipc_;
  gctx.dispatch_readiness_flags_offset = cfg_.dispatch_readiness_flags_offset();
  gctx.combine_readiness_flags_offset = cfg_.combine_readiness_flags_offset();

  gctx.world_size = cfg_.world_size;
  gctx.rank_me = comm_ctx_->rank_me();
  gctx.num_experts_per_rank = cfg_.num_experts_per_rank();
  gctx.model_hidden_dim = cfg_.model_hidden_dim;
  gctx.dispatch_hidden_elems = cfg_.dispatch_hidden_elems_effective();
  gctx.combine_hidden_elems = cfg_.combine_hidden_elems_effective();
  gctx.topk = cfg_.topk;
  gctx.max_slots = cfg_.max_slots();
  gctx.max_num_inp_token_per_rank = cfg_.max_num_inp_token_per_rank;
  gctx.dispatch_scale_dim = cfg_.dispatch_scale_dim;
  gctx.dispatch_scale_type_size = cfg_.dispatch_scale_type_size;
  gctx.combine_scale_dim = cfg_.combine_scale_dim;
  gctx.combine_scale_type_size = cfg_.combine_scale_type_size;
  gctx.dispatch_quant_format = cfg_.dispatch_quant_format;
  gctx.combine_quant_format = cfg_.combine_quant_format;
  gctx.dispatch_layout = cfg_.dispatch_buffer_layout();
  gctx.combine_layout = cfg_.combine_buffer_layout();

  gctx.d_recv_hidden = d_recv_hidden_;
  gctx.d_recv_indices = d_recv_indices_;
  gctx.d_recv_weights = d_recv_weights_;
  gctx.d_recv_scales = d_recv_scales_;
  gctx.d_recv_src_token_pos = d_recv_src_token_pos_;

  return gctx;
}

void ep_context_t::spawn_proxy_threads(int num_threads) {
  const int actual_num_threads = (num_threads == -1) ? cfg_.num_proxy_threads : num_threads;
  if (actual_num_threads <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_context_t::spawn_proxy_threads: num_threads=" +
                                std::to_string(actual_num_threads) + " must be positive");
  }
  if (actual_num_threads != cfg_.num_proxy_threads) {
    throw std::invalid_argument(
        "mori::kiwi::ep::ep_context_t::spawn_proxy_threads: num_threads=" +
        std::to_string(actual_num_threads) + " must equal config.num_proxy_threads=" +
        std::to_string(cfg_.num_proxy_threads) + " because each proxy thread owns one RDMA device");
  }
  comm_ctx_->spawn_proxy_threads(actual_num_threads);
}

void ep_context_t::join_proxy_threads() { comm_ctx_->join_proxy_threads(); }

void ep_context_t::set_debug_timing_enabled(bool enabled) {
  if (enabled && debug_time_buf_ == nullptr) {
    HIP_CHECK(hipMallocManaged(&debug_time_buf_, DEBUG_TIME_WORDS * sizeof(uint64_t)));
  }
  if (enabled && debug_time_meta_ == nullptr) {
    HIP_CHECK(hipMallocManaged(&debug_time_meta_, DEBUG_META_WORDS * sizeof(int32_t)));
  }
  debug_timing_enabled_ = enabled;
  if (debug_time_buf_) {
    memset(debug_time_buf_, 0, DEBUG_TIME_WORDS * sizeof(uint64_t));
  }
  if (debug_time_meta_) {
    memset(debug_time_meta_, 0, DEBUG_META_WORDS * sizeof(int32_t));
  }
}

void ep_context_t::reset(hipStream_t stream) {
  HIP_CHECK(
      hipMemsetAsync(d_recv_src_token_pos_, 0xFF, cfg_.max_slots() * sizeof(int32_t), stream));
  HIP_CHECK(
      hipMemsetAsync(d_device_workspace_, 0, device_workspace_layout_.get_num_bytes(), stream));
  // Initialize split route metadata to -1 (0xFF bytes):
  // producer_rank == -1 means "unused/invalid expert, skipped by combine".
  ep_device_workspace_t workspace = device_workspace();
  // protocol_epoch persists in the symmetric heap. reset() is allowed only
  // with no EP traffic in flight, so realign the barrier completion counter
  // after clearing resettable workspace. This also closes debug-only
  // dispatch replays that intentionally omit combine.
  HIP_CHECK(hipMemcpyAsync(workspace.barrier_counter(), d_heap_buf_ + cfg_.protocol_epoch_offset(),
                           sizeof(uint64_t), hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipMemsetAsync(workspace.token_to_producer_rank(), 0xFF,
                           cfg_.token_to_producer_rank_size(), stream));
  HIP_CHECK(hipMemsetAsync(workspace.token_to_dispatch_recv_slot(), 0xFF,
                           cfg_.token_to_dispatch_recv_slot_size(), stream));
  HIP_CHECK(hipMemsetAsync(workspace.recv_src_rank_map(), 0xFF, cfg_.max_slots() * sizeof(int32_t),
                           stream));
  HIP_CHECK(hipMemsetAsync(workspace.recv_local_slot_map(), 0xFF,
                           cfg_.max_slots() * sizeof(int32_t), stream));
  HIP_CHECK(hipMemsetAsync(d_heap_buf_ + cfg_.dispatch_readiness_flags_offset(), 0,
                           cfg_.payload_readiness_flag_count() * sizeof(uint8_t), stream));
  HIP_CHECK(hipMemsetAsync(d_heap_buf_ + cfg_.combine_readiness_flags_offset(), 0,
                           cfg_.payload_readiness_flag_count() * sizeof(uint8_t), stream));
}

void ep_all2all_callback(internal::ep_cmd_desc_all2all_t desc) {
  KIWI_EP_ASSERT(desc.ep_context != nullptr, "count alltoall descriptor has null EP context");
  desc.ep_context->all2all_callback(desc);
}

void ep_context_t::all2all_callback(internal::ep_cmd_desc_all2all_t desc) { all2all_->start(desc); }

void ep_context_t::copy_dest_token_counts_to_host(int32_t* host_counts, hipStream_t stream) const {
  HIP_CHECK(hipMemcpyAsync(host_counts, device_workspace().dispatch_dest_token_counts_snapshot(),
                           cfg_.world_size * sizeof(int32_t), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

void ep_context_t::copy_dispatch_recv_counts_to_host(int32_t* host_counts,
                                                     hipStream_t stream) const {
  HIP_CHECK(hipMemcpyAsync(host_counts, device_workspace().dispatch_recv_counts(),
                           (cfg_.world_size + 1) * sizeof(int32_t), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

int32_t ep_context_t::copy_recv_count_to_host(hipStream_t stream) const {
  int32_t host_count = 0;
  HIP_CHECK(hipMemcpyAsync(&host_count, device_workspace().recv_count(), sizeof(int32_t),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  return host_count;
}

void ep_context_t::copy_token_route_metadata_to_host(int32_t* host_producer_ranks,
                                                     int32_t* host_dispatch_recv_slots,
                                                     hipStream_t stream) const {
  if (host_producer_ranks != nullptr) {
    HIP_CHECK(hipMemcpyAsync(host_producer_ranks, device_workspace().token_to_producer_rank(),
                             cfg_.token_to_producer_rank_size(), hipMemcpyDeviceToHost, stream));
  }
  if (host_dispatch_recv_slots != nullptr) {
    HIP_CHECK(
        hipMemcpyAsync(host_dispatch_recv_slots, device_workspace().token_to_dispatch_recv_slot(),
                       cfg_.token_to_dispatch_recv_slot_size(), hipMemcpyDeviceToHost, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// Host-side stream synchronization wrapper.
// ---------------------------------------------------------------------------

void ep_context_t::stream_sync(hipStream_t stream) { HIP_CHECK(hipStreamSynchronize(stream)); }

}  // namespace mori::kiwi::ep
