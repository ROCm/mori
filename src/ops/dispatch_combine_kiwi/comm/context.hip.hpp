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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "comm/active_queue_state.hpp"
#include "comm/device_context.hip.hpp"
#include "mori/kiwi/invoke/invoke.hpp"
#include "mori/kiwi/rdma/bootstrap.hpp"
#include "mori/kiwi/rdma/ibverbs.hpp"

namespace mori::kiwi::ep {

namespace internal {

// Thread-local EP proxy state. EP callbacks use these values so EP owns its
// complete comm/proxy stack independently from generic P2P.
inline thread_local rdma::device_t* tl_device = nullptr;
inline thread_local int tl_device_idx = -1;
inline thread_local int tl_proxy_thread_idx = -1;

// All proxy devices share one protection domain, so one registration serves
// every device.
struct ep_heap_entry_t {
  rdma::mr_t mr;
  std::vector<rdma::rmr_t> rmrs;  // indexed by rank
};

using ep_heap_table_t = std::vector<ep_heap_entry_t>;

}  // namespace internal

class ep_comm_context_t {
 public:
  explicit ep_comm_context_t(std::shared_ptr<rdma::bootstrap_t> bootstrap, int num_endpoints = 1,
                             int num_devices = 1, std::string requested_device_name = {},
                             size_t queue_capacity = 2048);
  ~ep_comm_context_t();

  ep_comm_context_t(const ep_comm_context_t&) = delete;
  ep_comm_context_t& operator=(const ep_comm_context_t&) = delete;

  ep_comm_device_context_t get_device_context(uint8_t all2all_cb_id);

  int alloc_symmetric(void* buf, size_t size);
  // Must be called after join_proxy_threads(); raw-RDMA proxy devices drain
  // their own CQs while stopping and cannot be polled concurrently here.
  void deregister_symmetric(int heap_id);

  void spawn_proxy_threads(int num_threads = -1);
  void join_proxy_threads();
  void publish_active_queue_count(int active_count);

  template <typename Desc, typename Fn>
  uint8_t register_callback(Fn&& cb) {
    return gc_context_->register_callback(std::function<void(Desc)>(std::forward<Fn>(cb)));
  }

  rdma::device_t& get_device(int index = 0) const { return *devices_[index]; }
  int num_devices_count() const { return num_devices_; }

  rdma::bootstrap_t& bootstrap() const { return *bootstrap_; }
  int rank_me() const { return bootstrap_->rank(); }
  int rank_n() const { return bootstrap_->size(); }

  const internal::ep_heap_table_t& heap_table() const { return heap_table_; }

 private:
  void wait_drained();
  void proxy_thread_func(int thread_id, int num_proxy_threads, int selected_cpu);

  std::shared_ptr<rdma::bootstrap_t> bootstrap_;
  int num_queues_ = 0;
  int num_devices_ = 0;
  std::unique_ptr<rdma::net_context_t> net_;
  std::vector<std::unique_ptr<rdma::device_t>> devices_;
  std::unique_ptr<mori::kiwi::invoke::Context<ep_queue_t>> gc_context_;

  internal::ep_heap_table_t heap_table_;
  std::vector<std::thread> proxy_threads_;
  std::atomic<bool> stop_proxy_threads_{false};
  internal::active_queue_request_t active_queue_request_;
};

int ep_current_proxy_thread_index();

}  // namespace mori::kiwi::ep
