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
#include "comm/context.hip.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "assert.hpp"
#include "comm/proxy_network.hip.hpp"
#include "ep_context.hip.hpp"
#include "mori/kiwi/topology/topology.hpp"

namespace mori::kiwi::ep {

namespace internal {
namespace {

// Per-QP send depth for raw proxy WRITEs and the inline capacity needed for
// payload readiness flags.
constexpr size_t kRawNetworkMaxSends = 64;
constexpr size_t kRawNetworkMaxInline = 64;
constexpr size_t kRawNetworkMaxPolls = 32;

}  // namespace

static thread_local size_t tl_raw_network_outstanding = 0;

static void raw_network_report_bad_completion(const char* reason, const ibv_wc& wc,
                                              size_t outstanding) {
  std::fprintf(stderr,
               "kiwi EP raw network CQ %s: status=%s(%d) opcode=%d qp_num=%u "
               "wr_id=0x%llx byte_len=%u outstanding=%zu\n",
               reason, ibv_wc_status_str(wc.status), static_cast<int>(wc.status),
               static_cast<int>(wc.opcode), wc.qp_num, static_cast<unsigned long long>(wc.wr_id),
               wc.byte_len, outstanding);
}

static void raw_network_progress() {
  if (tl_raw_network_outstanding == 0) {
    return;
  }
  std::array<ibv_wc, kRawNetworkMaxPolls> wcs;
  const size_t count = tl_device->poll_cq(wcs.data(), wcs.size());
  for (size_t i = 0; i < count; ++i) {
    const ibv_wc& wc = wcs[i];
    if (tl_raw_network_outstanding == 0) {
      raw_network_report_bad_completion("has no tracked raw work request", wc,
                                        tl_raw_network_outstanding);
      KIWI_EP_ASSERT(false, "raw network CQ completion has no outstanding work request");
    }
    --tl_raw_network_outstanding;
    if (wc.status != IBV_WC_SUCCESS) {
      raw_network_report_bad_completion("reported an error", wc, tl_raw_network_outstanding);
      KIWI_EP_ASSERT(false, "raw network CQ reported an error");
    }
    if (wc.opcode != IBV_WC_RDMA_WRITE) {
      raw_network_report_bad_completion("received a CQE not owned by the raw proxy", wc,
                                        tl_raw_network_outstanding);
      KIWI_EP_ASSERT(false, "raw network CQ completion violates raw-device ownership");
    }
  }
}

static void raw_network_drain() {
  while (tl_raw_network_outstanding != 0) {
    raw_network_progress();
  }
}

void proxy_network_progress() { raw_network_progress(); }

void proxy_network_record_posted_write() { ++tl_raw_network_outstanding; }

void proxy_network_put(int rank, const void* local, size_t size, const rdma::mr_t& mr,
                       uint64_t remote_disp, const rdma::rmr_t& rmr) {
  while (tl_device->post_put(rank, local, size, mr.lkey(), remote_disp, rmr) ==
         rdma::post_status_t::retry) {
    raw_network_progress();
  }
  proxy_network_record_posted_write();
}

void proxy_network_put_inline(int rank, const void* local, size_t size, uint64_t remote_disp,
                              const rdma::rmr_t& rmr) {
  while (tl_device->post_puts(rank, local, size, remote_disp, rmr) == rdma::post_status_t::retry) {
    raw_network_progress();
  }
  // The inline source is consumed, but the signaled WRITE's CQE is still
  // pending and must be drained before this proxy can be torn down.
  proxy_network_record_posted_write();
}

}  // namespace internal

ep_comm_context_t::ep_comm_context_t(std::shared_ptr<rdma::bootstrap_t> bootstrap, int num_queues,
                                     int num_devices, std::string requested_device_name,
                                     size_t queue_capacity)
    : bootstrap_(std::move(bootstrap)),
      num_queues_(num_queues),
      num_devices_(num_devices),
      active_queue_request_(num_queues) {
  if (!bootstrap_) {
    throw std::invalid_argument("mori::kiwi::ep::ep_comm_context_t: bootstrap is null");
  }
  if (num_queues_ <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_comm_context_t: num_endpoints=" +
                                std::to_string(num_queues_) + " must be positive");
  }
  if (num_devices_ <= 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_comm_context_t: num_devices=" +
                                std::to_string(num_devices_) + " must be positive");
  }
  // Create EP-owned invoke context. queue_capacity is the per-queue
  // depth and must be a non-zero power of two for queue ring indexing.
  if (queue_capacity == 0 || (queue_capacity & (queue_capacity - 1)) != 0) {
    throw std::invalid_argument("mori::kiwi::ep::ep_comm_context_t: queue_capacity=" +
                                std::to_string(queue_capacity) + " is not a non-zero power of two");
  }

  gc_context_ =
      std::make_unique<mori::kiwi::invoke::Context<ep_queue_t>>(num_queues_, queue_capacity);

  // One NIC context per EP context; each proxy thread owns one device with
  // its own CQ and RC QPs to every rank. Device creation is collective.
  net_ = std::make_unique<rdma::net_context_t>(requested_device_name);
  devices_.reserve(num_devices_);
  for (int i = 0; i < num_devices_; i++) {
    devices_.push_back(std::make_unique<rdma::device_t>(
        *net_, *bootstrap_, internal::kRawNetworkMaxSends, internal::kRawNetworkMaxInline));
  }
}

ep_comm_context_t::~ep_comm_context_t() {
  for (auto& entry : heap_table_) {
    if (!entry.mr.empty()) {
      net_->deregister_memory(entry.mr);
    }
  }
  devices_.clear();

  gc_context_.reset();
}

ep_comm_device_context_t ep_comm_context_t::get_device_context(uint8_t all2all_cb_id) {
  return ep_comm_device_context_t(gc_context_->device_context(), gc_context_->num_endpoints(),
                                  all2all_cb_id);
}

int ep_comm_context_t::alloc_symmetric(void* buf, size_t size) {
  assert(!devices_.empty());

  const int nranks = bootstrap_->size();
  internal::ep_heap_entry_t heap_entry;
  heap_entry.mr = net_->register_memory(buf, size);
  const rdma::rmr_t my_rmr = heap_entry.mr.rmr();
  heap_entry.rmrs.resize(static_cast<size_t>(nranks));
  bootstrap_->allgather(&my_rmr, heap_entry.rmrs.data(), sizeof(rdma::rmr_t));

  int heap_id = static_cast<int>(heap_table_.size());
  heap_table_.push_back(std::move(heap_entry));
  return heap_id;
}

void ep_comm_context_t::wait_drained() {
  KIWI_EP_ASSERT(proxy_threads_.empty(),
                 "EP raw-network symmetric heap teardown requires joined proxy threads");
}

void ep_comm_context_t::deregister_symmetric(int heap_id) {
  assert(heap_id >= 0 && heap_id < static_cast<int>(heap_table_.size()));
  wait_drained();
  net_->deregister_memory(heap_table_[heap_id].mr);
}

void ep_comm_context_t::spawn_proxy_threads(int num_threads) {
  if (num_threads == -1) {
    num_threads = num_queues_;
  }
  if (num_threads <= 0) {
    throw std::invalid_argument(
        "mori::kiwi::ep::ep_comm_context_t::spawn_proxy_threads: num_threads=" +
        std::to_string(num_threads) + " must be positive");
  }
  assert(proxy_threads_.empty());
  stop_proxy_threads_.store(false, std::memory_order_relaxed);

  const std::vector<int> proxy_cpu_assignments =
      mori::kiwi::p2p::topology::choose_proxy_thread_cpus(num_threads);

  for (int i = 0; i < num_threads; i++) {
    const int selected_cpu = proxy_cpu_assignments[static_cast<size_t>(i)];
    proxy_threads_.emplace_back(&ep_comm_context_t::proxy_thread_func, this, i, num_threads,
                                selected_cpu);
  }
}

void ep_comm_context_t::join_proxy_threads() {
  stop_proxy_threads_.store(true, std::memory_order_relaxed);
  for (auto& t : proxy_threads_) {
    t.join();
  }
  proxy_threads_.clear();
}

void ep_comm_context_t::publish_active_queue_count(int active_count) {
  KIWI_EP_ASSERT(active_count > 0 && active_count <= num_queues_,
                 "active queue count is outside the provisioned endpoint range");
  active_queue_request_.publish(active_count);
}

void ep_comm_context_t::proxy_thread_func(int thread_id, int num_proxy_threads, int selected_cpu) {
  int device_idx = thread_id % num_devices_;
  mori::kiwi::p2p::topology::pin_proxy_thread_to_cpu(thread_id, num_proxy_threads, selected_cpu);

  internal::tl_device_idx = device_idx;
  internal::tl_device = devices_[device_idx].get();
  internal::tl_proxy_thread_idx = thread_id;

  std::vector<std::pair<int, ep_queue_t::HostPollState>> endpoint_poll_states;
  // Radix-fold ownership spreads active endpoint prefixes across proxies
  // without aliasing regularly spaced producer endpoints. Scan endpoint IDs
  // in order so this proxy's list remains sorted for the prefix break below.
  for (int q = 0; q < num_queues_; ++q) {
    if (internal::proxy_queue_owner(q, num_proxy_threads) != thread_id) {
      continue;
    }
    endpoint_poll_states.emplace_back(q, gc_context_->raw_queue(q)->host_poll_state());
  }
  // Poll every provisioned endpoint until a launch-driven shrink has drained
  // its retiring queues. This keeps the first launch safe before its control
  // callback is observed.
  internal::proxy_active_queue_state_t active_state(num_queues_);
  while (!stop_proxy_threads_.load(std::memory_order_relaxed)) {
    // A grow becomes visible immediately. A shrink leaves the current
    // polling range intact until this proxy drains its retiring queues.
    active_state.observe(active_queue_request_.load());

    bool had_ready_endpoint = false;
    bool retiring_queues_empty = true;
    for (auto& [q, poll_state] : endpoint_poll_states) {
      if (q >= active_state.polling_count()) {
        // endpoint_poll_states is ordered by increasing endpoint
        // index, so no later endpoint can be active.
        break;
      }
      poll_state.refresh();
      if (!poll_state.ready()) {
        continue;
      }

      had_ready_endpoint = true;
      gc_context_->progress(q, internal::kEpMaxCallbacksPerEndpointSweep);
      poll_state.refresh();
      internal::proxy_network_progress();
      if (active_state.is_retiring_queue(q) && poll_state.ready()) {
        retiring_queues_empty = false;
      }
    }

    if (active_state.has_retiring_queues() && retiring_queues_empty) {
      // Do not let a newer request be overwritten by completion of this
      // older shrink. The next loop observes the newer request and
      // grows immediately if needed.
      active_state.complete_shrink_if_current(active_queue_request_.load().generation);
    }

    if (!had_ready_endpoint) {
      internal::proxy_network_progress();
    }
  }

  bool made_progress = false;
  do {
    made_progress = false;
    for (auto& [q, poll_state] : endpoint_poll_states) {
      made_progress |= gc_context_->progress(q, internal::kEpMaxCallbacksPerEndpointSweep) != 0;
      poll_state.refresh();
      internal::proxy_network_progress();
    }
  } while (made_progress);
  internal::raw_network_drain();

  internal::tl_proxy_thread_idx = -1;
  internal::tl_device_idx = -1;
  internal::tl_device = nullptr;
}

int ep_current_proxy_thread_index() { return internal::tl_proxy_thread_idx; }

}  // namespace mori::kiwi::ep
