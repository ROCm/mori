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

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "mori/kiwi/invoke/invoke.hpp"

namespace mori::kiwi::ep {

// Device-to-host queue type behind every EP invoke endpoint.
using ep_queue_t = kiwi::queue::SPSCD2HQueue;

namespace internal {

// queue moved kiwi::invoke::DeviceContext under kiwi::invoke::internal in newer
// revisions.  Refer to the public Context::device_context() return type so EP
// builds against both layouts.
using ep_gc_device_context_t = typename std::remove_pointer<
    decltype(std::declval<kiwi::invoke::Context<ep_queue_t>&>().device_context())>::type;

}  // namespace internal

class ep_comm_handle_t {
 public:
  using DeviceHandle = internal::ep_gc_device_context_t::DeviceHandle;

  ep_comm_handle_t(const ep_comm_handle_t&) = delete;
  ep_comm_handle_t& operator=(const ep_comm_handle_t&) = delete;

  __device__ ep_comm_handle_t(internal::ep_gc_device_context_t* dev_ctx, DeviceHandle h,
                              size_t endpoint_id)
      : dev_ctx_(dev_ctx), handle_(h), endpoint_id_(endpoint_id), save_on_destroy_(true) {}

  __device__ ep_comm_handle_t(ep_comm_handle_t&& other) noexcept
      : dev_ctx_(other.dev_ctx_),
        handle_(other.handle_),
        endpoint_id_(other.endpoint_id_),
        save_on_destroy_(other.save_on_destroy_) {
    other.save_on_destroy_ = false;
  }

  // Force-inlined: an out-of-line destructor takes the handle's address,
  // which puts the whole handle (and every push's index update) in scratch.
  __device__ __forceinline__ ~ep_comm_handle_t() {
    if (save_on_destroy_) {
      dev_ctx_->save_device_handle(endpoint_id_, handle_);
    }
  }

  template <typename Desc>
  __device__ bool try_invoke(uint8_t cb_id, const Desc& desc) {
    return handle_.invoke(cb_id, desc);
  }

  template <typename Desc>
  __device__ void post_desc(uint8_t cb_id, const Desc& desc) {
    while (!handle_.invoke(cb_id, desc)) {
      // Queue full, spin-wait.
    }
  }

  template <typename Desc>
  __device__ void post_desc_shared(uint8_t cb_id, const Desc* lds_desc, size_t bytes) {
    while (!handle_.invoke_shared(cb_id, lds_desc, bytes)) {
      // Queue full, spin-wait.
    }
  }

 private:
  internal::ep_gc_device_context_t* dev_ctx_ = nullptr;
  DeviceHandle handle_;
  size_t endpoint_id_ = 0;
  bool save_on_destroy_ = false;
};

}  // namespace mori::kiwi::ep
