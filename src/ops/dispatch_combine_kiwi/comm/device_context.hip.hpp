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

#include "comm/handle.hip.hpp"

namespace mori::kiwi::ep {

class ep_comm_device_context_t {
 public:
  __host__ __device__ ep_comm_device_context_t() = default;

  __host__ __device__ ep_comm_device_context_t(internal::ep_gc_device_context_t* dev_ctx,
                                               size_t num_endpoints, uint8_t all2all_cb_id)
      : dev_ctx_(dev_ctx), num_endpoints_(num_endpoints), all2all_cb_id_(all2all_cb_id) {}

  __device__ ep_comm_handle_t get_handle() const {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    size_t ep = blockId % num_endpoints_;
    return get_handle(ep);
  }

  __device__ ep_comm_handle_t get_handle(size_t endpoint_id) const {
    return ep_comm_handle_t(dev_ctx_, dev_ctx_->get_device_handle(endpoint_id), endpoint_id);
  }

  __host__ __device__ size_t get_num_endpoints() const { return num_endpoints_; }

  __host__ __device__ uint8_t all2all_cb_id() const { return all2all_cb_id_; }

 private:
  internal::ep_gc_device_context_t* dev_ctx_ = nullptr;
  size_t num_endpoints_ = 0;
  uint8_t all2all_cb_id_ = 0;
};

}  // namespace mori::kiwi::ep
