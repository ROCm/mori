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

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mori/kiwi/rdma/bootstrap.hpp"

namespace mori::kiwi::ep {

class ep_context_t;
struct ep_gcontext_t;

// Construction parameters for ep_op_t. Element sizes are in bytes.
struct ep_op_config_t {
  int rank = 0;
  int world_size = 1;
  int hidden_dim = 0;  // logical hidden size; FP4 stores two values per byte
  int topk = 1;
  int num_experts_per_rank = 1;
  int max_num_inp_token_per_rank = 1;
  int dispatch_elem_size = 2;
  int combine_elem_size = 2;
  std::string dispatch_quant_format = "none";  // none, fp8, fp4
  std::string combine_quant_format = "none";
  int dispatch_scale_dim = 0;
  int dispatch_scale_type_size = 0;
  int num_blocks = 0;  // 0 selects the block count per call from the token count
  int num_proxy_threads = 4;
  int queue_capacity = 2048;
  std::string rcomp_backend = "auto";  // auto, rdma-only, hybrid
  std::string device_name;             // RDMA NIC; empty selects automatically
};

// MORI-compatible KIWI EP dispatch/combine operator. All tensors are device
// pointers; all work is enqueued on the given stream and is graph-capturable.
class ep_op_t {
 public:
  // Collective over `bootstrap`. The calling thread's current HIP device is
  // used.
  ep_op_t(const ep_op_config_t& config, std::shared_ptr<rdma::bootstrap_t> bootstrap);
  ~ep_op_t();

  ep_op_t(const ep_op_t&) = delete;
  ep_op_t& operator=(const ep_op_t&) = delete;

  void dispatch(const void* input, const float* weights, const void* scales, const int32_t* indices,
                int num_tokens, hipStream_t stream);

  // `output` holds [num_tokens, combine_hidden_elems]; `output_weights` holds
  // [num_tokens, topk] and is written only when `weights` is non-null.
  void combine(const void* input, const float* weights, const int32_t* indices, void* output,
               float* output_weights, int num_tokens, hipStream_t stream);

  // Clears per-round state. No EP traffic may be in flight.
  void reset(hipStream_t stream);

  // Operator-owned dispatch outputs, each with max_slots() rows.
  void* recv_hidden() const;
  float* recv_weights() const;
  int32_t* recv_indices() const;
  void* recv_scales() const;
  int32_t* recv_count() const;
  int32_t* recv_src_token_pos() const;
  int32_t copy_recv_count_to_host(hipStream_t stream) const;

  int max_slots() const;
  int dispatch_hidden_elems() const;
  int combine_hidden_elems() const;
  int max_num_tokens_to_recv() const;
  int max_num_tokens_to_recv_per_rank() const;
  int max_num_tokens_to_send() const;
  int max_num_tokens_to_send_per_rank() const;
  int rank() const;
  int world_size() const;
  int num_input_tokens() const { return num_input_tokens_; }

  void set_debug_timing_enabled(bool enabled);
  bool debug_timing_enabled() const;
  std::vector<uint64_t> debug_time_words() const;
  std::vector<int32_t> debug_meta_words() const;

 private:
  std::unique_ptr<ep_context_t> ctx_;
  std::unique_ptr<ep_gcontext_t> gctx_;
  int num_input_tokens_ = 0;
};

}  // namespace mori::kiwi::ep
