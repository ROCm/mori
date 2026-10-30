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
#include "ep_op.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include "ep_context.hip.hpp"
#include "hip_check.hpp"

namespace mori::kiwi::ep {

namespace {

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

quant_format_t parse_quant_format(const std::string& name) {
  const std::string value = lower(name);
  if (value.empty() || value == "none") return quant_format_t::NONE;
  if (value == "fp8") return quant_format_t::FP8;
  if (value == "fp4") return quant_format_t::FP4;
  throw std::invalid_argument("unsupported KIWI EP quant format '" + name +
                              "' (expected none, fp8, or fp4)");
}

rcomp_backend_t parse_rcomp_backend(const std::string& name) {
  const std::string value = lower(name);
  if (value.empty() || value == "auto") return rcomp_backend_t::AUTO;
  if (value == "rdma-only" || value == "rdma_only" || value == "rdma") {
    return rcomp_backend_t::RDMA_ONLY;
  }
  if (value == "hybrid") return rcomp_backend_t::HYBRID;
  throw std::invalid_argument("unsupported KIWI EP rcomp_backend '" + name +
                              "' (expected auto, rdma-only, or hybrid)");
}

int packed_hidden_elems(int hidden_dim, quant_format_t format, const char* which) {
  if (format != quant_format_t::FP4) return hidden_dim;
  if ((hidden_dim & 1) != 0) {
    throw std::invalid_argument(std::string("FP4 ") + which + " requires an even hidden_dim");
  }
  return hidden_dim / 2;
}

config_t make_config(const ep_op_config_t& in) {
  config_t cfg;
  cfg.world_size = in.world_size;
  cfg.num_experts = in.num_experts_per_rank * in.world_size;
  cfg.model_hidden_dim = in.hidden_dim;
  cfg.topk = in.topk;
  cfg.max_num_inp_token_per_rank = in.max_num_inp_token_per_rank;
  cfg.num_blocks = in.num_blocks;
  cfg.num_proxy_threads = in.num_proxy_threads;
  cfg.queue_capacity = in.queue_capacity;
  cfg.device_name = in.device_name;
  cfg.dispatch_elem_size = in.dispatch_elem_size;
  cfg.combine_elem_size = in.combine_elem_size;
  cfg.dispatch_quant_format = parse_quant_format(in.dispatch_quant_format);
  cfg.combine_quant_format = parse_quant_format(in.combine_quant_format);
  if (cfg.dispatch_quant_format != quant_format_t::NONE && cfg.dispatch_elem_size != 1) {
    throw std::invalid_argument("fp8/fp4 dispatch requires a one-byte dispatch element");
  }
  if (cfg.combine_quant_format != quant_format_t::NONE && cfg.combine_elem_size != 1) {
    throw std::invalid_argument("fp8/fp4 combine requires a one-byte combine element");
  }
  // MORI's public config carries the logical hidden size; FP4 rows are packed.
  cfg.dispatch_hidden_elems =
      packed_hidden_elems(in.hidden_dim, cfg.dispatch_quant_format, "dispatch");
  cfg.combine_hidden_elems =
      packed_hidden_elems(in.hidden_dim, cfg.combine_quant_format, "combine");
  cfg.dispatch_scale_dim = in.dispatch_scale_dim;
  cfg.dispatch_scale_type_size = in.dispatch_scale_type_size;
  cfg.combine_scale_dim = 0;
  cfg.combine_scale_type_size = 0;
  cfg.rcomp_backend = parse_rcomp_backend(in.rcomp_backend);
  return cfg;
}

}  // namespace

ep_op_t::ep_op_t(const ep_op_config_t& config,
                 std::shared_ptr<application::BootstrapNetwork> bootstrap) {
  if (!bootstrap) throw std::invalid_argument("KIWI EP bootstrap is null");
  if (bootstrap->GetWorldSize() != config.world_size || bootstrap->GetLocalRank() != config.rank) {
    throw std::invalid_argument("KIWI EP config rank/world_size (" + std::to_string(config.rank) +
                                "/" + std::to_string(config.world_size) +
                                ") does not match the bootstrap (" +
                                std::to_string(bootstrap->GetLocalRank()) + "/" +
                                std::to_string(bootstrap->GetWorldSize()) + ")");
  }
  const config_t cfg = make_config(config);
  ctx_ = std::make_unique<ep_context_t>(cfg, std::move(bootstrap));
  ctx_->spawn_proxy_threads(ctx_->config().num_proxy_threads);
  gctx_ = std::make_unique<ep_gcontext_t>(ctx_->get_device_context());

  hipStream_t stream = nullptr;
  HIP_CHECK_THROW(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  ctx_->reset(stream);
  ctx_->stream_sync(stream);
  HIP_CHECK_THROW(hipStreamDestroy(stream));
  // Every rank must finish resetting its heap before any peer writes to it.
  ctx_->bootstrap().Barrier();
}

ep_op_t::~ep_op_t() {
  if (ctx_) ctx_->join_proxy_threads();
}

void ep_op_t::dispatch(const void* input, const float* weights, const void* scales,
                       const int32_t* indices, int num_tokens, hipStream_t stream) {
  if (num_tokens < 0 || num_tokens > ctx_->config().max_num_inp_token_per_rank) {
    throw std::invalid_argument("dispatch num_tokens=" + std::to_string(num_tokens) +
                                " exceeds max_num_inp_token_per_rank=" +
                                std::to_string(ctx_->config().max_num_inp_token_per_rank));
  }
  num_input_tokens_ = num_tokens;
  dispatch_enqueue(*ctx_, *gctx_, input, indices, weights, scales, num_tokens, stream);
}

void ep_op_t::combine(const void* input, const float* weights, const int32_t* indices, void* output,
                      float* output_weights, int num_tokens, hipStream_t stream) {
  combine_enqueue(*ctx_, *gctx_, input, weights, nullptr, indices, output,
                  weights != nullptr ? output_weights : nullptr, num_tokens, stream);
}

void ep_op_t::reset(hipStream_t stream) {
  ctx_->reset(stream);
  ctx_->stream_sync(stream);
  num_input_tokens_ = 0;
}

void* ep_op_t::recv_hidden() const { return gctx_->d_recv_hidden; }
float* ep_op_t::recv_weights() const { return gctx_->d_recv_weights; }
int32_t* ep_op_t::recv_indices() const { return gctx_->d_recv_indices; }
void* ep_op_t::recv_scales() const { return gctx_->d_recv_scales; }
int32_t* ep_op_t::recv_count() const { return gctx_->device_workspace.recv_count(); }
int32_t* ep_op_t::recv_src_token_pos() const { return gctx_->d_recv_src_token_pos; }

int32_t ep_op_t::copy_recv_count_to_host(hipStream_t stream) const {
  return ctx_->copy_recv_count_to_host(stream);
}

int ep_op_t::max_slots() const { return ctx_->config().max_slots(); }
int ep_op_t::dispatch_hidden_elems() const {
  return ctx_->config().dispatch_hidden_elems_effective();
}
int ep_op_t::combine_hidden_elems() const {
  return ctx_->config().combine_hidden_elems_effective();
}
int ep_op_t::max_num_tokens_to_recv() const { return ctx_->config().max_recv_slots_total(); }
int ep_op_t::max_num_tokens_to_recv_per_rank() const {
  return ctx_->config().max_recv_slots_per_rank();
}
int ep_op_t::max_num_tokens_to_send() const {
  return ctx_->config().world_size * ctx_->config().max_num_inp_token_per_rank;
}
int ep_op_t::max_num_tokens_to_send_per_rank() const {
  return ctx_->config().max_num_inp_token_per_rank;
}
int ep_op_t::rank() const { return ctx_->rank_me(); }
int ep_op_t::world_size() const { return ctx_->config().world_size; }

}  // namespace mori::kiwi::ep
