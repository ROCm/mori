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

#include "xla/ffi/api/ffi.h"

#include <hip/hip_runtime.h>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/utils/data_types.hpp"
#include "src/ffi/mori_xla_ffi_handle_mgr.hpp"

namespace ffi = xla::ffi;

// Helper: convert XLA FFI DataType to hipDataType
static hipDataType FfiDataTypeToHip(ffi::DataType dtype) {
  switch (dtype) {
    case ffi::DataType::F32:
      return HIP_R_32F;
    case ffi::DataType::F16:
      return HIP_R_16F;
    case ffi::DataType::BF16:
      return HIP_R_16BF;
    default:
      throw std::runtime_error("mori FFI: unsupported element type");
  }
}

// ---------------------------------------------------------------------------
// mori_ep_dispatch
// ---------------------------------------------------------------------------
static ffi::Error MoriEpDispatchImpl(
    ffi::AnyBuffer input, ffi::AnyBuffer topk_ids,
    ffi::Result<ffi::AnyBuffer> out_tokens,
    ffi::Result<ffi::AnyBuffer> out_weights,
    ffi::Result<ffi::AnyBuffer> out_indices,
    ffi::Result<ffi::AnyBuffer> total_recv_token_num,
    int64_t handle_id, int64_t kernel_type,
    int64_t block_num, int64_t rdma_block_num, int64_t warp_per_block,
    int64_t hidden_dim) {
  auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
  auto hip_dtype = FfiDataTypeToHip(input.element_type());
  int num_tokens = static_cast<int>(input.dimensions()[0]);
  int h_dim = (hidden_dim > 0) ? static_cast<int>(hidden_dim)
                               : static_cast<int>(input.dimensions()[1]);

  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  handle->PrepareInference(
      hip_dtype, input.untyped_data(), nullptr,
      static_cast<float*>(out_weights->untyped_data()),
      static_cast<mori::moe::index_t*>(topk_ids.untyped_data()),
      num_tokens);

  handle->LaunchDispatch(
      static_cast<mori::moe::KernelType>(kernel_type),
      static_cast<int>(block_num), static_cast<int>(rdma_block_num),
      static_cast<int>(warp_per_block), stream, h_dim);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_dispatch, MoriEpDispatchImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()   // input
        .Arg<ffi::AnyBuffer>()   // topk_ids
        .Ret<ffi::AnyBuffer>()   // out_tokens
        .Ret<ffi::AnyBuffer>()   // out_weights
        .Ret<ffi::AnyBuffer>()   // out_indices
        .Ret<ffi::AnyBuffer>()   // total_recv_token_num
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("kernel_type")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("rdma_block_num")
        .Attr<int64_t>("warp_per_block")
        .Attr<int64_t>("hidden_dim"));

// ---------------------------------------------------------------------------
// mori_ep_combine
// ---------------------------------------------------------------------------
static ffi::Error MoriEpCombineImpl(
    ffi::AnyBuffer input, ffi::AnyBuffer topk_ids,
    ffi::Result<ffi::AnyBuffer> out_tokens,
    int64_t handle_id, int64_t kernel_type,
    int64_t block_num, int64_t rdma_block_num, int64_t warp_per_block,
    int64_t use_external_inp_buf, int64_t hidden_dim) {
  auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
  auto hip_dtype = FfiDataTypeToHip(input.element_type());
  int h_dim = (hidden_dim > 0) ? static_cast<int>(hidden_dim)
                               : static_cast<int>(input.dimensions()[1]);

  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  handle->PrepareInference(
      hip_dtype, input.untyped_data(), nullptr,
      nullptr,
      static_cast<mori::moe::index_t*>(topk_ids.untyped_data()),
      handle->curRankNumToken);

  handle->LaunchCombine(
      static_cast<mori::moe::KernelType>(kernel_type),
      static_cast<int>(block_num), static_cast<int>(rdma_block_num),
      static_cast<int>(warp_per_block),
      static_cast<int>(use_external_inp_buf), stream, h_dim);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_combine, MoriEpCombineImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()   // input
        .Arg<ffi::AnyBuffer>()   // topk_ids
        .Ret<ffi::AnyBuffer>()   // out_tokens
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("kernel_type")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("rdma_block_num")
        .Attr<int64_t>("warp_per_block")
        .Attr<int64_t>("use_external_inp_buf")
        .Attr<int64_t>("hidden_dim"));

// ---------------------------------------------------------------------------
// mori_ep_dispatch_recv
// ---------------------------------------------------------------------------
static ffi::Error MoriEpDispatchRecvImpl(
    int64_t handle_id, int64_t kernel_type,
    int64_t block_num, int64_t warp_per_block) {
  auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);
  handle->LaunchDispatchRecv(
      static_cast<mori::moe::KernelType>(kernel_type),
      static_cast<int>(block_num), static_cast<int>(warp_per_block), stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_dispatch_recv, MoriEpDispatchRecvImpl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("kernel_type")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("warp_per_block"));

// ---------------------------------------------------------------------------
// mori_ep_combine_recv
// ---------------------------------------------------------------------------
static ffi::Error MoriEpCombineRecvImpl(
    int64_t handle_id, int64_t kernel_type,
    int64_t block_num, int64_t warp_per_block) {
  auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);
  handle->LaunchCombineRecv(
      static_cast<mori::moe::KernelType>(kernel_type),
      static_cast<int>(block_num), static_cast<int>(warp_per_block), stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_combine_recv, MoriEpCombineRecvImpl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("kernel_type")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("warp_per_block"));

// ---------------------------------------------------------------------------
// mori_ep_reset
// ---------------------------------------------------------------------------
static ffi::Error MoriEpResetImpl(int64_t handle_id) {
  auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);
  handle->LaunchReset(stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_reset, MoriEpResetImpl,
    ffi::Ffi::Bind().Attr<int64_t>("handle_id"));

// ---------------------------------------------------------------------------
// Handle lifecycle (exported as plain C symbols for ctypes)
// ---------------------------------------------------------------------------
extern "C" {

int64_t mori_ffi_create_handle(
    int rank, int world_size, int hidden_dim,
    int scale_dim, int scale_type_size,
    int max_token_type_size, int max_num_inp_token_per_rank,
    int num_experts_per_rank, int num_experts_per_token,
    int warp_num_per_block, int block_num,
    int kernel_type, int gpu_per_node,
    int rdma_block_num, int num_qp_per_pe) {
  mori::moe::EpDispatchCombineConfig config;
  config.rank = rank;
  config.worldSize = world_size;
  config.hiddenDim = hidden_dim;
  config.scaleDim = scale_dim;
  config.scaleTypeSize = scale_type_size;
  config.maxTokenTypeSize = max_token_type_size;
  config.maxNumInpTokenPerRank = max_num_inp_token_per_rank;
  config.numExpertPerRank = num_experts_per_rank;
  config.numExpertPerToken = num_experts_per_token;
  config.warpNumPerBlock = warp_num_per_block;
  config.blockNum = block_num;
  config.kernelType = static_cast<mori::moe::KernelType>(kernel_type);
  config.gpuPerNode = gpu_per_node;
  config.rdmaBlockNum = rdma_block_num;
  config.numQpPerPe = num_qp_per_pe;
  return mori::ffi::HandleManager::Instance().CreateHandle(config);
}

void mori_ffi_destroy_handle(int64_t handle_id) {
  mori::ffi::HandleManager::Instance().DestroyHandle(handle_id);
}

}  // extern "C"
